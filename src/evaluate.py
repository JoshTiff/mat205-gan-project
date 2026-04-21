# Evaluate a trained model

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import Inception_V3_Weights, inception_v3
from torch_fidelity import calculate_metrics
from tqdm import tqdm

from utils import PadToSquare, ensure_dir, get_device

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Class to create image dataset
class RecursiveImageDataset(Dataset):
    def __init__(self, root_dir: str | Path, image_size: int) -> None:
        self.root_dir = Path(root_dir)
        self.image_paths = sorted(
            path for path in self.root_dir.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not self.image_paths:
            raise ValueError(f"No images found recursively under {self.root_dir}")

        self.transform = transforms.Compose(
            [
                PadToSquare(),
                transforms.Resize((image_size, image_size)),
            ]
        )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[Path, Image.Image]:
        image_path = self.image_paths[idx]
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            img = self.transform(img)
        return image_path, img

# Class to create image features dataset
class FeatureImageDataset(Dataset):
    def __init__(self, image_paths: list[Path], weights: Inception_V3_Weights) -> None:
        self.image_paths = image_paths
        self.normalize = transforms.Normalize(
            mean=weights.meta["mean"] if "mean" in weights.meta else (0.485, 0.456, 0.406),
            std=weights.meta["std"] if "std" in weights.meta else (0.229, 0.224, 0.225),
        )
        self.resize = transforms.Resize((299, 299))
        self.to_tensor = transforms.ToTensor()

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[str, torch.Tensor]:
        path = self.image_paths[idx]
        with Image.open(path) as img:
            img = img.convert("RGB")
            img = self.resize(img)
            tensor = self.to_tensor(img)
            tensor = self.normalize(tensor)
        return str(path), tensor

# Parse command lien arguments
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate generated images with FID/KID and nearest-neighbor matches."
    )
    parser.add_argument("--real-dir", type=str, required=True, help="Root folder of training/real images")
    parser.add_argument(
        "--fake-dir",
        type=str,
        required=True,
        help="Folder of generated images. Prefer the individual-image folder, not a grid-image folder.",
    )
    parser.add_argument("--image-size", type=int, required=True, help="Training image_size used by the GAN")
    parser.add_argument(
        "--metric",
        type=str,
        default="both",
        choices=["fid", "kid", "both"],
        help="Which metric(s) to compute",
    )
    parser.add_argument(
        "--kid-subset-size",
        type=int,
        default=1000,
        help="Subset size for KID estimation",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory for processed real images, metrics JSON, and side-by-side matches",
    )
    parser.add_argument(
        "--nn-subset",
        type=int,
        default=50,
        help="How many generated images to compare against the training set for nearest-neighbor search",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for feature extraction",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Dataloader workers for preprocessing and feature extraction",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device for feature extraction and metric computation: auto, cuda, or cpu",
    )
    parser.add_argument(
        "--reuse-processed",
        action="store_true",
        help="Reuse previously preprocessed real-image folders instead of rebuilding them",
    )
    return parser.parse_args()

# Find all image paths
def collect_paths(root_dir: str | Path) -> list[Path]:
    root = Path(root_dir)
    return sorted(path for path in root.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS)

# Reset directory if it contains data from an old run
def ensure_clean_dir(path: Path, reuse: bool) -> None:
    if path.exists() and not reuse:
        shutil.rmtree(path)
    ensure_dir(path)

# Recursively search folder for images
def preprocess_folder_recursive(
    input_dir: str | Path,
    output_dir: str | Path,
    image_size: int,
    reuse_processed: bool,
) -> list[Path]:
    output_dir = Path(output_dir)
    ensure_clean_dir(output_dir, reuse_processed)

    if reuse_processed:
        existing = collect_paths(output_dir)
        if existing:
            return existing

    dataset = RecursiveImageDataset(root_dir=input_dir, image_size=image_size)

    saved_paths: list[Path] = []
    to_tensor = transforms.ToTensor()

    for idx in tqdm(range(len(dataset)), desc=f"Preprocessing {Path(input_dir).name}"):
        _, processed_image = dataset[idx]
        tensor = to_tensor(processed_image)
        save_path = output_dir / f"img_{idx:06d}.png"
        transforms.functional.to_pil_image(tensor).save(save_path)
        saved_paths.append(save_path)

    return saved_paths

# Model to extract image features
class InceptionFeatureExtractor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        weights = Inception_V3_Weights.DEFAULT
        model = inception_v3(weights=weights)
        model.fc = nn.Identity()
        self.model = model
        self.weights = weights

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.model(x)
        if features.ndim == 1:
            features = features.unsqueeze(0)
        return features

# Extract feature vector from an image
def extract_features(
    image_paths: list[Path],
    device: torch.device,
    batch_size: int,
    workers: int,
) -> tuple[list[Path], torch.Tensor]:
    extractor = InceptionFeatureExtractor().to(device)
    extractor.eval()

    dataset = FeatureImageDataset(image_paths=image_paths, weights=extractor.weights)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers)

    ordered_paths: list[Path] = []
    feature_batches: list[torch.Tensor] = []

    with torch.no_grad():
        for paths, images in tqdm(loader, desc="Extracting features"):
            images = images.to(device, non_blocking=True)
            features = extractor(images)
            features = F.normalize(features, dim=1)
            feature_batches.append(features.cpu())
            ordered_paths.extend(Path(p) for p in paths)

    return ordered_paths, torch.cat(feature_batches, dim=0)

# Save two images next to each other
def save_side_by_side(
    left_path: Path,
    right_path: Path,
    save_path: Path,
) -> None:
    with Image.open(left_path) as left, Image.open(right_path) as right:
        left = left.convert("RGB")
        right = right.convert("RGB")

        width = left.width + right.width
        height = max(left.height, right.height)
        canvas = Image.new("RGB", (width, height), color=(255, 255, 255))
        canvas.paste(left, (0, 0))
        canvas.paste(right, (left.width, 0))
        canvas.save(save_path)

# Save most similar fake and real image next to each other
def build_nearest_neighbor_gallery(
    fake_paths: list[Path],
    real_paths: list[Path],
    subset_count: int,
    output_dir: Path,
    device: torch.device,
    batch_size: int,
    workers: int,
) -> list[dict[str, str | float]]:
    subset_fake_paths = fake_paths[: min(subset_count, len(fake_paths))]
    if not subset_fake_paths:
        return []

    ordered_real_paths, real_features = extract_features(real_paths, device, batch_size, workers)
    ordered_fake_paths, fake_features = extract_features(subset_fake_paths, device, batch_size, workers)

    similarities = fake_features @ real_features.T
    ensure_dir(output_dir)

    matches: list[dict[str, str | float]] = []
    for fake_idx, fake_path in enumerate(ordered_fake_paths):
        best_real_idx = int(similarities[fake_idx].argmax().item())
        best_score = float(similarities[fake_idx, best_real_idx].item())
        real_path = ordered_real_paths[best_real_idx]

        save_path = output_dir / f"match_{fake_idx:04d}.png"
        save_side_by_side(fake_path, real_path, save_path)

        matches.append(
            {
                "fake_image": str(fake_path),
                "nearest_real_image": str(real_path),
                "cosine_similarity": best_score,
                "side_by_side": str(save_path),
            }
        )

    return matches

def main() -> None:
    args = parse_args()

    if args.device == "auto":
        device = get_device()
    else:
        device = torch.device(args.device)

    output_dir = Path(args.output_dir)
    processed_real_dir = output_dir / f"processed_real_{args.image_size}"
    nn_dir = output_dir / "nearest_neighbors"
    ensure_dir(output_dir)

    print("Step 1/2: Preprocessing real images with the same square-pad + resize logic used in training...")
    real_processed_paths = preprocess_folder_recursive(
        input_dir=args.real_dir,
        output_dir=processed_real_dir,
        image_size=args.image_size,
        reuse_processed=args.reuse_processed,
    )

    print("Collecting generated images recursively (no fake-image preprocessing)...")
    fake_paths = collect_paths(args.fake_dir)
    if not fake_paths:
        raise ValueError(f"No generated images found recursively under {args.fake_dir}")

    compute_fid = args.metric in {"fid", "both"}
    compute_kid = args.metric in {"kid", "both"}
    kid_subset_size = min(args.kid_subset_size, len(real_processed_paths), len(fake_paths))

    print("Step 2/2: Computing metrics...")
    metrics = calculate_metrics(
        input1=str(processed_real_dir),
        input2=str(args.fake_dir),
        cuda=device.type == "cuda",
        fid=compute_fid,
        kid=compute_kid,
        kid_subset_size=kid_subset_size,
        verbose=False,
    )

    summary: dict[str, object] = {
        "recommended_primary_metric": "FID",
        "image_size": args.image_size,
        "real_count": len(real_processed_paths),
        "fake_count": len(fake_paths),
        "fake_preprocessing_skipped": True,
    }

    if compute_fid:
        summary["FID"] = float(metrics["frechet_inception_distance"])
    if compute_kid:
        summary["KID_mean"] = float(metrics["kernel_inception_distance_mean"])
        summary["KID_std"] = float(metrics["kernel_inception_distance_std"])
        summary["kid_subset_size"] = kid_subset_size

    print(json.dumps(summary, indent=2))

    print("Building nearest-neighbor side-by-side comparisons...")
    matches = build_nearest_neighbor_gallery(
        fake_paths=fake_paths,
        real_paths=real_processed_paths,
        subset_count=args.nn_subset,
        output_dir=nn_dir,
        device=device,
        batch_size=args.batch_size,
        workers=args.workers,
    )

    summary["nearest_neighbor_subset"] = len(matches)
    summary["nearest_neighbor_results"] = matches

    metrics_path = output_dir / "metrics_summary.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved metrics summary to: {metrics_path}")
    print(f"Saved nearest-neighbor comparisons to: {nn_dir}")

if __name__ == "__main__":
    main()
