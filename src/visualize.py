# Visualize GAN training results

from pathlib import Path
import argparse
import math
import re
import torch
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from utils import ensure_dir

# Parse command line arguments
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint file (for example: latest.pt)",
    )
    parser.add_argument(
        "--sample-dir",
        type=str,
        default=None,
        help="Directory containing saved sample grids (for example: outputs/.../samples)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save visualization outputs",
    )
    return parser.parse_args()

# Plot generator and discriminator loss curves
def plot_loss_curves(
    generator_losses: list[float],
    discriminator_losses: list[float],
    output_path: str | Path,
) -> None:
    epochs = list(range(1, len(generator_losses) + 1))

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, generator_losses, label="Generator Loss")
    plt.plot(epochs, discriminator_losses, label="Discriminator Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("GAN Training Loss Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

# Get epoch number from a filename like epoch_005.png
def extract_epoch_number(path: Path) -> int:
    match = re.search(r"epoch_(\d+)", path.stem)
    if match is None:
        return -1
    return int(match.group(1))

# Select a subset of sample images across training
def select_sample_images(sample_dir: str | Path, max_images: int = 6) -> list[Path]:
    sample_dir = Path(sample_dir)
    image_paths = sorted(sample_dir.glob("epoch_*.png"), key=extract_epoch_number)

    if len(image_paths) <= max_images:
        return image_paths

    selected = []
    for i in range(max_images):
        index = round(i * (len(image_paths) - 1) / (max_images - 1))
        selected.append(image_paths[index])

    return selected

# Build a horizontal comparison strip of saved sample grids
def build_sample_strip(
    image_paths: list[Path],
    output_path: str | Path,
) -> None:
    if not image_paths:
        return

    images = [Image.open(path).convert("RGB") for path in image_paths]

    tile_width = images[0].width
    tile_height = images[0].height
    label_height = 30

    canvas_width = tile_width * len(images)
    canvas_height = tile_height + label_height

    canvas = Image.new("RGB", (canvas_width, canvas_height), color="white")
    draw = ImageDraw.Draw(canvas)

    for i, (path, image) in enumerate(zip(image_paths, images)):
        x_offset = i * tile_width
        canvas.paste(image, (x_offset, label_height))

        epoch_num = extract_epoch_number(path)
        label = f"Epoch {epoch_num}"
        draw.text((x_offset + 10, 8), label, fill="black")

    canvas.save(output_path)

    for image in images:
        image.close()

# Create visuals
def visualize() -> None:
    args = parse_args()

    checkpoint_path = Path(args.checkpoint)

    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
    else:
        output_dir = checkpoint_path.parent.parent / "visualizations"

    ensure_dir(output_dir)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    generator_losses = checkpoint["generator_loss_history"]
    discriminator_losses = checkpoint["discriminator_loss_history"]

    plot_loss_curves(
        generator_losses=generator_losses,
        discriminator_losses=discriminator_losses,
        output_path=output_dir / "loss_curves.png",
    )

    print(f"Saved loss curves to: {output_dir / 'loss_curves.png'}")

    if args.sample_dir is not None:
        sample_images = select_sample_images(args.sample_dir, max_images=6)

        if sample_images:
            build_sample_strip(
                image_paths=sample_images,
                output_path=output_dir / "sample_progression.png",
            )
            print(f"Saved sample progression to: {output_dir / 'sample_progression.png'}")
        else:
            print("No sample images found to build progression strip.")

if __name__ == "__main__":
    visualize()