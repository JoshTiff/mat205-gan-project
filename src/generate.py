# Generate images from a trained GAN generator

from pathlib import Path
import argparse
import math
import torch
from torchvision.utils import save_image
import yaml
from models import build_models
from utils import set_random_seed, get_device, ensure_dir

# Parse command line arguments
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to generator checkpoint file",
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=None,
        help="Number of images to generate",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save generated images",
    )
    return parser.parse_args()

# Load arguments from YAML config file
def load_config(config_path: str | Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    return config

# Save generated images as a single grid
def save_generated_grid(
    fake_images: torch.Tensor,
    output_path: str | Path,
) -> None:
    nrow = int(math.sqrt(fake_images.size(0)))
    if nrow * nrow != fake_images.size(0):
        nrow = max(1, fake_images.size(0) // 2)

    save_image(
        fake_images,
        output_path,
        nrow=nrow,
        normalize=True,
        value_range=(-1, 1),
    )

# Save generated images individually
def save_individual_images(
    fake_images: torch.Tensor,
    output_dir: str | Path,
) -> None:
    output_dir = Path(output_dir)
    ensure_dir(output_dir)

    for index, image in enumerate(fake_images):
        save_image(
            image,
            output_dir / f"generated_{index:03d}.png",
            normalize=True,
            value_range=(-1, 1),
        )

# Generate images
def generate() -> None:
    args = parse_args()
    config = load_config(args.config)

    set_random_seed(config["random_seed"])
    device = get_device()

    num_images = args.num_images or config["num_sample_images"]

    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(config["output_dir"]) / "generated"

    ensure_dir(output_dir)

    # Build generator
    generator, _ = build_models(
        latent_dim=config["latent_dim"],
        image_channels=config["image_channels"],
        image_size=config["image_size"],
        model_name=config["model_name"],
    )
    generator = generator.to(device)

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    generator.load_state_dict(checkpoint["generator_state_dict"])
    generator.eval()

    # Sample noise and generate images
    noise = torch.randn(num_images, config["latent_dim"], device=device)

    with torch.no_grad():
        fake_images = generator(noise).cpu()

    # Save outputs
    save_generated_grid(fake_images, output_dir / "generated_grid.png")
    save_individual_images(fake_images, output_dir / "individual")

    print(f"Generated {num_images} images.")
    print(f"Saved grid to: {output_dir / 'generated_grid.png'}")
    print(f"Saved individual images to: {output_dir / 'individual'}")

if __name__ == "__main__":
    generate()