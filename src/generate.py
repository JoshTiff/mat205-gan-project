# Generate images from a trained GAN generator

from pathlib import Path
import argparse
import math
import torch
from torchvision.utils import save_image
from models import build_models
from utils import set_random_seed, get_device, ensure_dir, load_config

# Parse command line arguments
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint file")
    parser.add_argument("--num-images", type=int, default=None, help="Number of images to generate")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to save generated images")
    parser.add_argument("--batch-size", type=int, default=128, help="Generation batch size")
    parser.add_argument(
        "--save-grid-count",
        type=int,
        default=64,
        help="How many generated images to include in preview grid",
    )
    return parser.parse_args()

# Save a grid of images
def save_generated_grid(fake_images: torch.Tensor, output_path: str | Path) -> None:
    nrow = int(math.sqrt(fake_images.size(0)))
    if nrow * nrow != fake_images.size(0):
        nrow = max(1, int(math.sqrt(fake_images.size(0))))

    save_image(
        fake_images,
        output_path,
        nrow=nrow,
        normalize=True,
        value_range=(-1, 1),
    )

# Save individual images
def save_individual_images(
    fake_images: torch.Tensor,
    output_dir: str | Path,
    start_index: int,
) -> None:
    output_dir = Path(output_dir)
    ensure_dir(output_dir)

    for i, image in enumerate(fake_images):
        save_image(
            image,
            output_dir / f"generated_{start_index + i:05d}.png",
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

    individual_dir = output_dir / "individual"
    ensure_dir(output_dir)
    ensure_dir(individual_dir)

    generator, _ = build_models(
        latent_dim=config["latent_dim"],
        image_channels=config["image_channels"],
        image_size=config["image_size"],
        model_name=config["model_name"],
    )
    generator = generator.to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    generator.load_state_dict(checkpoint["generator_state_dict"])
    generator.eval()

    saved = 0
    preview_images = []

    with torch.no_grad():
        while saved < num_images:
            current_batch = min(args.batch_size, num_images - saved)
            noise = torch.randn(current_batch, config["latent_dim"], device=device)
            fake_images = generator(noise).cpu()

            if len(preview_images) < args.save_grid_count:
                needed = args.save_grid_count - len(preview_images)
                preview_images.extend(fake_images[:needed])

            save_individual_images(
                fake_images=fake_images,
                output_dir=individual_dir,
                start_index=saved,
            )

            saved += current_batch
            print(f"Saved {saved}/{num_images} images")

    if len(preview_images) > 0:
        preview_tensor = torch.stack(preview_images)
        save_generated_grid(preview_tensor, output_dir / "generated_grid.png")

    print(f"\nGenerated {num_images} images.")
    print(f"Saved individual images to: {individual_dir}")
    print(f"Saved preview grid to: {output_dir / 'generated_grid.png'}")

if __name__ == "__main__":
    generate()