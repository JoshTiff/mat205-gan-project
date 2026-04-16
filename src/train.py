# Train GAN models using config file

from pathlib import Path
import argparse
import math
import torch
import torch.optim as optim
from torchvision.utils import save_image
import yaml
from dataset import get_dataloader
from models import build_models
from losses import get_loss_functions
from utils import set_random_seed, get_device, ensure_dir, save_checkpoint, load_config

# Parse command line arguments
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file",
    )
    return parser.parse_args()

# Save a snapshot of images with fixed noise over time
def save_generated_samples(
    generator: torch.nn.Module,
    fixed_noise: torch.Tensor,
    sample_path: str | Path,
) -> None:
    generator.eval()

    with torch.no_grad():
        fake_images = generator(fixed_noise).cpu()

    nrow = int(math.sqrt(fake_images.size(0)))
    if nrow * nrow != fake_images.size(0):
        nrow = max(1, fake_images.size(0) // 2)

    save_image(
        fake_images,
        sample_path,
        nrow=nrow,
        normalize=True,
        value_range=(-1, 1),
    )

    generator.train()

# Train model
def train() -> None:
    args = parse_args()
    config = load_config(args.config)

    set_random_seed(config["random_seed"])
    device = get_device()

    output_dir = Path(config["output_dir"])
    sample_dir = output_dir / "samples"
    checkpoint_dir = output_dir / "checkpoints"

    ensure_dir(output_dir)
    ensure_dir(sample_dir)
    ensure_dir(checkpoint_dir)

    with open(output_dir / "used_config.yaml", "w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, sort_keys=False)

    # Set up dataloader
    dataloader = get_dataloader(
        root_dir=config["data_dir"],
        image_size=config["image_size"],
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
    )

    # Set up models
    generator, discriminator = build_models(
        latent_dim=config["latent_dim"],
        image_channels=config["image_channels"],
        image_size=config["image_size"],
        model_name=config["model_name"],
    )

    generator = generator.to(device)
    discriminator = discriminator.to(device)

    adversarial_loss, discriminator_loss_fn, generator_loss_fn = get_loss_functions(
        config["model_name"]
    )
    adversarial_loss = adversarial_loss.to(device)

    g_optimizer = optim.Adam(
        generator.parameters(),
        lr=config["learning_rate"],
        betas=(config["beta1"], config["beta2"]),
    )

    d_optimizer = optim.Adam(
        discriminator.parameters(),
        lr=config["learning_rate"],
        betas=(config["beta1"], config["beta2"]),
    )

    fixed_noise = torch.randn(
        config["num_sample_images"],
        config["latent_dim"],
        device=device,
    )

    history = {
        "generator_loss": [],
        "discriminator_loss": [],
    }

    print(f"Training on device: {device}")
    print(f"Number of training images: {len(dataloader.dataset)}")
    print(f"Steps per epoch: {len(dataloader)}")

    for epoch in range(1, config["num_epochs"] + 1):
        running_g_loss = 0.0
        running_d_loss = 0.0

        for real_images in dataloader:
            real_images = real_images.to(device, non_blocking=True)
            batch_size = real_images.size(0)

            # Train discriminator
            d_optimizer.zero_grad()

            noise = torch.randn(batch_size, config["latent_dim"], device=device)
            fake_images = generator(noise)

            d_loss = discriminator_loss_fn(
                discriminator=discriminator,
                real_images=real_images,
                fake_images=fake_images,
                adversarial_loss=adversarial_loss,
                device=device,
            )

            d_loss.backward()
            d_optimizer.step()

            # Train generator
            g_optimizer.zero_grad()

            noise = torch.randn(batch_size, config["latent_dim"], device=device)
            fake_images = generator(noise)

            g_loss = generator_loss_fn(
                discriminator=discriminator,
                fake_images=fake_images,
                adversarial_loss=adversarial_loss,
                device=device,
            )

            g_loss.backward()
            g_optimizer.step()

            running_d_loss += d_loss.item()
            running_g_loss += g_loss.item()

        # Evaluate losses
        epoch_d_loss = running_d_loss / len(dataloader)
        epoch_g_loss = running_g_loss / len(dataloader)

        history["discriminator_loss"].append(epoch_d_loss)
        history["generator_loss"].append(epoch_g_loss)

        print(
            f"Epoch [{epoch}/{config['num_epochs']}] | "
            f"D Loss: {epoch_d_loss:.4f} | "
            f"G Loss: {epoch_g_loss:.4f}"
        )

        if epoch % config["sample_every"] == 0 or epoch == 1:
            save_generated_samples(
                generator=generator,
                fixed_noise=fixed_noise,
                sample_path=sample_dir / f"epoch_{epoch:03d}.png",
            )

        if epoch % config["checkpoint_every"] == 0 or epoch == config["num_epochs"]:
            save_checkpoint(
                {
                    "epoch": epoch,
                    "generator_state_dict": generator.state_dict(),
                    "discriminator_state_dict": discriminator.state_dict(),
                    "g_optimizer_state_dict": g_optimizer.state_dict(),
                    "d_optimizer_state_dict": d_optimizer.state_dict(),
                    "generator_loss_history": history["generator_loss"],
                    "discriminator_loss_history": history["discriminator_loss"],
                    "config": config,
                },
                checkpoint_dir / f"epoch_{epoch:03d}.pt",
            )

    save_checkpoint(
        {
            "epoch": config["num_epochs"],
            "generator_state_dict": generator.state_dict(),
            "discriminator_state_dict": discriminator.state_dict(),
            "g_optimizer_state_dict": g_optimizer.state_dict(),
            "d_optimizer_state_dict": d_optimizer.state_dict(),
            "generator_loss_history": history["generator_loss"],
            "discriminator_loss_history": history["discriminator_loss"],
            "config": config,
        },
        checkpoint_dir / "latest.pt",
    )

    print("Training complete.")

if __name__ == "__main__":
    train()