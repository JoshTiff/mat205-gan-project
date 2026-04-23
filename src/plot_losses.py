# Plot generator and discriminator loss from a training checkpoint

from pathlib import Path
import argparse
import torch
import matplotlib.pyplot as plt

# Parse command line arguments
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot generator and discriminator loss from a saved checkpoint."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional output path for the plot image",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the plot window in addition to saving",
    )
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    generator_loss = checkpoint.get("generator_loss_history")
    discriminator_loss = checkpoint.get("discriminator_loss_history")

    if generator_loss is None or discriminator_loss is None:
        raise KeyError(
            "Checkpoint does not contain 'generator_loss_history' and "
            "'discriminator_loss_history'."
        )

    if len(generator_loss) != len(discriminator_loss):
        raise ValueError(
            "Generator and discriminator loss histories have different lengths."
        )

    config = checkpoint.get("config", {})
    model_name = config.get("model_name", "unknown_model")
    image_size = config.get("image_size", "unknown_size")

    pretty_model_name = model_name.replace("_", " ").upper()
    title = f"{pretty_model_name} ({image_size}x{image_size}) Training Losses"

    epochs = list(range(1, len(generator_loss) + 1))

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, generator_loss, label="Generator Loss")
    plt.plot(epochs, discriminator_loss, label="Discriminator Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.tight_layout()

    if args.output is not None:
        output_path = Path(args.output)
    else:
        output_path = checkpoint_path.with_name(checkpoint_path.stem + "_loss_plot.png")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"Saved plot to: {output_path}")

    if args.show:
        plt.show()
    else:
        plt.close()

if __name__ == "__main__":
    main()