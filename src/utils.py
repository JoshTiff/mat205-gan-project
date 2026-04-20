# Functions that may be used across multiple files

from pathlib import Path
import random
import numpy as np
from PIL import ImageOps
import torch
import yaml

# Set a random seed for all random functions for reproducibility
def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# Gets a device for training
def get_device() -> torch.device:
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create directory if it does not already exist
def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents = True, exist_ok = True)

# Save a model's state to disk
def save_checkpoint(state: dict, path: str | Path) -> None:
    ensure_dir(Path(path).parent)
    torch.save(state, path)

# Load arguments from YAML config file
def load_config(config_path: str | Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    return config

# Transformation to add zero padding to non square images.
class PadToSquare:
    def __call__(self, img):
        width, height = img.size
        max_side = max(width, height)

        pad_left = (max_side - width) // 2
        pad_right = max_side - width - pad_left
        pad_top = (max_side - height) // 2
        pad_bottom = max_side - height - pad_top

        return ImageOps.expand(
            img,
            border=(pad_left, pad_top, pad_right, pad_bottom),
            fill=0
        )