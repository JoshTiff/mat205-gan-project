# Custom dataset and dataloader for abstract art data

from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Dataset
class AbstractArtDataset(Dataset):
    # Constructor
    def __init__(self, root_dir: str | Path, image_size: int) -> None:
        self.root_dir = Path(root_dir)

        # Find image paths
        self.image_paths = sorted([
            path for path in self.root_dir.rglob("*")
            if path.suffix.lower() == ".jpg"
        ])
        
        # Validate that images were found
        if not self.image_paths:
            raise ValueError(f"No images found in {self.root_dir}")

        # Standardize images for training
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

    # Get number of samples in dataset
    def __len__(self) -> int:
        return len(self.image_paths)

    # Retrieve item from dataset
    def __getitem__(self, idx: int) -> torch.Tensor:
        image_path = self.image_paths[idx]

        with Image.open(image_path) as img:
            img = img.convert("RGB")
            img = self.transform(img)

        return img

# Create dataloader
def get_dataloader(
    root_dir: str | Path,
    image_size: int,
    batch_size: int = 64,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    
    dataset = AbstractArtDataset(root_dir=root_dir, image_size=image_size)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )

    return dataloader