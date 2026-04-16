# Generators and discriminators for various GAN implementations

import torch
import torch.nn as nn

# Vanilla generator model
class VanillaGenerator(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        image_channels: int,
        image_size: int,
    ) -> None:
        super().__init__()
        
        self.image_channels = image_channels
        self.image_size = image_size
        self.output_dim = image_channels * image_size * image_size
        
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2),
            
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            
            nn.Linear(1024, self.output_dim),
            nn.Tanh(),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model(x)
        x = x.view(-1, self.image_channels, self.image_size, self.image_size)
        return x
    
# Vanilla discriminator model
class VanillaDiscriminator(nn.Module):
    def __init__(
        self,
        image_channels: int,
        image_size: int,
    ) -> None:
        super().__init__()
        
        self.input_dim = image_channels * image_size * image_size
        
        self.model = nn.Sequential(
            nn.Flatten(),

            nn.Linear(self.input_dim, 1024),
            nn.LeakyReLU(0.2),

            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),

            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),

            nn.Linear(256, 1),
            nn.Sigmoid(),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
    
# Generator for DCGAN
class DCGenerator(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        image_channels: int,
        image_size: int,
        feature_maps: int = 64,
    ) -> None:
        super().__init__()

        if image_size != 128:
            raise ValueError("This DCGAN generator currently supports image_size=128 only.")

        self.latent_dim = latent_dim

        self.model = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, feature_maps * 16, 4, 1, 0, bias=False),
            nn.BatchNorm2d(feature_maps * 16),
            nn.ReLU(),

            nn.ConvTranspose2d(feature_maps * 16, feature_maps * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 8),
            nn.ReLU(),

            nn.ConvTranspose2d(feature_maps * 8, feature_maps * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 4),
            nn.ReLU(),

            nn.ConvTranspose2d(feature_maps * 4, feature_maps * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 2),
            nn.ReLU(),

            nn.ConvTranspose2d(feature_maps * 2, feature_maps, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps),
            nn.ReLU(),

            nn.ConvTranspose2d(feature_maps, image_channels, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.view(x.size(0), x.size(1), 1, 1)
        return self.model(x)

# Discriminator for DCGAN
class DCDiscriminator(nn.Module):
    def __init__(
        self,
        image_channels: int,
        image_size: int,
        feature_maps: int = 64,
    ) -> None:
        super().__init__()

        if image_size != 128:
            raise ValueError("This DCGAN discriminator currently supports image_size=128 only.")

        self.model = nn.Sequential(
            nn.Conv2d(image_channels, feature_maps, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2),

            nn.Conv2d(feature_maps, feature_maps * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 2),
            nn.LeakyReLU(0.2),

            nn.Conv2d(feature_maps * 2, feature_maps * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 4),
            nn.LeakyReLU(0.2),

            nn.Conv2d(feature_maps * 4, feature_maps * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 8),
            nn.LeakyReLU(0.2),

            nn.Conv2d(feature_maps * 8, feature_maps * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 16),
            nn.LeakyReLU(0.2),

            nn.Conv2d(feature_maps * 16, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model(x)
        return x.view(-1, 1)
    
def build_models(
    latent_dim: int,
    image_channels: int,
    image_size: int,
    model_name: str,
) -> tuple[nn.Module, nn.Module]:
    if model_name == "vanilla":
        generator = VanillaGenerator(
            latent_dim=latent_dim,
            image_channels=image_channels,
            image_size=image_size,
        )
        discriminator = VanillaDiscriminator(
            image_channels=image_channels,
            image_size=image_size,
        )
        
    elif model_name == "dcgan":
        generator = DCGenerator(
            latent_dim=latent_dim,
            image_channels=image_channels,
            image_size=image_size,
        )
        discriminator = DCDiscriminator(
            image_channels=image_channels,
            image_size=image_size,
        )
        
    # Add other models here when they are implemented
    
    else:
        raise ValueError(f"Unsupported model name: {model_name}")
    
    return generator, discriminator