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
        
    # Add other models here when they are implemented
    
    else:
        raise ValueError(f"Unsupported model name: {model_name}")
    
    return generator, discriminator