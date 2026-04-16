# Loss functions for various GAN implementations

import torch
import torch.nn as nn
from collections.abc import Callable

# Adversarial loss for vanilla GAN
def vanilla_adversarial_loss() -> nn.Module:
    return nn.BCELoss()

# Discriminator loss for vanilla GAN
def vanilla_discriminator_loss(
    discriminator: nn.Module,
    real_images: torch.Tensor,
    fake_images: torch.Tensor,
    adversarial_loss: nn.Module,
    device: torch.device,
) -> torch.Tensor:
    batch_size = real_images.size(0)

    real_labels = torch.ones((batch_size, 1), device=device)
    fake_labels = torch.zeros((batch_size, 1), device=device)

    real_preds = discriminator(real_images)
    fake_preds = discriminator(fake_images.detach())

    real_loss = adversarial_loss(real_preds, real_labels)
    fake_loss = adversarial_loss(fake_preds, fake_labels)

    return real_loss + fake_loss

# Generator loss for vanilla GAN
def vanilla_generator_loss(
    discriminator: nn.Module,
    fake_images: torch.Tensor,
    adversarial_loss: nn.Module,
    device: torch.device,
) -> torch.Tensor:
    batch_size = fake_images.size(0)

    target_labels = torch.ones((batch_size, 1), device=device)
    fake_preds = discriminator(fake_images)

    return adversarial_loss(fake_preds, target_labels)

# Adversarial loss for DCGAN
def dcgan_adversarial_loss() -> nn.Module:
    return nn.BCELoss()

# Discriminator loss for DCGAN
def dcgan_discriminator_loss(
    discriminator: nn.Module,
    real_images: torch.Tensor,
    fake_images: torch.Tensor,
    adversarial_loss: nn.Module,
    device: torch.device,
) -> torch.Tensor:
    return vanilla_discriminator_loss(
        discriminator=discriminator,
        real_images=real_images,
        fake_images=fake_images,
        adversarial_loss=adversarial_loss,
        device=device,
    )

# Generator loss for DCGAN
def dcgan_generator_loss(
    discriminator: nn.Module,
    fake_images: torch.Tensor,
    adversarial_loss: nn.Module,
    device: torch.device,
) -> torch.Tensor:
    return vanilla_generator_loss(
        discriminator=discriminator,
        fake_images=fake_images,
        adversarial_loss=adversarial_loss,
        device=device,
    )

# Select the correct loss functions for a given GAN type
def get_loss_functions(
    model_name: str,
) -> tuple[nn.Module, Callable, Callable]:
    if model_name == "vanilla":
        adversarial_loss = vanilla_adversarial_loss()
        return (
            adversarial_loss,
            vanilla_discriminator_loss,
            vanilla_generator_loss,
        )
            
    elif model_name == "dcgan":
        adversarial_loss = dcgan_adversarial_loss()
        return (
            adversarial_loss,
            dcgan_discriminator_loss,
            dcgan_generator_loss,
        )

    # Add other loss functions here when they are implemented

    raise ValueError(f"Unsupported model name: {model_name}")