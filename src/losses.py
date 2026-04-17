# Loss functions for various GAN implementations

import torch
import torch.nn as nn
from collections.abc import Callable
from typing import Optional

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
    
# Adversial loss for WGAN-GP
def wgan_gp_adversarial_loss() -> None:
    return None

# Gradient penalty for WGAN-GP
def compute_gradient_penalty(
    critic: nn.Module,
    real_images: torch.Tensor,
    fake_images: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    batch_size = real_images.size(0)

    alpha = torch.rand(batch_size, 1, 1, 1, device=device)
    interpolated = alpha * real_images + (1 - alpha) * fake_images.detach()
    interpolated.requires_grad_(True)

    critic_scores = critic(interpolated)

    gradients = torch.autograd.grad(
        outputs=critic_scores,
        inputs=interpolated,
        grad_outputs=torch.ones_like(critic_scores),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    gradients = gradients.view(batch_size, -1)
    gradient_norm = gradients.norm(2, dim=1)
    penalty = ((gradient_norm - 1) ** 2).mean()

    return penalty

# Discriminator loss for WGAN-GP
def wgan_gp_discriminator_loss(
    discriminator: nn.Module,
    real_images: torch.Tensor,
    fake_images: torch.Tensor,
    adversarial_loss: Optional[nn.Module],
    device: torch.device,
    lambda_gp: float = 10.0,
) -> torch.Tensor:
    del adversarial_loss  # unused for WGAN-GP

    real_scores = discriminator(real_images)
    fake_scores = discriminator(fake_images.detach())

    wasserstein_term = fake_scores.mean() - real_scores.mean()
    gp = compute_gradient_penalty(
        critic=discriminator,
        real_images=real_images,
        fake_images=fake_images,
        device=device,
    )

    return wasserstein_term + lambda_gp * gp

# Generator loss for WGAN-GP
def wgan_gp_generator_loss(
    discriminator: nn.Module,
    fake_images: torch.Tensor,
    adversarial_loss: Optional[nn.Module],
    device: torch.device,
) -> torch.Tensor:
    del adversarial_loss
    del device

    fake_scores = discriminator(fake_images)
    return -fake_scores.mean()

# Adversarial loss for LSGAN
def lsgan_adversarial_loss() -> nn.Module:
    return nn.MSELoss()

# Discriminator loss for LSGAN
def lsgan_discriminator_loss(
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

    real_loss = 0.5 * adversarial_loss(real_preds, real_labels)
    fake_loss = 0.5 * adversarial_loss(fake_preds, fake_labels)

    return real_loss + fake_loss

# Generator loss for LSGAN
def lsgan_generator_loss(
    discriminator: nn.Module,
    fake_images: torch.Tensor,
    adversarial_loss: nn.Module,
    device: torch.device,
) -> torch.Tensor:
    batch_size = fake_images.size(0)

    target_labels = torch.ones((batch_size, 1), device=device)
    fake_preds = discriminator(fake_images)

    return 0.5 * adversarial_loss(fake_preds, target_labels)

# Select the correct loss functions for a given GAN type
def get_loss_functions(
    model_name: str,
) -> tuple[Optional[nn.Module], Callable, Callable]:
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
        
    elif model_name == "wgan_gp":
        adversarial_loss = wgan_gp_adversarial_loss()
        return (
            adversarial_loss,
            wgan_gp_discriminator_loss,
            wgan_gp_generator_loss,
        )

    elif model_name == "wgan_gp":
        adversarial_loss = wgan_gp_adversarial_loss()
        return (
            adversarial_loss,
            wgan_gp_discriminator_loss,
            wgan_gp_generator_loss,
        )

    raise ValueError(f"Unsupported model name: {model_name}")