"""
Deep 1D Convolutional Autoencoder for Ultrasonic Signal Denoising

Based on "Ultrasonic signal noise reduction based on convolutional autoencoders for NDT applications"

Architecture:
- Encoder: 5 Conv1d layers with stride=2 downsampling
- Decoder: 5 ConvTranspose1d layers with stride=2 upsampling
- Regularization: BatchNorm + Dropout + LeakyReLU
"""

import math
import torch
import torch.nn as nn
from typing import Tuple


class DeepCAE(nn.Module):
    """
    Deep CAE with 5 encoder/decoder layers.

    Best for complex signal denoising tasks where deeper networks perform better.
    Uses LeakyReLU for better gradient flow.

    Architecture: 5 encoder + 5 decoder layers
    Input: (B, 1, 1000) → Latent: (B, 192, 32) → Output: (B, 1, 1000)
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 32,  # Restore wider channels for capacity
        kernel_size: int = 7,  # Larger kernel for better context
        dropout_rate: float = 0.1,  # Much lower dropout - 0.4 was too aggressive!
    ):
        super().__init__()

        padding = kernel_size // 2
        self.dropout_rate = dropout_rate

        # Encoder: (B, 1, 1000) → (B, 256, 32)
        # Only use dropout in middle layers, not all layers
        self.enc1 = nn.Sequential(
            nn.Conv1d(
                in_channels, base_channels, kernel_size, stride=2, padding=padding
            ),
            nn.BatchNorm1d(base_channels),
            nn.LeakyReLU(0.1, inplace=True),
            # No dropout in first layer - preserve input information
        )  # 1000 → 500

        self.enc2 = nn.Sequential(
            nn.Conv1d(
                base_channels, base_channels * 2, kernel_size, stride=2, padding=padding
            ),
            nn.BatchNorm1d(base_channels * 2),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout_rate),  # Use regular Dropout, not Dropout1d
        )  # 500 → 250

        self.enc3 = nn.Sequential(
            nn.Conv1d(
                base_channels * 2,
                base_channels * 4,
                kernel_size,
                stride=2,
                padding=padding,
            ),
            nn.BatchNorm1d(base_channels * 4),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout_rate),
        )  # 250 → 125

        self.enc4 = nn.Sequential(
            nn.Conv1d(
                base_channels * 4,
                base_channels * 8,
                kernel_size,
                stride=2,
                padding=padding,
            ),
            nn.BatchNorm1d(base_channels * 8),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout_rate),
        )  # 125 → 63

        self.enc5 = nn.Sequential(
            nn.Conv1d(
                base_channels * 8,
                base_channels * 8,
                kernel_size,
                stride=2,
                padding=padding,
            ),
            nn.BatchNorm1d(base_channels * 8),
            nn.LeakyReLU(0.1, inplace=True),
            # No dropout at bottleneck
        )  # 63 → 32

        # Decoder: (B, 256, 32) → (B, 1, 1000)
        self.dec5 = nn.Sequential(
            nn.ConvTranspose1d(
                base_channels * 8,
                base_channels * 8,
                kernel_size,
                stride=2,
                padding=padding,
                output_padding=0,
            ),
            nn.BatchNorm1d(base_channels * 8),
            nn.LeakyReLU(0.1, inplace=True),
            # No dropout right after bottleneck
        )  # 32 → 63

        self.dec4 = nn.Sequential(
            nn.ConvTranspose1d(
                base_channels * 8,
                base_channels * 4,
                kernel_size,
                stride=2,
                padding=padding,
                output_padding=0,
            ),
            nn.BatchNorm1d(base_channels * 4),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout_rate),
        )  # 63 → 125

        self.dec3 = nn.Sequential(
            nn.ConvTranspose1d(
                base_channels * 4,
                base_channels * 2,
                kernel_size,
                stride=2,
                padding=padding,
                output_padding=1,
            ),
            nn.BatchNorm1d(base_channels * 2),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout_rate),
        )  # 125 → 250

        self.dec2 = nn.Sequential(
            nn.ConvTranspose1d(
                base_channels * 2,
                base_channels,
                kernel_size,
                stride=2,
                padding=padding,
                output_padding=1,
            ),
            nn.BatchNorm1d(base_channels),
            nn.LeakyReLU(0.1, inplace=True),
            # No dropout near output
        )  # 250 → 500

        self.dec1 = nn.Sequential(
            nn.ConvTranspose1d(
                base_channels,
                in_channels,
                kernel_size,
                stride=2,
                padding=padding,
                output_padding=1,
            ),
            nn.Tanh(),
        )  # 500 → 1000

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="leaky_relu"
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)

        # Decoder
        d5 = self.dec5(e5)
        d4 = self.dec4(d5)
        d3 = self.dec3(d4)
        d2 = self.dec2(d3)
        d1 = self.dec1(d2)

        return d1


class DeepCAE_PINN(DeepCAE):
    """
    Physics-Informed DeepCAE for ultrasonic signal denoising.

    Inherits the full DeepCAE encoder/decoder architecture and adds
    a time-domain physics residual tailored to single-sensor ultrasonic
    traces. Because the current dataset only contains u(t) at one fixed
    receiver position, spatial derivatives ∂²u/∂x² are not observable.

    Instead of penalizing raw curvature, this model enforces a damped
    narrowband surrogate equation on the denoised trace:

        u_tt + 2ζω₀ u_t + ω₀² u = 0

    where ω₀ = 2πf₀ is the nominal angular frequency of the ultrasonic
    transducer and ζ is a small damping ratio. The residual is computed
    with finite differences, scaled by the true sampling interval dt,
    normalized by ω₀², and softly masked to focus on energetic parts
    of the waveform.

    Usage:
        - forward(x):         Standard inference (same as DeepCAE)
        - physics_forward(x): Returns (denoised, physics_residual) for PINN training
    """

    # Physical constants for ultrasonic NDT
    SAMPLING_RATE: float = 6.25e6  # 6.25 MHz
    DURATION: float = 160e-6  # 160 μs
    NUM_POINTS: int = 1000  # Total data points
    CENTER_FREQUENCY: float = 250e3  # 250 kHz nominal transducer frequency

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 32,
        kernel_size: int = 7,
        dropout_rate: float = 0.1,
        wave_speed: float = 5900.0,  # Speed of sound in steel (m/s)
        center_frequency: float = CENTER_FREQUENCY,
        damping_ratio: float = 0.05,
    ):
        super().__init__(
            in_channels=in_channels,
            base_channels=base_channels,
            kernel_size=kernel_size,
            dropout_rate=dropout_rate,
        )

        if wave_speed <= 0:
            raise ValueError("wave_speed must be positive")
        if center_frequency <= 0:
            raise ValueError("center_frequency must be positive")
        if damping_ratio < 0:
            raise ValueError("damping_ratio must be non-negative")

        # np.linspace(0, DURATION, NUM_POINTS) implies NUM_POINTS-1 intervals.
        self.dt = self.DURATION / (self.NUM_POINTS - 1)
        self.wave_speed = wave_speed
        self.center_frequency = center_frequency
        self.damping_ratio = damping_ratio
        self.omega0 = 2.0 * math.pi * center_frequency
        self.wavenumber = self.omega0 / wave_speed

    def _build_signal_mask(self, u: torch.Tensor) -> torch.Tensor:
        """
        Build a soft activity mask so physics loss focuses on actual echoes.

        A pure PDE residual applied uniformly would over-regularize silent
        segments. The local absolute amplitude provides a simple, fully
        differentiable proxy for where the ultrasonic trace contains energy.
        """
        local_energy = torch.nn.functional.avg_pool1d(
            u.abs(),
            kernel_size=9,
            stride=1,
            padding=4,
        )
        return local_energy / local_energy.amax(dim=-1, keepdim=True).clamp_min(1e-6)

    def compute_physics_residual(self, u: torch.Tensor) -> torch.Tensor:
        """
        Compute a normalized damped-oscillator residual using finite differences.

        Residual definition:
            r = (u_tt + 2ζω₀ u_t + ω₀² u) / ω₀²

        This is not the full spatial wave equation, but it is a better physics
        surrogate than an unscaled curvature penalty for single-point ultrasonic
        traces where only time-domain observations are available.

        Args:
            u: Denoised signal tensor of shape (B, 1, T)

        Returns:
            Physics residual tensor of shape (B, 1, T-2)
        """

        u_center = u[:, :, 1:-1]
        u_t = (u[:, :, 2:] - u[:, :, :-2]) / (2.0 * self.dt)
        u_tt = (u[:, :, 2:] - 2.0 * u_center + u[:, :, :-2]) / (self.dt**2)

        residual = (
            u_tt
            + 2.0 * self.damping_ratio * self.omega0 * u_t
            + (self.omega0**2) * u_center
        ) / (self.omega0**2)

        signal_mask = self._build_signal_mask(u)[:, :, 1:-1]
        return residual * signal_mask

    def compute_wave_equation_residual(self, u: torch.Tensor) -> torch.Tensor:
        """
        Backward-compatible wrapper for older training code.
        """
        return self.compute_physics_residual(u)

    def physics_forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass that returns both denoised output and physics residual.

        Used during PINN training to compute combined loss:
            L = L_data(denoised, clean) + λ · L_physics(residual)

        Args:
            x: Noisy input tensor of shape (B, 1, 1000)

        Returns:
            Tuple of:
                - denoised: Denoised output (B, 1, 1000)
                - residual: Normalized physics residual (B, 1, 998)
        """
        # Standard forward pass (inherited from DeepCAE)
        denoised = self.forward(x)

        # Compute physics residual on denoised output
        residual = self.compute_physics_residual(denoised)

        return denoised, residual


def count_parameters(model: nn.Module) -> int:
    """
    Count total trainable parameters in a model.

    Args:
        model: PyTorch model

    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_summary(
    model: nn.Module, input_size: Tuple[int, ...] = (1, 1, 1000)
) -> None:
    """
    Print a summary of the model architecture.

    Args:
        model: PyTorch model
        input_size: Input tensor size (batch, channels, length)
    """
    print(f"{'=' * 60}")
    print(f"Model: {model.__class__.__name__}")
    print(f"{'=' * 60}")
    print(f"Total parameters: {count_parameters(model):,}")

    # Test forward pass
    device = next(model.parameters()).device
    x = torch.randn(*input_size).to(device)

    # Trace through encoder
    encoder = getattr(model, "encoder", None)
    if callable(encoder):
        z = encoder(x)
        if isinstance(z, torch.Tensor):
            print(f"Input shape:  {tuple(x.shape)}")
            print(f"Latent shape: {tuple(z.shape)}")

    # Full forward pass
    y = model(x)
    print(f"Output shape: {tuple(y.shape)}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    # Test DeepCAE
    print("Testing DeepCAE...")
    model = DeepCAE()
    print_model_summary(model)

    # Verify input/output shapes match
    x = torch.randn(4, 1, 1000)
    y = model(x)
    assert x.shape == y.shape, f"Shape mismatch: {x.shape} vs {y.shape}"
    print("✓ Shape verification passed")
