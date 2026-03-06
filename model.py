"""
Deep 1D Convolutional Autoencoder for Ultrasonic Signal Denoising

Based on "Ultrasonic signal noise reduction based on convolutional autoencoders for NDT applications"

Architecture:
- Encoder: 5 Conv1d layers with stride=2 downsampling
- Decoder: 5 ConvTranspose1d layers with stride=2 upsampling
- Regularization: BatchNorm + Dropout + LeakyReLU
"""

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
        kernel_size: int = 7,     # Larger kernel for better context
        dropout_rate: float = 0.1  # Much lower dropout - 0.4 was too aggressive!
    ):
        super().__init__()
        
        padding = kernel_size // 2
        self.dropout_rate = dropout_rate
        
        # Encoder: (B, 1, 1000) → (B, 256, 32)
        # Only use dropout in middle layers, not all layers
        self.enc1 = nn.Sequential(
            nn.Conv1d(in_channels, base_channels, kernel_size, stride=2, padding=padding),
            nn.BatchNorm1d(base_channels),
            nn.LeakyReLU(0.1, inplace=True),
            # No dropout in first layer - preserve input information
        )  # 1000 → 500
        
        self.enc2 = nn.Sequential(
            nn.Conv1d(base_channels, base_channels * 2, kernel_size, stride=2, padding=padding),
            nn.BatchNorm1d(base_channels * 2),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout_rate),  # Use regular Dropout, not Dropout1d
        )  # 500 → 250
        
        self.enc3 = nn.Sequential(
            nn.Conv1d(base_channels * 2, base_channels * 4, kernel_size, stride=2, padding=padding),
            nn.BatchNorm1d(base_channels * 4),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout_rate),
        )  # 250 → 125
        
        self.enc4 = nn.Sequential(
            nn.Conv1d(base_channels * 4, base_channels * 8, kernel_size, stride=2, padding=padding),
            nn.BatchNorm1d(base_channels * 8),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout_rate),
        )  # 125 → 63
        
        self.enc5 = nn.Sequential(
            nn.Conv1d(base_channels * 8, base_channels * 8, kernel_size, stride=2, padding=padding),
            nn.BatchNorm1d(base_channels * 8),
            nn.LeakyReLU(0.1, inplace=True),
            # No dropout at bottleneck
        )  # 63 → 32
        
        # Decoder: (B, 256, 32) → (B, 1, 1000)
        self.dec5 = nn.Sequential(
            nn.ConvTranspose1d(base_channels * 8, base_channels * 8, kernel_size,
                              stride=2, padding=padding, output_padding=0),
            nn.BatchNorm1d(base_channels * 8),
            nn.LeakyReLU(0.1, inplace=True),
            # No dropout right after bottleneck
        )  # 32 → 63
        
        self.dec4 = nn.Sequential(
            nn.ConvTranspose1d(base_channels * 8, base_channels * 4, kernel_size,
                              stride=2, padding=padding, output_padding=0),
            nn.BatchNorm1d(base_channels * 4),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout_rate),
        )  # 63 → 125
        
        self.dec3 = nn.Sequential(
            nn.ConvTranspose1d(base_channels * 4, base_channels * 2, kernel_size,
                              stride=2, padding=padding, output_padding=1),
            nn.BatchNorm1d(base_channels * 2),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout_rate),
        )  # 125 → 250
        
        self.dec2 = nn.Sequential(
            nn.ConvTranspose1d(base_channels * 2, base_channels, kernel_size,
                              stride=2, padding=padding, output_padding=1),
            nn.BatchNorm1d(base_channels),
            nn.LeakyReLU(0.1, inplace=True),
            # No dropout near output
        )  # 250 → 500
        
        self.dec1 = nn.Sequential(
            nn.ConvTranspose1d(base_channels, in_channels, kernel_size,
                              stride=2, padding=padding, output_padding=1),
            nn.Tanh(),
        )  # 500 → 1000
        
        self._initialize_weights()
        
    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
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
    a physics constraint based on the 1D acoustic wave equation:
    
        ∂²u/∂t² = c² · ∂²u/∂x²
    
    For a fixed-position receiver, the denoised output u(t) should
    satisfy smoothness constraints derived from the wave equation.
    The physics residual (second-order time derivative) is computed
    via finite differences and penalized during training.
    
    Usage:
        - forward(x):         Standard inference (same as DeepCAE)
        - physics_forward(x): Returns (denoised, physics_residual) for PINN training
    """
    
    # Physical constants for ultrasonic NDT
    SAMPLING_RATE: float = 6.25e6   # 6.25 MHz
    DURATION: float = 160e-6        # 160 μs
    NUM_POINTS: int = 1000          # Total data points
    
    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 32,
        kernel_size: int = 7,
        dropout_rate: float = 0.1,
        wave_speed: float = 5900.0,  # Speed of sound in steel (m/s)
    ):
        super().__init__(
            in_channels=in_channels,
            base_channels=base_channels,
            kernel_size=kernel_size,
            dropout_rate=dropout_rate,
        )
        # Time step between consecutive samples
        self.dt = self.DURATION / self.NUM_POINTS  # 160e-6 / 1000 = 1.6e-7 s
        self.wave_speed = wave_speed
    
    def compute_wave_equation_residual(self, u: torch.Tensor) -> torch.Tensor:
        """
        Compute the wave equation physics residual using finite differences.
        
        Computes the second-order central difference (proportional to ∂²u/∂t²):
            Δ²u = u[t+1] - 2·u[t] + u[t-1]
        
        We omit the 1/dt² scaling factor to keep the residual in a
        numerically stable range (dt=1.6e-7 would cause ~1e14 scaling).
        The physics_weight hyperparameter absorbs this scale.
        
        Args:
            u: Denoised signal tensor of shape (B, 1, T)
            
        Returns:
            Physics residual tensor of shape (B, 1, T-2)
        """
        # Second-order central finite difference (curvature measure)
        d2u = u[:, :, 2:] - 2 * u[:, :, 1:-1] + u[:, :, :-2]
        return d2u
    
    def physics_forward(self, x: torch.Tensor) -> tuple:
        """
        Forward pass that returns both denoised output and physics residual.
        
        Used during PINN training to compute combined loss:
            L = L_data(denoised, clean) + λ · L_physics(residual)
        
        Args:
            x: Noisy input tensor of shape (B, 1, 1000)
            
        Returns:
            Tuple of:
                - denoised: Denoised output (B, 1, 1000)
                - residual: Wave equation residual (B, 1, 998)
        """
        # Standard forward pass (inherited from DeepCAE)
        denoised = self.forward(x)
        
        # Compute physics residual on denoised output
        residual = self.compute_wave_equation_residual(denoised)
        
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


def print_model_summary(model: nn.Module, input_size: Tuple[int, ...] = (1, 1, 1000)) -> None:
    """
    Print a summary of the model architecture.
    
    Args:
        model: PyTorch model
        input_size: Input tensor size (batch, channels, length)
    """
    print(f"{'='*60}")
    print(f"Model: {model.__class__.__name__}")
    print(f"{'='*60}")
    print(f"Total parameters: {count_parameters(model):,}")
    
    # Test forward pass
    device = next(model.parameters()).device
    x = torch.randn(*input_size).to(device)
    
    # Trace through encoder
    if hasattr(model, 'encoder'):
        z = model.encoder(x)
        print(f"Input shape:  {tuple(x.shape)}")
        print(f"Latent shape: {tuple(z.shape)}")
    
    # Full forward pass
    y = model(x)
    print(f"Output shape: {tuple(y.shape)}")
    print(f"{'='*60}")


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
