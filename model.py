"""
Lightweight 1D Convolutional Autoencoder for Ultrasonic Signal Denoising

Based on "Ultrasonic signal noise reduction based on convolutional autoencoders for NDT applications"
Scaled down for consumer-grade hardware while maintaining effectiveness.

Architecture:
- Encoder: 3 Conv1d layers with stride=2 downsampling
- Decoder: 3 ConvTranspose1d layers with stride=2 upsampling
- Regularization: BatchNorm + Dropout (as mentioned in paper)
"""

import torch
import torch.nn as nn
from typing import Tuple


class LightweightCAE(nn.Module):
    """
    Lightweight 1D Convolutional Autoencoder for signal denoising.
    
    Architecture:
    - Encoder: Conv1d (1→16→32→64 channels), stride=2, ReLU, BatchNorm, Dropout
    - Decoder: ConvTranspose1d (64→32→16→1 channels), stride=2, ReLU, BatchNorm
    
    Input/Output shape: (Batch, 1, 1000)
    Latent shape: (Batch, 64, 125)
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 16,
        kernel_size: int = 5,
        dropout_rate: float = 0.1
    ):
        """
        Initialize the Convolutional Autoencoder.
        
        Args:
            in_channels: Number of input channels (1 for 1D signal)
            base_channels: Base number of channels (16), doubled at each layer
            kernel_size: Convolution kernel size (3 or 5 recommended)
            dropout_rate: Dropout probability for regularization (paper mentions dropout)
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.kernel_size = kernel_size
        
        # Calculate padding for 'same' convolution before stride
        padding = kernel_size // 2
        
        # ============================================================
        # Encoder: Progressively downsample and increase channels
        # Input: (B, 1, 1000) → Output: (B, 64, 125)
        # ============================================================
        self.encoder = nn.Sequential(
            # Layer 1: (B, 1, 1000) → (B, 16, 500)
            nn.Conv1d(in_channels, base_channels, kernel_size, 
                     stride=2, padding=padding),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            
            # Layer 2: (B, 16, 500) → (B, 32, 250)
            nn.Conv1d(base_channels, base_channels * 2, kernel_size, 
                     stride=2, padding=padding),
            nn.BatchNorm1d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            
            # Layer 3: (B, 32, 250) → (B, 64, 125)
            nn.Conv1d(base_channels * 2, base_channels * 4, kernel_size, 
                     stride=2, padding=padding),
            nn.BatchNorm1d(base_channels * 4),
            nn.ReLU(inplace=True),
        )
        
        # ============================================================
        # Decoder: Progressively upsample and decrease channels
        # Input: (B, 64, 125) → Output: (B, 1, 1000)
        # ============================================================
        self.decoder = nn.Sequential(
            # Layer 1: (B, 64, 125) → (B, 32, 250)
            nn.ConvTranspose1d(base_channels * 4, base_channels * 2, kernel_size, 
                              stride=2, padding=padding, output_padding=1),
            nn.BatchNorm1d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            
            # Layer 2: (B, 32, 250) → (B, 16, 500)
            nn.ConvTranspose1d(base_channels * 2, base_channels, kernel_size, 
                              stride=2, padding=padding, output_padding=1),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            
            # Layer 3: (B, 16, 500) → (B, 1, 1000)
            nn.ConvTranspose1d(base_channels, in_channels, kernel_size, 
                              stride=2, padding=padding, output_padding=1),
            # Tanh activation: output in [-1, 1] range (matches normalized signal)
            nn.Tanh(),
        )
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self) -> None:
        """
        Initialize model weights using Kaiming (He) initialization.
        
        Kaiming init is preferred for networks with ReLU activations.
        """
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input signal to latent representation.
        
        Args:
            x: Input tensor of shape (B, 1, 1000)
            
        Returns:
            Latent tensor of shape (B, 64, 125)
        """
        return self.encoder(x)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent representation to reconstructed signal.
        
        Args:
            z: Latent tensor of shape (B, 64, 125)
            
        Returns:
            Reconstructed tensor of shape (B, 1, 1000)
        """
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: encode noisy signal then decode to denoised signal.
        
        The autoencoder learns to map noisy signals to clean signals
        by learning a compressed representation that preserves signal
        features while discarding noise.
        
        Args:
            x: Input noisy signal of shape (B, 1, 1000)
            
        Returns:
            Denoised signal of shape (B, 1, 1000)
        """
        z = self.encode(x)
        return self.decode(z)
    
    def get_latent_dim(self) -> Tuple[int, int]:
        """
        Get latent space dimensions.
        
        Returns:
            Tuple of (channels, length) = (64, 125)
        """
        return (self.base_channels * 4, 125)


class DeeperCAE(nn.Module):
    """
    Deeper variant of the CAE for better performance on complex signals.
    
    Architecture: 4 encoder + 4 decoder layers
    Use this if LightweightCAE doesn't achieve desired PSNR.
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 16,
        kernel_size: int = 5,
        dropout_rate: float = 0.1
    ):
        super().__init__()
        
        padding = kernel_size // 2
        
        # Encoder: (B, 1, 1000) → (B, 128, 63)
        self.encoder = nn.Sequential(
            # Layer 1: 1→16, 1000→500
            nn.Conv1d(in_channels, base_channels, kernel_size, stride=2, padding=padding),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            
            # Layer 2: 16→32, 500→250
            nn.Conv1d(base_channels, base_channels * 2, kernel_size, stride=2, padding=padding),
            nn.BatchNorm1d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            
            # Layer 3: 32→64, 250→125
            nn.Conv1d(base_channels * 2, base_channels * 4, kernel_size, stride=2, padding=padding),
            nn.BatchNorm1d(base_channels * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            
            # Layer 4: 64→128, 125→63
            nn.Conv1d(base_channels * 4, base_channels * 8, kernel_size, stride=2, padding=padding),
            nn.BatchNorm1d(base_channels * 8),
            nn.ReLU(inplace=True),
        )
        
        # Decoder: (B, 128, 63) → (B, 1, 1000)
        self.decoder = nn.Sequential(
            # Layer 1: 128→64, 63→125
            nn.ConvTranspose1d(base_channels * 8, base_channels * 4, kernel_size, 
                              stride=2, padding=padding, output_padding=0),
            nn.BatchNorm1d(base_channels * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            
            # Layer 2: 64→32, 125→250
            nn.ConvTranspose1d(base_channels * 4, base_channels * 2, kernel_size, 
                              stride=2, padding=padding, output_padding=1),
            nn.BatchNorm1d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            
            # Layer 3: 32→16, 250→500
            nn.ConvTranspose1d(base_channels * 2, base_channels, kernel_size, 
                              stride=2, padding=padding, output_padding=1),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            
            # Layer 4: 16→1, 500→1000
            nn.ConvTranspose1d(base_channels, in_channels, kernel_size, 
                              stride=2, padding=padding, output_padding=1),
            nn.Tanh(),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.decoder(z)


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
    # Test LightweightCAE
    print("Testing LightweightCAE...")
    model = LightweightCAE()
    print_model_summary(model)
    
    # Verify input/output shapes match
    x = torch.randn(4, 1, 1000)
    y = model(x)
    assert x.shape == y.shape, f"Shape mismatch: {x.shape} vs {y.shape}"
    print("✓ Shape verification passed\n")
    
    # Test DeeperCAE
    print("Testing DeeperCAE...")
    deeper_model = DeeperCAE()
    print_model_summary(deeper_model)
    
    y_deep = deeper_model(x)
    assert x.shape == y_deep.shape, f"Shape mismatch: {x.shape} vs {y_deep.shape}"
    print("✓ Shape verification passed")
