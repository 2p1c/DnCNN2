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
from typing import Tuple, Optional


# Shared physical constants across all model variants.
SAMPLING_RATE: float = 6.25e6  # Hz
DURATION: float = 160e-6  # seconds
NUM_POINTS: int = 1000  # time samples
CENTER_FREQUENCY: float = 250e3  # Hz
DEFAULT_WAVE_SPEED: float = 5900.0  # m/s (steel)
DEFAULT_DX: float = 1e-3  # 1 mm grid spacing
DEFAULT_DY: float = 1e-3  # 1 mm grid spacing


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
        base_channels: int = 4,  # Restore wider channels for capacity
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


class UnsupervisedDeepCAE(DeepCAE):
    """
    Unsupervised autoencoder variant with the same architecture as DeepCAE.

    This model reuses the exact encoder-decoder structure from DeepCAE and is
    intended for reconstruction-style training where the target is the input
    itself (x -> x), i.e., no clean label is required for the optimization
    objective.

    Compared with DeepCAE, this variant uses an additional channel bottleneck
    at the latent stage (after enc5) to further reduce representation capacity,
    which is useful for stronger compression-style denoising in unsupervised
    reconstruction.
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 32,
        kernel_size: int = 7,
        dropout_rate: float = 0.1,
        bottleneck_channels: int = 2,
    ):
        super().__init__(
            in_channels=in_channels,
            base_channels=base_channels,
            kernel_size=kernel_size,
            dropout_rate=dropout_rate,
        )

        latent_channels = base_channels * 8
        if bottleneck_channels <= 0:
            raise ValueError("bottleneck_channels must be positive")
        if bottleneck_channels >= latent_channels:
            raise ValueError(
                "bottleneck_channels must be smaller than latent channels "
                f"({latent_channels})"
            )

        self.bottleneck_channels = bottleneck_channels
        self.bottleneck_compress = nn.Sequential(
            nn.Conv1d(latent_channels, bottleneck_channels, kernel_size=1),
            nn.BatchNorm1d(bottleneck_channels),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.bottleneck_expand = nn.Sequential(
            nn.Conv1d(bottleneck_channels, latent_channels, kernel_size=1),
            nn.BatchNorm1d(latent_channels),
            nn.LeakyReLU(0.1, inplace=True),
        )

        self._initialize_unsupervised_bottleneck()

    def _initialize_unsupervised_bottleneck(self) -> None:
        for module in (self.bottleneck_compress, self.bottleneck_expand):
            for layer in module.modules():
                if isinstance(layer, nn.Conv1d):
                    nn.init.kaiming_normal_(
                        layer.weight, mode="fan_out", nonlinearity="leaky_relu"
                    )
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, 0)
                elif isinstance(layer, nn.BatchNorm1d):
                    nn.init.constant_(layer.weight, 1)
                    nn.init.constant_(layer.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)

        compressed = self.bottleneck_compress(e5)
        expanded = self.bottleneck_expand(compressed)

        d5 = self.dec5(expanded)
        d4 = self.dec4(d5)
        d3 = self.dec3(d4)
        d2 = self.dec2(d3)
        d1 = self.dec1(d2)

        return d1


# Backward-compatible aliases used by scripts/train/train.py.
LightweightCAE = DeepCAE
DeeperCAE = DeepCAE


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


class SignalEncoder(nn.Module):
    """Conv1d encoder for per-point set signals used by DeepSets variants."""

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 32,
        kernel_size: int = 7,
        dropout_rate: float = 0.1,
        embed_dim: int = 256,
    ):
        super().__init__()
        padding = kernel_size // 2

        self.enc1 = nn.Sequential(
            nn.Conv1d(
                in_channels, base_channels, kernel_size, stride=2, padding=padding
            ),
            nn.BatchNorm1d(base_channels),
            nn.LeakyReLU(0.1, inplace=True),
        )

        self.enc2 = nn.Sequential(
            nn.Conv1d(
                base_channels, base_channels * 2, kernel_size, stride=2, padding=padding
            ),
            nn.BatchNorm1d(base_channels * 2),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout_rate),
        )

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
        )

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
        )

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
        )

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Linear(base_channels * 8, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.enc1(x)
        h = self.enc2(h)
        h = self.enc3(h)
        h = self.enc4(h)
        h = self.enc5(h)
        h = self.pool(h).squeeze(-1)
        h = self.proj(h)
        return h


class CoordinateMLP(nn.Module):
    """Encode normalized 2D coordinates to a compact embedding."""

    def __init__(self, coord_dim: int = 2, embed_dim: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(coord_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, embed_dim),
        )

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        return self.mlp(coords)


class PointEncoder(nn.Module):
    """Fuse per-point signal and coordinate embeddings."""

    def __init__(
        self, signal_dim: int = 256, coord_dim: int = 64, output_dim: int = 256
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(signal_dim + coord_dim, output_dim),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(output_dim, output_dim),
        )

    def forward(
        self, signal_emb: torch.Tensor, coord_emb: torch.Tensor
    ) -> torch.Tensor:
        return self.mlp(torch.cat([signal_emb, coord_emb], dim=-1))


class SignalDecoder(nn.Module):
    """ConvTranspose1d decoder used by set-based denoising models."""

    def __init__(
        self,
        input_dim: int = 512,
        base_channels: int = 32,
        kernel_size: int = 7,
        dropout_rate: float = 0.1,
        latent_channels: int = 256,
        latent_length: int = 32,
    ):
        super().__init__()
        padding = kernel_size // 2
        self.latent_channels = latent_channels
        self.latent_length = latent_length

        self.pre = nn.Linear(input_dim, latent_channels * latent_length)

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
        )

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
        )

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
        )

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
        )

        self.dec1 = nn.Sequential(
            nn.ConvTranspose1d(
                base_channels,
                1,
                kernel_size,
                stride=2,
                padding=padding,
                output_padding=1,
            ),
            nn.Tanh(),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        h = self.pre(feat)
        h = h.view(*feat.shape[:-1], self.latent_channels, self.latent_length)
        h = self.dec5(h)
        h = self.dec4(h)
        h = self.dec3(h)
        h = self.dec2(h)
        h = self.dec1(h)
        return h


class DeepSetsPINN(nn.Module):
    """Set-invariant encoder-decoder with 2D wave-equation residual."""

    def __init__(
        self,
        signal_embed_dim: int = 256,
        coord_embed_dim: int = 64,
        point_dim: int = 256,
        base_channels: int = 32,
        kernel_size: int = 7,
        dropout_rate: float = 0.1,
        wave_speed: float = DEFAULT_WAVE_SPEED,
        center_frequency: float = CENTER_FREQUENCY,
        dx: float = DEFAULT_DX,
        dy: float = DEFAULT_DY,
        patch_size: int = 5,
    ):
        super().__init__()

        self.signal_encoder = SignalEncoder(
            base_channels=base_channels,
            kernel_size=kernel_size,
            dropout_rate=dropout_rate,
            embed_dim=signal_embed_dim,
        )
        self.coord_mlp = CoordinateMLP(
            coord_dim=2,
            embed_dim=coord_embed_dim,
        )
        self.point_encoder = PointEncoder(
            signal_dim=signal_embed_dim,
            coord_dim=coord_embed_dim,
            output_dim=point_dim,
        )
        self.decoder = SignalDecoder(
            input_dim=point_dim * 2,
            base_channels=base_channels,
            kernel_size=kernel_size,
            dropout_rate=dropout_rate,
            latent_channels=base_channels * 8,
            latent_length=32,
        )

        self.wave_speed = wave_speed
        self.center_frequency = center_frequency
        self.omega0 = 2.0 * math.pi * center_frequency
        self.dt = DURATION / (NUM_POINTS - 1)
        self.dx = dx
        self.dy = dy
        self.patch_size = patch_size

        self._precompute_patch_topology(patch_size)
        self._initialize_weights()

    def _precompute_patch_topology(self, P: int) -> None:
        half = P // 2
        interior, left, right, down, up = [], [], [], [], []
        for dc in range(-half, half + 1):
            for dr in range(-half, half + 1):
                if (
                    abs(dc - 1) <= half
                    and abs(dc + 1) <= half
                    and abs(dr - 1) <= half
                    and abs(dr + 1) <= half
                ):
                    idx = (dc + half) * P + (dr + half)
                    interior.append(idx)
                    left.append((dc - 1 + half) * P + (dr + half))
                    right.append((dc + 1 + half) * P + (dr + half))
                    down.append((dc + half) * P + (dr - 1 + half))
                    up.append((dc + half) * P + (dr + 1 + half))

        self.register_buffer("_interior_idx", torch.tensor(interior, dtype=torch.long))
        self.register_buffer("_left_idx", torch.tensor(left, dtype=torch.long))
        self.register_buffer("_right_idx", torch.tensor(right, dtype=torch.long))
        self.register_buffer("_down_idx", torch.tensor(down, dtype=torch.long))
        self.register_buffer("_up_idx", torch.tensor(up, dtype=torch.long))

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
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_in", nonlinearity="leaky_relu"
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(
        self,
        noisy_signals: torch.Tensor,
        coordinates: torch.Tensor,
    ) -> torch.Tensor:
        B, R, T = noisy_signals.shape

        sig_flat = noisy_signals.reshape(B * R, 1, T)
        sig_emb = self.signal_encoder(sig_flat)
        sig_emb = sig_emb.view(B, R, -1)

        coord_emb = self.coord_mlp(coordinates)
        point_feat = self.point_encoder(sig_emb, coord_emb)

        global_feat = point_feat.mean(dim=1, keepdim=True)
        global_feat = global_feat.expand(-1, R, -1)

        dec_input = torch.cat([point_feat, global_feat], dim=-1)
        dec_flat = dec_input.reshape(B * R, -1)
        out_flat = self.decoder(dec_flat)
        denoised = out_flat.view(B, R, T)

        return denoised

    def compute_wave_equation_residual(
        self,
        denoised: torch.Tensor,
        grid_indices: Optional[torch.Tensor] = None,
        grid_cols: int = 41,
        grid_rows: int = 41,
    ) -> torch.Tensor:
        del grid_indices, grid_cols, grid_rows

        B, _, T = denoised.shape
        c2 = self.wave_speed**2

        u_c = denoised[:, self._interior_idx, :]
        u_l = denoised[:, self._left_idx, :]
        u_r = denoised[:, self._right_idx, :]
        u_d = denoised[:, self._down_idx, :]
        u_u = denoised[:, self._up_idx, :]

        u_tt = (u_c[:, :, 2:] - 2.0 * u_c[:, :, 1:-1] + u_c[:, :, :-2]) / (self.dt**2)
        u_xx = (u_r[:, :, 1:-1] - 2.0 * u_c[:, :, 1:-1] + u_l[:, :, 1:-1]) / (
            self.dx**2
        )
        u_yy = (u_u[:, :, 1:-1] - 2.0 * u_c[:, :, 1:-1] + u_d[:, :, 1:-1]) / (
            self.dy**2
        )
        residual = (u_tt - c2 * (u_xx + u_yy)) / (self.omega0**2)

        with torch.no_grad():
            n_int = u_c.shape[1]
            u_flat = u_c.reshape(B * n_int, 1, T)
            energy = torch.nn.functional.avg_pool1d(
                u_flat.abs(), kernel_size=9, stride=1, padding=4
            )
            mask = energy / energy.amax(dim=-1, keepdim=True).clamp_min(1e-6)
            mask = mask.view(B, n_int, T)[:, :, 1:-1]

        return residual * mask

    def physics_forward(
        self,
        noisy_signals: torch.Tensor,
        coordinates: torch.Tensor,
        grid_indices: Optional[torch.Tensor] = None,
        grid_cols: int = 41,
        grid_rows: int = 41,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        denoised = self.forward(noisy_signals, coordinates)
        residual = self.compute_wave_equation_residual(
            denoised, grid_indices=grid_indices, grid_cols=grid_cols, grid_rows=grid_rows
        )
        return denoised, residual


class SpatialAuxiliaryCAE(nn.Module):
    """DeepCAE-style model with spatial modulation at the bottleneck."""

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 32,
        kernel_size: int = 7,
        dropout_rate: float = 0.1,
        coord_dim: int = 64,
        wave_speed: float = DEFAULT_WAVE_SPEED,
        center_frequency: float = CENTER_FREQUENCY,
        dx: float = DEFAULT_DX,
        dy: float = DEFAULT_DY,
        patch_size: int = 5,
        signal_embed_dim: int = 256,
        coord_embed_dim: int = 64,
        point_dim: int = 256,
        global_dim: int = 128,
    ):
        del signal_embed_dim, coord_embed_dim, point_dim, global_dim

        super().__init__()
        padding = kernel_size // 2
        self.coord_dim = coord_dim

        self.enc1 = nn.Sequential(
            nn.Conv1d(
                in_channels, base_channels, kernel_size, stride=2, padding=padding
            ),
            nn.BatchNorm1d(base_channels),
            nn.LeakyReLU(0.1, inplace=True),
        )

        self.enc2 = nn.Sequential(
            nn.Conv1d(
                base_channels, base_channels * 2, kernel_size, stride=2, padding=padding
            ),
            nn.BatchNorm1d(base_channels * 2),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout_rate),
        )

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
        )

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
        )

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
        )

        self.latent_channels = base_channels * 8
        self.latent_length = 32

        self.coord_mlp = CoordinateMLP(coord_dim=2, embed_dim=coord_dim)
        self.spatial_modulator = nn.Sequential(
            nn.Linear(self.latent_channels + coord_dim, self.latent_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(self.latent_channels, self.latent_channels),
        )

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
        )

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
        )

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
        )

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
        )

        self.dec1 = nn.Sequential(
            nn.ConvTranspose1d(
                base_channels,
                1,
                kernel_size,
                stride=2,
                padding=padding,
                output_padding=1,
            ),
            nn.Tanh(),
        )

        self.wave_speed = wave_speed
        self.center_frequency = center_frequency
        self.omega0 = 2.0 * math.pi * center_frequency
        self.dt = DURATION / (NUM_POINTS - 1)
        self.dx = dx
        self.dy = dy
        self.patch_size = patch_size

        self._precompute_patch_topology(patch_size)
        self._initialize_weights()

    def _precompute_patch_topology(self, P: int) -> None:
        half = P // 2
        interior, left, right, down, up = [], [], [], [], []
        for dc in range(-half, half + 1):
            for dr in range(-half, half + 1):
                if (
                    abs(dc - 1) <= half
                    and abs(dc + 1) <= half
                    and abs(dr - 1) <= half
                    and abs(dr + 1) <= half
                ):
                    idx = (dc + half) * P + (dr + half)
                    interior.append(idx)
                    left.append((dc - 1 + half) * P + (dr + half))
                    right.append((dc + 1 + half) * P + (dr + half))
                    down.append((dc + half) * P + (dr - 1 + half))
                    up.append((dc + half) * P + (dr + 1 + half))

        self.register_buffer("_interior_idx", torch.tensor(interior, dtype=torch.long))
        self.register_buffer("_left_idx", torch.tensor(left, dtype=torch.long))
        self.register_buffer("_right_idx", torch.tensor(right, dtype=torch.long))
        self.register_buffer("_down_idx", torch.tensor(down, dtype=torch.long))
        self.register_buffer("_up_idx", torch.tensor(up, dtype=torch.long))

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
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_in", nonlinearity="leaky_relu"
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(
        self,
        noisy_signals: torch.Tensor,
        coordinates: torch.Tensor,
    ) -> torch.Tensor:
        B, R, T = noisy_signals.shape

        x = noisy_signals.view(B * R, 1, T)
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)

        local_channel_desc = e5.mean(dim=2)
        local_channel_desc = local_channel_desc.view(B, R, self.latent_channels)

        global_ctx, _ = torch.max(local_channel_desc, dim=1, keepdim=True)
        global_ctx = global_ctx.expand(-1, R, -1)
        coord_emb = self.coord_mlp(coordinates)

        mod_input = torch.cat([global_ctx, coord_emb], dim=-1)
        shift = self.spatial_modulator(mod_input)
        shift_expanded = shift.view(B * R, self.latent_channels, 1)
        modified_latent = e5 + shift_expanded

        d5 = self.dec5(modified_latent)
        d4 = self.dec4(d5)
        d3 = self.dec3(d4)
        d2 = self.dec2(d3)
        d1 = self.dec1(d2)
        return d1.view(B, R, T)

    def compute_wave_equation_residual(
        self,
        denoised: torch.Tensor,
        grid_indices: Optional[torch.Tensor] = None,
        grid_cols: int = 41,
        grid_rows: int = 41,
    ) -> torch.Tensor:
        del grid_indices, grid_cols, grid_rows

        B, _, T = denoised.shape
        c2 = self.wave_speed**2

        u_c = denoised[:, self._interior_idx, :]
        u_l = denoised[:, self._left_idx, :]
        u_r = denoised[:, self._right_idx, :]
        u_d = denoised[:, self._down_idx, :]
        u_u = denoised[:, self._up_idx, :]

        u_tt = (u_c[:, :, 2:] - 2.0 * u_c[:, :, 1:-1] + u_c[:, :, :-2]) / (self.dt**2)
        u_xx = (u_r[:, :, 1:-1] - 2.0 * u_c[:, :, 1:-1] + u_l[:, :, 1:-1]) / (
            self.dx**2
        )
        u_yy = (u_u[:, :, 1:-1] - 2.0 * u_c[:, :, 1:-1] + u_d[:, :, 1:-1]) / (
            self.dy**2
        )
        residual = (u_tt - c2 * (u_xx + u_yy)) / (self.omega0**2)

        with torch.no_grad():
            n_int = u_c.shape[1]
            u_flat = u_c.reshape(B * n_int, 1, T)
            energy = torch.nn.functional.avg_pool1d(
                u_flat.abs(), kernel_size=9, stride=1, padding=4
            )
            mask = energy / energy.amax(dim=-1, keepdim=True).clamp_min(1e-6)
            mask = mask.view(B, n_int, T)[:, :, 1:-1]

        return residual * mask

    def physics_forward(
        self,
        noisy_signals: torch.Tensor,
        coordinates: torch.Tensor,
        grid_indices: Optional[torch.Tensor] = None,
        grid_cols: int = 41,
        grid_rows: int = 41,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        denoised = self.forward(noisy_signals, coordinates)
        residual = self.compute_wave_equation_residual(
            denoised, grid_indices=grid_indices, grid_cols=grid_cols, grid_rows=grid_rows
        )
        return denoised, residual


# Improved names with compatibility aliases retained below.
class SetInvariantWavePINN(DeepSetsPINN):
    """Preferred name for the set-invariant wave-equation PINN model."""


class SpatialContextCAE(SpatialAuxiliaryCAE):
    """Preferred name for the spatially modulated DeepCAE variant."""


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
