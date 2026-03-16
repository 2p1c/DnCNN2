"""
DeepSets + PINN Model for Ultrasonic Signal Denoising

Architecture overview:
    1. SignalEncoder   — 5-layer Conv1d (mirrors DeepCAE encoder)
    2. CoordinateMLP   — 2-layer MLP for (x, y) position encoding
    3. PointEncoder    — Fuses signal + coordinate embeddings
    4. MeanPooling     — Symmetric set aggregation
    5. PointDecoder    — Reconstructs per-point denoised signal

Physics:
    Computes the 2D acoustic wave equation residual on the denoised
    output grid:
        r = u_tt − c² (u_xx + u_yy)
    using finite differences in time (dt) and space (dx, dy).

Usage:
    model = DeepSetsPINN()
    denoised = model(noisy_signals, coordinates)
    denoised, residual = model.physics_forward(
        noisy_signals, coordinates, grid_indices,
        grid_cols=41, grid_rows=41
    )
"""

import math
import torch
import torch.nn as nn
from typing import Tuple, Optional


# ============================================================
# Physical Constants (shared with model.py / DeepCAE_PINN)
# ============================================================
SAMPLING_RATE: float = 6.25e6      # Hz
DURATION: float = 160e-6           # seconds
NUM_POINTS: int = 1000             # time samples
CENTER_FREQUENCY: float = 250e3   # Hz
DEFAULT_WAVE_SPEED: float = 5900.0 # m/s  (steel)
DEFAULT_DX: float = 1e-3          # 1 mm grid spacing
DEFAULT_DY: float = 1e-3          # 1 mm grid spacing


# ============================================================
# Sub-Modules
# ============================================================

class SignalEncoder(nn.Module):
    """
    5-layer 1D convolutional encoder matching DeepCAE's encoder.

    Input:  (*, 1, T)   where T = 1000
    Output: (*, D_s)    signal embedding vector

    Architecture:
        Conv1d layers: 1→32→64→128→256→256, stride=2, kernel=7
        + BatchNorm + LeakyReLU(0.1) + selective Dropout
        → adaptive avg pool to length 1 → flatten → linear → D_s
    """

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
            nn.Conv1d(in_channels, base_channels, kernel_size,
                      stride=2, padding=padding),
            nn.BatchNorm1d(base_channels),
            nn.LeakyReLU(0.1, inplace=True),
        )  # T=1000 → 500

        self.enc2 = nn.Sequential(
            nn.Conv1d(base_channels, base_channels * 2, kernel_size,
                      stride=2, padding=padding),
            nn.BatchNorm1d(base_channels * 2),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout_rate),
        )  # 500 → 250

        self.enc3 = nn.Sequential(
            nn.Conv1d(base_channels * 2, base_channels * 4, kernel_size,
                      stride=2, padding=padding),
            nn.BatchNorm1d(base_channels * 4),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout_rate),
        )  # 250 → 125

        self.enc4 = nn.Sequential(
            nn.Conv1d(base_channels * 4, base_channels * 8, kernel_size,
                      stride=2, padding=padding),
            nn.BatchNorm1d(base_channels * 8),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout_rate),
        )  # 125 → 63

        self.enc5 = nn.Sequential(
            nn.Conv1d(base_channels * 8, base_channels * 8, kernel_size,
                      stride=2, padding=padding),
            nn.BatchNorm1d(base_channels * 8),
            nn.LeakyReLU(0.1, inplace=True),
        )  # 63 → 32

        # Collapse spatial dim and project to embedding
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Linear(base_channels * 8, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (*, 1, T) signal tensor
        Returns:
            (*, D_s) embedding
        """
        h = self.enc1(x)
        h = self.enc2(h)
        h = self.enc3(h)
        h = self.enc4(h)
        h = self.enc5(h)
        h = self.pool(h).squeeze(-1)   # (*, 256)
        h = self.proj(h)               # (*, D_s)
        return h


class CoordinateMLP(nn.Module):
    """
    Small MLP that encodes 2D spatial coordinates.

    Input:  (*, 2)     normalised (x, y) in [0, 1]
    Output: (*, D_c)   coordinate embedding
    """

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
    """
    Fuse signal embedding and coordinate embedding into a single
    point feature vector.

    Input:  signal_emb (*, D_s),  coord_emb (*, D_c)
    Output: (*, D)   fused point feature
    """

    def __init__(self, signal_dim: int = 256, coord_dim: int = 64,
                 output_dim: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(signal_dim + coord_dim, output_dim),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, signal_emb: torch.Tensor,
                coord_emb: torch.Tensor) -> torch.Tensor:
        return self.mlp(torch.cat([signal_emb, coord_emb], dim=-1))


class SignalDecoder(nn.Module):
    """
    5-layer 1D transposed convolutional decoder that reconstructs
    a denoised signal from a feature vector.

    Input:  (*, D_in)   feature vector (typically 2*D = 512)
    Output: (*, 1, T)   reconstructed signal  (T = 1000)

    The decoder mirrors the encoder:
        linear → reshape → 5× ConvTranspose1d → Tanh
    """

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

        # Project feature vector → latent spatial map
        self.pre = nn.Linear(input_dim, latent_channels * latent_length)

        self.dec5 = nn.Sequential(
            nn.ConvTranspose1d(base_channels * 8, base_channels * 8,
                               kernel_size, stride=2, padding=padding,
                               output_padding=0),
            nn.BatchNorm1d(base_channels * 8),
            nn.LeakyReLU(0.1, inplace=True),
        )  # 32 → 63

        self.dec4 = nn.Sequential(
            nn.ConvTranspose1d(base_channels * 8, base_channels * 4,
                               kernel_size, stride=2, padding=padding,
                               output_padding=0),
            nn.BatchNorm1d(base_channels * 4),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout_rate),
        )  # 63 → 125

        self.dec3 = nn.Sequential(
            nn.ConvTranspose1d(base_channels * 4, base_channels * 2,
                               kernel_size, stride=2, padding=padding,
                               output_padding=1),
            nn.BatchNorm1d(base_channels * 2),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout_rate),
        )  # 125 → 250

        self.dec2 = nn.Sequential(
            nn.ConvTranspose1d(base_channels * 2, base_channels,
                               kernel_size, stride=2, padding=padding,
                               output_padding=1),
            nn.BatchNorm1d(base_channels),
            nn.LeakyReLU(0.1, inplace=True),
        )  # 250 → 500

        self.dec1 = nn.Sequential(
            nn.ConvTranspose1d(base_channels, 1, kernel_size,
                               stride=2, padding=padding,
                               output_padding=1),
            nn.Tanh(),
        )  # 500 → 1000

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feat: (*, D_in)
        Returns:
            (*, 1, T)
        """
        h = self.pre(feat)  # (*, latent_channels * latent_length)
        h = h.view(*feat.shape[:-1], self.latent_channels,
                    self.latent_length)
        h = self.dec5(h)
        h = self.dec4(h)
        h = self.dec3(h)
        h = self.dec2(h)
        h = self.dec1(h)
        return h


# ============================================================
# Main Model
# ============================================================

class DeepSetsPINN(nn.Module):
    """
    DeepSets encoder-decoder with Physics-Informed wave equation
    constraint for ultrasonic signal denoising.

    Forward signature:
        denoised = model(noisy_signals, coordinates)
            noisy_signals: (B, R, T)   batch of sets, each R points
            coordinates:   (B, R, 2)   normalised (x, y)
            → denoised:    (B, R, T)

    Physics forward (for training):
        denoised, residual = model.physics_forward(
            noisy_signals, coordinates, grid_indices,
            grid_cols, grid_rows)

    Args:
        signal_embed_dim: Signal encoder output dimension. Default: 256.
        coord_embed_dim:  Coordinate MLP output dimension. Default: 64.
        point_dim:        Fused point feature dimension. Default: 256.
        base_channels:    Base channel count for Conv layers. Default: 32.
        kernel_size:      Conv kernel size. Default: 7.
        dropout_rate:     Dropout rate in middle layers. Default: 0.1.
        wave_speed:       Acoustic wave speed (m/s). Default: 5900.
        center_frequency: Transducer centre frequency (Hz). Default: 250e3.
        dx:               Grid spacing in x (m). Default: 1e-3.
        dy:               Grid spacing in y (m). Default: 1e-3.
    """

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

        # Sub-modules
        self.signal_encoder = SignalEncoder(
            base_channels=base_channels,
            kernel_size=kernel_size,
            dropout_rate=dropout_rate,
            embed_dim=signal_embed_dim,
        )
        self.coord_mlp = CoordinateMLP(
            coord_dim=2, embed_dim=coord_embed_dim,
        )
        self.point_encoder = PointEncoder(
            signal_dim=signal_embed_dim,
            coord_dim=coord_embed_dim,
            output_dim=point_dim,
        )
        self.decoder = SignalDecoder(
            input_dim=point_dim * 2,  # concat(point_feat, global_feat)
            base_channels=base_channels,
            kernel_size=kernel_size,
            dropout_rate=dropout_rate,
            latent_channels=base_channels * 8,
            latent_length=32,
        )

        # Physics parameters
        self.wave_speed = wave_speed
        self.center_frequency = center_frequency
        self.omega0 = 2.0 * math.pi * center_frequency
        self.dt = DURATION / (NUM_POINTS - 1)
        self.dx = dx
        self.dy = dy
        self.patch_size = patch_size

        # Precompute interior / neighbour index arrays for vectorised
        # wave-equation residual.  For a P×P patch with element ordering
        #   flat_idx = (dc + half) * P + (dr + half)
        # an interior point needs all 4 spatial neighbours inside the patch.
        self._precompute_patch_topology(patch_size)

        self._initialize_weights()

    def _precompute_patch_topology(self, P: int) -> None:
        """
        Build index arrays so that spatial derivatives can be computed
        with a single gather instead of a Python loop.

        Registers:
            _interior_idx:  (N_int,)   flat indices of interior points
            _left_idx:      (N_int,)   neighbour: col-1
            _right_idx:     (N_int,)   neighbour: col+1
            _down_idx:      (N_int,)   neighbour: row-1
            _up_idx:        (N_int,)   neighbour: row+1

        Element ordering (same as data_deepsets._extract_patch_indices):
            for dc in range(-half, half+1):      # col direction
                for dr in range(-half, half+1):  # row direction
            → flat_idx = (dc + half) * P + (dr + half)
        """
        half = P // 2
        interior, left, right, down, up = [], [], [], [], []
        for dc in range(-half, half + 1):
            for dr in range(-half, half + 1):
                # Check all 4 neighbours are within the patch
                if (abs(dc - 1) <= half and abs(dc + 1) <= half
                        and abs(dr - 1) <= half and abs(dr + 1) <= half):
                    idx = (dc + half) * P + (dr + half)
                    interior.append(idx)
                    left.append((dc - 1 + half) * P + (dr + half))
                    right.append((dc + 1 + half) * P + (dr + half))
                    down.append((dc + half) * P + (dr - 1 + half))
                    up.append((dc + half) * P + (dr + 1 + half))

        # Register as non-parameter buffers (move with model.to(device))
        self.register_buffer('_interior_idx', torch.tensor(interior, dtype=torch.long))
        self.register_buffer('_left_idx',     torch.tensor(left,     dtype=torch.long))
        self.register_buffer('_right_idx',    torch.tensor(right,    dtype=torch.long))
        self.register_buffer('_down_idx',     torch.tensor(down,     dtype=torch.long))
        self.register_buffer('_up_idx',       torch.tensor(up,       dtype=torch.long))

    def _initialize_weights(self) -> None:
        """Kaiming initialisation for Conv and Linear layers."""
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    # ==============================================================
    # Forward Pass
    # ==============================================================

    def forward(
        self,
        noisy_signals: torch.Tensor,
        coordinates: torch.Tensor,
    ) -> torch.Tensor:
        """
        Standard forward pass — per-point denoising with set context.

        Args:
            noisy_signals: (B, R, T)  noisy time-series for each point
            coordinates:   (B, R, 2)  normalised (x, y) positions

        Returns:
            denoised: (B, R, T) denoised signals
        """
        B, R, T = noisy_signals.shape

        # --- Per-point encoding ---
        # Reshape to (B*R, 1, T) for conv encoder
        sig_flat = noisy_signals.reshape(B * R, 1, T)
        sig_emb = self.signal_encoder(sig_flat)          # (B*R, D_s)
        sig_emb = sig_emb.view(B, R, -1)                # (B, R, D_s)

        coord_emb = self.coord_mlp(coordinates)          # (B, R, D_c)
        point_feat = self.point_encoder(sig_emb, coord_emb)  # (B, R, D)

        # --- Set aggregation ---
        global_feat = point_feat.mean(dim=1, keepdim=True)    # (B, 1, D)
        global_feat = global_feat.expand(-1, R, -1)           # (B, R, D)

        # --- Per-point decoding ---
        dec_input = torch.cat([point_feat, global_feat], dim=-1)  # (B, R, 2D)
        dec_flat = dec_input.reshape(B * R, -1)           # (B*R, 2D)
        out_flat = self.decoder(dec_flat)                  # (B*R, 1, T)
        denoised = out_flat.view(B, R, T)                 # (B, R, T)

        return denoised

    # ==============================================================
    # Physics Residual  (fully vectorised — no Python loops)
    # ==============================================================

    def compute_wave_equation_residual(
        self,
        denoised: torch.Tensor,
        grid_indices: torch.Tensor = None,
        grid_cols: int = 41,
        grid_rows: int = 41,
    ) -> torch.Tensor:
        """
        Vectorised 2D acoustic wave equation residual:
            r = (u_tt − c² (u_xx + u_yy)) / ω₀²

        Uses precomputed index arrays from _precompute_patch_topology
        so everything is pure tensor ops — no Python loops.

        Args:
            denoised:     (B, R, T) denoised signals (R = patch_size²)
            grid_indices: unused (kept for API compat), may be None
            grid_cols:    unused (kept for API compat)
            grid_rows:    unused (kept for API compat)

        Returns:
            residual: (B, N_interior, T-2)
        """
        B, R, T = denoised.shape
        c2 = self.wave_speed ** 2

        # Gather interior and neighbour signals using precomputed indices
        # Each index tensor is (N_int,); advanced indexing broadcasts over B
        u_c = denoised[:, self._interior_idx, :]   # (B, N_int, T)
        u_l = denoised[:, self._left_idx, :]       # (B, N_int, T)
        u_r = denoised[:, self._right_idx, :]      # (B, N_int, T)
        u_d = denoised[:, self._down_idx, :]       # (B, N_int, T)
        u_u = denoised[:, self._up_idx, :]         # (B, N_int, T)

        # Time derivative: central difference  (T → T-2)
        u_tt = (u_c[:, :, 2:] - 2.0 * u_c[:, :, 1:-1] + u_c[:, :, :-2]) \
               / (self.dt ** 2)

        # Spatial derivatives: central difference across grid neighbours
        u_xx = (u_r[:, :, 1:-1] - 2.0 * u_c[:, :, 1:-1] + u_l[:, :, 1:-1]) \
               / (self.dx ** 2)
        u_yy = (u_u[:, :, 1:-1] - 2.0 * u_c[:, :, 1:-1] + u_d[:, :, 1:-1]) \
               / (self.dy ** 2)

        # Wave equation residual, normalised by ω₀²
        residual = (u_tt - c2 * (u_xx + u_yy)) / (self.omega0 ** 2)

        # Energy mask — focus on signal-rich regions (computed without grad)
        with torch.no_grad():
            # (B, N_int, T) → (B*N_int, 1, T) for avg_pool1d
            N_int = u_c.shape[1]
            u_flat = u_c.reshape(B * N_int, 1, T)
            energy = torch.nn.functional.avg_pool1d(
                u_flat.abs(), kernel_size=9, stride=1, padding=4)
            mask = energy / energy.amax(dim=-1, keepdim=True).clamp_min(1e-6)
            mask = mask.view(B, N_int, T)[:, :, 1:-1]  # (B, N_int, T-2)

        return residual * mask   # (B, N_int, T-2)

    def physics_forward(
        self,
        noisy_signals: torch.Tensor,
        coordinates: torch.Tensor,
        grid_indices: torch.Tensor = None,
        grid_cols: int = 41,
        grid_rows: int = 41,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning both denoised output and physics residual.

        Used during PINN training:
            L = L_data(denoised, clean) + λ · L_physics(residual)

        Args:
            noisy_signals: (B, R, T)
            coordinates:   (B, R, 2)
            grid_indices:  (B, R, 2)  integer (col, row) — unused by
                           vectorised residual but kept for API compat
            grid_cols:     unused
            grid_rows:     unused

        Returns:
            denoised:  (B, R, T)
            residual:  (B, N_interior, T-2)
        """
        denoised = self.forward(noisy_signals, coordinates)
        residual = self.compute_wave_equation_residual(denoised)
        return denoised, residual


# ============================================================
# Spatial-Auxiliary DeepCAE
# ============================================================

class SpatialAuxiliaryCAE(nn.Module):
    """
    1D Convolutional Autoencoder with Spatial Bottleneck Modulation
    for Ultrasonic Signal Denoising.

    This architecture prioritises learning single-trace signal features
    via a deep 1D CNN Encoder-Decoder (DeepCAE backbone), while injecting
    spatial context (neighbourhood aggregation + coordinates) strictly at
    the lowest-dimensional bottleneck to aid the denoising process.

    Architecture:
        1. Encoder: 5-layer 1D Conv (processes each point independently)
        2. Bottleneck Fusion:
           - Local latent shape: (256, 32) -> flatten to 8192
           - Global context: MaxPool over patch signals
           - Coordinates: MLP(x, y) -> 64
           - Modulation: [latent + shift(global, coord)]
        3. Decoder: 5-layer 1D ConvTranspose (processes each point independently)

    Forward signature:
        denoised = model(noisy_signals, coordinates)
            noisy_signals: (B, R, T=1000)
            coordinates:   (B, R, 2)
            → denoised:    (B, R, T=1000)
    """

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
        # Unused but accepted for API compatibility with train_deepsets_pinn.py
        signal_embed_dim: int = 256,
        coord_embed_dim: int = 64,
        point_dim: int = 256,
        global_dim: int = 128,
    ):
        super().__init__()
        padding = kernel_size // 2
        self.dropout_rate = dropout_rate
        self.coord_dim = coord_dim

        # --- 1. Signal Encoder (DeepCAE backbone) ---
        # Input: (B*R, 1, 1000)
        self.enc1 = nn.Sequential(
            nn.Conv1d(in_channels, base_channels, kernel_size, stride=2, padding=padding),
            nn.BatchNorm1d(base_channels),
            nn.LeakyReLU(0.1, inplace=True),
        )  # 1000 → 500

        self.enc2 = nn.Sequential(
            nn.Conv1d(base_channels, base_channels * 2, kernel_size, stride=2, padding=padding),
            nn.BatchNorm1d(base_channels * 2),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout_rate),
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
        )  # 63 → 32  | Output: (B*R, 256, 32)
        
        # Bottleneck size: 256 channels * 32 spatial length
        self.latent_channels = base_channels * 8
        self.latent_length = 32

        # --- 2. Spatial Auxiliary Fusion ---
        self.coord_mlp = CoordinateMLP(coord_dim=2, embed_dim=coord_dim)

        # Modulator takes [global_channel_ctx (latent_channels) + coord_emb (coord_dim)]
        # and outputs a channel-wise shift vector.
        self.spatial_modulator = nn.Sequential(
            nn.Linear(self.latent_channels + coord_dim, self.latent_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(self.latent_channels, self.latent_channels),
        )

        # --- 3. Signal Decoder (DeepCAE backbone) ---
        # Input: (B*R, 256, 32)
        self.dec5 = nn.Sequential(
            nn.ConvTranspose1d(base_channels * 8, base_channels * 8, kernel_size,
                               stride=2, padding=padding, output_padding=0),
            nn.BatchNorm1d(base_channels * 8),
            nn.LeakyReLU(0.1, inplace=True),
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
        )  # 250 → 500

        self.dec1 = nn.Sequential(
            nn.ConvTranspose1d(base_channels, 1, kernel_size,
                               stride=2, padding=padding, output_padding=1),
            nn.Tanh(),
        )  # 500 → 1000

        # Physics parameters
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
        """Build index arrays for vectorised 2D wave-equation residual."""
        half = P // 2
        interior, left, right, down, up = [], [], [], [], []
        for dc in range(-half, half + 1):
            for dr in range(-half, half + 1):
                if (abs(dc - 1) <= half and abs(dc + 1) <= half
                        and abs(dr - 1) <= half and abs(dr + 1) <= half):
                    idx = (dc + half) * P + (dr + half)
                    interior.append(idx)
                    left.append((dc - 1 + half) * P + (dr + half))
                    right.append((dc + 1 + half) * P + (dr + half))
                    down.append((dc + half) * P + (dr - 1 + half))
                    up.append((dc + half) * P + (dr + 1 + half))

        self.register_buffer('_interior_idx', torch.tensor(interior, dtype=torch.long))
        self.register_buffer('_left_idx',     torch.tensor(left,     dtype=torch.long))
        self.register_buffer('_right_idx',    torch.tensor(right,    dtype=torch.long))
        self.register_buffer('_down_idx',     torch.tensor(down,     dtype=torch.long))
        self.register_buffer('_up_idx',       torch.tensor(up,       dtype=torch.long))

    def _initialize_weights(self) -> None:
        """Kaiming initialisation for Conv and Linear layers."""
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    # ==============================================================
    # Forward Pass
    # ==============================================================

    def forward(
        self,
        noisy_signals: torch.Tensor,
        coordinates: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            noisy_signals: (B, R, 1000)
            coordinates:   (B, R, 2)
        """
        B, R, T = noisy_signals.shape

        # 1. Independent Signal Encoding
        # Reshape to (B*R, 1, 1000) for Conv1d
        x = noisy_signals.view(B * R, 1, T)
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)  # (B*R, 256, 32)
        
        # 2. Spatial Auxiliary Fusion at Bottleneck
        # Pool temporal dimension to get channel descriptors: (B*R, latent_channels)
        local_channel_desc = e5.mean(dim=2)
        local_channel_desc = local_channel_desc.view(B, R, self.latent_channels)
        
        # Aggregate global context across the patch: (B, 1, latent_channels)
        global_ctx, _ = torch.max(local_channel_desc, dim=1, keepdim=True)
        global_ctx = global_ctx.expand(-1, R, -1)  # (B, R, latent_channels)
        
        # Coordinate embedding: (B, R, coord_dim)
        coord_emb = self.coord_mlp(coordinates)
        
        # Modulation: Shift vector based on global context + coords
        mod_input = torch.cat([global_ctx, coord_emb], dim=-1)  # (B, R, latent_channels + coord_dim)
        shift = self.spatial_modulator(mod_input)               # (B, R, latent_channels)
        
        # Additive soft modulation (broadcast along the length dimension 32)
        # shift: (B, R, latent_channels) -> (B*R, latent_channels, 1)
        shift_expanded = shift.view(B * R, self.latent_channels, 1)
        
        modified_latent = e5 + shift_expanded                   # (B*R, latent_channels, 32)
        
        # 3. Independent Signal Decoding
        d5 = self.dec5(modified_latent)
        d4 = self.dec4(d5 + e4)  # Skip connections (U-Net style optional, currently disabled in DeepCAE, kept linear here)
        # DeepCAE does not use UNet skip connections, preserving linear flow:
        d4 = self.dec4(d5)
        d3 = self.dec3(d4)
        d2 = self.dec2(d3)
        d1 = self.dec1(d2)  # (B*R, 1, 1000)
        
        denoised = d1.view(B, R, T)
        return denoised

    # ==============================================================
    # Physics Residual
    # ==============================================================

    def compute_wave_equation_residual(
        self,
        denoised: torch.Tensor,
        grid_indices: torch.Tensor = None,
        grid_cols: int = 41,
        grid_rows: int = 41,
    ) -> torch.Tensor:
        """
        Vectorised 2D acoustic wave equation residual.
        u_tt - c^2 (u_xx + u_yy) = 0
        """
        B, R, T = denoised.shape
        c2 = self.wave_speed ** 2

        u_c = denoised[:, self._interior_idx, :]
        u_l = denoised[:, self._left_idx, :]
        u_r = denoised[:, self._right_idx, :]
        u_d = denoised[:, self._down_idx, :]
        u_u = denoised[:, self._up_idx, :]

        u_tt = (u_c[:, :, 2:] - 2.0 * u_c[:, :, 1:-1] + u_c[:, :, :-2]) \
               / (self.dt ** 2)
        u_xx = (u_r[:, :, 1:-1] - 2.0 * u_c[:, :, 1:-1] + u_l[:, :, 1:-1]) \
               / (self.dx ** 2)
        u_yy = (u_u[:, :, 1:-1] - 2.0 * u_c[:, :, 1:-1] + u_d[:, :, 1:-1]) \
               / (self.dy ** 2)

        residual = (u_tt - c2 * (u_xx + u_yy)) / (self.omega0 ** 2)

        with torch.no_grad():
            N_int = u_c.shape[1]
            u_flat = u_c.reshape(B * N_int, 1, T)
            energy = torch.nn.functional.avg_pool1d(
                u_flat.abs(), kernel_size=9, stride=1, padding=4)
            mask = energy / energy.amax(dim=-1, keepdim=True).clamp_min(1e-6)
            mask = mask.view(B, N_int, T)[:, :, 1:-1]

        return residual * mask

    def physics_forward(
        self,
        noisy_signals: torch.Tensor,
        coordinates: torch.Tensor,
        grid_indices: torch.Tensor = None,
        grid_cols: int = 41,
        grid_rows: int = 41,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        denoised = self.forward(noisy_signals, coordinates)
        residual = self.compute_wave_equation_residual(denoised)
        return denoised, residual


# ============================================================
# Utility Functions
# ============================================================

def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_summary(model: nn.Module,
                        B: int = 2, R: int = 25, T: int = 1000) -> None:
    """Print model architecture summary with a test forward pass."""
    print(f"{'=' * 60}")
    print(f"Model: {model.__class__.__name__}")
    print(f"{'=' * 60}")
    print(f"Total parameters: {count_parameters(model):,}")

    device = next(model.parameters()).device
    signals = torch.randn(B, R, T, device=device)
    coords = torch.rand(B, R, 2, device=device)

    out = model(signals, coords)
    print(f"Input  signals: ({B}, {R}, {T})")
    print(f"Input  coords:  ({B}, {R}, 2)")
    print(f"Output shape:   {tuple(out.shape)}")
    print(f"{'=' * 60}")


# ============================================================
# Self-Test
# ============================================================

if __name__ == "__main__":
    B, R, T = 4, 25, 1000
    signals = torch.randn(B, R, T)
    coords = torch.rand(B, R, 2)

    # --- Test 1: DeepSetsPINN ---
    print("=" * 60)
    print("Testing DeepSetsPINN...")
    print("=" * 60)

    model_old = DeepSetsPINN()
    print_model_summary(model_old)

    out = model_old(signals, coords)
    assert out.shape == (B, R, T), f"Shape mismatch: {out.shape}"
    print("✓ DeepSetsPINN forward shape passed")

    grid_indices = torch.zeros(B, R, 2, dtype=torch.long)
    idx = 0
    for dc in range(-2, 3):
        for dr in range(-2, 3):
            grid_indices[:, idx, 0] = 10 + dc
            grid_indices[:, idx, 1] = 10 + dr
            idx += 1

    denoised, residual = model_old.physics_forward(
        signals, coords, grid_indices, grid_cols=41, grid_rows=41)
    print(f"✓ Physics residual shape: {residual.shape}")

    # --- Test 2: SpatialAuxiliaryCAE ---
    print("\n" + "=" * 60)
    print("Testing SpatialAuxiliaryCAE...")
    print("=" * 60)

    model_sac = SpatialAuxiliaryCAE(base_channels=16, coord_dim=64)
    print_model_summary(model_sac)

    out_sac = model_sac(signals, coords)
    assert out_sac.shape == (B, R, T), f"Shape mismatch: {out_sac.shape}"
    print("✓ SpatialAuxiliaryCAE forward shape passed")

    denoised_sac, residual_sac = model_sac.physics_forward(
        signals, coords, grid_indices, grid_cols=41, grid_rows=41)
    print(f"✓ Physics residual shape: {residual_sac.shape}")
    print(f"  Residual mean: {residual_sac.mean().item():.6f}")

    print("\n✓ All tests passed!")

