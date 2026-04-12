"""Core model definitions for the unified ultrasonic training pipeline."""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn


SAMPLING_RATE: float = 6.25e6
DURATION: float = 160e-6
NUM_POINTS: int = 1000
CENTER_FREQUENCY: float = 250e3
DEFAULT_WAVE_SPEED: float = 5900.0
DEFAULT_DX: float = 1e-3
DEFAULT_DY: float = 1e-3


class DeepCAE(nn.Module):
    """Deep 1D convolutional autoencoder used by CAE/PINN pipeline."""

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 32,
        kernel_size: int = 7,
        dropout_rate: float = 0.1,
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

        self.dec5 = nn.Sequential(
            nn.ConvTranspose1d(
                base_channels * 8,
                base_channels * 8,
                kernel_size,
                stride=2,
                padding=padding,
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
                in_channels,
                kernel_size,
                stride=2,
                padding=padding,
                output_padding=1,
            ),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.enc4(x)
        x = self.enc5(x)
        x = self.dec5(x)
        x = self.dec4(x)
        x = self.dec3(x)
        x = self.dec2(x)
        x = self.dec1(x)
        return x


class DeepCAE_PINN(DeepCAE):
    """Physics-informed DeepCAE with damped narrowband residual."""

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 32,
        kernel_size: int = 7,
        dropout_rate: float = 0.1,
        wave_speed: float = DEFAULT_WAVE_SPEED,
        center_frequency: float = CENTER_FREQUENCY,
        damping_ratio: float = 0.05,
    ):
        super().__init__(
            in_channels=in_channels,
            base_channels=base_channels,
            kernel_size=kernel_size,
            dropout_rate=dropout_rate,
        )
        if wave_speed <= 0 or center_frequency <= 0:
            raise ValueError("wave_speed and center_frequency must be positive")
        if damping_ratio < 0:
            raise ValueError("damping_ratio must be non-negative")

        self.dt = DURATION / (NUM_POINTS - 1)
        self.wave_speed = wave_speed
        self.center_frequency = center_frequency
        self.damping_ratio = damping_ratio
        self.omega0 = 2.0 * math.pi * center_frequency

    def _build_signal_mask(self, u: torch.Tensor) -> torch.Tensor:
        local_energy = torch.nn.functional.avg_pool1d(
            u.abs(), kernel_size=9, stride=1, padding=4
        )
        return local_energy / local_energy.amax(dim=-1, keepdim=True).clamp_min(1e-6)

    def compute_physics_residual(self, u: torch.Tensor) -> torch.Tensor:
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

    def physics_forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        denoised = self.forward(x)
        residual = self.compute_physics_residual(denoised)
        return denoised, residual


class SignalEncoder(nn.Module):
    def __init__(
        self,
        base_channels: int = 32,
        kernel_size: int = 7,
        dropout_rate: float = 0.1,
        embed_dim: int = 256,
    ):
        super().__init__()
        padding = kernel_size // 2
        self.enc1 = nn.Sequential(
            nn.Conv1d(1, base_channels, kernel_size, stride=2, padding=padding),
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
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.enc4(x)
        x = self.enc5(x)
        x = self.pool(x).squeeze(-1)
        return self.proj(x)


class TimeFreqEncoder(SignalEncoder):
    """Encoder for 1D time-frequency proxy sequences."""


class CoordinateMLP(nn.Module):
    def __init__(self, embed_dim: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2, 64), nn.ReLU(inplace=True), nn.Linear(64, embed_dim)
        )

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        return self.mlp(coords)


class PointEncoder(nn.Module):
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


class GatedFeatureFusion(nn.Module):
    """Adaptive gate that blends time and time-frequency embeddings."""

    def __init__(self, embed_dim: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(embed_dim, embed_dim),
            nn.Sigmoid(),
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, f_time: torch.Tensor, f_tf: torch.Tensor) -> torch.Tensor:
        if f_time.shape != f_tf.shape:
            raise ValueError(
                f"f_time and f_tf shape mismatch: {f_time.shape} vs {f_tf.shape}"
            )
        gate = self.gate(torch.cat([f_time, f_tf], dim=-1))
        fused = gate * f_time + (1.0 - gate) * f_tf
        return self.norm(fused)


class ConcatFeatureFusion(nn.Module):
    """Concat + projection fusion for ablation experiments."""

    def __init__(self, embed_dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(embed_dim, embed_dim),
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, f_time: torch.Tensor, f_tf: torch.Tensor) -> torch.Tensor:
        if f_time.shape != f_tf.shape:
            raise ValueError(
                f"f_time and f_tf shape mismatch: {f_time.shape} vs {f_tf.shape}"
            )
        fused = self.proj(torch.cat([f_time, f_tf], dim=-1))
        return self.norm(fused)


class SignalDecoder(nn.Module):
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
    """Set-invariant model with 2D wave-equation residual for patch training."""

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
            base_channels, kernel_size, dropout_rate, signal_embed_dim
        )
        self.coord_mlp = CoordinateMLP(coord_embed_dim)
        self.point_encoder = PointEncoder(signal_embed_dim, coord_embed_dim, point_dim)
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

    def _precompute_patch_topology(self, p: int) -> None:
        half = p // 2
        interior, left, right, down, up = [], [], [], [], []
        for dc in range(-half, half + 1):
            for dr in range(-half, half + 1):
                if (
                    abs(dc - 1) <= half
                    and abs(dc + 1) <= half
                    and abs(dr - 1) <= half
                    and abs(dr + 1) <= half
                ):
                    idx = (dc + half) * p + (dr + half)
                    interior.append(idx)
                    left.append((dc - 1 + half) * p + (dr + half))
                    right.append((dc + 1 + half) * p + (dr + half))
                    down.append((dc + half) * p + (dr - 1 + half))
                    up.append((dc + half) * p + (dr + 1 + half))

        self.register_buffer("_interior_idx", torch.tensor(interior, dtype=torch.long))
        self.register_buffer("_left_idx", torch.tensor(left, dtype=torch.long))
        self.register_buffer("_right_idx", torch.tensor(right, dtype=torch.long))
        self.register_buffer("_down_idx", torch.tensor(down, dtype=torch.long))
        self.register_buffer("_up_idx", torch.tensor(up, dtype=torch.long))

    def forward(
        self, noisy_signals: torch.Tensor, coordinates: torch.Tensor
    ) -> torch.Tensor:
        b, r, t = noisy_signals.shape
        sig_flat = noisy_signals.reshape(b * r, 1, t)
        sig_emb = self.signal_encoder(sig_flat).view(b, r, -1)
        coord_emb = self.coord_mlp(coordinates)
        point_feat = self.point_encoder(sig_emb, coord_emb)
        global_feat = point_feat.mean(dim=1, keepdim=True).expand(-1, r, -1)
        dec_input = torch.cat([point_feat, global_feat], dim=-1)
        out = self.decoder(dec_input.reshape(b * r, -1)).view(b, r, t)
        return out

    def compute_wave_equation_residual(
        self,
        denoised: torch.Tensor,
        grid_indices: Optional[torch.Tensor] = None,
        grid_cols: int = 41,
        grid_rows: int = 41,
    ) -> torch.Tensor:
        del grid_indices, grid_cols, grid_rows
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
            b, n_int, t = u_c.shape
            u_flat = u_c.reshape(b * n_int, 1, t)
            energy = torch.nn.functional.avg_pool1d(
                u_flat.abs(), kernel_size=9, stride=1, padding=4
            )
            mask = energy / energy.amax(dim=-1, keepdim=True).clamp_min(1e-6)
            mask = mask.view(b, n_int, t)[:, :, 1:-1]

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
            denoised, grid_indices, grid_cols, grid_rows
        )
        return denoised, residual


class DeepSetsPINN_TF(DeepSetsPINN):
    """DeepSetsPINN variant with an additional time-frequency branch."""

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
        tf_embed_dim: Optional[int] = None,
        fusion_mode: str = "gated",
        debug_numerics: bool = False,
    ):
        super().__init__(
            signal_embed_dim=signal_embed_dim,
            coord_embed_dim=coord_embed_dim,
            point_dim=point_dim,
            base_channels=base_channels,
            kernel_size=kernel_size,
            dropout_rate=dropout_rate,
            wave_speed=wave_speed,
            center_frequency=center_frequency,
            dx=dx,
            dy=dy,
            patch_size=patch_size,
        )
        resolved_tf_dim = signal_embed_dim if tf_embed_dim is None else tf_embed_dim
        if resolved_tf_dim <= 0:
            raise ValueError("tf_embed_dim must be positive")
        if fusion_mode not in {"gated", "concat"}:
            raise ValueError(f"Unsupported fusion_mode: {fusion_mode}")

        self.tf_encoder = TimeFreqEncoder(
            base_channels=base_channels,
            kernel_size=kernel_size,
            dropout_rate=dropout_rate,
            embed_dim=resolved_tf_dim,
        )
        self.tf_project = (
            nn.Identity()
            if resolved_tf_dim == signal_embed_dim
            else nn.Linear(resolved_tf_dim, signal_embed_dim)
        )
        if fusion_mode == "gated":
            self.fusion = GatedFeatureFusion(embed_dim=signal_embed_dim)
        else:
            self.fusion = ConcatFeatureFusion(embed_dim=signal_embed_dim)
        self.debug_numerics = debug_numerics

    def _check_finite(self, name: str, x: torch.Tensor) -> None:
        if self.debug_numerics and not torch.isfinite(x).all():
            raise ValueError(f"{name} contains NaN or Inf")

    def forward(
        self,
        noisy_signals: torch.Tensor,
        tf_signals: torch.Tensor,
        coordinates: torch.Tensor,
    ) -> torch.Tensor:
        if noisy_signals.shape != tf_signals.shape:
            raise ValueError(
                "noisy_signals and tf_signals must have the same shape, got "
                f"{noisy_signals.shape} vs {tf_signals.shape}"
            )
        b, r, t = noisy_signals.shape
        sig_emb = self.signal_encoder(noisy_signals.reshape(b * r, 1, t)).view(b, r, -1)
        tf_emb = self.tf_encoder(tf_signals.reshape(b * r, 1, t)).view(b, r, -1)
        tf_emb = self.tf_project(tf_emb)
        self._check_finite("sig_emb", sig_emb)
        self._check_finite("tf_emb", tf_emb)
        fused_emb = self.fusion(sig_emb, tf_emb)
        self._check_finite("fused_emb", fused_emb)
        coord_emb = self.coord_mlp(coordinates)
        point_feat = self.point_encoder(fused_emb, coord_emb)
        global_feat = point_feat.mean(dim=1, keepdim=True).expand(-1, r, -1)
        dec_input = torch.cat([point_feat, global_feat], dim=-1)
        out = self.decoder(dec_input.reshape(b * r, -1)).view(b, r, t)
        self._check_finite("denoised", out)
        return out

    def physics_forward(
        self,
        noisy_signals: torch.Tensor,
        tf_signals: torch.Tensor,
        coordinates: torch.Tensor,
        grid_indices: Optional[torch.Tensor] = None,
        grid_cols: int = 41,
        grid_rows: int = 41,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        denoised = self.forward(noisy_signals, tf_signals, coordinates)
        residual = self.compute_wave_equation_residual(
            denoised, grid_indices, grid_cols, grid_rows
        )
        self._check_finite("residual", residual)
        return denoised, residual


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
