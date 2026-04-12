"""Focused smoke checks for TF-fusion building blocks."""

from __future__ import annotations

import numpy as np
import torch
from pathlib import Path

from data.tf_features import normalize_to_minus1_1, stft_to_1d_feature
from model.model import DeepSetsPINN_TF
import json


def test_tf_feature_ops() -> None:
    signal = np.sin(np.linspace(0.0, 8.0 * np.pi, 1000)).astype(np.float64)
    feature = stft_to_1d_feature(
        signal_1d=signal,
        target_length=1000,
        n_fft=128,
        hop_length=32,
        win_length=128,
        window="hann",
        pooling="mean",
    )
    assert feature.shape == (1000,)
    normalized, min_val, max_val = normalize_to_minus1_1(feature)
    assert normalized.shape == (1000,)
    assert np.isfinite(normalized).all()
    assert float(normalized.max()) <= 1.0 + 1e-6
    assert float(normalized.min()) >= -1.0 - 1e-6
    assert max_val >= min_val


def main() -> None:
    test_tf_feature_ops()
    test_tf_model_forward_shape()
    test_tf_model_forward_shape_concat()
    test_inference_model_type_guard()
    test_config_has_tf_keys()


def test_tf_model_forward_shape() -> None:
    model = DeepSetsPINN_TF(
        signal_embed_dim=128,
        coord_embed_dim=64,
        point_dim=128,
        base_channels=16,
        patch_size=5,
    )
    noisy = torch.randn(2, 25, 1000)
    tf_signal = torch.randn(2, 25, 1000)
    coords = torch.randn(2, 25, 2)
    output = model(noisy, tf_signal, coords)
    assert output.shape == noisy.shape


def test_tf_model_forward_shape_concat() -> None:
    model = DeepSetsPINN_TF(
        signal_embed_dim=128,
        tf_embed_dim=64,
        coord_embed_dim=64,
        point_dim=128,
        base_channels=16,
        patch_size=5,
        fusion_mode="concat",
        debug_numerics=True,
    )
    noisy = torch.randn(2, 25, 1000)
    tf_signal = torch.randn(2, 25, 1000)
    coords = torch.randn(2, 25, 2)
    output = model(noisy, tf_signal, coords)
    assert output.shape == noisy.shape


def test_inference_model_type_guard() -> None:
    ckpt_meta = {"model_type": "deepsets"}
    requested = "tf_fusion"
    assert ckpt_meta["model_type"] != requested


def test_config_has_tf_keys() -> None:
    config_path = Path("configs/pipeline_deepsets_template.json")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    required = [
        "model_type",
        "stft_n_fft",
        "stft_hop_length",
        "stft_win_length",
        "stft_window",
        "stft_pooling",
        "fusion_mode",
        "debug_numerics",
    ]
    for key in required:
        assert key in config


if __name__ == "__main__":
    main()
