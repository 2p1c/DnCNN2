"""Backward-compatible DeepSets model imports.

This module now re-exports DeepSets-related classes from model.model,
so legacy imports keep working after model script consolidation.
"""

from .model import (
    CENTER_FREQUENCY,
    DEFAULT_DX,
    DEFAULT_DY,
    DEFAULT_WAVE_SPEED,
    DURATION,
    NUM_POINTS,
    SAMPLING_RATE,
    CoordinateMLP,
    DeepSetsPINN,
    PointEncoder,
    SetInvariantWavePINN,
    SignalDecoder,
    SignalEncoder,
    SpatialAuxiliaryCAE,
    SpatialContextCAE,
    count_parameters,
    print_model_summary,
)

__all__ = [
    "SAMPLING_RATE",
    "DURATION",
    "NUM_POINTS",
    "CENTER_FREQUENCY",
    "DEFAULT_WAVE_SPEED",
    "DEFAULT_DX",
    "DEFAULT_DY",
    "SignalEncoder",
    "CoordinateMLP",
    "PointEncoder",
    "SignalDecoder",
    "DeepSetsPINN",
    "SpatialAuxiliaryCAE",
    "SetInvariantWavePINN",
    "SpatialContextCAE",
    "count_parameters",
    "print_model_summary",
]
