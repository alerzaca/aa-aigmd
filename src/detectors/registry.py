# The runner uses this contract; model-specific preprocessing stays in adapters.
from __future__ import annotations

import os
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Protocol

import torch

from src.detectors.deezer import DeezerDetector
from src.detectors.deezer import DeezerDetector
from src.detectors.lofcz import LofczDetector

if TYPE_CHECKING:
    from src.detectors.sonics import SonicsDetector


class DetectorMetadata(Protocol):
    # None lengths mean the detector does not require a fixed-duration window.
    sample_rate: Optional[int]
    input_length_seconds: Optional[float]
    max_len_samples: Optional[int]
    num_classes: Optional[int]
    fake_label_idx: Optional[int]
    real_label_idx: Optional[int]


class Detector(Protocol):
    meta: DetectorMetadata

    def prepare_audio(self, waveform: torch.Tensor) -> torch.Tensor: ...

    def __call__(self, inputs: torch.Tensor) -> torch.Tensor: ...

    def eval(self) -> Detector: ...

    def to(self, *args, **kwargs) -> Detector: ...


DetectorFactory = Callable[[], Detector]


def _create_sonics(
    checkpoint: str = "awsaf49/sonics-spectttra-gamma-5s",
) -> "SonicsDetector":
    from src.detectors.sonics import (
        SonicsDetector,
        inspect_sonics_meta,
        load_sonics,
    )

    model = load_sonics(checkpoint)
    return SonicsDetector(model, inspect_sonics_meta(model))


def _create_lofcz() -> Detector:
    return LofczDetector()


def _create_deezer() -> DeezerDetector:
    checkpoint = os.environ.get("AA_AIMD_DEEZER_CHECKPOINT")
    if checkpoint is None:
        checkpoint = (
            Path(__file__).resolve().parents[3]
            / "deepfake-detector"
            / "weights"
            / "final"
            / "specnn_amplitude"
        )
    return DeezerDetector(checkpoint)


def _create_ai_music_deepfake() -> AiMusicDeepFakeDetector:
    checkpoint = os.environ.get("AA_AIMD_AI_MUSIC_DEEPFAKE_CHECKPOINT")
    if checkpoint is None:
        local_checkpoint = (
            Path(__file__).resolve().parents[3]
            / "AI-Music-DeepFake-Detector"
            / "outputs"
            / "models"
            / "best_model.pth"
        )
        checkpoint = local_checkpoint if local_checkpoint.is_file() else None
    return AiMusicDeepFakeDetector(checkpoint)


# Distinct checkpoint IDs also get distinct results/<detector>/ directories.
DETECTOR_REGISTRY: dict[str, DetectorFactory] = {
    "sonics": _create_sonics,
    "sonics-alpha-5s": partial(_create_sonics, "awsaf49/sonics-spectttra-alpha-5s"),
    "sonics-alpha-120s": partial(
        _create_sonics, "awsaf49/sonics-spectttra-alpha-120s"
    ),
    "sonics-gamma-120s": partial(
        _create_sonics, "awsaf49/sonics-spectttra-gamma-120s"
    ),
    "lofcz": _create_lofcz,
    "deezer": _create_deezer,
    "ai-music-deepfake": _create_ai_music_deepfake,
}


def load_detector(name: str) -> Detector:
    try:
        factory = DETECTOR_REGISTRY[name]
    except KeyError as error:
        available = ", ".join(sorted(DETECTOR_REGISTRY))
        raise ValueError(
            f"Unknown detector {name!r}. Available detectors: {available}"
        ) from error
    return factory()
