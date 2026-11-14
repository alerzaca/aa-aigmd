"""Inference adapter for Deezer's research specnn_amplitude checkpoint."""
from __future__ import annotations

import dataclasses
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

SAMPLE_RATE = 44_100
WINDOW_SECONDS = 0.78
N_FFT = 2_048
HOP_LENGTH = 512
HIGH_FREQUENCY_CUTOFF = 16_000
INFERENCE_BATCH_SIZE = 32


@dataclasses.dataclass
class DeezerMeta:
    sample_rate: int = SAMPLE_RATE
    input_length_seconds: float = WINDOW_SECONDS
    max_len_samples: None = None
    num_classes: int = 2
    fake_label_idx: int = 1
    real_label_idx: int = 0


class DeezerDetector(nn.Module):
    """Run the SavedModel checkpoint; predictions are intentionally inference-only."""

    supports_attacks = False

    def __init__(self, checkpoint: str | Path):
        super().__init__()
        checkpoint = Path(checkpoint).expanduser().resolve()
        if not (checkpoint / "saved_model.pb").is_file():
            raise FileNotFoundError(
                f"Deezer SavedModel not found at {checkpoint}. "
                "Set AA_AIMD_DEEZER_CHECKPOINT to its extracted directory."
            )

        try:
            import tensorflow as tf
        except ImportError as error:
            raise RuntimeError(
                "The Deezer detector requires TensorFlow. "
                "Install its optional dependencies with "
                "`python -m pip install -r requirements-deezer.txt`."
            ) from error

        self._tf = tf
        self._saved_model = tf.saved_model.load(str(checkpoint))
        try:
            self._signature = self._saved_model.signatures["serving_default"]
        except KeyError as error:
            raise RuntimeError(
                f"No serving_default signature in Deezer checkpoint: {checkpoint}"
            ) from error

        positional, keyword = self._signature.structured_input_signature
        if len(positional) == 1 and not keyword:
            self._input_name = None
        elif not positional and len(keyword) == 1:
            self._input_name = next(iter(keyword))
        else:
            raise RuntimeError(
                "Expected Deezer SavedModel to accept exactly one spectrogram input."
            )

        output_specs = self._signature.structured_outputs
        if not isinstance(output_specs, dict):
            output_specs = {"output": output_specs}
        named_outputs = [
            name for name in output_specs if "deepfake" in name.lower()
        ]
        if len(named_outputs) == 1:
            self._output_name = named_outputs[0]
        elif len(output_specs) == 1:
            self._output_name = next(iter(output_specs))
        else:
            scalar_outputs = [
                name
                for name, spec in output_specs.items()
                if spec.shape.rank is not None
                and spec.shape.rank >= 2
                and spec.shape[-1] == 1
            ]
            if len(scalar_outputs) != 1:
                raise RuntimeError(
                    "Could not identify the deepfake probability output in "
                    f"Deezer checkpoint outputs: {tuple(output_specs)}"
                )
            self._output_name = scalar_outputs[0]

        self.meta = DeezerMeta()
        self.checkpoint = checkpoint

    def prepare_audio(self, waveform: torch.Tensor) -> torch.Tensor:
        # Deezer was trained on short random crops; forward scores each crop in
        # the supplied clip rather than discarding the rest of a full song.
        return waveform

    def _make_spectrograms(
        self, waveform: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        waveforms = waveform.detach().to(device="cpu", dtype=torch.float32)
        if waveforms.ndim == 1:
            waveforms = waveforms.unsqueeze(0)
        if waveforms.ndim != 2:
            raise ValueError(
                f"Expected audio shaped [batch, samples], got {tuple(waveforms.shape)}."
            )

        window_samples = round(SAMPLE_RATE * WINDOW_SECONDS)
        windows: list[torch.Tensor] = []
        owners: list[int] = []
        for owner, audio in enumerate(waveforms):
            if audio.numel() == 0:
                raise ValueError("Cannot run Deezer inference on empty audio.")
            if audio.numel() <= window_samples:
                starts = [0]
            else:
                n_windows = math.ceil(audio.numel() / window_samples)
                last_start = audio.numel() - window_samples
                starts = (
                    torch.linspace(0, last_start, n_windows)
                    .round()
                    .to(torch.long)
                    .tolist()
                )
            for start in starts:
                segment = audio[start : start + window_samples]
                segment = F.pad(segment, (0, window_samples - segment.numel()))
                windows.append(segment)
                owners.append(owner)

        batches = torch.stack(windows)
        window = torch.hann_window(N_FFT)
        power = torch.stft(
            batches,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            win_length=N_FFT,
            window=window,
            center=False,
            return_complex=True,
        ).abs().square()
        # Match EvalAugmenter: log10(power), normalize, then keep frequencies
        # below 16 kHz. TensorFlow's STFT emits time before frequency.
        spectrograms = (torch.log10(power.clamp(1e-10, 1e6)) + 4.0) / 3.0
        spectrograms = spectrograms.transpose(1, 2)
        n_bins = int(
            (HIGH_FREQUENCY_CUTOFF * 2 / SAMPLE_RATE) * (N_FFT // 2 + 1)
        )
        spectrograms = spectrograms[:, :, :n_bins].unsqueeze(-1).contiguous()
        return spectrograms, torch.tensor(owners, dtype=torch.long)

    def _call_model(self, inputs: Any):
        if self._input_name is None:
            outputs = self._signature(inputs)
        else:
            outputs = self._signature(**{self._input_name: inputs})
        if isinstance(outputs, dict):
            return outputs[self._output_name]
        return outputs

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.requires_grad:
            raise RuntimeError(
                "The Deezer SavedModel adapter is inference-only and does not "
                "support gradients or adversarial attacks."
            )

        spectrograms, owners = self._make_spectrograms(waveform)
        probabilities = []
        for batch in spectrograms.split(INFERENCE_BATCH_SIZE):
            output = self._call_model(
                self._tf.convert_to_tensor(batch.numpy(), dtype=self._tf.float32)
            )
            output = output.numpy().reshape(-1)
            if output.size != batch.shape[0]:
                raise RuntimeError(
                    "Deezer checkpoint returned an unexpected output shape: "
                    f"{output.shape} for batch size {batch.shape[0]}."
                )
            # The research loader labels real audio 1 and reconstructed audio 0,
            # despite the Keras output being named "deepfake".
            probabilities.append(torch.from_numpy(1.0 - output).to(torch.float32))

        probabilities_tensor = torch.cat(probabilities)
        if (
            not torch.isfinite(probabilities_tensor).all()
            or (probabilities_tensor < 0).any()
            or (probabilities_tensor > 1).any()
        ):
            raise RuntimeError(
                "Deezer checkpoint output was not a finite probability in [0, 1]."
            )

        batch_size = 1 if waveform.ndim == 1 else waveform.shape[0]
        sums = torch.zeros(batch_size, dtype=torch.float32)
        counts = torch.zeros(batch_size, dtype=torch.float32)
        sums.index_add_(0, owners, probabilities_tensor)
        counts.index_add_(0, owners, torch.ones_like(probabilities_tensor))
        mean_probability = sums / counts
        eps = torch.finfo(mean_probability.dtype).eps
        logit = torch.logit(mean_probability.clamp(eps, 1 - eps))
        logits = torch.stack((torch.zeros_like(logit), logit), dim=-1)
        return logits.to(device=waveform.device)
