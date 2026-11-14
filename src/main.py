## Main CLI interface
from __future__ import annotations

import argparse
import importlib
import sys
from collections.abc import Sequence
from pathlib import Path

import pandas as pd
import torch

from src.attacks.registry import ATTACK_REGISTRY
from src.audio import load_audio
from src.detectors.registry import DETECTOR_REGISTRY, load_detector
from src.experiment import run_attack_sweep, run_baseline

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def _add_detector_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--detector", choices=tuple(DETECTOR_REGISTRY), required=True)


def _prepare_sample(args):
    ## infer/check-gradient inspect one sample, not a baseline or an attack sweep.
    detector = load_detector(args.detector).eval()

    sample_path = args.sample_file
    if sample_path is None:
        for label in ("fake", "real"):
            sample_path = next(
                (
                    path
                    for path in sorted((DEFAULT_DATA_DIR / label).rglob("*"))
                    if path.suffix.lower() in {".mp3", ".wav"}
                ),
                None,
            )
            if sample_path is not None:
                break

    if sample_path is None:
        raise FileNotFoundError(
            f"No audio file found in {DEFAULT_DATA_DIR / 'fake'} or {DEFAULT_DATA_DIR / 'real'}"
        )

    waveform, _ = load_audio(str(sample_path), target_sr=detector.meta.sample_rate)
    waveform = detector.prepare_audio(waveform)
    return detector, sample_path, waveform.unsqueeze(0)


def _run_check_env(_args) -> None:
    required_packages = [
        "torch",
        "torchaudio",
        "librosa",
        "soundfile",
        "numpy",
        "pandas",
        "matplotlib",
    ]
    optional_packages = [
        "pesq",
        "pystoi",
        "sonics",
        "huggingface_hub",
        "tensorflow",
    ]
    missing = []
    for name, required in [(name, True) for name in required_packages] + [
        (name, False) for name in optional_packages
    ]:
        try:
            module = importlib.import_module(name)
            version = getattr(module, "__version__", "no __version__ attribute")
            print(f"[OK]   {name:<20} version: {version}")
        except ImportError as error:
            status = "ERROR" if required else "missing (optional)"
            print(f"[{status}] {name:<20} -> {error}")
            if required:
                missing.append(name)

    print(f"\nCUDA available: {torch.cuda.is_available()}")
    print(f"MPS available: {torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else 'n/a'}")
    print(f"Python version: {sys.version}")
    if missing:
        raise SystemExit(f"Missing required packages: {', '.join(missing)}")


def _run_infer(args) -> None:
    detector, sample_path, inputs = _prepare_sample(args)
    with torch.no_grad():
        logits = detector(inputs)
    probabilities = torch.softmax(logits, dim=-1)[0]
    prediction = logits.argmax(dim=-1).item()
    print(f"{sample_path}: real={probabilities[0]:.4f}, fake={probabilities[1]:.4f}, pred={prediction}")


def _run_check_gradient(args) -> None:
    detector, _, inputs = _prepare_sample(args)
    if not getattr(detector, "supports_attacks", True):
        raise ValueError(f"{args.detector} does not support gradient checks.")
    inputs.requires_grad_(True)
    logits = detector(inputs)
    fake_label = (detector.meta.fake_label_idx if detector.meta.fake_label_idx is not None else 1)

    ## This checks d(logit)/d(waveform) connectivity through the full detector;
    ## it does not run an attack or prove that an attack will succeed.
    logits[:, fake_label].sum().backward()
    if inputs.grad is None or inputs.grad.abs().sum().item() == 0:
        raise RuntimeError("Gradient does not reach the waveform input")
    print(f"sum(|gradient|) = {inputs.grad.abs().sum().item():.6f}")


def _device() -> torch.device:
    return torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )


def _run_baseline(args) -> None:
    print(f"Loading {args.detector} detector for {args.representation} baseline...", flush=True)
    detector = load_detector(args.detector).eval()
    device = _device()
    print(f"Detector ready on {device}. Scanning audio under {args.data_dir}...", flush=True)
    result = run_baseline(
        detector.to(device),
        detector.meta,
        args.data_dir,
        args.results_dir,
        device,
        args.representation,
        args.reconstruction,
    )
    ## Podsumowanie grup zostało zebrane w trakcie zapisu, bez wczytywania całego CSV.
    print(result.summary.rename(columns={"size": "n"}).to_string())
    print(f"Baseline saved to {result.metrics_path}")


def _run_attack(args) -> None:
    detector = load_detector(args.detector).eval()
    if not getattr(detector, "supports_attacks", True):
        raise ValueError(f"{args.detector} does not support adversarial attacks.")
    meta = detector.meta
    device = _device()
    detector.to(device)
    files = sorted(
        p
        for p in (args.data_dir / "fake" / args.source).rglob("*")
        if p.suffix.lower() in {".mp3", ".wav"}
    )
    if not files:
        raise FileNotFoundError(f"No fake audio files found in {args.data_dir / 'fake'}")

    ## Epsilon, C&W c and noise SNR are different quantities, not one shared scale.
    budget_name = ATTACK_REGISTRY[args.attack].budget_name
    budgets = (
        args.snr_db
        if budget_name == "snr_db_target"
        else [args.c] if budget_name == "c" else args.epsilon
    )
    ## Atakujemy pliki fake, kierując detektor zawsze na klasę real.
    target = meta.real_label_idx if meta.real_label_idx is not None else 0
    result = run_attack_sweep(
        detector,
        meta,
        files,
        args.data_dir,
        args.attack,
        budgets,
        args.results_dir,
        device,
        representation=args.representation,
        reconstruction=args.reconstruction,
        alpha=args.alpha,
        steps=args.steps,
        target_label=target,
    )
    print(f"Metrics saved to {result.metrics_path}")
    print(pd.Series(result.mean_p_fake_by_budget, name="p_fake_after").sort_index())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)

    env_parser = commands.add_parser("check-env", help="Check dependencies and hardware")
    env_parser.set_defaults(handler=_run_check_env)

    infer_parser = commands.add_parser("infer", help="Run detector inference on one audio file")
    _add_detector_arguments(infer_parser)
    infer_parser.add_argument("--sample-file", type=Path)
    infer_parser.set_defaults(handler=_run_infer)

    gradient_parser = commands.add_parser("check-gradient", help="Check waveform gradient flow")
    _add_detector_arguments(gradient_parser)
    gradient_parser.add_argument("--sample-file", type=Path)
    gradient_parser.set_defaults(handler=_run_check_gradient)

    baseline_parser = commands.add_parser("baseline", help="Clean detection on all real and fake data (no attacks)")
    _add_detector_arguments(baseline_parser)
    baseline_parser.add_argument("--representation", choices=("waveform", "mel"), default="waveform")
    baseline_parser.add_argument("--reconstruction", choices=("source_phase", "griffin_lim"), default="source_phase", help="Mel to waveform reconstruction")
    baseline_parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    baseline_parser.add_argument("--results-dir", type=Path, required=True)
    baseline_parser.set_defaults(handler=_run_baseline)

    attack_parser = commands.add_parser("attack", help="Run an adversarial attack sweep")
    _add_detector_arguments(attack_parser)
    attack_parser.add_argument("--representation", choices=("waveform", "mel"), default="waveform", help="Perturb waveform or mel features")
    attack_parser.add_argument("--attack", choices=tuple(ATTACK_REGISTRY), required=True)
    attack_parser.add_argument("--reconstruction", choices=("source_phase", "griffin_lim"), default="source_phase", help="Mel to waveform reconstruction (mel representation only)",)
    attack_parser.add_argument("--epsilon", type=float, nargs="+", default=[0.005])
    attack_parser.add_argument("--alpha", type=float, default=0.001, help="PGD step size")
    attack_parser.add_argument("--steps", type=int, default=20)
    attack_parser.add_argument("--c", type=float, default=1.0, help="C&W classification-margin weight")
    attack_parser.add_argument("--snr-db", type=float, nargs="+", default=[20.0])
    attack_parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    attack_parser.add_argument("--source", default="", help="Subfolder of data/fake to attack (e.g. sonics, echoes); default: all",)
    attack_parser.add_argument("--results-dir", type=Path, required=True)
    attack_parser.set_defaults(handler=_run_attack)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.handler(args)


if __name__ == "__main__":
    main()
