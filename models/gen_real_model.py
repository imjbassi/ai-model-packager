#!/usr/bin/env python3
"""Generate a real, pre-trained model for demonstration purposes.

Downloads a torchvision classifier and saves it as a ``.pth`` file, optionally
exporting the same network to ONNX so the other supported formats can be
exercised end to end.
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings

import torch
import torchvision.models as models

DEFAULT_ARCH = "resnet18"
SUPPORTED_ARCHS = ("resnet18", "resnet34", "mobilenet_v2")


def _load_pretrained(arch: str):
    """Load a pre-trained torchvision model, across torchvision versions.

    Args:
        arch: Architecture name from :data:`SUPPORTED_ARCHS`.

    Returns:
        torch.nn.Module: The pre-trained network.
    """
    factory = getattr(models, arch)
    try:
        # torchvision >= 0.13 uses the weights enum API.
        weights_enum = models.get_model_weights(arch).DEFAULT
        return factory(weights=weights_enum)
    except (AttributeError, ValueError):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            return factory(pretrained=True)


def _default_output(arch: str) -> str:
    """Default ``.pth`` path in the repository root."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(os.path.join(script_dir, "..", f"{arch}_full.pth"))


def _print_model_info(output_path: str, arch: str, parameters: int) -> None:
    """Print a short summary of the saved model."""
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print("\nModel saved successfully!")
    print(f"File: {output_path}")
    print(f"Size: {size_mb:.1f} MB")
    print(f"Architecture: {arch}")
    print(f"Parameters: {parameters / 1e6:.1f} million")
    print("Input: 224x224 RGB images")
    print("Output: 1000 ImageNet classes")


def _export_onnx(model, onnx_path: str) -> None:
    """Export ``model`` to ONNX with a dynamic batch dimension."""
    print(f"Exporting ONNX model to {onnx_path} ...")
    dummy = torch.randn(1, 3, 224, 224)
    torch.onnx.export(
        model,
        dummy,
        onnx_path,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
    )
    print(f"ONNX model saved: {onnx_path}")


def generate_model(arch: str = DEFAULT_ARCH, output_path=None, export_onnx: bool = False) -> str:
    """Download a pre-trained model and save it for packaging demos.

    Args:
        arch: Architecture name from :data:`SUPPORTED_ARCHS`.
        output_path: Destination ``.pth`` path. Defaults to the repo root.
        export_onnx: Also write a sibling ``.onnx`` file.

    Returns:
        Path to the saved ``.pth`` file.

    Raises:
        RuntimeError: If the download or save fails.
    """
    output_path = output_path or _default_output(arch)
    try:
        print(f"Generating pre-trained {arch} model...")
        print("Downloading weights from PyTorch Hub (cached after the first run)...")
        model = _load_pretrained(arch)
        model.eval()

        output_dir = os.path.dirname(os.path.abspath(output_path))
        os.makedirs(output_dir, exist_ok=True)

        torch.save(model, output_path)
        if not os.path.exists(output_path):
            raise RuntimeError(f"Failed to save model to {output_path}")

        _print_model_info(
            output_path, arch, sum(p.numel() for p in model.parameters())
        )

        if export_onnx:
            _export_onnx(model, os.path.splitext(output_path)[0] + ".onnx")

        return output_path
    except Exception as exc:
        raise RuntimeError(f"Failed to generate {arch} model: {exc}") from exc


def main(argv=None) -> int:
    """Entry point for ``python models/gen_real_model.py``."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--arch", choices=SUPPORTED_ARCHS, default=DEFAULT_ARCH, help="Architecture to download"
    )
    parser.add_argument("--output", "-o", help="Destination .pth path")
    parser.add_argument(
        "--onnx", action="store_true", help="Also export the model to ONNX"
    )
    args = parser.parse_args(argv)

    try:
        generate_model(args.arch, args.output, export_onnx=args.onnx)
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
