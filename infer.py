"""Inference entry point.

Runs inside the generated Docker image, inside the portable Python package, or
directly from a checkout. Model loading is delegated to ``model_loader`` so all
supported frameworks go through one code path.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Optional, Sequence

from formats import describe_support, require_format
from model_loader import load_model

DEFAULT_IMAGE_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# --------------------------------------------------------------------------- #
# Input preparation
# --------------------------------------------------------------------------- #
def _load_image_array(input_image: str, size: int = DEFAULT_IMAGE_SIZE):
    """Load an image as a float32 HWC array scaled to [0, 1]."""
    import numpy as np
    from PIL import Image

    image = Image.open(input_image).convert("RGB").resize((size, size))
    return np.asarray(image, dtype="float32") / 255.0


def _nchw_input(input_image: Optional[str], size: int = DEFAULT_IMAGE_SIZE):
    """Build a normalized NCHW batch for PyTorch/ONNX, or random data."""
    import numpy as np

    if input_image and os.path.exists(input_image):
        print(f"Processing image: {input_image}")
        array = _load_image_array(input_image, size)
        mean = np.asarray(IMAGENET_MEAN, dtype="float32")
        std = np.asarray(IMAGENET_STD, dtype="float32")
        array = (array - mean) / std
        return np.expand_dims(array.transpose(2, 0, 1), axis=0)

    if input_image:
        print(f"Warning: image file not found: {input_image}", file=sys.stderr)
    print("Using dummy input tensor (no image provided)")
    return np.random.rand(1, 3, size, size).astype("float32")


def _nhwc_input(input_image: Optional[str], size: int = DEFAULT_IMAGE_SIZE):
    """Build an NHWC batch in [0, 1] for TensorFlow, or random data."""
    import numpy as np

    if input_image and os.path.exists(input_image):
        print(f"Processing image: {input_image}")
        return np.expand_dims(_load_image_array(input_image, size), axis=0)

    if input_image:
        print(f"Warning: image file not found: {input_image}", file=sys.stderr)
    print("Using dummy input tensor (no image provided)")
    return np.random.rand(1, size, size, 3).astype("float32")


# --------------------------------------------------------------------------- #
# Output formatting
# --------------------------------------------------------------------------- #
def _softmax(scores):
    """Numerically stable softmax over a 1-D array."""
    import numpy as np

    shifted = np.asarray(scores, dtype="float64") - np.max(scores)
    exp = np.exp(shifted)
    return exp / exp.sum()


def top_predictions(scores, top_k: int = 5, labels: Optional[Sequence[str]] = None):
    """Return the ``top_k`` highest-scoring classes as dictionaries.

    Args:
        scores: 1-D sequence of logits or probabilities.
        top_k: How many predictions to return (clamped to the class count).
        labels: Optional class names, indexed by class id.

    Returns:
        A list of ``{"rank", "class_id", "label", "probability"}`` dicts,
        highest probability first.
    """
    import numpy as np

    probabilities = _softmax(np.ravel(scores))
    top_k = max(1, min(top_k, probabilities.shape[0]))
    order = np.argsort(probabilities)[::-1][:top_k]

    results = []
    for rank, idx in enumerate(order, start=1):
        idx = int(idx)
        label = labels[idx] if labels and idx < len(labels) else f"Class {idx}"
        results.append(
            {
                "rank": rank,
                "class_id": idx,
                "label": label,
                "probability": float(probabilities[idx]),
            }
        )
    return results


def _load_labels(labels_path: Optional[str]) -> Optional[List[str]]:
    """Read newline- or JSON-delimited class names, tolerating a missing file."""
    if not labels_path:
        return None
    if not os.path.exists(labels_path):
        print(f"Warning: labels file not found: {labels_path}", file=sys.stderr)
        return None

    text = open(labels_path, encoding="utf-8").read().strip()
    if text.startswith("["):
        return [str(item) for item in json.loads(text)]
    return [line.strip() for line in text.splitlines() if line.strip()]


def _report(predictions, as_json: bool) -> None:
    """Print predictions as a table or as a JSON document."""
    if as_json:
        print(json.dumps({"predictions": predictions}, indent=2))
        return

    print(f"\nTop {len(predictions)} predictions:")
    for item in predictions:
        pct = item["probability"] * 100
        print(f"   {item['rank']}. {item['label']}: {item['probability']:.4f} ({pct:.1f}%)")


# --------------------------------------------------------------------------- #
# Per-framework inference
# --------------------------------------------------------------------------- #
def run_pytorch_inference(model_path, input_image=None, top_k=5, labels=None, as_json=False):
    """Run inference with a PyTorch model and report the top predictions."""
    import torch

    model = load_model(model_path)
    tensor = torch.from_numpy(_nchw_input(input_image))

    with torch.no_grad():
        outputs = model(tensor)

    scores = outputs.detach().cpu().numpy()
    _report(top_predictions(scores, top_k, labels), as_json)
    return scores


def run_tensorflow_inference(model_path, input_image=None, top_k=5, labels=None, as_json=False):
    """Run inference with a TensorFlow/Keras model."""
    model = load_model(model_path)
    outputs = model(_nhwc_input(input_image), training=False)

    scores = outputs.numpy() if hasattr(outputs, "numpy") else outputs
    _report(top_predictions(scores, top_k, labels), as_json)
    return scores


def run_onnx_inference(model_path, input_image=None, top_k=5, labels=None, as_json=False):
    """Run inference with an ONNX model via onnxruntime."""
    session = load_model(model_path)
    spec = session.get_inputs()[0]

    # ONNX exports are usually NCHW, but fall back to NHWC when the static
    # shape says the channel dimension is last.
    shape = spec.shape
    channels_last = len(shape) == 4 and shape[-1] == 3
    batch = _nhwc_input(input_image) if channels_last else _nchw_input(input_image)

    outputs = session.run(None, {spec.name: batch})
    _report(top_predictions(outputs[0], top_k, labels), as_json)
    return outputs[0]


def run_sklearn_inference(model_path, input_image=None, top_k=5, labels=None, as_json=False):
    """Run inference with a scikit-learn estimator on flattened image pixels."""
    import numpy as np

    model = load_model(model_path)
    features = _nchw_input(input_image).reshape(1, -1)

    if hasattr(model, "predict_proba"):
        scores = np.asarray(model.predict_proba(features))
        _report(top_predictions(scores, top_k, labels), as_json)
        return scores

    prediction = model.predict(features)
    if as_json:
        print(json.dumps({"prediction": np.asarray(prediction).tolist()}, indent=2))
    else:
        print(f"\nPrediction: {np.asarray(prediction).tolist()}")
    return prediction


_RUNNERS = {
    "pytorch": run_pytorch_inference,
    "tensorflow": run_tensorflow_inference,
    "onnx": run_onnx_inference,
    "sklearn": run_sklearn_inference,
}


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the inference CLI."""
    parser = argparse.ArgumentParser(
        description="Run inference on a packaged model. "
        f"Supported formats: {describe_support()}"
    )
    parser.add_argument("--model", required=True, help="Path to the model file")
    parser.add_argument("--test-input", help="Path to an input image for testing")
    parser.add_argument("--labels", help="Optional class-name file (text or JSON list)")
    parser.add_argument(
        "--top-k", type=int, default=5, help="Number of predictions to show (default: 5)"
    )
    parser.add_argument("--json", action="store_true", help="Emit predictions as JSON")
    return parser


def main(argv=None) -> int:
    """Entry point: dispatch to the runner matching the model's format."""
    args = build_parser().parse_args(argv)

    if not os.path.exists(args.model):
        print(f"Error: model file not found: {args.model}", file=sys.stderr)
        return 1

    try:
        fmt = require_format(args.model)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    try:
        _RUNNERS[fmt.key](
            args.model,
            args.test_input,
            top_k=args.top_k,
            labels=_load_labels(args.labels),
            as_json=args.json,
        )
    except ImportError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001 - surface any framework failure cleanly
        print(f"Error during inference: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
