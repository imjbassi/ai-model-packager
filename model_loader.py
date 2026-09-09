"""Framework-agnostic model loading.

Heavy frameworks are imported lazily inside each branch so that packaging a
PyTorch model never requires TensorFlow to be installed (and vice versa).
"""
from __future__ import annotations

import os

from formats import require_format


def load_model(path, weights_only: bool = False):
    """Load an AI model from ``path``.

    Supports PyTorch (``.pth``/``.pt``), TensorFlow/Keras (``.h5``/``.keras``),
    ONNX (``.onnx``) and scikit-learn/joblib (``.joblib``/``.pkl``) artifacts.
    PyTorch models are loaded on CPU in evaluation mode.

    Args:
        path: Path to the model file.
        weights_only: For PyTorch only. ``True`` restricts unpickling to plain
            tensors, which is safer but cannot restore a full ``nn.Module``
            saved with ``torch.save(model)``.

    Returns:
        The loaded model object. For ONNX this is an
        ``onnxruntime.InferenceSession``.

    Raises:
        FileNotFoundError: If the model file does not exist.
        ValueError: If the model format is not supported.
        ImportError: If the framework for this format is not installed.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")

    fmt = require_format(path)

    if fmt.key == "pytorch":
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError(
                "PyTorch is required to load .pth/.pt models: pip install torch"
            ) from exc
        model = torch.load(
            path,
            map_location=torch.device("cpu"),
            weights_only=weights_only,
        )
        if hasattr(model, "eval"):
            model.eval()
        return model

    if fmt.key == "tensorflow":
        try:
            import tensorflow as tf
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError(
                "TensorFlow is required to load .h5/.keras models: "
                "pip install tensorflow"
            ) from exc
        return tf.keras.models.load_model(path)

    if fmt.key == "onnx":
        try:
            import onnxruntime as ort
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError(
                "onnxruntime is required to load .onnx models: "
                "pip install onnxruntime"
            ) from exc
        return ort.InferenceSession(path, providers=["CPUExecutionProvider"])

    if fmt.key == "sklearn":
        try:
            import joblib
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError(
                "joblib is required to load .joblib/.pkl models: "
                "pip install joblib"
            ) from exc
        return joblib.load(path)

    # require_format() only returns registered formats, so this is unreachable
    # unless a format is added to formats.py without a branch here.
    raise ValueError(f"No loader implemented for format: {fmt.key}")
