"""Model format detection shared by the loader, the packager and the CLI.

Keeping the format table in one place means adding a new framework only
requires a new :class:`ModelFormat` entry here plus a loader branch in
``model_loader.py``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass(frozen=True)
class ModelFormat:
    """Describes one supported serialization format."""

    key: str
    label: str
    extensions: tuple
    requirements: tuple
    # Extra pip packages needed only when running inference in a container.
    inference_requirements: tuple = field(default=("Pillow", "numpy"))

    @property
    def pip_requirements(self) -> List[str]:
        """Full dependency list for a container/package built for this format."""
        seen: Dict[str, None] = {}
        for req in tuple(self.requirements) + tuple(self.inference_requirements):
            seen.setdefault(req, None)
        return list(seen)


FORMATS: tuple = (
    ModelFormat(
        key="pytorch",
        label="PyTorch",
        extensions=(".pth", ".pt"),
        requirements=("torch", "torchvision"),
    ),
    ModelFormat(
        key="tensorflow",
        label="TensorFlow / Keras",
        extensions=(".h5", ".keras"),
        requirements=("tensorflow",),
    ),
    ModelFormat(
        key="onnx",
        label="ONNX",
        extensions=(".onnx",),
        requirements=("onnxruntime",),
    ),
    ModelFormat(
        key="sklearn",
        label="scikit-learn / joblib",
        extensions=(".joblib", ".pkl"),
        requirements=("scikit-learn", "joblib"),
    ),
)

_BY_EXTENSION: Dict[str, ModelFormat] = {
    ext: fmt for fmt in FORMATS for ext in fmt.extensions
}


def supported_extensions() -> List[str]:
    """Every extension the tool knows how to package, in registration order."""
    return list(_BY_EXTENSION)


def detect_format(path) -> Optional[ModelFormat]:
    """Return the :class:`ModelFormat` for ``path``, or ``None`` if unknown."""
    return _BY_EXTENSION.get(Path(path).suffix.lower())


def require_format(path) -> ModelFormat:
    """Like :func:`detect_format` but raises for unsupported files.

    Raises:
        ValueError: If the extension is not one of :func:`supported_extensions`.
    """
    fmt = detect_format(path)
    if fmt is None:
        ext = Path(path).suffix or "(no extension)"
        raise ValueError(
            f"Unsupported model format: {ext}. "
            f"Supported extensions: {', '.join(supported_extensions())}."
        )
    return fmt


def describe_support() -> str:
    """Human-readable summary used in CLI help text."""
    return "; ".join(
        f"{fmt.label} ({', '.join(fmt.extensions)})" for fmt in FORMATS
    )
