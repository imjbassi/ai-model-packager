import pytest

from formats import detect_format, describe_support, require_format, supported_extensions


@pytest.mark.parametrize(
    "filename,expected",
    [
        ("model.pth", "pytorch"),
        ("MODEL.PT", "pytorch"),
        ("model.h5", "tensorflow"),
        ("model.keras", "tensorflow"),
        ("model.onnx", "onnx"),
        ("model.joblib", "sklearn"),
        ("nested/dir/model.pkl", "sklearn"),
    ],
)
def test_detect_format(filename, expected):
    assert detect_format(filename).key == expected


def test_detect_format_unknown_returns_none():
    assert detect_format("model.bin") is None


def test_require_format_raises_for_unknown():
    with pytest.raises(ValueError, match="Unsupported model format"):
        require_format("model.bin")


def test_pip_requirements_are_unique_and_include_inference_deps():
    reqs = require_format("model.pth").pip_requirements
    assert reqs == list(dict.fromkeys(reqs))
    assert "torch" in reqs and "Pillow" in reqs


def test_tensorflow_requirements_exclude_torch():
    assert "torch" not in require_format("model.h5").pip_requirements


def test_supported_extensions_and_description_agree():
    description = describe_support()
    for ext in supported_extensions():
        assert ext in description
