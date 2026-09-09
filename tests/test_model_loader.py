import pytest

from model_loader import load_model


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_model(tmp_path / "missing.pth")


def test_unsupported_format_raises(tmp_path):
    model = tmp_path / "model.bin"
    model.write_bytes(b"x")

    with pytest.raises(ValueError, match="Unsupported model format"):
        load_model(model)


def test_sklearn_roundtrip(tmp_path):
    joblib = pytest.importorskip("joblib")
    path = tmp_path / "model.joblib"
    joblib.dump({"weights": [1, 2, 3]}, path)

    assert load_model(path) == {"weights": [1, 2, 3]}


def test_pytorch_roundtrip_returns_eval_mode_module(tmp_path):
    torch = pytest.importorskip("torch")
    path = tmp_path / "model.pth"
    torch.save(torch.nn.Linear(4, 2), path)

    model = load_model(path)
    assert isinstance(model, torch.nn.Module)
    assert model.training is False
