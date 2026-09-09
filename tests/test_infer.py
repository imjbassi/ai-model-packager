import json

import pytest

import infer

np = pytest.importorskip("numpy")


def test_top_predictions_sorted_and_normalized():
    results = infer.top_predictions([1.0, 3.0, 2.0], top_k=3)

    assert [item["class_id"] for item in results] == [1, 2, 0]
    assert [item["rank"] for item in results] == [1, 2, 3]
    assert sum(item["probability"] for item in results) == pytest.approx(1.0)


def test_top_predictions_clamped_to_class_count():
    assert len(infer.top_predictions([0.1, 0.9], top_k=10)) == 2


def test_top_predictions_uses_labels():
    labels = ["cat", "dog", "bird"]
    assert infer.top_predictions([0.0, 5.0, 0.0], top_k=1, labels=labels)[0]["label"] == "dog"


def test_top_predictions_falls_back_when_labels_are_short():
    assert infer.top_predictions([0.0, 5.0], top_k=1, labels=["cat"])[0]["label"] == "Class 1"


def test_softmax_is_stable_for_large_logits():
    probabilities = infer._softmax([1000.0, 1001.0])

    assert np.isfinite(probabilities).all()
    assert probabilities.sum() == pytest.approx(1.0)


def test_load_labels_reads_text_and_json(tmp_path):
    text_file = tmp_path / "labels.txt"
    text_file.write_text("cat\ndog\n\nbird\n", encoding="utf-8")
    json_file = tmp_path / "labels.json"
    json_file.write_text(json.dumps(["cat", "dog"]), encoding="utf-8")

    assert infer._load_labels(str(text_file)) == ["cat", "dog", "bird"]
    assert infer._load_labels(str(json_file)) == ["cat", "dog"]
    assert infer._load_labels(None) is None
    assert infer._load_labels(str(tmp_path / "missing.txt")) is None


def test_dummy_inputs_have_expected_shapes():
    assert infer._nchw_input(None).shape == (1, 3, 224, 224)
    assert infer._nhwc_input(None).shape == (1, 224, 224, 3)


def test_image_input_is_normalized(tmp_path):
    Image = pytest.importorskip("PIL.Image", reason="Pillow not installed")
    path = tmp_path / "sample.jpg"
    Image.new("RGB", (300, 300), "steelblue").save(path)

    nhwc = infer._nhwc_input(str(path))
    assert nhwc.shape == (1, 224, 224, 3)
    assert 0.0 <= nhwc.min() and nhwc.max() <= 1.0

    assert infer._nchw_input(str(path)).shape == (1, 3, 224, 224)


def test_main_rejects_missing_model(tmp_path, capsys):
    assert infer.main(["--model", str(tmp_path / "missing.pth")]) == 1
    assert "not found" in capsys.readouterr().err


def test_main_rejects_unsupported_format(tmp_path, capsys):
    model = tmp_path / "model.bin"
    model.write_bytes(b"x")

    assert infer.main(["--model", str(model)]) == 1
    assert "Unsupported model format" in capsys.readouterr().err
