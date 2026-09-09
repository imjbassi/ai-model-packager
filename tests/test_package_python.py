import zipfile

import pytest

import package_python
from package_python import create_python_package


@pytest.fixture
def fake_model(tmp_path):
    model = tmp_path / "demo.pth"
    model.write_bytes(b"not-a-real-model")
    return model


def test_creates_package_directory_and_zip(fake_model, tmp_path):
    zip_path = create_python_package(fake_model, "my_model", output_dir=tmp_path)
    package_dir = tmp_path / "my_model_package"

    assert package_dir.is_dir()
    for expected in ("run.py", "run.bat", "run.sh", "requirements.txt", "README.md", "demo.pth"):
        assert (package_dir / expected).exists(), expected
    for script in package_python.SUPPORT_FILES:
        assert (package_dir / script).exists(), script

    with zipfile.ZipFile(zip_path) as archive:
        names = archive.namelist()
    assert all(name.startswith("my_model_package/") for name in names)


def test_requirements_match_model_format(fake_model, tmp_path):
    create_python_package(fake_model, "my_model", output_dir=tmp_path)
    requirements = (tmp_path / "my_model_package" / "requirements.txt").read_text()

    assert "torch" in requirements
    assert "tensorflow" not in requirements


def test_tensorflow_model_gets_tensorflow_requirements(tmp_path):
    model = tmp_path / "demo.h5"
    model.write_bytes(b"x")

    create_python_package(model, "tf_model", output_dir=tmp_path)
    requirements = (tmp_path / "tf_model_package" / "requirements.txt").read_text()

    assert "tensorflow" in requirements
    assert "torch" not in requirements


def test_zip_name_survives_dots_in_package_name(fake_model, tmp_path):
    zip_path = create_python_package(fake_model, "my_model_1.0", output_dir=tmp_path)
    assert zip_path.endswith("my_model_1.0_package.zip")


def test_run_script_points_at_packaged_model(fake_model, tmp_path):
    create_python_package(fake_model, "my_model", output_dir=tmp_path)
    run_script = (tmp_path / "my_model_package" / "run.py").read_text()

    assert 'MODEL = "demo.pth"' in run_script
    assert "sys.argv[1:]" in run_script


def test_rebuilding_replaces_stale_files(fake_model, tmp_path):
    create_python_package(fake_model, "my_model", output_dir=tmp_path)
    stale = tmp_path / "my_model_package" / "stale.txt"
    stale.write_text("old")

    create_python_package(fake_model, "my_model", output_dir=tmp_path)
    assert not stale.exists()


def test_rejects_unsupported_format(tmp_path):
    model = tmp_path / "model.bin"
    model.write_bytes(b"x")

    with pytest.raises(ValueError):
        create_python_package(model, "bad", output_dir=tmp_path)


def test_missing_model_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        create_python_package(tmp_path / "missing.pth", "nope", output_dir=tmp_path)
