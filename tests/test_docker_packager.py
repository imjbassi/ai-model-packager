import pytest

import docker_packager
from docker_packager import build_context, package_model, validate_image_name


@pytest.fixture
def fake_model(tmp_path):
    """A file with a supported extension; contents are irrelevant here."""
    model = tmp_path / "demo.pth"
    model.write_bytes(b"not-a-real-model")
    return model


@pytest.mark.parametrize("name", ["my_model:1.0", "registry.io/team/model:latest", "model"])
def test_validate_image_name_accepts_valid_references(name):
    assert validate_image_name(name) == name


@pytest.mark.parametrize("name", ["", None, "My_Model:1.0", "my model:1.0", "model:"])
def test_validate_image_name_rejects_invalid_references(name):
    with pytest.raises(ValueError):
        validate_image_name(name)


def test_build_context_writes_expected_files(fake_model, tmp_path):
    context = build_context(fake_model, tmp_path / "ctx")

    for expected in ("Dockerfile", "requirements.txt", ".dockerignore", "demo.pth"):
        assert (context / expected).exists(), expected
    for script in docker_packager.SUPPORT_FILES:
        assert (context / script).exists(), script


def test_build_context_requirements_match_model_format(fake_model, tmp_path):
    context = build_context(fake_model, tmp_path / "ctx")
    requirements = (context / "requirements.txt").read_text()

    assert "torch" in requirements
    assert "tensorflow" not in requirements


def test_dockerfile_references_model_and_runs_as_non_root(fake_model, tmp_path):
    context = build_context(fake_model, tmp_path / "ctx", python_version="3.12")
    dockerfile = (context / "Dockerfile").read_text()

    assert "FROM python:3.12-slim" in dockerfile
    assert "demo.pth" in dockerfile
    assert "USER appuser" in dockerfile


def test_build_context_rejects_unsupported_format(tmp_path):
    model = tmp_path / "model.bin"
    model.write_bytes(b"x")

    with pytest.raises(ValueError):
        build_context(model, tmp_path / "ctx")


def test_build_context_missing_model(tmp_path):
    with pytest.raises(FileNotFoundError):
        build_context(tmp_path / "missing.pth", tmp_path / "ctx")


def test_docker_build_command_includes_flags(tmp_path):
    cmd = docker_packager._docker_build_command(
        "img:1.0", tmp_path, platform="linux/amd64", no_cache=True
    )

    assert cmd[:4] == ["docker", "build", "-t", "img:1.0"]
    assert "--platform" in cmd and "linux/amd64" in cmd
    assert "--no-cache" in cmd
    assert cmd[-1] == str(tmp_path)


def test_package_model_raises_when_docker_unavailable(fake_model, monkeypatch):
    monkeypatch.setattr(docker_packager, "check_docker_available", lambda *a, **k: False)

    with pytest.raises(docker_packager.DockerUnavailableError):
        package_model(fake_model, "img:1.0")


def test_package_model_cleans_up_temp_context(fake_model, monkeypatch):
    contexts = []

    monkeypatch.setattr(docker_packager, "check_docker_available", lambda *a, **k: True)
    monkeypatch.setattr(
        docker_packager.subprocess,
        "run",
        lambda *a, **k: type("R", (), {"returncode": 0, "stdout": "img:1.0"})(),
    )

    real_build_context = docker_packager.build_context

    def spy(model_path, context_dir, **kwargs):
        contexts.append(context_dir)
        return real_build_context(model_path, context_dir, **kwargs)

    monkeypatch.setattr(docker_packager, "build_context", spy)

    assert package_model(fake_model, "img:1.0") is True
    assert contexts and not contexts[0].exists()
