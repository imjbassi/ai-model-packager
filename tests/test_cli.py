import pytest

import cli
import docker_packager


@pytest.fixture
def fake_model(tmp_path):
    model = tmp_path / "demo.pth"
    model.write_bytes(b"not-a-real-model")
    return model


def test_requires_input_and_image():
    with pytest.raises(SystemExit):
        cli.main([])


def test_python_mode_skips_docker(fake_model, tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("Docker must not be consulted in python mode")

    monkeypatch.setattr(cli, "check_docker_available", fail)

    exit_code = cli.main(
        ["-i", str(fake_model), "-t", "my_model:1.0", "--mode", "python", "-o", str(tmp_path)]
    )

    assert exit_code == 0
    assert (tmp_path / "my_model_1.0_package").is_dir()


def test_auto_mode_falls_back_to_python_package(fake_model, tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "check_docker_available", lambda *a, **k: False)

    exit_code = cli.main(["-i", str(fake_model), "-t", "my_model:1.0", "-o", str(tmp_path)])

    assert exit_code == 0
    assert (tmp_path / "my_model_1.0_package.zip").exists()


def test_docker_mode_reports_missing_docker(fake_model, monkeypatch, capsys):
    monkeypatch.setattr(docker_packager, "check_docker_available", lambda *a, **k: False)

    exit_code = cli.main(["-i", str(fake_model), "-t", "my_model:1.0", "--mode", "docker"])

    assert exit_code == 1
    assert "--mode python" in capsys.readouterr().err


def test_missing_model_returns_error(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(docker_packager, "check_docker_available", lambda *a, **k: True)

    exit_code = cli.main(
        ["-i", str(tmp_path / "missing.pth"), "-t", "my_model:1.0", "--mode", "docker"]
    )

    assert exit_code == 1
    assert "not found" in capsys.readouterr().err


def test_invalid_image_name_returns_error(fake_model, capsys):
    exit_code = cli.main(["-i", str(fake_model), "-t", "Bad_Name:1.0", "--mode", "docker"])

    assert exit_code == 1
    assert "lowercase" in capsys.readouterr().err


def test_package_name_is_filesystem_safe():
    assert cli._package_name_from_image("registry.io/team/model:1.0") == (
        "registry.io_team_model_1.0"
    )
