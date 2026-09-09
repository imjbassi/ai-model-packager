# Changelog

All notable changes to this project are documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [1.1.0]

### Added
- `formats.py`: a single registry mapping file extensions to frameworks and their
  pip dependencies, shared by the loader, packager and CLI.
- ONNX (`.onnx`) and scikit-learn/joblib (`.joblib`, `.pkl`) support alongside
  PyTorch and TensorFlow.
- CLI flags: `--mode {auto,docker,python}`, `--output-dir`, `--python-version`,
  `--platform`, `--no-cache`, `--keep-context`, `--version`.
- `infer.py` flags: `--top-k`, `--labels`, `--json`.
- Non-root `appuser` and a generated `.dockerignore` in produced images.
- `run.sh` and a per-package `README.md` in the portable Python package; extra
  arguments passed to `run.py` are now forwarded to `infer.py`.
- `models/gen_real_model.py`: `--arch`, `--output` and `--onnx` export options.
- pytest suite (`tests/`) and a GitHub Actions CI workflow across Python 3.9–3.12.
- `pyproject.toml` with per-framework extras and an `ai-model-packager` console script.

### Changed
- Docker build contexts are assembled in a temporary directory and removed after
  the build, so stale files no longer leak between runs.
- Container base image moved from `python:3.9-slim` to `python:3.11-slim`
  (configurable via `--python-version`), with an `ENTRYPOINT`/`CMD` split so
  inference arguments can be overridden at `docker run` time.
- Packaging failures now raise typed exceptions (`DockerUnavailableError`,
  `PackagingError`) instead of returning booleans that the CLI ignored.
- `infer.py` routes every framework through `model_loader.load_model()` rather
  than reimplementing model loading.
- `model_loader.py` imports frameworks lazily, so PyTorch users no longer need
  TensorFlow installed.

### Fixed
- The CLI exited `0` even when Docker packaging failed.
- The Python-package fallback wrote `torch` requirements for every model,
  including TensorFlow ones.
- Generated `Dockerfile` unconditionally copied `sample.jpg`, breaking the build
  when Pillow was unavailable to create it.
- `run.py` in the generated package never passed the sample input to `infer.py`
  and broke when invoked from another directory.
- Zip archive names were truncated for package names containing a dot.
- `infer.py` only recognized `.pth` and `.h5`, unlike `model_loader.py`.
- Sample-image creation spawned a `python -c` subprocess instead of using Pillow
  in-process.
- `.gitignore` excluded `test_*.py`, which would have ignored the test suite.
- Mojibake in the README and console output from mis-encoded emoji.

## [1.0.0]

- Initial capstone release: PyTorch/TensorFlow model packaging into Docker
  images, with a portable Python package fallback.
