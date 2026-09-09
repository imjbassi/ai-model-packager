# AI Model Packaging Library

[![CI](https://github.com/imjbassi/ai-model-packager/actions/workflows/ci.yml/badge.svg)](https://github.com/imjbassi/ai-model-packager/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

## Abstract

The AI Model Packaging Library is a command-line utility that automates the packaging of trained machine learning models into deployable Docker container images. Where a container runtime is unavailable, the same command produces an equivalent self-contained Python package. The tool was developed as the capstone project (Milestone 5) for the Master of Science in Software Engineering at Grand Canyon University.

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Scope and Capabilities](#2-scope-and-capabilities)
3. [System Architecture](#3-system-architecture)
4. [Installation](#4-installation)
5. [Usage](#5-usage)
6. [Command Reference](#6-command-reference)
7. [Representative Output](#7-representative-output)
8. [Testing and Continuous Integration](#8-testing-and-continuous-integration)
9. [Security Considerations](#9-security-considerations)
10. [Troubleshooting](#10-troubleshooting)
11. [Limitations and Future Work](#11-limitations-and-future-work)
12. [References](#12-references)
13. [Project Information](#13-project-information)

---

## 1. Problem Statement

The deployment of trained models into production remains a recurring source of friction in machine learning workflows. Practitioners are conventionally required to author a Dockerfile by hand, reconcile framework-specific dependencies, integrate the model with an inference entry point, and possess working knowledge of container tooling and DevOps practice. These requirements fall outside the competency ordinarily expected of a data scientist and constitute a barrier to reproducible deployment.

This project addresses that gap by reducing the packaging process to a single command that accepts a trained model and emits a deployable artifact.

### Demonstration

![AI Model Packager in action](assets/demo.gif)

*A single command converts a trained model into a deployable artifact. Where Docker is unavailable, the tool falls back automatically to a portable Python package. A screen recording is also available in [MP4 format](assets/demo.mp4).*

---

## 2. Scope and Capabilities

The library provides the following capabilities:

| Capability | Description |
| --- | --- |
| Format detection | The serialization format is inferred from the file extension across four frameworks. |
| Dependency resolution | Only the dependencies required by the detected framework are installed, so a PyTorch image does not carry TensorFlow. |
| Container generation | The Dockerfile, requirements manifest, build context, and image are produced without user intervention. |
| Image hardening | Images use a pinned slim base image, an unprivileged runtime user, and a generated `.dockerignore`. |
| Build isolation | The build context is assembled in a temporary directory, preventing artifacts from one run from contaminating the next. |
| Container-free fallback | A self-contained Python package is produced with `run.py`, `run.sh`, and `run.bat` launchers. |
| Cross-platform builds | A target architecture may be specified through `--platform`. |
| Machine-readable output | Predictions may be emitted as JSON for consumption by downstream tooling. |
| Deterministic exit codes | `0` denotes success, `1` failure, and `130` user cancellation. |
| Verification | An automated test suite executes under continuous integration on Python 3.9, 3.11, and 3.12. |

### 2.1 Supported Model Formats

| Framework | Extensions | Dependencies installed |
| --- | --- | --- |
| PyTorch | `.pth`, `.pt` | `torch`, `torchvision` |
| TensorFlow / Keras | `.h5`, `.keras` | `tensorflow` |
| ONNX | `.onnx` | `onnxruntime` |
| scikit-learn / joblib | `.joblib`, `.pkl` | `scikit-learn`, `joblib` |

Support for an additional framework requires one entry in `formats.py` and one corresponding branch in `model_loader.py`.

---

## 3. System Architecture

### 3.1 Module Organization

```
ai-model-packager/
├── cli.py                  # Command-line entry point and mode selection
├── formats.py              # Format registry: extension → framework → dependencies
├── docker_packager.py      # Build-context assembly and image construction
├── package_python.py       # Portable Python-package fallback
├── model_loader.py         # Framework-agnostic model loading
├── infer.py                # Inference pipeline and prediction reporting
├── final_demo.py           # Read-only project walkthrough
├── models/
│   └── gen_real_model.py   # Generation of a pretrained demonstration model
├── tests/                  # Automated test suite
├── .github/workflows/ci.yml
├── assets/                 # Demonstration media
├── pyproject.toml          # Packaging metadata and optional extras
├── requirements.txt        # Development dependencies
├── CHANGELOG.md
├── LICENSE
└── README.md
```

### 3.2 Processing Pipeline

1. The model format is detected from the file extension by `formats.py`.
2. The dependency set corresponding to the detected framework is resolved.
3. A build context is assembled, comprising the Dockerfile, the requirements manifest, the model, and the inference scripts.
4. The container image is constructed and configured to execute as an unprivileged user.
5. The resulting image is verified; where Docker is unavailable, a portable Python package is produced instead.

`formats.py` serves as the single authoritative registry consulted by the loader, the packager, and the command-line interface, thereby preventing divergence between the formats each component accepts.

---

## 4. Installation

### 4.1 Prerequisites

- Python 3.9 or later
- Docker Desktop or Docker Engine (optional; absent a container runtime, the tool falls back to Python packaging)
- Git

### 4.2 Repository Acquisition

```bash
git clone https://github.com/imjbassi/ai-model-packager.git
cd ai-model-packager
```

### 4.3 Dependency Installation

```bash
pip install -r requirements.txt
```

Alternatively, the tool may be installed with only the framework required for a given workflow:

```bash
pip install -e ".[pytorch]"    # or .[tensorflow] / .[onnx] / .[sklearn] / .[dev]
```

Installation additionally provides an `ai-model-packager` console command equivalent to `python cli.py`.

### 4.4 Generation of a Demonstration Model

```bash
python models/gen_real_model.py
```

This produces `resnet18_full.pth` in the repository root. The `--onnx` option additionally exports `resnet18_full.onnx`, and `--arch mobilenet_v2` selects a smaller architecture.

---

## 5. Usage

### 5.1 Packaging a Model

```bash
python cli.py --input resnet18_full.pth --image my_ai_model:1.0
```

### 5.2 Executing Inference Within the Container

```bash
docker run --rm my_ai_model:1.0
```

An alternative input may be supplied by mounting a directory and overriding the default arguments:

```bash
docker run --rm -v "$PWD:/data" my_ai_model:1.0 --test-input /data/your_image.jpg --top-k 3
```

### 5.3 Representative Invocations

```bash
# Standard container build
python cli.py -i resnet18_full.pth -t resnet_inference:latest

# ONNX model targeting x86 from an ARM host
python cli.py -i model.onnx -t onnx_model:1.0 --platform linux/amd64

# Container-free packaging, written to dist/
python cli.py -i model.h5 -t tf_model:1.0 --mode python -o dist/

# Retention of the generated build context for inspection
python cli.py -i model.pth -t my_model:1.0 --keep-context
```

### 5.4 Direct Inference

```bash
python infer.py --model resnet18_full.pth --test-input sample.jpg --top-k 3
python infer.py --model resnet18_full.pth --labels imagenet_classes.txt --json
```

---

## 6. Command Reference

### 6.1 `cli.py`

| Argument | Description |
| --- | --- |
| `--input`, `-i` | Path to the model file. Required. |
| `--image`, `-t` | Docker image name and tag, for example `my_model:1.0`. Also determines the Python package name. Required. |
| `--mode` | `auto` (default) selects Docker where available and the Python package otherwise; `docker` requires Docker; `python` bypasses Docker entirely. |
| `--output-dir`, `-o` | Destination directory for the Python package. Default: `.` |
| `--python-version` | Python version of the container base image. Default: `3.11` |
| `--platform` | Target platform for the build, for example `linux/amd64`. |
| `--no-cache` | Constructs the image without the Docker layer cache. |
| `--keep-context` | Retains the generated `build_context/` directory. |
| `--version` | Reports the tool version. |

### 6.2 `infer.py`

| Argument | Description |
| --- | --- |
| `--model` | Path to the model file. Required. |
| `--test-input` | Path to an input image. Random input is used when omitted. |
| `--labels` | Class-name file, supplied either as newline-delimited text or as a JSON array. |
| `--top-k` | Number of predictions to report. Default: `5` |
| `--json` | Emits predictions as JSON. |

---

## 7. Representative Output

### 7.1 Image Construction

```
Checking Docker daemon availability...
SUCCESS: Docker daemon is running and responsive
Creating Docker build context in: /tmp/ai_model_packager_ab12cd
Building Docker image: my_ai_model:1.0
============================================================
...
============================================================
SUCCESS: built Docker image: my_ai_model:1.0
VERIFIED: image my_ai_model:1.0 exists and is ready to use
```

### 7.2 Inference

```
Processing image: sample.jpg

Top 5 predictions:
   1. Class 741: 0.0720 (7.2%)
   2. Class 539: 0.0510 (5.1%)
   3. Class 735: 0.0430 (4.3%)
   4. Class 604: 0.0380 (3.8%)
   5. Class 892: 0.0290 (2.9%)
```

---

## 8. Testing and Continuous Integration

```bash
pip install -e ".[dev]"
pytest
```

The suite exercises format detection, build-context generation, image-name validation, the Python-package fallback, prediction ranking, and command-line exit codes. Tests requiring an optional framework are skipped automatically where that framework is not installed. The continuous integration workflow executes the suite on Python 3.9, 3.11, and 3.12, together with an end-to-end packaging job that runs without a container runtime.

---

## 9. Security Considerations

- Generated images execute as an unprivileged user (`appuser`, UID 10001) rather than as root.
- Container isolation constrains the packaged model at runtime.
- Dependencies are installed exclusively from the Python Package Index.
- The use of private container registries is recommended for proprietary or sensitive models.
- Deserialization of a PyTorch `.pth` file executes arbitrary pickled code. Models should therefore be packaged only from trusted sources; `model_loader.load_model()` accepts `weights_only=True` for artifacts containing only a state dictionary.
- Images should be scanned with a vulnerability scanner such as Docker Scout or Trivy prior to deployment.

---

## 10. Troubleshooting

| Symptom | Recommended action |
| --- | --- |
| The image build fails | Confirm that the daemon is running with `docker ps`, inspect the daemon logs, verify available disk space, and retry with `--no-cache` where a cached layer may be stale. |
| The model file is not found | Verify the supplied path and confirm that the model was generated successfully by `models/gen_real_model.py`. |
| The model format is unsupported | Compare the file extension against Section 2.1. Additional frameworks may be registered in `formats.py`. |
| Permission errors occur | On Linux and macOS, either invoke Docker with `sudo` or add the current user to the `docker` group. |

---

## 11. Limitations and Future Work

The present implementation targets image-classification workloads and assumes a 224×224 RGB input convention for its bundled sample inference path. The following extensions are identified for future work:

- Support for additional frameworks, including XGBoost, LightGBM, and safetensors
- Integration with managed cloud deployment platforms (AWS, Azure, GCP)
- Automated model versioning and container registry management
- Generation of a REST API wrapper around the containerized model
- Multi-architecture image manifests by way of `docker buildx`

---

## 12. References

Amazon Web Services. (2024). *Amazon Elastic Container Service documentation*. https://docs.aws.amazon.com/AmazonECS/latest/developerguide/Welcome.html

Docker, Inc. (2024). *Docker overview*. https://docs.docker.com/get-started/overview/

Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep learning*. MIT Press.

Paszke, A., et al. (2019). PyTorch: An imperative style, high-performance deep learning library. *NeurIPS 2019*. https://doi.org/10.48550/arXiv.1912.01703

Red Hat. (2023). *Introduction to containers, Kubernetes, and Red Hat OpenShift*. https://www.redhat.com/en/topics/containers

---

## 13. Project Information

### 13.1 Academic Context

This project was developed as the capstone requirement for the Master of Science in Software Engineering at Grand Canyon University. It demonstrates the application of software engineering principles, DevOps integration, and applied machine learning deployment.

| Field | Value |
| --- | --- |
| Author | Imjot Bassi |
| Institution | Grand Canyon University |
| Program | Master of Science in Software Engineering |
| Project | Capstone Milestone 5 |

### 13.2 Contributing

Contributions are welcomed. The following procedure is requested:

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/your-feature`).
3. Add or update the corresponding tests and confirm that `pytest` passes.
4. Commit the changes (`git commit -m 'Add new feature'`).
5. Push the branch (`git push origin feature/your-feature`).
6. Open a pull request.

A record of changes between releases is maintained in [CHANGELOG.md](CHANGELOG.md).

### 13.3 License

This project is released under the MIT License. The full text is provided in [LICENSE](LICENSE).

### 13.4 Contact

- GitHub: [@imjbassi](https://github.com/imjbassi)
- Repository: [ai-model-packager](https://github.com/imjbassi/ai-model-packager)
