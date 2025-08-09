# AI Model Packager

Turn your machine learning models into Docker containers with just one command! 🚀

## What it does

This tool takes your trained PyTorch or TensorFlow models and packages them into production-ready Docker containers. No more "it works on my machine" - your models will run anywhere Docker runs.

## Quick Start

```bash
# Package your model into a Docker container
python cli.py --input your_model.pth --image my_ai_model:1.0

# Run the containerized model
docker run --rm my_ai_model:1.0
```

### Overview
This library provides a command-line tool that automatically packages PyTorch models into deployable Docker containers with a single command.

### Files Structure
```
capstone5/
├── cli.py                  # Main CLI interface
├── docker_packager.py      # Docker containerization engine
├── model_loader.py         # PyTorch model loading utilities
├── infer.py               # Model inference script
├── package_python.py      # Python packaging alternative
├── test_suite.py          # Comprehensive test suite
├── final_demo.py          # Capstone demonstration script
├── requirements.txt       # Project dependencies
├── resnet18_full.pth      # Pre-trained ResNet-18 model (47MB)
├── sample.jpg            # Test image for inference
└── build_context/        # Docker build context
    ├── Dockerfile
    ├── requirements.txt
    ├── resnet18_full.pth
    ├── infer.py
    └── sample.jpg
```

### Usage

#### Package a model into Docker container:
```bash
python cli.py --input resnet18_full.pth --image my_model:1.0
```

#### Run demonstration:
```bash
python final_demo.py
```

#### Run tests:
```bash
python -m pytest test_suite.py -v
```

#### Create Python package (alternative to Docker):
```bash
python package_python.py --input resnet18_full.pth --name my_package
```

### Key Features
- ✅ One-command model containerization
- ✅ Automatic dependency detection
- ✅ PyTorch model support (.pth files)
- ✅ Docker build context generation
- ✅ Python packaging fallback
- ✅ Comprehensive error handling
- ✅ Cross-platform deployment ready

### Innovation
Transforms complex ML model deployment from a multi-step manual process into a single command operation, enabling rapid containerization and deployment of PyTorch models.

### Technologies
- Python 3.12+
- Docker
- PyTorch
- Subprocess automation
- CLI interface design

### Author
Capstone Project - AI Model Packaging Library
