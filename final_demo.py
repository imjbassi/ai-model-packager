#!/usr/bin/env python3
"""Read-only walkthrough of the AI Model Packaging Library.

Prints the project structure, features and workflow without building anything,
so it is safe to run during a screencast or a live demo.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Tuple

from formats import FORMATS

SEPARATOR = "=" * 62
CHECK = "[x]"
CROSS = "[ ]"


def format_file_size(size_bytes: int) -> str:
    """Format a byte count in human-readable units (e.g. ``47.2MB``)."""
    if size_bytes < 0:
        return "0B"

    units: List[Tuple[str, int]] = [("GB", 1024**3), ("MB", 1024**2), ("KB", 1024)]
    for unit, threshold in units:
        if size_bytes >= threshold:
            return f"{size_bytes / threshold:.1f}{unit}"
    return f"{size_bytes}B"


def display_project_files() -> None:
    """Show the project files with their sizes and roles."""
    print("\n=== Step 1: Project Structure ===")

    files: Dict[str, str] = {
        "cli.py": "Command-line interface for model packaging",
        "formats.py": "Model format registry and dependency mapping",
        "docker_packager.py": "Docker build context generation and image build",
        "model_loader.py": "Framework-agnostic model loading",
        "infer.py": "Inference pipeline and prediction reporting",
        "package_python.py": "Portable Python package fallback (no Docker)",
        "resnet18_full.pth": "Demo model - pre-trained ResNet-18",
    }

    for filename, description in files.items():
        if not os.path.exists(filename):
            print(f"   {CROSS} {filename:22} {'(missing)':>12}  {description}")
            continue
        try:
            size_str = format_file_size(os.path.getsize(filename))
        except OSError:
            size_str = "unavailable"
        print(f"   {CHECK} {filename:22} {size_str:>12}  {description}")


def display_supported_formats() -> None:
    """List every model format the packager understands."""
    print("\n=== Step 2: Supported Model Formats ===")
    for fmt in FORMATS:
        extensions = ", ".join(fmt.extensions)
        deps = ", ".join(fmt.requirements)
        print(f"   - {fmt.label:24} {extensions:20} -> {deps}")


def display_build_context() -> None:
    """Show the contents of a retained build context, if one exists."""
    print("\n=== Step 3: Docker Build Context ===")
    print("   build_context/  (created with --keep-context)")

    build_context_path = Path("build_context")
    if not build_context_path.exists():
        print("      (not yet generated - run the CLI with --keep-context)")
        return

    try:
        items = sorted(build_context_path.iterdir())
    except OSError as exc:
        print(f"      (error reading directory: {exc})")
        return

    if not items:
        print("      (empty directory)")
        return

    for item in items:
        if item.is_dir():
            print(f"      |-- {item.name}/ (directory)")
            continue
        try:
            print(f"      |-- {item.name} ({format_file_size(item.stat().st_size)})")
        except OSError:
            print(f"      |-- {item.name} (size unavailable)")


def display_cli_usage() -> None:
    """Show the CLI signature and the packaging workflow."""
    print("\n=== Step 4: CLI Interface ===")
    print("   Usage: python cli.py --input MODEL --image NAME:TAG [--mode auto|docker|python]")
    print("\n   Workflow:")

    for idx, step in enumerate(
        [
            "Detect the model format from its extension",
            "Resolve the dependency set for that framework",
            "Assemble a Docker build context (Dockerfile, model, scripts)",
            "Build the container image as an unprivileged user",
            "Verify the image, or fall back to a portable Python package",
        ],
        start=1,
    ):
        print(f"      {idx}. {step}")


def display_features() -> None:
    """List the tool's core features."""
    print("\n=== Step 5: Core Features ===")
    for feature in [
        "Automatic model format detection (PyTorch, TensorFlow, ONNX, scikit-learn)",
        "Per-framework dependency resolution - no unused frameworks in the image",
        "Reproducible, isolated Docker build contexts",
        "Non-root container user and pinned slim base image",
        "Portable Python package fallback when Docker is unavailable",
        "Cross-platform builds via --platform",
        "JSON-formatted predictions for downstream tooling",
        "Structured error handling with meaningful exit codes",
    ]:
        print(f"   {CHECK} {feature}")


def display_technology_stack() -> None:
    """List the technologies the project builds on."""
    print("\n=== Step 6: Technology Stack ===")
    technologies: Dict[str, str] = {
        "PyTorch / TensorFlow": "Supported deep learning frameworks",
        "ONNX Runtime": "Framework-neutral inference",
        "Docker": "Containerization and deployment",
        "Python (argparse)": "Core implementation and CLI",
        "pytest": "Automated test suite",
    }
    for tech_name, purpose in technologies.items():
        print(f"   - {tech_name:22} -> {purpose}")


def display_summary() -> None:
    """Print the closing summary block."""
    print(f"\n{SEPARATOR}")
    print("PROJECT SUMMARY".center(62))
    print(SEPARATOR)
    print("Project:        AI Model Packaging Library")
    print("Goal:           Automate ML model containerization")
    print("Status:         Functional CLI with Docker and Python packaging modes")
    print("Key Innovation: One-command model packaging")
    print("Output:         Production-ready Docker images or portable packages")
    print(SEPARATOR)


def main() -> None:
    """Run the full walkthrough."""
    print(SEPARATOR)
    print("AI Model Packaging Library - Demonstration".center(62))
    print(SEPARATOR)

    display_project_files()
    display_supported_formats()
    display_build_context()
    display_cli_usage()
    display_features()
    display_technology_stack()
    display_summary()


if __name__ == "__main__":
    main()
