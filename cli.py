"""Command-line entry point for the AI Model Packaging Library."""
from __future__ import annotations

import argparse
import re
import sys

from docker_packager import (
    DEFAULT_PYTHON_VERSION,
    DockerUnavailableError,
    PackagingError,
    check_docker_available,
    package_model,
)
from formats import describe_support
from package_python import create_python_package

__version__ = "1.1.0"


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser."""
    parser = argparse.ArgumentParser(
        prog="ai-model-packager",
        description="Package ML models into Docker containers for easy deployment. "
        "Supported formats: " + describe_support(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s -i resnet18_full.pth -t my_ai_model:1.0
  %(prog)s -i model.onnx -t onnx_model:latest --platform linux/amd64
  %(prog)s -i model.h5 -t tf_model:1.0 --mode python -o dist/
  %(prog)s -i model.pth -t my_model:1.0 --keep-context
        """,
    )
    parser.add_argument(
        "--input", "-i", required=True, metavar="PATH", help="Path to the model file"
    )
    parser.add_argument(
        "--image",
        "-t",
        required=True,
        metavar="NAME:TAG",
        help="Docker image name and tag (also used to name the Python package)",
    )
    parser.add_argument(
        "--mode",
        choices=("auto", "docker", "python"),
        default="auto",
        help="auto: use Docker when available, otherwise build a portable Python "
        "package (default). docker: require Docker. python: skip Docker entirely",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default=".",
        help="Where to write the Python package in python/fallback mode (default: .)",
    )
    parser.add_argument(
        "--python-version",
        default=DEFAULT_PYTHON_VERSION,
        metavar="X.Y",
        help=f"Python version for the container base image (default: {DEFAULT_PYTHON_VERSION})",
    )
    parser.add_argument(
        "--platform",
        metavar="OS/ARCH",
        help="Target platform for the build, e.g. linux/amd64",
    )
    parser.add_argument(
        "--no-cache", action="store_true", help="Build the image without the Docker cache"
    )
    parser.add_argument(
        "--keep-context",
        action="store_true",
        help="Keep the generated build_context/ directory for inspection",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def _package_name_from_image(image_name: str) -> str:
    """Turn a Docker reference into a filesystem-safe package name."""
    return re.sub(r"[^A-Za-z0-9._-]", "_", image_name)


def _run_python_mode(args) -> int:
    """Build the portable Python package and report the outcome."""
    create_python_package(
        args.input, _package_name_from_image(args.image), args.output_dir
    )
    return 0


def main(argv=None) -> int:
    """Parse arguments and run the requested packaging mode.

    Returns:
        0 on success, 1 on failure, 130 if interrupted by the user.
    """
    args = build_parser().parse_args(argv)

    try:
        if args.mode == "python":
            return _run_python_mode(args)

        if args.mode == "auto" and not check_docker_available():
            print("INFO: Docker unavailable - falling back to a portable Python package")
            return _run_python_mode(args)

        package_model(
            args.input,
            args.image,
            python_version=args.python_version,
            platform=args.platform,
            no_cache=args.no_cache,
            keep_context=args.keep_context,
        )
        return 0

    except DockerUnavailableError:
        print(
            "Error: Docker is required for --mode docker. "
            "Start Docker, or rerun with --mode python.",
            file=sys.stderr,
        )
        return 1
    except (FileNotFoundError, ValueError, PackagingError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nOperation cancelled by user", file=sys.stderr)
        return 130
    except OSError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
