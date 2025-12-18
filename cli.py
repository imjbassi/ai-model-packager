```python
import argparse
import sys
from docker_packager import package_model


def main():
    """
    Main entry point for the AI model packager CLI.
    
    Parses command-line arguments and invokes the Docker packaging process
    to containerize a machine learning model.
    
    Returns:
        int: Exit code (0 for success, 130 for user cancellation, 1 for errors)
    """
    parser = argparse.ArgumentParser(
        description="Package ML models into Docker containers for easy deployment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s -i resnet18_full.pth -t my_ai_model:1.0
  %(prog)s --input model.h5 --image ml_model:latest
        """
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        metavar="PATH",
        help="Path to the model file (e.g., resnet18_full.pth, my_model.h5)"
    )
    parser.add_argument(
        "--image",
        "-t",
        required=True,
        metavar="NAME:TAG",
        help="Docker image name and tag (e.g., my_ai_model:1.0)"
    )
    
    args = parser.parse_args()
    
    try:
        package_model(args.input, args.image)
        return 0
    except KeyboardInterrupt:
        print("\nOperation cancelled by user", file=sys.stderr)
        return 130
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
```