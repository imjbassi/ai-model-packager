import argparse
from docker_packager import package_model


def main():
    """
    Main entry point for the AI model packager CLI.
    
    Parses command-line arguments and invokes the Docker packaging process
    to containerize a machine learning model.
    """
    parser = argparse.ArgumentParser(
        description="Package ML models into Docker containers for easy deployment"
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Path to the model file (e.g., resnet18_full.pth, my_model.h5)"
    )
    parser.add_argument(
        "--image",
        "-t",
        required=True,
        help="Docker image name and tag (e.g., my_ai_model:1.0)"
    )
    args = parser.parse_args()
    
    package_model(args.input, args.image)


if __name__ == "__main__":
    main()