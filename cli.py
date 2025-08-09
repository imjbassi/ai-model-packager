import argparse
from docker_packager import package_model

def main():
    parser = argparse.ArgumentParser(
        description="Turn your ML model into a Docker container (pretty neat, right?)"
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to your model file (like resnet18_full.pth or my_model.h5)"
    )
    parser.add_argument(
        "--image", "-t",
        required=True,
        help="What to call your Docker image (something like my_ai_model:1.0)"
    )
    args = parser.parse_args()
    package_model(args.input, args.image)

if __name__ == "__main__":
    main()
