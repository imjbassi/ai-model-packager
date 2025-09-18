import argparse
import sys
from docker_packager import package_model, check_docker

def parse_args():
    p = argparse.ArgumentParser(description="AI Model Packaging Library (GCU Capstone)")
    p.add_argument("--input", required=True, help="Path to trained model file (e.g., .pth or .h5)")
    p.add_argument("--image", required=True, help="Docker image name:tag (e.g., my_model:1.0)")
    p.add_argument("--framework", choices=["auto", "pytorch", "tensorflow"], default="auto",
                   help="Force a framework or auto-detect from file")
    p.add_argument("--timeout", type=int, default=1800, help="Max seconds to allow docker build to run")
    p.add_argument("--verbose", action="store_true", help="Stream docker build logs")
    p.add_argument("--no-docker", action="store_true", help="Package w/o Docker (fallback to wheel)")
    p.add_argument("--push", action="store_true", help="Push image to registry after build")
    p.add_argument("--registry", default="", help="Optional registry prefix (e.g., ghcr.io/user)")
    return p.parse_args()

def main():
    args = parse_args()

    if not args.no_docker:
        ok, msg = check_docker()
        if not ok:
            print(f"ERROR: {msg}", file=sys.stderr)
            print("Tip: Start Docker Desktop and re-run. Or use --no-docker to build a Python wheel.")
            sys.exit(1)

    try:
        package_model(
            model_path=args.input,
            image_name=args.image if not args.registry else f"{args.registry}/{args.image}",
            framework=args.framework,
            timeout=args.timeout,
            verbose=args.verbose,
            no_docker=args.no_docker,
            push=args.push
        )
    except Exception as e:
        print(f"\nFATAL: {e}", file=sys.stderr)
        sys.exit(2)

if __name__ == "__main__":
    main()
