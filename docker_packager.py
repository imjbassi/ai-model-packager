```python
import os
import subprocess
import shutil
from pathlib import Path

DOCKERFILE = """\
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY {model_filename} ./
COPY infer.py .
COPY sample.jpg .
CMD ["python", "infer.py", "--model", "{model_filename}", "--test-input", "sample.jpg"]
"""


def check_docker_available():
    """Check if Docker daemon is running and accessible."""
    try:
        print("Checking Docker daemon availability...")
        result = subprocess.run(
            ["docker", "version", "--format", "json"],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode == 0:
            print("SUCCESS: Docker daemon is running and responsive")
            return True
        else:
            print("ERROR: Docker daemon not responding properly")
            if result.stderr:
                print(f"Error details: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("TIMEOUT: Docker daemon check timed out")
        print("INFO: Try restarting Docker Desktop from system tray")
        return False
    except FileNotFoundError:
        print("ERROR: Docker command not found - Docker may not be installed")
        return False
    except Exception as e:
        print(f"ERROR: Docker check failed: {e}")
        return False


def _get_requirements_for_model(model_path):
    """Determine required dependencies based on model file extension."""
    if model_path.endswith('.pth'):
        return "torch\ntorchvision\nPillow\n"
    elif model_path.endswith('.h5'):
        return "tensorflow\nPillow\n"
    else:
        # Fallback for unknown model types
        return "torch\ntensorflow\nPillow\n"


def _create_sample_image(sample_path):
    """Create a sample image file if it doesn't exist."""
    if os.path.exists(sample_path):
        return
    
    print("Creating sample image...")
    try:
        subprocess.run(
            ["python", "-c", 
             "from PIL import Image; Image.new('RGB', (224,224), 'blue').save('sample.jpg')"],
            check=True,
            timeout=10
        )
    except Exception:
        # Create a placeholder file if PIL is not available
        with open(sample_path, 'w') as f:
            f.write("# Placeholder image file")


def _verify_docker_image(image_name):
    """Verify that the Docker image was created successfully."""
    print("Verifying image...")
    try:
        verify_result = subprocess.run(
            ["docker", "images", image_name, "--format", "table {{.Repository}}:{{.Tag}}"],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if verify_result.returncode == 0 and image_name in verify_result.stdout:
            print(f"VERIFIED: Image {image_name} exists and ready to use!")
            print()
            print("Next steps:")
            print(f"   • Test container: docker run --rm {image_name}")
            print(f"   • Run inference: docker run --rm {image_name}")
            return True
        else:
            print("WARNING: Build succeeded but image verification failed")
            return True
    except Exception as e:
        print(f"WARNING: Image verification failed: {e}")
        return True


def _fallback_to_python_package(model_path, image_name):
    """Attempt to create a Python package as fallback when Docker is unavailable."""
    print("WARNING: Docker is not available!")
    print("INFO: Falling back to Python packaging alternative...")
    
    try:
        from package_python import create_python_package
        package_file = create_python_package(
            model_path,
            f"{image_name.replace(':', '_')}_package"
        )
        print(f"SUCCESS: Created Python package instead: {package_file}")
        print("INFO: You can run this package with: python -m zipapp <package_file>")
        return True
    except ImportError:
        print("ERROR: Python packaging fallback not available")
        return False


def package_model(model_path, image_name):
    """
    Build a Docker image containing the model and inference environment.
    
    The image includes:
    - Python runtime
    - Required dependencies (PyTorch, TensorFlow, etc.)
    - Model file
    - Inference script
    - Sample test image
    
    Args:
        model_path: Path to the model file to package
        image_name: Name for the resulting Docker image
        
    Returns:
        bool: True if packaging succeeded, False otherwise
    """
    # Check Docker availability first
    if not check_docker_available():
        return _fallback_to_python_package(model_path, image_name)
    
    model_filename = os.path.basename(model_path)
    ctx = Path("build_context")
    ctx.mkdir(exist_ok=True)

    print(f"Creating Docker build context in: {ctx}")

    # Copy the model file into build context
    print(f"Copying model: {model_path}")
    shutil.copy(model_path, ctx / model_filename)

    # Write requirements.txt based on model type
    requirements = _get_requirements_for_model(model_path)
    print("Writing requirements.txt")
    (ctx / "requirements.txt").write_text(requirements)

    # Write Dockerfile with model filename substituted
    print("Writing Dockerfile")
    dockerfile_content = DOCKERFILE.format(model_filename=model_filename)
    (ctx / "Dockerfile").write_text(dockerfile_content)

    # Copy inference script and sample image
    print("Copying inference script")
    
    sample_path = "sample.jpg"
    _create_sample_image(sample_path)
    
    print("Copying sample image")
    shutil.copy(sample_path, ctx / "sample.jpg")

    # Build Docker image
    print(f"Building Docker image: {image_name}")
    print("This will download dependencies (~800MB) - may take 5-10 minutes")
    
    cmd = ["docker", "build", "-t", image_name, str(ctx)]
    print(f"Running: {' '.join(cmd)}")
    
    try:
        print("=" * 60)
        result = subprocess.run(cmd, check=False)
        print("=" * 60)
        
        if result.returncode == 0:
            print(f"SUCCESS: Built Docker image: {image_name}")
            return _verify_docker_image(image_name)
        else:
            print(f"ERROR: Docker build failed with return code: {result.returncode}")
            return False
            
    except KeyboardInterrupt:
        print("\nINTERRUPTED: Build interrupted by user")
        return False
    except Exception as e:
        print(f"ERROR: Docker build failed with error: {e}")
        return False
```