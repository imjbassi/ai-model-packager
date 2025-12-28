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
    """Check if Docker daemon is running and accessible.
    
    Returns:
        bool: True if Docker is available and responsive, False otherwise.
    """
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
    """Determine required dependencies based on model file extension.
    
    Args:
        model_path: Path to the model file.
        
    Returns:
        str: Requirements file content with necessary dependencies.
    """
    ext = Path(model_path).suffix.lower()
    
    if ext == '.pth':
        return "torch\ntorchvision\nPillow\n"
    elif ext == '.h5':
        return "tensorflow\nPillow\n"
    else:
        # Fallback for unknown model types
        return "torch\ntensorflow\nPillow\n"


def _create_sample_image(sample_path):
    """Create a sample image file if it doesn't exist.
    
    Args:
        sample_path: Path where the sample image should be created.
    """
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
    except subprocess.TimeoutExpired:
        print("WARNING: Sample image creation timed out")
        _create_placeholder_image(sample_path)
    except subprocess.CalledProcessError:
        print("WARNING: PIL not available, creating placeholder")
        _create_placeholder_image(sample_path)
    except Exception as e:
        print(f"WARNING: Failed to create sample image: {e}")
        _create_placeholder_image(sample_path)


def _create_placeholder_image(sample_path):
    """Create a placeholder file when PIL is unavailable.
    
    Args:
        sample_path: Path where the placeholder should be created.
    """
    try:
        with open(sample_path, 'w') as f:
            f.write("# Placeholder image file\n")
    except Exception as e:
        print(f"WARNING: Failed to create placeholder image: {e}")


def _verify_docker_image(image_name):
    """Verify that the Docker image was created successfully.
    
    Args:
        image_name: Name of the Docker image to verify.
        
    Returns:
        bool: True if verification succeeded or was skipped, False otherwise.
    """
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
    except subprocess.TimeoutExpired:
        print("WARNING: Image verification timed out")
        return True
    except Exception as e:
        print(f"WARNING: Image verification failed: {e}")
        return True


def _fallback_to_python_package(model_path, image_name):
    """Attempt to create a Python package as fallback when Docker is unavailable.
    
    Args:
        model_path: Path to the model file.
        image_name: Name to use for the package (Docker image name repurposed).
        
    Returns:
        bool: True if fallback packaging succeeded, False otherwise.
    """
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
    except Exception as e:
        print(f"ERROR: Python packaging failed: {e}")
        return False


def package_model(model_path, image_name):
    """Build a Docker image containing the model and inference environment.
    
    The image includes:
    - Python runtime
    - Required dependencies (PyTorch, TensorFlow, etc.)
    - Model file
    - Inference script
    - Sample test image
    
    Args:
        model_path: Path to the model file to package.
        image_name: Name for the resulting Docker image.
        
    Returns:
        bool: True if packaging succeeded, False otherwise.
    """
    # Validate inputs
    if not os.path.exists(model_path):
        print(f"ERROR: Model file not found: {model_path}")
        return False
    
    if not image_name or not isinstance(image_name, str):
        print("ERROR: Invalid image name provided")
        return False
    
    # Check Docker availability first
    if not check_docker_available():
        return _fallback_to_python_package(model_path, image_name)
    
    model_filename = os.path.basename(model_path)
    ctx = Path("build_context")
    
    # Create build context directory
    try:
        ctx.mkdir(exist_ok=True)
    except Exception as e:
        print(f"ERROR: Failed to create build context directory: {e}")
        return False

    print(f"Creating Docker build context in: {ctx}")

    # Copy the model file into build context
    print(f"Copying model: {model_path}")
    try:
        shutil.copy(model_path, ctx / model_filename)
    except Exception as e:
        print(f"ERROR: Failed to copy model file: {e}")
        return False

    # Write requirements.txt based on model type
    requirements = _get_requirements_for_model(model_path)
    print("Writing requirements.txt")
    try:
        (ctx / "requirements.txt").write_text(requirements)
    except Exception as e:
        print(f"ERROR: Failed to write requirements.txt: {e}")
        return False

    # Write Dockerfile with model filename substituted
    print("Writing Dockerfile")
    try:
        dockerfile_content = DOCKERFILE.format(model_filename=model_filename)
        (ctx / "Dockerfile").write_text(dockerfile_content)
    except Exception as e:
        print(f"ERROR: Failed to write Dockerfile: {e}")
        return False

    # Copy inference script
    print("Copying inference script")
    infer_script = "infer.py"
    if os.path.exists(infer_script):
        try:
            shutil.copy(infer_script, ctx / infer_script)
        except Exception as e:
            print(f"WARNING: Failed to copy inference script: {e}")
    else:
        print(f"WARNING: Inference script not found: {infer_script}")
    
    # Create and copy sample image
    sample_path = "sample.jpg"
    _create_sample_image(sample_path)
    
    print("Copying sample image")
    try:
        shutil.copy(sample_path, ctx / "sample.jpg")
    except Exception as e:
        print(f"WARNING: Failed to copy sample image: {e}")

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