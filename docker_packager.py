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
    """Let's see if Docker is actually running and ready to go"""
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

def package_model(model_path, image_name):
    """
    This builds a Docker image
      - Python runtime
      - requirements.txt dependencies (PyTorch, etc.)
      - your model file (the important bit!)
      - infer.py script (to actually use the model)
    """
    # First check if Docker is available
    if not check_docker_available():
        print("WARNING: Docker is not available!")
        print("INFO: Falling back to Python packaging alternative...")
        
        # Import and use Python packaging as fallback
        try:
            from package_python import create_python_package
            package_file = create_python_package(model_path, f"{image_name.replace(':', '_')}_package")
            print(f"SUCCESS: Created Python package instead: {package_file}")
            print("INFO: You can run this package with: python -m zipapp <package_file>")
            return True
        except ImportError:
            print("ERROR: Python packaging fallback not available")
            return False
    
    model_filename = os.path.basename(model_path)
    ctx = Path("build_context")
    ctx.mkdir(exist_ok=True)

    print(f"Creating Docker build context in: {ctx}")

    # Copy the model file into build_context/
    print(f"Copying model: {model_path}")
    shutil.copy(model_path, ctx / model_filename)

    # Write requirements.txt (only what we actually need for each model type)
    if model_path.endswith('.pth'):
        requirements = "torch\ntorchvision\nPillow\n"
    elif model_path.endswith('.h5'):
        requirements = "tensorflow\nPillow\n"
    else:
        requirements = "torch\ntensorflow\nPillow\n"  # Just in case we get something weird
    
    print("Writing requirements.txt")
    (ctx / "requirements.txt").write_text(requirements)

    # Write Dockerfile, substituting your model's filename
    print("Writing Dockerfile")
    dockerfile_content = DOCKERFILE.format(model_filename=model_filename)
    (ctx / "Dockerfile").write_text(dockerfile_content)

    # Copy your inference script and sample image
    print("Copying inference script")
    
    # Create sample image if it doesn't exist
    sample_path = "sample.jpg"
    if not os.path.exists(sample_path):
        print("Creating sample image...")
        try:
            subprocess.run(["python", "-c", "from PIL import Image; Image.new('RGB', (224,224), 'blue').save('sample.jpg')"], 
                   check=True, timeout=10)
        except:
            # Create a placeholder
            with open(sample_path, 'w') as f:
                f.write("# Placeholder image file")
    
    print("Copying sample image")
    shutil.copy(sample_path, ctx / "sample.jpg")

    # Run docker build (this is where the magic happens!)
    print(f"Building Docker image: {image_name}")
    print("This will download PyTorch (~800MB) - grab a coffee, it takes 5-10 minutes")
    
    # Build the command
    cmd = ["docker", "build", "-t", image_name, str(ctx)]
    print(f"Running: {' '.join(cmd)}")
    
    try:
        print("=" * 60)
        result = subprocess.run(cmd, check=False)
        print("=" * 60)
        
        if result.returncode == 0:
            print(f"SUCCESS: Built Docker image: {image_name}")
            
            # Let's double-check that worked
            print("Verifying image...")
            verify_result = subprocess.run(
                ["docker", "images", image_name, "--format", "table {{.Repository}}:{{.Tag}}"],
                capture_output=True, text=True, timeout=10
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
        else:
            print(f"ERROR: Docker build failed with return code: {result.returncode}")
            return False
            
    except KeyboardInterrupt:
        print("\nINTERRUPTED: Build interrupted by user")
        return False
            
    except Exception as e:
        print(f"ERROR: Docker build failed with error: {e}")
        return False
