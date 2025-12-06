```python
import argparse
import shutil
import zipfile
from pathlib import Path
from typing import Optional


def create_python_package(model_path: str, package_name: str) -> None:
    """
    Create a portable Python package for model deployment.
    
    This function creates a self-contained package that includes:
    - The model file
    - Required Python scripts (infer.py, model_loader.py)
    - Dependencies specification (requirements.txt)
    - Cross-platform run scripts (run.py for Unix/Mac, run.bat for Windows)
    
    Args:
        model_path: Path to the model file to be packaged
        package_name: Name for the output package (without extension)
    
    Raises:
        FileNotFoundError: If model_path or required scripts don't exist
        PermissionError: If unable to create package directory or files
    """
    model_file = Path(model_path)
    if not model_file.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    model_filename = model_file.name
    package_dir = Path(f"{package_name}_package")
    package_dir.mkdir(exist_ok=True)
    
    print(f"📦 Creating Python package: {package_name}")
    
    # Copy required files
    _copy_required_files(model_file, model_filename, package_dir)
    
    # Create requirements.txt
    _create_requirements_file(package_dir)
    
    # Create run scripts for different platforms
    _create_run_script(package_dir, model_filename)
    _create_batch_script(package_dir)
    
    # Create zip package
    zip_path = _create_zip_archive(package_dir, package_name)
    
    # Print usage instructions
    _print_usage_instructions(zip_path, package_dir)


def _copy_required_files(model_file: Path, model_filename: str, package_dir: Path) -> None:
    """
    Copy model and required Python scripts to the package directory.
    
    Args:
        model_file: Path object for the model file
        model_filename: Name of the model file
        package_dir: Destination package directory
    
    Raises:
        FileNotFoundError: If required scripts are missing
    """
    required_scripts = ["infer.py", "model_loader.py"]
    
    # Copy model file
    shutil.copy(model_file, package_dir / model_filename)
    
    # Copy required scripts
    for script in required_scripts:
        script_path = Path(script)
        if not script_path.exists():
            raise FileNotFoundError(f"Required script not found: {script}")
        shutil.copy(script_path, package_dir / script)


def _create_requirements_file(package_dir: Path) -> None:
    """
    Create requirements.txt with necessary dependencies.
    
    Args:
        package_dir: Package directory where requirements.txt will be created
    """
    requirements = "torch\ntorchvision\n"
    (package_dir / "requirements.txt").write_text(requirements, encoding="utf-8")


def _create_run_script(package_dir: Path, model_filename: str) -> None:
    """
    Create a Python run script for Unix/Mac systems.
    
    Args:
        package_dir: Package directory where run.py will be created
        model_filename: Name of the model file to pass to infer.py
    """
    run_script = f"""#!/usr/bin/env python3
\"\"\"
Automated run script for model inference.
This script installs dependencies and runs inference on the packaged model.
\"\"\"
import sys
import subprocess


def main():
    print("Installing dependencies...")
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"]
        )
    except subprocess.CalledProcessError as e:
        print(f"Error installing dependencies: {{e}}", file=sys.stderr)
        return 1
    
    print("Running inference...")
    try:
        subprocess.check_call(
            [sys.executable, "infer.py", "--model", "{model_filename}"]
        )
    except subprocess.CalledProcessError as e:
        print(f"Error running inference: {{e}}", file=sys.stderr)
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
"""
    run_script_path = package_dir / "run.py"
    run_script_path.write_text(run_script, encoding="utf-8")
    
    # Make executable on Unix-like systems
    try:
        run_script_path.chmod(0o755)
    except (OSError, NotImplementedError):
        # Windows or other systems that don't support chmod
        pass


def _create_batch_script(package_dir: Path) -> None:
    """
    Create a Windows batch script for easy execution.
    
    Args:
        package_dir: Package directory where run.bat will be created
    """
    batch_script = """@echo off
echo Running model inference...
python run.py
if %ERRORLEVEL% NEQ 0 (
    echo Error occurred during execution.
    pause
    exit /b %ERRORLEVEL%
)
echo.
echo Execution completed successfully.
pause
"""
    (package_dir / "run.bat").write_text(batch_script, encoding="utf-8")


def _create_zip_archive(package_dir: Path, package_name: str) -> str:
    """
    Create a zip archive of the package directory.
    
    Args:
        package_dir: Directory to be zipped
        package_name: Base name for the zip file
    
    Returns:
        Path to the created zip file
    """
    zip_path = f"{package_name}_package.zip"
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file in package_dir.rglob("*"):
            if file.is_file():
                zipf.write(file, file.relative_to(package_dir))
    
    return zip_path


def _print_usage_instructions(zip_path: str, package_dir: Path) -> None:
    """
    Print usage instructions for the created package.
    
    Args:
        zip_path: Path to the created zip file
        package_dir: Path to the package directory
    """
    print(f"✅ Created package: {zip_path}")
    print(f"📁 Package contents: {package_dir}")
    print("\n🚀 To use:")
    print(f"   1. Extract {zip_path}")
    print(f"   2. Run: python run.py (Unix/Mac/Linux)")
    print(f"   3. Or double-click: run.bat (Windows)")


def main() -> None:
    """
    Main entry point for the package creation script.
    
    Parses command-line arguments and creates a portable Python package.
    """
    parser = argparse.ArgumentParser(
        description="Create portable Python package for model deployment (alternative to Docker)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --input model.pth --name my_model
  %(prog)s -i /path/to/model.pt -n production_model
        """
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to model file to be packaged"
    )
    parser.add_argument(
        "--name", "-n",
        required=True,
        help="Package name (without extension)"
    )
    
    args = parser.parse_args()
    
    try:
        create_python_package(args.input, args.name)
    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        return 1
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
```