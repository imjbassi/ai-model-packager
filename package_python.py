import argparse
import shutil
import zipfile
from pathlib import Path

def create_python_package(model_path: str, package_name: str) -> None:
    """Create a portable Python package instead of Docker"""
    
    model_filename = Path(model_path).name
    package_dir = Path(f"{package_name}_package")
    package_dir.mkdir(exist_ok=True)
    
    print(f"📦 Creating Python package: {package_name}")
    
    # Copy files
    shutil.copy(model_path, package_dir / model_filename)
    shutil.copy("infer.py", package_dir / "infer.py")
    shutil.copy("model_loader.py", package_dir / "model_loader.py")
    
    # Create requirements.txt
    (package_dir / "requirements.txt").write_text("torch\ntorchvision\n")
    
    # Create run script
    run_script = f"""#!/usr/bin/env python3
import sys
import subprocess

# Install requirements
subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

# Run inference
subprocess.check_call([sys.executable, "infer.py", "--model", "{model_filename}"])
"""
    
    (package_dir / "run.py").write_text(run_script)
    
    # Create batch file for Windows
    batch_script = f"""@echo off
python run.py
pause
"""
    (package_dir / "run.bat").write_text(batch_script)
    
    # Create zip package
    zip_path = f"{package_name}_package.zip"
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        for file in package_dir.rglob("*"):
            if file.is_file():
                zipf.write(file, file.relative_to(package_dir))
    
    print(f"✅ Created package: {zip_path}")
    print(f"📁 Package contents: {package_dir}")
    print("\n🚀 To use:")
    print(f"   1. Extract {zip_path}")
    print(f"   2. Run: python run.py")
    print(f"   3. Or double-click: run.bat")

def main():
    parser = argparse.ArgumentParser(description="Create portable Python package (alternative to Docker)")
    parser.add_argument("--input", "-i", required=True, help="Path to model file")
    parser.add_argument("--name", "-n", required=True, help="Package name")
    args = parser.parse_args()
    
    create_python_package(args.input, args.name)

if __name__ == "__main__":
    main()
