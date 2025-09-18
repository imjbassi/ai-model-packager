import os, shutil, subprocess, sys, time
from pathlib import Path

DOCKERFILE_TPL = """\
FROM python:3.10-slim
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps kept minimal
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./requirements.txt
RUN python -m pip install --upgrade pip && pip install -r requirements.txt

# Add app files
COPY infer.py ./infer.py
COPY {model_filename} ./model.bin
{extra_files}

# Drop privileges
RUN useradd -m appuser
USER appuser

HEALTHCHECK --interval=30s --timeout=5s --retries=3 CMD python -c "print('ok')" || exit 1

CMD ["python", "infer.py", "--model", "model.bin"]
"""

REQS_PYTORCH = "torch==2.3.*\ttorchvision==0.18.*\tpillow\n"
REQS_TF = "tensorflow==2.16.*\tpillow\n"

def _detect_framework(model_path: str, forced: str) -> str:
    if forced and forced != "auto":
        return forced
    ext = Path(model_path).suffix.lower()
    if ext in (".pth", ".pt", ".bin"):
        return "pytorch"
    if ext in (".h5", ".hdf5", ".keras", ".pb", ".savedmodel"):
        return "tensorflow"
    # default to pytorch if unknown
    return "pytorch"

def _write_requirements(context: Path, framework: str):
    reqs = REQS_PYTORCH if framework == "pytorch" else REQS_TF
    (context / "requirements.txt").write_text(reqs, encoding="utf-8")

def _write_dockerfile(context: Path, model_filename: str, extra_files: str = ""):
    df = DOCKERFILE_TPL.format(model_filename=model_filename, extra_files=extra_files)
    (context / "Dockerfile").write_text(df, encoding="utf-8")

def _write_dockerignore(context: Path):
    ignore = """\
**/__pycache__/
**/.pytest_cache/
**/.venv/
**/venv/
**/.git/
**/.DS_Store
*.log
*.egg-info/
build/
dist/
"""
    (context / ".dockerignore").write_text(ignore, encoding="utf-8")

def check_docker():
    try:
        p = subprocess.run(
            ["docker", "version"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=15,
        )
        return (p.returncode == 0, p.stderr if p.returncode else "OK")
    except Exception as e:
        return (False, str(e))

def _stream_build(cmd, timeout, verbose):
    start = time.time()
    # Stream output to avoid Windows decode issues
    with subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    ) as proc:
        for line in proc.stdout:
            if verbose:
                print(line, end="")
            if time.time() - start > timeout:
                proc.kill()
                raise TimeoutError("Docker build timed out.")
        rc = proc.wait()
        if rc != 0:
            raise RuntimeError("Docker build failed (see logs above).")

def package_model(model_path: str, image_name: str, framework: str = "auto",
                  timeout: int = 1800, verbose: bool = True, no_docker: bool = False, push: bool = False):
    model_path = Path(model_path).resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    fw = _detect_framework(str(model_path), framework)
    print(f"Framework: {fw}")

    context = Path("build_context")
    context.mkdir(exist_ok=True)

    # copy model & required files
    target_model = context / model_path.name
    shutil.copy2(model_path, target_model)
    shutil.copy2("infer.py", context / "infer.py")
    _write_requirements(context, fw)
    _write_dockerfile(context, model_path.name)
    _write_dockerignore(context)

    if no_docker:
        # optional: create a wheel/sdist here
        print("Docker disabled (--no-docker). Packaging step skipped.")
        return

    cmd = ["docker", "build", "-t", image_name, str(context)]
    print("Building Docker image:", " ".join(cmd))
    _stream_build(cmd, timeout=timeout, verbose=verbose)

    # Verify image exists
    verify = subprocess.run(
        ["docker", "image", "inspect", image_name],
        capture_output=True, text=True, encoding="utf-8", errors="replace"
    )
    if verify.returncode != 0:
        raise RuntimeError(f"Image not found after build: {image_name}")

    print(f"SUCCESS: Built Docker image: {image_name}")

    if push:
        print("Pushing image…")
        pushp = subprocess.run(["docker", "push", image_name],
                               text=True, encoding="utf-8", errors="replace")
        if pushp.returncode != 0:
            raise RuntimeError("Failed to push image")
