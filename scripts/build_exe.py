"""
Build script to package Halotec Container OCR Desktop Application via PyInstaller.
Usage:
    python scripts/build_exe.py [--include-local-token]
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


def build(include_local_token: bool = False):
    print("[1/3] Checking dependencies...")
    try:
        import PySide6
        import onnxruntime
        import PyInstaller
        print(f" - PySide6: {PySide6.__version__}")
        print(f" - ONNX Runtime: {onnxruntime.__version__}")
        print(f" - PyInstaller: {PyInstaller.__version__}")
    except ImportError as e:
        print(f"Error: missing dependency {e}")
        sys.exit(1)

    print("\n[2/3] Cleaning previous dist output and running PyInstaller desktop.spec...")
    spec_file = BASE_DIR / "desktop.spec"
    if not spec_file.is_file():
        print(f"Error: {spec_file} not found")
        sys.exit(1)

    # Ensure config/config.json exists for desktop.spec (bootstrap from config.example.json if fresh clone)
    cfg_file = BASE_DIR / "config" / "config.json"
    cfg_example = BASE_DIR / "config" / "config.example.json"
    if not cfg_file.is_file() and cfg_example.is_file():
        shutil.copy2(cfg_example, cfg_file)
        print(f"Bootstrapped {cfg_file.relative_to(BASE_DIR)} from {cfg_example.relative_to(BASE_DIR)}")

    # Terminate any stale HalotecContainerOCR instance holding locks on dist folder
    try:
        subprocess.run(["taskkill", "/F", "/IM", "HalotecContainerOCR.exe"], capture_output=True)
    except Exception:
        pass

    dist_dir = BASE_DIR / "dist" / "HalotecContainerOCR"
    if dist_dir.exists():
        import time
        for attempt in range(5):
            try:
                shutil.rmtree(dist_dir)
                break
            except Exception as e:
                time.sleep(1)

    cmd = [sys.executable, "-m", "PyInstaller", "--noconfirm", str(spec_file)]
    res = subprocess.run(cmd, cwd=str(BASE_DIR))
    if res.returncode != 0:
        print(f"PyInstaller build failed with code {res.returncode}")
        sys.exit(res.returncode)

    print("\n[3/3] Verifying output directory and preserving runtime state...")
    dist_dir = BASE_DIR / "dist" / "HalotecContainerOCR"
    exe_file = dist_dir / "HalotecContainerOCR.exe"
    if exe_file.is_file():
        print(f"SUCCESS: Executable created at {exe_file}")
        
        # Ensure runtime data and outputs directories exist
        dist_data = dist_dir / "data"
        dist_outputs = dist_dir / "outputs"
        dist_data.mkdir(parents=True, exist_ok=True)
        dist_outputs.mkdir(parents=True, exist_ok=True)

        # Only copy local DPAPI token when explicitly requested for local testing
        if include_local_token:
            src_token = BASE_DIR / "data" / ".token.dpapi"
            dst_token = dist_data / ".token.dpapi"
            if src_token.is_file():
                shutil.copy2(src_token, dst_token)
                print(f"Preserved active DPAPI authentication token to {dst_token}")
    else:
        print(f"Warning: Executable not found at {exe_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build Halotec Container OCR Desktop executable.")
    parser.add_argument(
        "--include-local-token",
        action="store_true",
        help="Copy local data/.token.dpapi into dist/ for local-only testing (do not distribute).",
    )
    args = parser.parse_args()
    build(include_local_token=args.include_local_token)
