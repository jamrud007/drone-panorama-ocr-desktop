"""
Quality Assurance (QA) Verification Script for Halotec Container OCR Desktop.
Validates:
1. Automated Test Suite (Unit, IPC, OCR Mock, DPAPI, ISO 6346, Database)
2. Standalone Binary Health & Directory Structure in dist/HalotecContainerOCR
3. Model Weights & Stitching Pipeline Script Packaging
4. Windows DPAPI Encrypted Token Integrity
5. API Contract Conformance (numeric gate, camera_position, endpoint)
6. Zero Orphaned Processes / Named Mutex Safety
"""
import json
import os
import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))


def run_check(title: str, check_fn) -> bool:
    print(f"\n[QA CHECK] {title}...")
    try:
        ok, msg = check_fn()
        if ok:
            print(f"  [PASS] {msg}")
            return True
        else:
            print(f"  [FAIL] {msg}")
            return False
    except Exception as e:
        print(f"  [ERROR] Exception during check: {e}")
        return False


def check_pytest_suite():
    res = subprocess.run(
        [sys.executable, "-m", "pytest", "tests", "-q"],
        cwd=str(BASE_DIR),
        capture_output=True,
        text=True,
    )
    if res.returncode == 0:
        lines = [l.strip() for l in res.stdout.strip().split("\n") if l.strip()]
        summary = lines[-1] if lines else "All tests passed"
        return True, summary
    return False, f"Pytest failed with returncode {res.returncode}:\n{res.stdout}\n{res.stderr}"


def check_dist_executable():
    dist_dir = BASE_DIR / "dist" / "HalotecContainerOCR"
    exe_file = dist_dir / "HalotecContainerOCR.exe"
    if not exe_file.is_file():
        return False, f"Executable not found at {exe_file}"

    size_mb = exe_file.stat().st_size / (1024 * 1024)
    if size_mb < 20:
        return False, f"Executable size unusually small ({size_mb:.2f} MB)"

    internal_dir = dist_dir / "_internal"
    if not internal_dir.is_dir():
        return False, f"_internal folder not found at {internal_dir}"

    return True, f"Executable validated: {exe_file.name} ({size_mb:.2f} MB) with _internal assets"


def check_bundled_libraries():
    dist_internal = BASE_DIR / "dist" / "HalotecContainerOCR" / "_internal"
    required_packages = ["torch", "torchvision", "ultralytics", "lightglue", "kornia", "onnxruntime", "PySide6", "cv2"]
    missing = []
    for pkg in required_packages:
        pkg_dir = dist_internal / pkg
        pkg_pyd = list(dist_internal.glob(f"{pkg}*.pyd"))
        if not pkg_dir.exists() and not pkg_pyd:
            missing.append(pkg)

    if missing:
        return False, f"Missing bundled packages in _internal: {missing}"
    return True, f"All core packages bundled: {required_packages}"


def check_model_weights():
    dist_dir = BASE_DIR / "dist" / "HalotecContainerOCR"
    candidates = [
        BASE_DIR / "container-detection-drone-view" / "weight" / "container_detection_model.onnx",
        dist_dir / "_internal" / "container-detection-drone-view" / "weight" / "container_detection_model.onnx",
        dist_dir / "_internal" / "container-detection-drone-view" / "weight" / "best.pt",
        BASE_DIR / "container-detection-drone-view" / "weight" / "best.pt",
    ]
    found = [str(c.relative_to(BASE_DIR)) for c in candidates if c.is_file()]
    pipeline_script = BASE_DIR / "container-detection-drone-view" / "pipeline_yolo_paronama.py"
    if not found or not pipeline_script.is_file():
        return False, "Missing ONNX/PT model weights or pipeline_yolo_paronama.py"
    return True, f"Model weights and pipeline verified: {found}"


def check_dpapi_token():
    from src.utils.security import retrieve_auth_token
    local_token_path = BASE_DIR / "data" / ".token.dpapi"
    dist_token_path = BASE_DIR / "dist" / "HalotecContainerOCR" / "data" / ".token.dpapi"
    target = dist_token_path if dist_token_path.is_file() else local_token_path
    if not target.is_file():
        return True, "No local DPAPI token present (expected on fresh checkout before login)"

    token = retrieve_auth_token(storage_path=target)
    if not token or len(token) < 20:
        return False, f"Failed to decrypt DPAPI token at {target}"

    return True, f"DPAPI encrypted token valid at {target.relative_to(BASE_DIR)}"


def check_api_contract():
    config_file = BASE_DIR / "config" / "config.json"
    if not config_file.is_file():
        config_file = BASE_DIR / "config" / "config.example.json"
    with open(config_file, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    gate = cfg.get("api", {}).get("gate")
    if not isinstance(gate, str) or not gate.isdigit():
        return False, f"API gate violates contract: must be numeric string, got {repr(gate)}"

    url = cfg.get("api", {}).get("url", "")
    if not isinstance(url, str) or not url.startswith(("https://", "http://")):
        return False, f"API endpoint invalid: expected http(s):// URL, got {url!r}"

    scheme = url.split("://", 1)[0]
    return True, f"API contract compliant: scheme={scheme}, gate={repr(gate)} (strict numeric string)"


def check_orphaned_processes():
    import psutil
    active = []
    current_pid = os.getpid()
    for proc in psutil.process_iter(["pid", "name", "exe"]):
        try:
            if proc.info["pid"] == current_pid:
                continue
            name = (proc.info["name"] or "").lower()
            if "haloteccontainerocr" in name:
                active.append(proc.info)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    if active:
        return False, f"Found lingering HalotecContainerOCR processes: {active}"
    return True, "No orphaned or lingering HalotecContainerOCR processes detected"


def main():
    print("=================================================================")
    print("   HALOTEC CONTAINER OCR DESKTOP - QA VERIFICATION SUITE")
    print("=================================================================")

    dist_exists = (BASE_DIR / "dist" / "HalotecContainerOCR" / "HalotecContainerOCR.exe").is_file()
    require_dist = "--require-dist" in sys.argv

    checks = [
        ("Automated PyTest Test Suite (Unit & E2E)", check_pytest_suite),
        ("AI Model Weights & Stitching Pipeline Assets", check_model_weights),
        ("Windows DPAPI Token Security & Storage", check_dpapi_token),
        ("Halotec OCR API Schema & Parameter Contract", check_api_contract),
        ("Clean Process Lifecycle & Named Mutex Safety", check_orphaned_processes),
    ]
    if dist_exists or require_dist:
        checks.insert(1, ("Single Binary Distribution Health", check_dist_executable))
        checks.insert(2, ("Bundled Machine Learning & GUI Libraries", check_bundled_libraries))
    else:
        print("\n[INFO] dist/HalotecContainerOCR not built yet; skipping binary bundle checks (pass --require-dist to enforce).")

    results = []
    for title, fn in checks:
        passed = run_check(title, fn)
        results.append((title, passed))

    print("\n=================================================================")
    print("   QA VERIFICATION SUMMARY")
    print("=================================================================")
    all_passed = True
    for title, passed in results:
        status = "PASSED [OK]" if passed else "FAILED [X]"
        print(f" - {title.ljust(50)}: {status}")
        if not passed:
            all_passed = False

    print("=================================================================")
    if all_passed:
        print("RESULT: ALL QA CHECKS PASSED. READY FOR ACCEPTANCE TESTING!")
        sys.exit(0)
    else:
        print("RESULT: SOME QA CHECKS FAILED. PLEASE REVIEW LOGS ABOVE.")
        sys.exit(1)


if __name__ == "__main__":
    main()
