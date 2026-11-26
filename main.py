"""
Application Entry Point (Windows Desktop PySide6).
Complies with TDD_Container_OCR_Windows_v1.9.md:
- Enforces Windows Named Mutex (Single instance per user session).
- Starts PySide6 event loop in main process.
- Starts Orchestrator as a background master thread with thread-safe queues.
- Graceful shutdown of workers and database.
"""
from __future__ import annotations

import os
os.environ.setdefault("ULTRALYTICS_SKIP_REQUIREMENTS_CHECKS", "1")
os.environ.setdefault("YOLO_AUTOINSTALL", "0")

import queue
import sys
from pathlib import Path

# Ensure bundled CUDA/cuDNN & ONNX Runtime DLL dirs are searchable in frozen Windows builds
if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
    _meipass = Path(getattr(sys, "_MEIPASS"))
    _dll_dirs = [_meipass / "torch" / "lib", _meipass / "onnxruntime" / "capi"]
    if (_meipass / "nvidia").is_dir():
        _dll_dirs.extend(p / "bin" for p in (_meipass / "nvidia").iterdir() if (p / "bin").is_dir())
    for _d in _dll_dirs:
        if _d.is_dir():
            try:
                os.add_dll_directory(str(_d))
            except Exception:
                pass
            if str(_d) not in os.environ.get("PATH", ""):
                os.environ["PATH"] = str(_d) + os.pathsep + os.environ.get("PATH", "")

from PySide6.QtWidgets import QApplication, QMessageBox

from src.config import BASE_DIR, AppConfig
from src.database.db import Database
from src.orchestrator.engine import Orchestrator
from src.ui.main_window import MainWindow
from src.utils.windows_job import SingleInstanceMutex


def main() -> int:
    import multiprocessing
    multiprocessing.freeze_support()

    if "--self-check" in sys.argv:
        import json
        import traceback
        out_file = BASE_DIR / "self_check.json"
        try:
            import torch
            from src.config import BUNDLE_DIR
            for c_root in [BASE_DIR, BUNDLE_DIR, BASE_DIR.parent]:
                cd = c_root / "container-detection-drone-view"
                if cd.is_dir() and str(cd) not in sys.path:
                    sys.path.insert(0, str(cd))
            from src.recognition.detector import describe_compute_plan, run_gpu_preflight_probe
            from pipeline_yolo_paronama import run_stitching_pipeline

            cfg = AppConfig.load()
            plan = describe_compute_plan("auto")
            probe = run_gpu_preflight_probe(cfg.detection.resolved_model_path(), "auto")
            payload = {
                "torch_version": str(torch.__version__),
                "cuda_built": bool(torch.backends.cuda.is_built()),
                "pipeline_ok": callable(run_stitching_pipeline),
                "plan": plan,
                "probe": probe,
            }
            out_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            return 0 if probe.get("probe_ok") else 2
        except Exception as exc:
            out_file.write_text(json.dumps({"error": str(exc), "trace": traceback.format_exc()}, indent=2), encoding="utf-8")
            return 3

    # 1. Enforce single running application instance
    mutex = SingleInstanceMutex("Local\\HalotecContainerOCRDesktopAppMutex")
    if mutex.is_already_running():
        # Temporary fallback app to show warning
        dummy_app = QApplication.instance() or QApplication(sys.argv)
        QMessageBox.warning(
            None,
            "Aplikasi Sedang Berjalan",
            "Aplikasi Halotec Container OCR Desktop sudah terbuka di sesi pengguna ini.\n"
            "Hanya satu instance aplikasi yang diperbolehkan aktif.",
        )
        return 1

    # 2. Initialize configuration & folders
    config = AppConfig.load()
    config.ensure_directories()

    # 3. Initialize PySide6 GUI Application
    app = QApplication(sys.argv)
    app.setApplicationName("HalotecContainerOCR")
    app.setOrganizationName("Halotec")
    app.setQuitOnLastWindowClosed(True)

    # 4. Check Authentication Session (Windows DPAPI)
    from src.ocr_client.client import HalotecOcrClient
    from src.ui.login_dialog import LoginDialog

    ocr_checker = HalotecOcrClient(base_url=config.api.url, gate=config.api.gate)
    valid_session = False
    saved_token = ocr_checker.get_token()
    if saved_token:
        try:
            is_valid, _, validated_token = ocr_checker.validate_token(saved_token)
            if is_valid and validated_token:
                valid_session = True
                from src.utils.security import store_auth_token
                store_auth_token(validated_token)
            else:
                from src.utils.security import store_auth_token
                store_auth_token("")
                valid_session = False
        except Exception:
            valid_session = False

    if not valid_session:
        login_dlg = LoginDialog(config=config)
        if login_dlg.exec() != LoginDialog.DialogCode.Accepted:
            mutex.release()
            import os
            os._exit(0)

    # 5. Initialize Database & Queues
    db = Database()
    cmd_queue = queue.Queue()
    evt_queue = queue.Queue()

    # 6. Start In-Process Orchestrator Thread (Master Scheduler)
    orchestrator = Orchestrator(
        command_queue=cmd_queue,
        event_queue=evt_queue,
        config=config,
        db=db,
    )
    orchestrator.start()

    # 7. Initialize & Show Main Window
    window = MainWindow(
        cmd_queue=cmd_queue,
        evt_queue=evt_queue,
        config=config,
        db=db,
        orchestrator=orchestrator,
    )
    window.show()

    # 8. Run event loop with guaranteed clean process termination
    exit_code = 0
    try:
        exit_code = app.exec()
    finally:
        orchestrator.stop()
        orchestrator.join(timeout=1.5)
        mutex.release()
        import os
        os._exit(exit_code)


if __name__ == "__main__":
    sys.exit(main())
