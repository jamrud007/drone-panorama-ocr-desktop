import sys
from pathlib import Path
import pytest
from unittest.mock import MagicMock, patch

from src.config import AppConfig
from src.database.db import Database
from src.orchestrator.engine import ApiAuditLogger

def test_api_audit_logger_initializes_files(tmp_path):
    txt_path = tmp_path / "logs" / "api_requests.txt"
    json_path = tmp_path / "logs" / "api_requests.json"
    evt_q = MagicMock()
    logger = ApiAuditLogger(txt_path, json_path, evt_q, job_id=99)
    assert txt_path.exists()
    assert json_path.exists()
    assert "Job #99" in txt_path.read_text(encoding="utf-8")
    assert json_path.read_text(encoding="utf-8") == "[]"
    logger.finalize()
    assert "Ringkasan Sesi Job #99" in txt_path.read_text(encoding="utf-8")

def test_folder_resolution_logic(tmp_path):
    out_dir = tmp_path / "outputs"
    job_folder = out_dir / "20261123_212101_sample_video"
    img_folder = job_folder / "TOP" / "images"
    img_folder.mkdir(parents=True)
    img_path = img_folder / "sample_video_region_001.jpg"
    img_path.write_bytes(b"dummy")

    # Traverse upward algorithm
    p = img_path.resolve()
    target_dir = out_dir
    found = False
    for parent in [p] + list(p.parents):
        if parent.parent.resolve() == out_dir.resolve():
            target_dir = parent
            found = True
            break
    assert found is True
    assert target_dir.resolve() == job_folder.resolve()


def test_resolve_onnx_providers_fallback_safe():
    from src.recognition.detector import resolve_onnx_providers
    import onnxruntime as ort
    avail = set(ort.get_available_providers())
    for mode in ("cpu", "auto", "cuda", "cuda:0", "dml"):
        provs, active = resolve_onnx_providers(mode)
        assert "CPUExecutionProvider" in provs
        assert active == provs[0]
        if mode == "cpu":
            assert provs == ["CPUExecutionProvider"]
        if mode in ("dml", "cuda") and "DmlExecutionProvider" in avail and "CUDAExecutionProvider" not in avail:
            assert active == "DmlExecutionProvider"


def test_describe_compute_plan_modes():
    import onnxruntime as ort
    from src.recognition.detector import describe_compute_plan

    avail = set(ort.get_available_providers())
    plan_auto = describe_compute_plan("auto")
    assert plan_auto["requested"] == "auto"
    if not plan_auto["cuda_available"]:
        assert plan_auto["track_provider"] == "CPUExecutionProvider"
        assert plan_auto["uses_gpu"] is False
        if "DmlExecutionProvider" in avail:
            assert "Auto" in plan_auto["hint"]

    plan_dml = describe_compute_plan("dml")
    if "DmlExecutionProvider" in avail:
        assert plan_dml["track_provider"] == "DmlExecutionProvider"
        assert plan_dml["onnx_provider"] == "DmlExecutionProvider"
        assert plan_dml["uses_gpu"] is True
        if not plan_dml["cuda_available"]:
            assert "CPU" in plan_dml["stitch_device"]

    plan_cpu = describe_compute_plan("cpu")
    assert plan_cpu["track_provider"] == "CPUExecutionProvider"
    assert plan_cpu["onnx_provider"] == "CPUExecutionProvider"
    assert "CPU" in plan_cpu["stitch_device"]


def test_run_gpu_preflight_probe_missing_and_real(tmp_path):
    from src.recognition.detector import run_gpu_preflight_probe

    missing = run_gpu_preflight_probe(tmp_path / "does_not_exist.onnx", "auto")
    assert missing["probe_ok"] is False
    assert "probe_error" in missing

    real_model = Path("container-detection-drone-view/weight/container_detection_model.onnx")
    if real_model.is_file():
        res = run_gpu_preflight_probe(real_model, "auto")
        assert res["probe_ok"] is True
        assert res["active_provider"] in ("CUDAExecutionProvider", "CPUExecutionProvider")
        assert res["infer_ms"] >= 0.0
        assert res["warmup_ms"] >= 0.0


def test_settings_dialog_compute_preview_and_test_gpu_button(monkeypatch):
    import onnxruntime as ort
    from PySide6.QtWidgets import QApplication, QGroupBox, QMessageBox
    from src.recognition.detector import describe_compute_plan
    import src.ui.settings_dialog as sd_mod

    _app = QApplication.instance() or QApplication(["test", "-platform", "offscreen"])
    monkeypatch.setattr(sd_mod, "retrieve_auth_token", lambda: "")

    dlg = sd_mod.SettingsDialog(AppConfig())
    try:
        titles = [gb.title() for gb in dlg.findChildren(QGroupBox)]
        assert "Model, Perangkat Komputasi && Deteksi AI" in titles
        assert dlg.btn_test_gpu.text() == "Cek && Uji GPU"

        avail = set(ort.get_available_providers())
        if "DmlExecutionProvider" in avail:
            dlg.device_combo.setCurrentIndex(dlg.device_combo.findData("dml"))
            assert "DmlExecutionProvider" in dlg.lbl_compute_preview.text()
        else:
            dlg.device_combo.setCurrentIndex(dlg.device_combo.findData("cuda"))
            assert describe_compute_plan("cuda")["track_provider"] in dlg.lbl_compute_preview.text()

        dlg.device_combo.setCurrentIndex(dlg.device_combo.findData("cpu"))
        assert "CPUExecutionProvider" in dlg.lbl_compute_preview.text()

        dlg.device_combo.setCurrentIndex(dlg.device_combo.findData("auto"))
        assert describe_compute_plan("auto")["track_provider"] in dlg.lbl_compute_preview.text()

        captured = []
        monkeypatch.setattr(
            sd_mod,
            "run_gpu_preflight_probe",
            lambda model_p, dev: {
                "probe_ok": True,
                "active_provider": "DmlExecutionProvider",
                "stitch_device": "CPU (PyTorch)",
                "infer_ms": 12.3,
                "warmup_ms": 45.6,
                "cpu_infer_ms": 10.1,
                "hint": "",
            },
        )
        monkeypatch.setattr(QMessageBox, "information", lambda parent, title, msg: captured.append((title, msg)))

        dlg.btn_test_gpu.click()
        assert len(captured) == 1
        assert "DmlExecutionProvider" in captured[0][1]
        assert "12.3" in captured[0][1]
    finally:
        dlg.close()
        dlg.deleteLater()
        _app.processEvents()


def test_main_window_prerun_confirmation_cancel_and_proceed(tmp_path, monkeypatch):
    import queue
    from PySide6.QtWidgets import QApplication, QMessageBox
    import src.ui.main_window as mw_mod

    _app = QApplication.instance() or QApplication(["test", "-platform", "offscreen"])
    cmd_q: queue.Queue = queue.Queue()
    evt_q: queue.Queue = queue.Queue()
    db = Database(db_path=tmp_path / "ui_confirm.db")

    win = mw_mod.MainWindow(cmd_q, evt_q, AppConfig(), db, orchestrator=None)
    try:
        win.selected_video_path = str(tmp_path / "DJI_TEST.MP4")
        seen = {}
        target_label = {"text": "Batal"}

        def fake_exec(box):
            seen["title"] = box.windowTitle()
            seen["text"] = box.text()
            for btn in box.buttons():
                if btn.text() == target_label["text"]:
                    box._fake_clicked = btn
                    break
            return 0

        monkeypatch.setattr(QMessageBox, "exec", fake_exec)
        monkeypatch.setattr(QMessageBox, "clickedButton", lambda box: getattr(box, "_fake_clicked", None))

        win._on_start_job()
        assert seen["title"] == "Konfirmasi Perangkat & Mulai Proses"
        assert "Tracking Frame" in seen["text"]
        assert cmd_q.empty()

        target_label["text"] = "Mulai Proses"
        win._on_start_job()
        queued = cmd_q.get_nowait()
        assert queued["action"] == "START_JOB"
        assert queued["name"] == "DJI_TEST"
    finally:
        win.is_job_running = False
        win.close()
        win.deleteLater()
        _app.processEvents()


def test_rtx_cuda_and_cpu_switching_with_mocked_rtx(monkeypatch):
    import onnxruntime as ort
    import torch
    from src.recognition import detector

    monkeypatch.setattr(ort, "get_available_providers", lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"])
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda idx=0: "NVIDIA GeForce RTX 4060", raising=False)

    provs_auto, act_auto = detector.resolve_onnx_providers("auto")
    assert act_auto == "CUDAExecutionProvider"
    assert "CPUExecutionProvider" in provs_auto

    provs_cuda, act_cuda = detector.resolve_onnx_providers("cuda")
    assert act_cuda == "CUDAExecutionProvider"

    provs_cpu, act_cpu = detector.resolve_onnx_providers("cpu")
    assert provs_cpu == ["CPUExecutionProvider"]
    assert act_cpu == "CPUExecutionProvider"

    plan_auto = detector.describe_compute_plan("auto")
    assert plan_auto["uses_gpu"] is True
    assert plan_auto["track_provider"] == "CUDAExecutionProvider"
    assert "RTX 4060" in plan_auto["stitch_device"]

    plan_cpu = detector.describe_compute_plan("cpu")
    assert plan_cpu["uses_gpu"] is False
    assert plan_cpu["track_provider"] == "CPUExecutionProvider"
    assert "CPU" in plan_cpu["stitch_device"]

    # When no NVIDIA GPU is present on the PC, auto & cuda fall back cleanly to CPU
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert detector.resolve_onnx_providers("auto")[1] == "CPUExecutionProvider"
    assert detector.resolve_onnx_providers("cuda")[1] == "CPUExecutionProvider"
    plan_no_gpu = detector.describe_compute_plan("auto")
    assert plan_no_gpu["uses_gpu"] is False
    assert plan_no_gpu["track_provider"] == "CPUExecutionProvider"
