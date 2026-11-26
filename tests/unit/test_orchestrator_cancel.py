import queue
import threading
import time
from pathlib import Path
import pytest

from src.config import AppConfig
from src.database.db import Database
from src.orchestrator.engine import Orchestrator, OrchestratorEvent


def test_orchestrator_cancel_sets_flag_and_interrupts(tmp_path):
    db_path = tmp_path / "test.db"
    db = Database(db_path=db_path)
    cmd_q = queue.Queue()
    evt_q = queue.Queue()

    orch = Orchestrator(command_queue=cmd_q, event_queue=evt_q, db=db)
    assert not orch._cancel_requested.is_set()

    orch.cancel_current_job()
    assert orch._cancel_requested.is_set()


def test_main_window_start_cancel_button_state(tmp_path):
    from PySide6.QtWidgets import QApplication
    from src.ui.main_window import MainWindow

    app = QApplication.instance() or QApplication(["test", "-platform", "offscreen"])
    cmd_q = queue.Queue()
    evt_q = queue.Queue()
    cfg = AppConfig()
    db = Database(db_path=tmp_path / "test_ui.db")

    win = MainWindow(cmd_queue=cmd_q, evt_queue=evt_q, config=cfg, db=db)

    # Initial state
    assert win.btn_start.text() == "Mulai Proses"
    assert "0078D4" in win.btn_start.styleSheet()

    # When job starts -> Batalkan Proses (Red)
    win._set_start_button_state(True)
    assert win.is_job_running is True
    assert win.btn_start.text() == "Batalkan Proses"
    assert "D83B01" in win.btn_start.styleSheet()
    assert win.btn_start.isEnabled()

    # When job finishes or cancelled -> Mulai Proses (Blue)
    win._set_start_button_state(False)
    assert win.is_job_running is False
    assert win.btn_start.text() == "Mulai Proses"
    assert "0078D4" in win.btn_start.styleSheet()
    assert win.btn_start.isEnabled()

    win.close()
