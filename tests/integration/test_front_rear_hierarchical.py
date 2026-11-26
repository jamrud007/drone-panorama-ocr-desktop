import queue
import cv2
import numpy as np
from pathlib import Path
import pytest

from src.config import AppConfig
from src.database.db import Database
from src.orchestrator.engine import Orchestrator, OrchestratorEvent


def test_front_rear_hierarchical_generates_annotated_images(tmp_path, monkeypatch):
    # 1. Create a synthetic test video with 30 frames
    video_path = tmp_path / "test_synth.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(video_path), fourcc, 10.0, (640, 480))
    for i in range(30):
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        # Draw a simulated container box
        cv2.rectangle(img, (100, 100), (400, 400), (120, 120, 120), -1)
        # Draw simulated text
        cv2.putText(img, "MSMU1234567", (120, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        out.write(img)
    out.release()

    # 2. Setup mock DB and Orchestrator
    db_path = tmp_path / "test_app.db"
    db = Database(db_path=db_path)
    cmd_q = queue.Queue()
    evt_q = queue.Queue()
    cfg = AppConfig()
    cfg.api.url = "https://mock.ocr.example.com"
    cfg.paths.output_dir = str(tmp_path / "outputs")

    orch = Orchestrator(command_queue=cmd_q, event_queue=evt_q, config=cfg, db=db)

    # 3. Mock OnnxDetector to return a container and a number inside it
    from src.recognition.detector import DetectionBox

    class MockDetector:
        def detect(self, img, conf_thresh=0.2, imgsz=1280, classes=None):
            return [
                DetectionBox(class_id=3, class_name="container_rear_front", confidence=0.90, x1=100, y1=100, x2=400, y2=400),
                DetectionBox(class_id=2, class_name="container_number", confidence=0.85, x1=120, y1=130, x2=280, y2=170),
            ]

    monkeypatch.setattr(orch, "_get_detector", lambda: MockDetector())

    # 4. Mock OCR client to return a valid ISO 6346 number
    monkeypatch.setattr(orch.ocr_client, "recognize_image_file", lambda *args, **kwargs: ("MSMU1234566", 0.95))

    # 5. Run _process_front_rear_job
    job_id = db.create_job("Synthetic Job", str(video_path), "FRONT_REAR")
    orch._process_front_rear_job(job_id=job_id, video_path=str(video_path), camera_position="rear")

    # 6. Verify containers table
    containers = db.get_containers_for_job(job_id)
    assert len(containers) >= 1
    c0 = containers[0]
    assert c0["container_type"] == "FRONT_REAR"
    assert c0["annotated_image_path"] is not None
    assert Path(c0["annotated_image_path"]).is_file()
    assert Path(c0["image_path"]).is_file()

    # 7. Verify detections table
    dets = db.get_detections_for_container(c0["id"])
    assert len(dets) >= 1
    d0 = dets[0]
    assert d0["detection_type"] == "container_number"
    assert d0["ocr_clean_text"] == "MSMU1234566"
    assert d0["is_valid_iso"] == 1
    assert Path(d0["crop_image_path"]).is_file()
