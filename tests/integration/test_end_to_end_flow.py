"""
End-to-End Integration Flow Test for Orchestrator, Database, and OCR Client.
Simulates Stage 4 labels handoff, RAW image cropping, OCR recognition,
ISO 6346 validation, SQLite storage, and manual override review.
"""
import json
import queue
import shutil
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from src.config import AppConfig
from src.database.db import Database
from src.orchestrator.engine import Orchestrator
from src.utils.image_io import imwrite_unicode


class TestEndToEndFlow(unittest.TestCase):
    def setUp(self):
        self._tmp_dir = tempfile.TemporaryDirectory()
        self.test_data_dir = Path(self._tmp_dir.name)
        self.test_db_path = self.test_data_dir / "test_e2e.db"

        # Generate a test RAW panorama image (1000x500 px)
        self.raw_pano_path = self.test_data_dir / "test_region_001.jpg"
        dummy_pano = np.zeros((500, 1000, 3), dtype=np.uint8)
        # Draw simulated container area
        dummy_pano[100:400, 200:800] = [180, 100, 50]
        # Draw simulated text regions
        dummy_pano[120:160, 600:780] = [255, 255, 255]
        imwrite_unicode(self.raw_pano_path, dummy_pano)

        # Generate Stage 4 JSON label file
        self.stage4_json_path = self.test_data_dir / "test_region_001.json"
        stage4_content = {
            "region_id": "R001",
            "container_id": 1,
            "status": "stable",
            "image_path": str(self.raw_pano_path),
            "panorama_coordinates": {"x": 200, "y": 100, "width": 600, "height": 300},
            "detections": {
                "container_number": {
                    "confidence": 0.94,
                    "bbox": {"x": 600, "y": 120, "width": 180, "height": 40},
                },
                "iso_code": {
                    "confidence": 0.91,
                    "bbox": {"x": 600, "y": 170, "width": 80, "height": 30},
                },
            },
        }
        with open(self.stage4_json_path, "w", encoding="utf-8") as f:
            json.dump(stage4_content, f)

        # Setup configuration and DB (isolated from repo config/config.json)
        self.cfg = AppConfig()
        self.cfg.api.url = "https://mock.ocr.example.com"
        self.cfg.paths.data_dir = str(self.test_data_dir)
        self.cfg.paths.output_dir = str(self.test_data_dir / "outputs")
        self.db = Database(db_path=self.test_db_path)

        self.cmd_queue = queue.Queue()
        self.evt_queue = queue.Queue()
        self.orchestrator = None

    def tearDown(self):
        if self.orchestrator is not None:
            self.orchestrator.stop()
            self.orchestrator.join(timeout=5)
        self._tmp_dir.cleanup()

    @patch("src.ocr_client.client.requests.Session.post")
    def test_full_pipeline_flow(self, mock_post):
        # 1. Mock OCR API response for container_number and iso_code
        def mock_post_handler(url, json=None, headers=None, timeout=None):
            resp = MagicMock()
            resp.status_code = 200
            # Check detection type by inspection
            resp.json.return_value = {
                "id": json.get("id", 1),
                "gate": 1,
                "camera_position": json.get("camera_position", "left"),
                "results": {
                    "container_number": [{"text": "MSCU1234566", "score_text": 0.97}],
                    "iso_code": [{"text": "45G1", "score_text": 0.95}],
                },
            }
            return resp

        mock_post.side_effect = mock_post_handler

        # 2. Initialize orchestrator
        orchestrator = Orchestrator(
            command_queue=self.cmd_queue,
            event_queue=self.evt_queue,
            config=self.cfg,
            db=self.db,
        )
        self.orchestrator = orchestrator

        # Mock stitching adapter so it directly returns our pre-generated Stage 4 JSON
        orchestrator.stitching_adapter.run_stitching = MagicMock(return_value=[self.stage4_json_path])

        orchestrator.start()

        try:
            # 3. Dispatch START_JOB command
            self.cmd_queue.put({
                "action": "START_JOB",
                "name": "IntegrationPilot",
                "video_path": str(self.raw_pano_path),
                "mode": "LEFT_RIGHT",
                "camera_position": "left",
            })

            # Wait for job completion event (allow headroom for cold torch/ONNX session init)
            completed = False
            timeout = 30
            start_t = time.time()
            events = []

            while time.time() - start_t < timeout:
                try:
                    evt = self.evt_queue.get(timeout=0.2)
                    events.append(evt.type)
                    if evt.type == "JOB_COMPLETED":
                        completed = True
                        break
                except queue.Empty:
                    continue

            self.assertTrue(completed, f"Job did not complete in time. Events received: {events}")

            # 4. Verify database state
            jobs = self.db.get_all_jobs()
            self.assertEqual(len(jobs), 1)
            self.assertEqual(jobs[0]["status"], "COMPLETED")
            self.assertEqual(jobs[0]["mode"], "LEFT_RIGHT")

            containers = self.db.get_full_container_view(jobs[0]["id"])
            self.assertEqual(len(containers), 1)
            c = containers[0]
            self.assertEqual(c["route_order"], 1)

            # Check container_number detection
            num_det = c["detections"].get("container_number")
            self.assertIsNotNone(num_det)
            self.assertEqual(num_det["ocr_clean_text"], "MSCU1234566")
            self.assertTrue(num_det["is_valid_iso"])

            # Check iso_code detection
            iso_det = c["detections"].get("iso_code")
            self.assertIsNotNone(iso_det)
            self.assertEqual(iso_det["ocr_clean_text"], "45G1")

            # Verify that crop files were physically generated and saved to disk
            crop_path = Path(num_det["crop_image_path"])
            self.assertTrue(crop_path.is_file(), f"Crop image not found at {crop_path}")

            # Verify unified single job_activity.log captures UI + OCR service + evaluation summary without split files
            job_out_dirs = [p for p in Path(self.cfg.paths.resolved_output_dir()).glob(f"*_{self.raw_pano_path.stem}")]
            if job_out_dirs:
                latest_logs = sorted(job_out_dirs, key=lambda p: p.stat().st_mtime)[-1] / "logs"
                unified = latest_logs / "job_activity.log"
                self.assertTrue(unified.is_file(), f"Missing unified log at {unified}")
                txt = unified.read_text(encoding="utf-8")
                self.assertIn("MULAI JOB", txt)
                self.assertIn("[KIRIM KE SERVER]", txt)
                self.assertIn("[SERVER OK]", txt)
                self.assertIn("RINGKASAN EVALUASI JOB", txt)
                self.assertIn("Durasi Total", txt)
                self.assertFalse((latest_logs / "api_requests.txt").exists())
                self.assertFalse((latest_logs / "stitching_pipeline.log").exists())

            # 5. Test manual review override
            self.cmd_queue.put({
                "action": "MANUAL_OVERRIDE",
                "detection_id": num_det["id"],
                "text": "TEMU9876543",
            })

            time.sleep(0.5)

            # Re-fetch view
            updated_view = self.db.get_full_container_view(jobs[0]["id"])
            updated_det = updated_view[0]["detections"]["container_number"]
            self.assertEqual(updated_det["manual_override_text"], "TEMU9876543")
            self.assertEqual(updated_det["is_reviewed"], 1)

        finally:
            orchestrator.stop()
            orchestrator.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
