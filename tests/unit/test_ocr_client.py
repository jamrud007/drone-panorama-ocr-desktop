"""
Unit tests for Halotec OCR HTTP Client.
Complies with OCR_SERVICE_DOCUMENTATION.md & TDD v1.9 contracts:
- Numeric gate enforcement
- DataDetection results parsing (container_number, iso_code)
- camera_position parameterization
- Token refresh on 401
"""
import base64
import json
import queue
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from src.ocr_client.client import HalotecOcrClient, OcrClientError
from src.orchestrator.engine import ApiAuditLogger
from src.utils.image_io import imwrite_unicode


class TestHalotecOcrClient(unittest.TestCase):
    def setUp(self):
        self.client = HalotecOcrClient(
            base_url="https://mock.ocr.example.com",
            gate="1",
            timeout=5,
            max_retries=2,
        )

    def test_gate_numeric_enforcement(self):
        # Non-numeric gate string must fallback safely to "1"
        c1 = HalotecOcrClient(gate="GATE-01")
        self.assertEqual(c1.gate, "1")

        c2 = HalotecOcrClient(gate="2")
        self.assertEqual(c2.gate, "2")

        c3 = HalotecOcrClient(gate="05")
        self.assertEqual(c3.gate, "5")

    def test_camera_position_validation(self):
        # Ensure invalid camera positions fallback to "rear"
        with patch.object(self.client._session, "post") as mock_post:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {
                "id": 1,
                "gate": 1,
                "camera_position": "rear",
                "results": {
                    "container_number": [{"text": "MSCU1234566", "score_text": 0.98}]
                },
            }
            mock_post.return_value = mock_resp

            text, conf = self.client.recognize_base64(
                base64_str="dummy_base64",
                detection_type="container_number",
                camera_position="invalid_pos",
            )
            self.assertEqual(text, "MSCU1234566")
            self.assertEqual(conf, 0.98)

            # Check that payload sent camera_position="rear"
            sent_payload = mock_post.call_args[1]["json"]
            self.assertEqual(sent_payload["camera_position"], "rear")
            self.assertEqual(sent_payload["gate"], "1")

    def test_parse_data_detection_iso_code(self):
        with patch.object(self.client._session, "post") as mock_post:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {
                "id": 2,
                "gate": 1,
                "camera_position": "left",
                "results": {
                    "iso_code": [
                        {"text": "22G1", "score_text": 0.70},
                        {"text": "45G1", "score_text": 0.96},
                    ]
                },
            }
            mock_post.return_value = mock_resp

            text, conf = self.client.recognize_base64(
                base64_str="dummy_base64",
                detection_type="iso_code",
                camera_position="left",
            )
            # Must pick item with highest score_text (45G1 with 0.96)
            self.assertEqual(text, "45G1")
            self.assertEqual(conf, 0.96)

    def test_retry_on_401_with_refresh_token(self):
        with patch.object(self.client._session, "post") as mock_post, \
             patch.object(self.client, "refresh_token", return_value="new_mock_token") as mock_refresh:

            resp_401 = MagicMock()
            resp_401.status_code = 401

            resp_200 = MagicMock()
            resp_200.status_code = 200
            resp_200.json.return_value = {
                "results": {
                    "container_number": [{"text": "TGHU9876543", "score_text": 0.99}]
                }
            }

            mock_post.side_effect = [resp_401, resp_200]

            text, conf = self.client.recognize_base64(
                base64_str="dummy_base64",
                detection_type="container_number",
            )
            self.assertEqual(text, "TGHU9876543")
            self.assertEqual(mock_refresh.call_count, 1)

    def test_server_500_raises_ocr_error(self):
        with patch.object(self.client._session, "post") as mock_post:
            mock_resp = MagicMock()
            mock_resp.status_code = 500
            mock_resp.text = "Internal Server Error in model parser"
            mock_post.return_value = mock_resp

            with self.assertRaises(OcrClientError) as ctx:
                self.client.recognize_base64("dummy")
            self.assertIn("500", str(ctx.exception))

    @patch("time.sleep", return_value=None)
    def test_log_callback_and_rate_limit_retry(self, mock_sleep):
        with patch.object(self.client._session, "post") as mock_post:
            resp_429 = MagicMock()
            resp_429.status_code = 429
            resp_429.text = "Too Many Requests"
            resp_429.json.return_value = {"message": "Rate limit exceeded. Try again in 2 seconds."}

            resp_200 = MagicMock()
            resp_200.status_code = 200
            resp_200.json.return_value = {
                "results": {
                    "container_number": [{"text": "MEDU1234567", "score_text": 0.97}]
                }
            }

            mock_post.side_effect = [resp_429, resp_200]
            logs = []

            def cb(data):
                logs.append(data)

            text, conf = self.client.recognize_base64(
                base64_str="dummy_base64",
                detection_type="container_number",
                log_callback=cb,
                context_info="Kontainer R001",
            )

            self.assertEqual(text, "MEDU1234567")
            self.assertEqual(conf, 0.97)
            self.assertEqual(mock_sleep.call_count, 1)

            # Check logged events
            events = [l.get("event") for l in logs]
            self.assertEqual(events, ["REQUEST_SENDING", "RESPONSE_RECEIVED", "REQUEST_SENDING", "RESPONSE_RECEIVED"])
            self.assertEqual(logs[1]["status"], "RATE_LIMIT")
            self.assertEqual(logs[1]["status_code"], 429)
            self.assertIn("Rate limit", logs[1]["server_message"])
            self.assertEqual(logs[3]["status"], "SUCCESS")
            self.assertEqual(logs[3]["status_code"], 200)

    @patch("time.sleep", return_value=None)
    def test_rate_limit_exceeded_raises_ocr_error(self, mock_sleep):
        with patch.object(self.client._session, "post") as mock_post:
            resp_429 = MagicMock()
            resp_429.status_code = 429
            resp_429.text = "Too Many Requests"
            resp_429.json.return_value = {"message": "Rate limit quota exhausted"}
            mock_post.return_value = resp_429

            with self.assertRaises(OcrClientError) as ctx:
                self.client.recognize_base64("dummy")

            self.assertIn("Rate Limit (429)", str(ctx.exception))


class TestApiAuditLogger(unittest.TestCase):
    def test_audit_logger_writes_txt_json_and_emits_event(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            txt_file = tmp_path / "api_requests.txt"
            json_file = tmp_path / "api_requests.json"
            evt_q = queue.Queue()

            logger = ApiAuditLogger(txt_file, json_file, evt_q, job_id=99)

            # 1. Test Request Sending Event
            logger.on_api_event({
                "event": "REQUEST_SENDING",
                "timestamp": "2026-11-22 10:00:00",
                "attempt": 1,
                "max_retries": 3,
                "req_id": 101,
                "context_info": "Kontainer R001 (Track 1)",
                "url": "https://ocr.example.com/api/ocr",
                "gate": "1",
                "detection_type": "container_number",
                "camera_position": "left",
                "payload_kb": 45.2,
            })

            # Check queue
            self.assertFalse(evt_q.empty())
            evt1 = evt_q.get()
            self.assertEqual(evt1.type, "LOG")
            self.assertIn("[KIRIM KE SERVER]", evt1.data["log"])
            self.assertIn("Kontainer R001", evt1.data["log"])

            # 2. Test Response Received (Rate limit)
            logger.on_api_event({
                "event": "RESPONSE_RECEIVED",
                "status": "RATE_LIMIT",
                "status_code": 429,
                "duration_ms": 120.5,
                "attempt": 1,
                "server_message": "Rate limit exceeded (10 req/min)",
                "context_info": "Kontainer R001 (Track 1)",
                "detection_type": "container_number",
            })

            evt2 = evt_q.get()
            self.assertIn("[SERVER RATE LIMIT]", evt2.data["log"])
            self.assertIn("Rate limit exceeded", evt2.data["log"])

            # 3. Test Response Received (Success)
            logger.on_api_event({
                "event": "RESPONSE_RECEIVED",
                "status": "SUCCESS",
                "status_code": 200,
                "duration_ms": 140.2,
                "attempt": 2,
                "server_message": "OK",
                "text": "TBJU7377270",
                "confidence": 0.99,
                "context_info": "Kontainer R001 (Track 1)",
                "detection_type": "container_number",
            })

            evt3 = evt_q.get()
            self.assertIn("[SERVER OK]", evt3.data["log"])
            self.assertIn("TBJU7377270", evt3.data["log"])

            # Verify files on disk
            self.assertTrue(txt_file.is_file())
            self.assertTrue(json_file.is_file())

            with open(txt_file, "r", encoding="utf-8") as f:
                txt_content = f.read()
            self.assertIn("[KIRIM KE SERVER]", txt_content)
            self.assertIn("[SERVER RATE LIMIT]", txt_content)
            self.assertIn("[SERVER OK]", txt_content)

            with open(json_file, "r", encoding="utf-8") as f:
                json_content = json.load(f)
            self.assertEqual(len(json_content), 3)
            self.assertEqual(json_content[1]["status"], "RATE_LIMIT")
            self.assertEqual(json_content[2]["text"], "TBJU7377270")
