import tempfile
import unittest
from pathlib import Path
from src.database.db import Database


class TestDatabase(unittest.TestCase):
    def setUp(self):
        self._tmp_dir = tempfile.TemporaryDirectory()
        self.test_db_path = Path(self._tmp_dir.name) / "test_db.db"
        self.db = Database(db_path=self.test_db_path)

    def tearDown(self):
        self._tmp_dir.cleanup()

    def test_job_and_container_flow(self):
        # 1. Create job
        job_id = self.db.create_job(name="Test_Video_001", video_path="sample.mp4", mode="LEFT_RIGHT")
        self.assertEqual(job_id, 1)

        job = self.db.get_job(job_id)
        self.assertIsNotNone(job)
        self.assertEqual(job["status"], "QUEUED")

        # 2. Add container
        c_id = self.db.add_container(
            job_id=job_id,
            route_order=1,
            container_id=10,
            status="stable",
            image_path="outputs/images/reg_001.jpg",
        )
        self.assertEqual(c_id, 1)

        # 3. Add detection
        d_id = self.db.add_detection(
            container_id=c_id,
            detection_type="container_number",
            bbox_x=10,
            bbox_y=10,
            bbox_w=100,
            bbox_h=50,
            confidence=0.95,
            ocr_raw_text="MSCU1234566",
            ocr_clean_text="MSCU1234566",
            is_valid_iso=True,
        )
        self.assertEqual(d_id, 1)

        # 4. Verify full view
        view = self.db.get_full_container_view(job_id)
        self.assertEqual(len(view), 1)
        self.assertIn("container_number", view[0]["detections"])
        self.assertTrue(view[0]["detections"]["container_number"]["is_valid_iso"])

        # 5. Manual override
        self.db.update_detection_manual_override(
            detection_id=d_id,
            manual_override_text="MSCU1234567",
            is_valid_iso=False,
        )
        view2 = self.db.get_full_container_view(job_id)
        det = view2[0]["detections"]["container_number"]
        self.assertEqual(det["manual_override_text"], "MSCU1234567")
        self.assertEqual(det["is_reviewed"], 1)

        # 6. Output video path persistence
        self.db.update_job_output_video(job_id, "outputs/test_job_annotated.mp4")
        updated_job = self.db.get_job(job_id)
        self.assertEqual(updated_job["output_video_path"], "outputs/test_job_annotated.mp4")


if __name__ == "__main__":
    unittest.main()
