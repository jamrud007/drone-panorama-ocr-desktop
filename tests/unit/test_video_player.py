import tempfile
import unittest
from pathlib import Path
from PySide6.QtWidgets import QApplication
from src.database.db import Database
from src.ui.video_player_dialog import VideoPlayerDialog


class TestVideoPlayer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not QApplication.instance():
            cls.app = QApplication(["test", "-platform", "offscreen"])
        else:
            cls.app = QApplication.instance()

    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tmp_dir.name) / "test.db"
        self.db = Database(db_path=self.db_path)

    def tearDown(self):
        try:
            self.tmp_dir.cleanup()
        except Exception:
            pass

    def test_video_player_dialog_instantiation(self):
        from src.ui.video_player_dialog import format_ms
        self.assertEqual(format_ms(0), "00:00")
        self.assertEqual(format_ms(65000), "01:05")

        dummy_vid = Path(self.tmp_dir.name) / "dummy.mp4"
        dummy_vid.touch()
        for i in range(3):
            dlg = VideoPlayerDialog(None, str(dummy_vid), f"Test Video Player {i}")
            self.assertIsNotNone(dlg.video_widget)
            self.assertIsNotNone(dlg.slider)
            self.assertIsNotNone(dlg.btn_play)
            if i == 0:
                dlg.reject()
            elif i == 1:
                dlg.accept()
            else:
                dlg.close()
            self.assertTrue(dlg._is_cleaned_up)
            self.assertIsNone(dlg.player)
            self.assertIsNone(dlg.audio)

    def test_database_video_field(self):
        job_id = self.db.create_job("Test Vid Job", "vid.mp4", "TOP")
        job = self.db.get_job(job_id)
        self.assertIn("output_video_path", job)
        self.assertIsNone(job["output_video_path"])

        self.db.update_job_output_video(job_id, "outputs/session/vid_annotated.mp4")
        updated = self.db.get_job(job_id)
        self.assertEqual(updated["output_video_path"], "outputs/session/vid_annotated.mp4")


if __name__ == "__main__":
    unittest.main()
