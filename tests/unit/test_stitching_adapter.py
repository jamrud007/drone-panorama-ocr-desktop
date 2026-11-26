"""
Unit test for StitchingAdapter and Path Resolution.
Validates TDD v1.9 stitching adapter behavior, path fallback, and input validation.
"""
import unittest
from pathlib import Path

from src.config import AppConfig, BASE_DIR, BUNDLE_DIR
from src.stitching.adapter import StitchingAdapter


class TestStitchingAdapter(unittest.TestCase):
    def setUp(self):
        self.cfg = AppConfig()
        self.adapter = StitchingAdapter(self.cfg)

    def test_adapter_initialization(self):
        self.assertIsNotNone(self.adapter.config)
        self.assertIsNotNone(self.adapter.job_object)

    def test_model_and_script_path_resolution(self):
        model_path = self.cfg.detection.resolved_model_path()
        self.assertTrue(model_path.exists(), f"Model path must exist: {model_path}")

        script_path = self.cfg.stitching.resolved_script_path()
        self.assertTrue(script_path.exists(), f"Script path must exist: {script_path}")

    def test_invalid_video_raises_filenotfound(self):
        non_existent_video = BASE_DIR / "non_existent_drone_video.mp4"
        with self.assertRaises(FileNotFoundError):
            self.adapter.run_stitching(
                video_path=non_existent_video,
                camera_position="LEFT_RIGHT",
            )


if __name__ == "__main__":
    unittest.main()
