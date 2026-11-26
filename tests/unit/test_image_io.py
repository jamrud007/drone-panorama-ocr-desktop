"""
Unit tests for Unicode-safe Image I/O.
"""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.utils.image_io import imread_unicode, imwrite_unicode


class TestImageIO(unittest.TestCase):
    def setUp(self):
        # Create a directory with spaces and Unicode characters inside an isolated temp dir
        self._tmp_dir = tempfile.TemporaryDirectory()
        self.test_dir = Path(self._tmp_dir.name) / "test_unicode_üñîçødé_spaces"
        self.test_dir.mkdir(parents=True, exist_ok=True)
        self.test_file = self.test_dir / "sample_kontainer_ü.png"

    def tearDown(self):
        self._tmp_dir.cleanup()

    def test_write_and_read_unicode_path(self):
        # Generate dummy 100x100 RGB image
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[20:80, 20:80] = [0, 255, 0]  # Green box

        # 1. Write image (lossless PNG)
        success = imwrite_unicode(self.test_file, img)
        self.assertTrue(success)
        self.assertTrue(self.test_file.is_file())

        # 2. Read image back
        loaded = imread_unicode(self.test_file)
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.shape, (100, 100, 3))
        # Center pixel should match green color exactly
        self.assertEqual(list(loaded[50, 50]), [0, 255, 0])

    def test_read_nonexistent_file(self):
        res = imread_unicode(self.test_dir / "not_found.jpg")
        self.assertIsNone(res)

    def test_write_invalid_image(self):
        self.assertFalse(imwrite_unicode(self.test_file, None))
        self.assertFalse(imwrite_unicode(self.test_file, np.array([])))
