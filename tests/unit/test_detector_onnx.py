"""
Unit tests for Local ONNX Container Detector.
"""
import unittest
from pathlib import Path

import numpy as np

from src.recognition.detector import DetectionBox, OnnxDetector, letterbox, nms_cpu


class TestDetector(unittest.TestCase):
    def test_crop_box_boundaries(self):
        # 100x100 dummy image
        img = np.zeros((100, 100, 3), dtype=np.uint8)

        # Normal crop
        crop = OnnxDetector.crop_box(img, {"x": 10, "y": 10, "width": 30, "height": 40})
        self.assertEqual(crop.shape, (40, 30, 3))

        # Negative x, y must clamp to 0
        crop_neg = OnnxDetector.crop_box(img, {"x": -20, "y": -10, "width": 50, "height": 50})
        self.assertEqual(crop_neg.shape, (50, 50, 3))

        # Out-of-bounds width, height must clamp to image dimensions
        crop_oob = OnnxDetector.crop_box(img, {"x": 80, "y": 80, "width": 50, "height": 50})
        self.assertEqual(crop_oob.shape, (20, 20, 3))

    def test_letterbox_shapes(self):
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        padded, ratio, (dw, dh) = letterbox(img, new_shape=(640, 640))
        self.assertEqual(padded.shape, (640, 640, 3))
        self.assertAlmostEqual(ratio, 1.0, places=2)

    def test_nms_cpu(self):
        boxes = np.array([
            [10, 10, 50, 50],
            [12, 12, 52, 52],  # High overlap with box 0
            [100, 100, 150, 150],  # Independent box
        ], dtype=float)
        scores = np.array([0.9, 0.8, 0.95], dtype=float)

        keep = nms_cpu(boxes, scores, iou_threshold=0.5)
        # Should keep box 2 (score 0.95) and box 0 (score 0.9), suppressing box 1
        self.assertEqual(len(keep), 2)
        self.assertIn(2, keep)
        self.assertIn(0, keep)
        self.assertNotIn(1, keep)

    def test_onnx_model_inference_if_weights_exist(self):
        model_path = Path("container-detection-drone-view/weight/container_detection_model.onnx")
        if not model_path.is_file():
            self.skipTest("ONNX model weights file not present in repository")

        detector = OnnxDetector(model_path)
        dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
        results = detector.detect(dummy_img, conf_thresh=0.25)
        self.assertIsInstance(results, list)
