"""
tests/test_haar_detector.py
────────────────────────────
HaarDetector birim testleri.
Çalıştırma: python -m pytest tests/ -v
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from core.haar_detector import HaarDetector
from core.result_model  import DetectionResult


def make_blank_frame(h=480, w=640):
    return np.zeros((h, w, 3), dtype=np.uint8)


class TestHaarDetector:

    def test_init_default(self):
        det = HaarDetector()
        assert det.get_name() == "Haar Cascade"

    def test_detect_returns_result(self):
        det = HaarDetector()
        frame = make_blank_frame()
        result = det.detect(frame, frame_index=0)
        assert isinstance(result, DetectionResult)
        assert result.algorithm_name == "Haar Cascade"
        assert isinstance(result.bounding_boxes, list)
        assert isinstance(result.inference_time_ms, float)
        assert result.inference_time_ms >= 0

    def test_invalid_frame_raises(self):
        det = HaarDetector()
        with pytest.raises(ValueError):
            det.detect(None)

    def test_set_params(self):
        det = HaarDetector()
        det.set_params(scale_factor=1.2, min_neighbors=3)
        assert det.scale_factor == 1.2
        assert det.min_neighbors == 3

    def test_set_params_invalid_scale(self):
        det = HaarDetector()
        with pytest.raises(ValueError):
            det.set_params(scale_factor=0.9)

    def test_frame_index_propagated(self):
        det = HaarDetector()
        frame = make_blank_frame()
        result = det.detect(frame, frame_index=42)
        assert result.frame_index == 42