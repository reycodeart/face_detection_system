"""
tests/test_dnn_detector.py
───────────────────────────
DNNDetector birim testleri (model dosyaları gerektirir).
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from core.dnn_detector import DNNDetector
from core.result_model import DetectionResult


def make_blank_frame(h=300, w=300):
    return np.zeros((h, w, 3), dtype=np.uint8)


class TestDNNDetector:

    def test_init(self):
        det = DNNDetector(auto_download=True)
        assert det.get_name() == "DNN (SSD+ResNet)"

    def test_detect_returns_result(self):
        det = DNNDetector(auto_download=True)
        frame = make_blank_frame()
        result = det.detect(frame, frame_index=1)
        assert isinstance(result, DetectionResult)
        assert result.inference_time_ms >= 0
        assert result.algorithm_name == "DNN (SSD+ResNet)"

    def test_invalid_frame_raises(self):
        det = DNNDetector(auto_download=True)
        with pytest.raises(ValueError):
            det.detect(None)

    def test_confidence_threshold_setter(self):
        det = DNNDetector(auto_download=True)
        det.set_confidence_threshold(0.7)
        assert det.confidence_thr == 0.7

    def test_confidence_threshold_invalid(self):
        det = DNNDetector(auto_download=True)
        with pytest.raises(ValueError):
            det.set_confidence_threshold(1.5)

    def test_all_scores_above_threshold(self):
        det = DNNDetector(confidence_thr=0.5, auto_download=True)
        frame = make_blank_frame(480, 640)
        result = det.detect(frame)
        for score in result.confidence_scores:
            assert score >= 0.5