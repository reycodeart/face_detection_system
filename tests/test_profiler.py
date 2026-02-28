"""
tests/test_profiler.py
───────────────────────
PerformanceProfiler birim testleri.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import time
import pytest
from core.performance_profiler import PerformanceProfiler, compute_iou
from core.result_model         import DetectionResult


def make_result(algo="Haar Cascade", faces=2, time_ms=20.0, idx=0):
    return DetectionResult(
        algorithm_name=algo,
        bounding_boxes=[(10, 10, 50, 50)] * faces,
        confidence_scores=[0.9] * faces,
        inference_time_ms=time_ms,
        frame_index=idx,
    )


class TestComputeIoU:

    def test_perfect_overlap(self):
        box = (0, 0, 100, 100)
        assert compute_iou(box, box) == pytest.approx(1.0)

    def test_no_overlap(self):
        assert compute_iou((0, 0, 10, 10), (20, 20, 10, 10)) == pytest.approx(0.0)

    def test_partial_overlap(self):
        iou = compute_iou((0, 0, 20, 20), (10, 0, 20, 20))
        assert 0.0 < iou < 1.0


class TestPerformanceProfiler:

    def test_record_and_stats(self):
        p = PerformanceProfiler()
        for i in range(5):
            p.record(make_result(idx=i, time_ms=10.0 + i))
        stats = p.get_statistics()
        assert "Haar Cascade" in stats
        s = stats["Haar Cascade"]
        assert s.num_frames == 5
        assert s.avg_time_ms == pytest.approx(12.0, abs=0.1)

    def test_live_fps(self):
        p = PerformanceProfiler(fps_window=10)
        for i in range(15):
            p.record(make_result(idx=i))
            time.sleep(0.005)
        fps = p.get_live_fps("Haar Cascade")
        assert fps > 0

    def test_reset(self):
        p = PerformanceProfiler()
        p.record(make_result())
        p.reset()
        assert p.get_statistics() == {}

    def test_precision_recall(self):
        p = PerformanceProfiler()
        p.record(make_result("DNN (SSD+ResNet)", idx=0))
        p.record_gt_match(
            "DNN (SSD+ResNet)",
            predicted=[(10, 10, 50, 50)],
            ground_truth=[(10, 10, 50, 50)],
            iou_thr=0.5,
        )
        stats = p.get_statistics()
        s = stats["DNN (SSD+ResNet)"]
        assert s.precision == pytest.approx(1.0)
        assert s.recall    == pytest.approx(1.0)
        assert s.f1_score  == pytest.approx(1.0)

    def test_multiple_algorithms(self):
        p = PerformanceProfiler()
        p.record(make_result("Haar Cascade", idx=0))
        p.record(make_result("DNN (SSD+ResNet)", idx=0))
        stats = p.get_statistics()
        assert len(stats) == 2