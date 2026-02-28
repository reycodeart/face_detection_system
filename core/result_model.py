"""
core/result_model.py
────────────────────
Tespit sonucunu taşıyan merkezi veri modeli.
Tüm dedektörler bu modeli üretir; GUI ve profiler bu modeli tüketir.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple
import time


@dataclass
class DetectionResult:
    """
    Tek bir frame'e ait tespit sonucunu taşır.

    Attributes:
        algorithm_name   : Algoritmayı tanımlayan dize  ("Haar Cascade", "DNN …").
        bounding_boxes   : [(x, y, w, h)] koordinat listesi.
        confidence_scores: Her bbox için [0,1] güven skoru.
        inference_time_ms: Yalnızca çıkarım süresi (ms).
        frame_index      : İşlenen frame'in sıra numarası.
        image_shape      : Giriş frame boyutu (H, W, C).
        timestamp        : Tespiti oluşturulma zamanı (Unix).
    """

    algorithm_name: str
    bounding_boxes: List[Tuple[int, int, int, int]] = field(default_factory=list)
    confidence_scores: List[float] = field(default_factory=list)
    inference_time_ms: float = 0.0
    frame_index: int = 0
    image_shape: Tuple[int, ...] = (0, 0, 0)
    timestamp: float = field(default_factory=time.time)

    # ── Erişim kolaylıkları ──────────────────────

    @property
    def num_detections(self) -> int:
        """Tespit edilen yüz sayısı."""
        return len(self.bounding_boxes)

    @property
    def avg_confidence(self) -> float:
        """Ortalama güven skoru; skor yoksa 0.0."""
        if not self.confidence_scores:
            return 0.0
        return sum(self.confidence_scores) / len(self.confidence_scores)

    # ── Dönüşümler ───────────────────────────────

    def to_dict(self) -> dict:
        """CSV / JSON çıktısı için sözlüğe dönüştürür."""
        return {
            "algorithm": self.algorithm_name,
            "frame_index": self.frame_index,
            "num_detections": self.num_detections,
            "bounding_boxes": self.bounding_boxes,
            "confidence_scores": [round(s, 4) for s in self.confidence_scores],
            "inference_time_ms": round(self.inference_time_ms, 4),
            "avg_confidence": round(self.avg_confidence, 4),
            "timestamp": round(self.timestamp, 6),
        }

    def __repr__(self) -> str:
        return (
            f"DetectionResult(algo={self.algorithm_name!r}, "
            f"faces={self.num_detections}, "
            f"time={self.inference_time_ms:.2f}ms, "
            f"frame={self.frame_index})"
        )