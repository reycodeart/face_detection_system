"""
core/haar_detector.py
─────────────────────
Viola-Jones / Haar Cascade tabanlı yüz dedektörü.
OpenCV'nin CascadeClassifier sınıfını kullanır.
"""

from __future__ import annotations

import os
import time
from typing import List, Optional, Tuple

import cv2
import numpy as np

from core.detector_base import DetectorBase
from core.result_model import DetectionResult


class HaarDetector(DetectorBase):
    """
    OpenCV Haar Cascade Classifier ile yüz tespiti.

    Args:
        model_path   : .xml cascade dosyasının yolu.
                       None → OpenCV dahili varsayılan model kullanılır.
        scale_factor : Her piramit adımında görüntü küçültme oranı (>1.0).
        min_neighbors: Tespit kabul eşiği (komşu sayısı).
        min_size     : Minimum yüz boyutu (px).
        max_size     : Maksimum yüz boyutu; None = sınırsız.
        equalize_hist: Gri görüntüye histogram eşitleme uygulansın mı?
    """

    _NAME = "Haar Cascade"

    def __init__(
        self,
        model_path: Optional[str] = None,
        scale_factor: float = 1.1,
        min_neighbors: int = 5,
        min_size: Tuple[int, int] = (30, 30),
        max_size: Optional[Tuple[int, int]] = None,
        equalize_hist: bool = False,
    ) -> None:
        self.model_path    = model_path
        self.scale_factor  = scale_factor
        self.min_neighbors = min_neighbors
        self.min_size      = min_size
        self.max_size      = max_size or ()
        self.equalize_hist = equalize_hist
        self._clf: Optional[cv2.CascadeClassifier] = None
        self.load_model()

    # ── Soyut Metodların Uygulaması ──────────────

    def get_name(self) -> str:
        return self._NAME

    def load_model(self) -> None:
        """
        Cascade modelini yükler.

        Raises:
            FileNotFoundError: Model dosyası bulunamazsa.
            RuntimeError     : CascadeClassifier boş yüklenirse.
        """
        if self.model_path is None:
            self.model_path = (
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )

        if not os.path.isfile(self.model_path):
            raise FileNotFoundError(
                f"Haar model dosyası bulunamadı: {self.model_path}"
            )

        self._clf = cv2.CascadeClassifier(self.model_path)
        if self._clf.empty():
            raise RuntimeError(f"CascadeClassifier yüklenemedi: {self.model_path}")

        print(f"[HaarDetector] Model yüklendi ✓  ({self.model_path})")

    def detect(self, frame: np.ndarray, frame_index: int = 0) -> DetectionResult:
        """
        Frame üzerinde Haar Cascade ile yüz tespiti yapar.

        Args:
            frame      : BGR görüntü.
            frame_index: Frame numarası.

        Returns:
            DetectionResult

        Raises:
            ValueError : Geçersiz frame.
            RuntimeError: Model yüklü değilse.
        """
        if not self.validate_frame(frame):
            raise ValueError("Geçersiz frame (None veya boş).")
        if self._clf is None or self._clf.empty():
            raise RuntimeError("Model yüklü değil.")

        gray = self._preprocess(frame)

        # maxSize bos tuple/None ise kwarg a ekleme (OpenCV 4.x uyumlulugu)
        detect_kwargs: dict = dict(
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=self.min_size,
        )
        if self.max_size:
            detect_kwargs["maxSize"] = self.max_size

        t0 = time.perf_counter()
        detections = self._clf.detectMultiScale(gray, **detect_kwargs)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        bboxes: List[Tuple[int, int, int, int]] = []
        scores: List[float] = []

        # detectMultiScale guven skoru dondurmez; sabit 1.0 atanir
        if len(detections) > 0:
            for (x, y, w, h) in detections:
                bboxes.append((int(x), int(y), int(w), int(h)))
                scores.append(1.0)

        return DetectionResult(
            algorithm_name=self._NAME,
            bounding_boxes=bboxes,
            confidence_scores=scores,
            inference_time_ms=elapsed_ms,
            frame_index=frame_index,
            image_shape=frame.shape,
        )

    # ── Yardımcılar ──────────────────────────────

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """BGR → gri dönüşümü ve opsiyonel histogram eşitleme."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.equalize_hist:
            gray = cv2.equalizeHist(gray)
        return gray

    def set_params(
        self,
        scale_factor: Optional[float] = None,
        min_neighbors: Optional[int] = None,
        min_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        """
        Çalışma zamanında parametreleri günceller.

        Args:
            scale_factor : Yeni ölçek faktörü (>1.0).
            min_neighbors: Yeni komşu eşiği (>=1).
            min_size     : Yeni minimum boyut.
        """
        if scale_factor is not None:
            if scale_factor <= 1.0:
                raise ValueError("scale_factor > 1.0 olmalı.")
            self.scale_factor = scale_factor

        if min_neighbors is not None:
            if min_neighbors < 1:
                raise ValueError("min_neighbors >= 1 olmalı.")
            self.min_neighbors = min_neighbors

        if min_size is not None:
            self.min_size = min_size

    def __repr__(self) -> str:
        return (
            f"HaarDetector(scale={self.scale_factor}, "
            f"neighbors={self.min_neighbors}, min_size={self.min_size})"
        )