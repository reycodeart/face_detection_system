"""
core/detector_base.py
─────────────────────
Tüm dedektörlerin uyması gereken ABC (Abstract Base Class) arayüzü.
Ortak çizim ve doğrulama yardımcıları burada tanımlanır.
"""

from __future__ import annotations

import abc
from typing import Tuple

import cv2
import numpy as np

from core.result_model import DetectionResult


class DetectorBase(abc.ABC):
    """
    Yüz dedektörleri için soyut temel sınıf.

    Alt sınıflar şu metodları uygulamak ZORUNDADIR:
      • load_model()
      • detect(frame, frame_index) → DetectionResult
      • get_name()                 → str
    """

    # ── Soyut Metodlar ──────────────────────────

    @abc.abstractmethod
    def load_model(self) -> None:
        """Model dosyasını / ağırlıklarını belleğe yükler."""

    @abc.abstractmethod
    def detect(self, frame: np.ndarray, frame_index: int = 0) -> DetectionResult:
        """
        Verilen frame üzerinde yüz tespiti yapar.

        Args:
            frame      : BGR formatında NumPy dizisi.
            frame_index: Sıra numarası (metrik/log için).

        Returns:
            DetectionResult nesnesi.
        """

    @abc.abstractmethod
    def get_name(self) -> str:
        """Algoritmayı tanımlayan kısa ad döner."""

    # ── Paylaşılan Yardımcılar ───────────────────

    @staticmethod
    def validate_frame(frame: np.ndarray) -> bool:
        """
        Frame'in işlenebilir durumda olup olmadığını kontrol eder.

        Args:
            frame: Kontrol edilecek NumPy dizisi.

        Returns:
            True ise geçerli, False ise geçersiz.
        """
        if frame is None:
            return False
        if not isinstance(frame, np.ndarray):
            return False
        if frame.size == 0:
            return False
        if len(frame.shape) < 2:
            return False
        return True

    @staticmethod
    def draw_results(
        frame: np.ndarray,
        result: DetectionResult,
        color: Tuple[int, int, int] = (0, 220, 80),
        thickness: int = 2,
        show_label: bool = True,
    ) -> np.ndarray:
        """
        Tespit sonuçlarını (bounding box + etiket) frame kopyası üzerine çizer.

        Args:
            frame      : Kaynak BGR görüntü (değiştirilmez).
            result     : Çizilecek DetectionResult.
            color      : BGR çizgi rengi.
            thickness  : Çizgi kalınlığı (px).
            show_label : Algoritma adı + skor etiketi gösterilsin mi?

        Returns:
            Çizim yapılmış yeni frame.
        """
        output = frame.copy()
        for i, (x, y, w, h) in enumerate(result.bounding_boxes):
            cv2.rectangle(output, (x, y), (x + w, y + h), color, thickness)
            if show_label:
                label = result.algorithm_name
                if result.confidence_scores and i < len(result.confidence_scores):
                    label += f"  {result.confidence_scores[i]:.2f}"
                cv2.putText(
                    output, label, (x, max(y - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color,
                    thickness - 1, cv2.LINE_AA,
                )

        # Köşe bilgi satırı
        info = f"{result.algorithm_name} | {result.inference_time_ms:.1f} ms | {result.num_detections} yuz"
        cv2.putText(output, info, (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
        return output

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.get_name()!r})"