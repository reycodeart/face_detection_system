"""
utils/video_capture.py
──────────────────────
Thread-safe kamera ve video dosyası akış yöneticisi.
Webcam, video dosyası ve statik görüntüyü tek arayüzden sunar.
"""

from __future__ import annotations

import threading
import queue
from enum import Enum, auto
from typing import Optional, Tuple

import cv2
import numpy as np


class SourceType(Enum):
    """Kaynak türü sabitleri."""
    WEBCAM = auto()
    VIDEO  = auto()
    IMAGE  = auto()


class VideoCaptureManager:
    """
    OpenCV VideoCapture sargısı — thread-safe frame üretici.

    Arka plan thread'i sürekli frame okur ve bir kuyruğa koyar.
    Ana thread (GUI) get_frame() ile en güncel frame'i alır.

    Args:
        source      : Kamera indeksi (int) veya dosya yolu (str).
        queue_size  : Frame kuyruğu maksimum boyutu.
        target_fps  : Kamera için hedef FPS (None = donanım varsayılanı).
        target_size : Hedef (genişlik, yükseklik); None = kaynak boyutu.
    """

    def __init__(
        self,
        source: int | str,
        queue_size: int = 4,
        target_fps: Optional[int] = None,
        target_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        self.source      = source
        self.target_fps  = target_fps
        self.target_size = target_size

        self._cap: Optional[cv2.VideoCapture] = None
        self._queue: queue.Queue[Optional[np.ndarray]] = queue.Queue(maxsize=queue_size)
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._source_type = self._detect_source_type()
        self._single_frame: Optional[np.ndarray] = None   # statik görüntü

    # ── Kaynak Tipi Tespiti ──────────────────────

    def _detect_source_type(self) -> SourceType:
        if isinstance(self.source, int):
            return SourceType.WEBCAM
        ext = str(self.source).lower().rsplit(".", 1)[-1]
        if ext in ("jpg", "jpeg", "png", "bmp", "webp", "tiff"):
            return SourceType.IMAGE
        return SourceType.VIDEO

    # ── Başlatma / Durdurma ──────────────────────

    def start(self) -> "VideoCaptureManager":
        """
        Akışı başlatır. Akış türüne göre arka plan thread'i oluşturur.

        Returns:
            self (method chaining için).

        Raises:
            RuntimeError: Kamera/video açılamazsa.
        """
        self._stop_event.clear()

        if self._source_type == SourceType.IMAGE:
            frame = cv2.imread(str(self.source))
            if frame is None:
                raise RuntimeError(f"Görüntü okunamadı: {self.source}")
            self._single_frame = frame
            return self

        self._cap = cv2.VideoCapture(self.source)
        if not self._cap.isOpened():
            raise RuntimeError(f"Kaynak açılamadı: {self.source}")

        # Kamera yapılandırması
        if self._source_type == SourceType.WEBCAM:
            if self.target_size:
                self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.target_size[0])
                self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.target_size[1])
            if self.target_fps:
                self._cap.set(cv2.CAP_PROP_FPS, self.target_fps)

        self._thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        """Akışı durdurur ve kaynağı serbest bırakır."""
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)
        if self._cap:
            self._cap.release()
            self._cap = None

    # ── Frame Okuma ──────────────────────────────

    def get_frame(self) -> Optional[np.ndarray]:
        """
        En güncel frame'i döner.

        Returns:
            BGR frame veya None (henüz hazır değilse / bittiyse).
        """
        if self._source_type == SourceType.IMAGE:
            return self._single_frame

        try:
            return self._queue.get_nowait()
        except queue.Empty:
            return None

    def is_running(self) -> bool:
        """Okuma thread'inin hâlâ çalışıp çalışmadığını döner."""
        if self._source_type == SourceType.IMAGE:
            return self._single_frame is not None
        return self._thread is not None and self._thread.is_alive()

    # ── Meta Bilgi ───────────────────────────────

    @property
    def fps(self) -> float:
        """Kaynak FPS değeri."""
        if self._cap:
            return self._cap.get(cv2.CAP_PROP_FPS) or 25.0
        return 25.0

    @property
    def frame_size(self) -> Tuple[int, int]:
        """(genişlik, yükseklik) çifti."""
        if self._cap:
            return (
                int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            )
        if self._single_frame is not None:
            h, w = self._single_frame.shape[:2]
            return w, h
        return (640, 480)

    @property
    def total_frames(self) -> int:
        """Video dosyasındaki toplam frame sayısı (webcam için -1)."""
        if self._cap and self._source_type == SourceType.VIDEO:
            return int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return -1

    # ── Arka Plan Okuyucu ────────────────────────

    def _reader_loop(self) -> None:
        """
        Arka plan thread'inde çalışır.
        Her frame'i okuyup kuyruğa koyar; kuyruk doluysa eski frame'i atar.
        Thread sonunda kuyruğa None koyar (sinyal).
        """
        while not self._stop_event.is_set():
            if not self._cap or not self._cap.isOpened():
                break
            ret, frame = self._cap.read()
            if not ret:
                break   # Dosya sonu veya kamera hatası

            # Eski frame'i at
            if self._queue.full():
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    pass
            self._queue.put(frame)

        # Bitiş sinyali
        self._queue.put(None)

    # ── Context Manager ──────────────────────────

    def __enter__(self) -> "VideoCaptureManager":
        return self.start()

    def __exit__(self, *_) -> None:
        self.stop()

    def __repr__(self) -> str:
        return (
            f"VideoCaptureManager(source={self.source!r}, "
            f"type={self._source_type.name}, running={self.is_running()})"
        )