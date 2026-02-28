"""
core/dnn_detector.py
────────────────────
SSD + ResNet-10 tabanlı yüz dedektörü (OpenCV DNN modülü).
Caffe formatı: deploy.prototxt + res10_300x300_ssd_iter_140000.caffemodel

Model dosyaları yoksa models/ klasörüne otomatik indirilir.
"""

from __future__ import annotations

import os
import time
import urllib.request
from typing import List, Optional, Tuple

import cv2
import numpy as np

from core.detector_base import DetectorBase
from core.result_model  import DetectionResult


# ── Model URL'leri ────────────────────────────────────────────────────────────
_PROTOTXT_URL = (
    "https://raw.githubusercontent.com/opencv/opencv/master/"
    "samples/dnn/face_detector/deploy.prototxt"
)
_CAFFE_URL = (
    "https://github.com/opencv/opencv_3rdparty/raw/"
    "dnn_samples_face_detector_20170830/"
    "res10_300x300_ssd_iter_140000.caffemodel"
)

# ── Varsayılan yerel yollar ───────────────────────────────────────────────────
_HERE       = os.path.dirname(os.path.abspath(__file__))
_MODEL_DIR  = os.path.normpath(os.path.join(_HERE, "..", "models"))
_PROTO_PATH = os.path.join(_MODEL_DIR, "deploy.prototxt")
_CAFFE_PATH = os.path.join(_MODEL_DIR, "res10_300x300_ssd_iter_140000.caffemodel")


class DNNDetector(DetectorBase):
    """
    OpenCV DNN + Caffe SSD modeli ile yüz tespiti.

    Args:
        prototxt_path   : .prototxt dosya yolu (None -> models/ varsayılanı).
        caffemodel_path : .caffemodel dosya yolu (None -> models/ varsayılanı).
        confidence_thr  : Minimum guven skoru (0.0 - 1.0).
        input_size      : Ag giris boyutu (SSD icin 300x300).
        mean_vals       : BGR piksel ortalamalari (blob normalizasyonu).
        use_gpu         : CUDA backend denensin mi?
        auto_download   : Model eksikse otomatik indirilsin mi?
    """

    _NAME = "DNN (SSD+ResNet)"

    def __init__(
        self,
        prototxt_path: Optional[str]   = None,
        caffemodel_path: Optional[str] = None,
        confidence_thr: float          = 0.5,
        input_size: Tuple[int, int]    = (300, 300),
        mean_vals: Tuple[float, float, float] = (104.0, 177.0, 123.0),
        use_gpu: bool    = False,
        auto_download: bool = True,
    ) -> None:
        self.prototxt_path   = os.path.abspath(prototxt_path   or _PROTO_PATH)
        self.caffemodel_path = os.path.abspath(caffemodel_path or _CAFFE_PATH)
        self.confidence_thr  = confidence_thr
        self.input_size      = input_size
        self.mean_vals       = mean_vals
        self.use_gpu         = use_gpu
        self.auto_download   = auto_download
        self._net: Optional[cv2.dnn.Net] = None
        self._loaded: bool = False
        self.load_model()

    # ─────────────────────────────────────────────
    # Arayuz metodlari
    # ─────────────────────────────────────────────

    def get_name(self) -> str:
        return self._NAME

    def load_model(self) -> None:
        """
        Model dosyalarini indirir (gerekirse) ve agi belleğe yukler.

        Raises:
            FileNotFoundError : Dosyalar eksik ve auto_download=False.
            RuntimeError      : readNetFromCaffe basarisiz olursa.
        """
        self._loaded = False
        self._net    = None

        # 1. Dosyalari hazirla (yoksa indir)
        self._ensure_models()

        # 2. Dosya boyutu kontrolu (bozuk/sifir bayt indirme tespiti)
        self._check_file_sizes()

        # 3. Agi yukle
        try:
            net = cv2.dnn.readNetFromCaffe(self.prototxt_path, self.caffemodel_path)
        except cv2.error as exc:
            raise RuntimeError(
                f"readNetFromCaffe basarisiz:\n"
                f"  prototxt  : {self.prototxt_path}\n"
                f"  caffemodel: {self.caffemodel_path}\n"
                f"  Hata: {exc}"
            ) from exc

        # 4. Bos ag kontrolu
        if net.empty():
            raise RuntimeError(
                "DNN agi bos yuklendi (empty). "
                "Model dosyalari bozuk olabilir; models/ klasorunu silin ve yeniden calistirin."
            )

        # 5. Backend sec
        if self.use_gpu:
            try:
                net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
                print("[DNNDetector] CUDA backend etkin.")
            except Exception:
                print("[DNNDetector] CUDA kullanilamiyor -> CPU'ya gecildi.")
                net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        else:
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        self._net    = net
        self._loaded = True
        print("[DNNDetector] Model yuklendi OK")
        print(f"  prototxt  : {self.prototxt_path}")
        print(f"  caffemodel: {self.caffemodel_path}")

    def detect(self, frame: np.ndarray, frame_index: int = 0) -> DetectionResult:
        """
        SSD + ResNet-10 ile yuz tespiti yapar.

        Args:
            frame      : BGR goruntu dizisi.
            frame_index: Sira numarasi.

        Returns:
            DetectionResult nesnesi.

        Raises:
            ValueError  : Gecersiz frame.
            RuntimeError: Model yuklu degil veya ag bos.
        """
        if not self.validate_frame(frame):
            raise ValueError("Gecersiz frame (None veya bos dizi).")

        if not self._loaded or self._net is None:
            raise RuntimeError(
                "DNN modeli yuklu degil. load_model() basariyla tamamlanmamis."
            )

        if self._net.empty():
            raise RuntimeError(
                "DNN agi bos (empty). Model dosyalarini kontrol edin."
            )

        h, w = frame.shape[:2]

        blob = cv2.dnn.blobFromImage(
            frame,
            scalefactor=1.0,
            size=self.input_size,
            mean=self.mean_vals,
            swapRB=False,
            crop=False,
        )

        t0 = time.perf_counter()
        self._net.setInput(blob)
        out = self._net.forward()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        bboxes: List[Tuple[int, int, int, int]] = []
        scores: List[float] = []

        for i in range(out.shape[2]):
            conf = float(out[0, 0, i, 2])
            if conf < self.confidence_thr:
                continue

            x1 = max(0,     int(out[0, 0, i, 3] * w))
            y1 = max(0,     int(out[0, 0, i, 4] * h))
            x2 = min(w - 1, int(out[0, 0, i, 5] * w))
            y2 = min(h - 1, int(out[0, 0, i, 6] * h))

            box_w = x2 - x1
            box_h = y2 - y1
            if box_w <= 0 or box_h <= 0:
                continue

            bboxes.append((x1, y1, box_w, box_h))
            scores.append(round(conf, 4))

        return DetectionResult(
            algorithm_name=self._NAME,
            bounding_boxes=bboxes,
            confidence_scores=scores,
            inference_time_ms=elapsed_ms,
            frame_index=frame_index,
            image_shape=frame.shape,
        )

    # ─────────────────────────────────────────────
    # Yardimci metodlar
    # ─────────────────────────────────────────────

    def _ensure_models(self) -> None:
        """Eksik model dosyalarini indirir."""
        missing: List[Tuple[str, str]] = []

        if not os.path.isfile(self.prototxt_path):
            missing.append((self.prototxt_path, _PROTOTXT_URL))

        if not os.path.isfile(self.caffemodel_path):
            missing.append((self.caffemodel_path, _CAFFE_URL))

        if not missing:
            return

        if not self.auto_download:
            paths_str = "\n".join(f"  - {p}" for p, _ in missing)
            raise FileNotFoundError(
                f"DNN model dosyalari bulunamadi:\n{paths_str}\n"
                "Manuel olarak models/ klasorune koyun veya auto_download=True yapin."
            )

        os.makedirs(_MODEL_DIR, exist_ok=True)
        for dest, url in missing:
            self._download_file(url, dest)

    def _check_file_sizes(self) -> None:
        """Bozuk (cok kucuk) dosyalari silip yeniden indirir."""
        min_sizes = {
            self.prototxt_path:   1_000,
            self.caffemodel_path: 1_000_000,
        }
        needs_redownload = False
        for path, min_size in min_sizes.items():
            if os.path.isfile(path):
                size = os.path.getsize(path)
                if size < min_size:
                    print(
                        f"[DNNDetector] Uyari: {os.path.basename(path)} "
                        f"cok kucuk ({size} bayt) - yeniden indiriliyor."
                    )
                    os.remove(path)
                    needs_redownload = True

        if needs_redownload:
            self._ensure_models()

    @staticmethod
    def _download_file(url: str, dest: str) -> None:
        """URL'den dosyayi indirir."""
        print(f"[DNNDetector] Indiriliyor: {os.path.basename(dest)}")
        print(f"              URL        : {url}")
        try:
            urllib.request.urlretrieve(url, dest)
            size_mb = os.path.getsize(dest) / 1_048_576
            print(f"[DNNDetector] Tamamlandi : {dest}  ({size_mb:.2f} MB)")
        except Exception as exc:
            if os.path.exists(dest):
                os.remove(dest)
            raise RuntimeError(
                f"Model indirilemedi: {os.path.basename(dest)}\n"
                f"URL: {url}\n"
                f"Hata: {exc}\n\n"
                "Cozum: Asagidaki dosyayi manuel olarak models/ klasorune koyun:\n"
                f"  prototxt  : {_PROTOTXT_URL}\n"
                f"  caffemodel: {_CAFFE_URL}"
            ) from exc

    def set_confidence_threshold(self, thr: float) -> None:
        """Guven esigini gunceller (0.0 - 1.0)."""
        if not (0.0 <= thr <= 1.0):
            raise ValueError(f"Esik 0.0-1.0 arasinda olmali, alinan: {thr}")
        self.confidence_thr = thr

    @property
    def is_loaded(self) -> bool:
        """Model basariyla yuklenmis mi?"""
        return self._loaded and self._net is not None and not self._net.empty()

    def __repr__(self) -> str:
        return (
            f"DNNDetector("
            f"conf={self.confidence_thr}, "
            f"size={self.input_size}, "
            f"gpu={self.use_gpu}, "
            f"loaded={self.is_loaded})"
        )