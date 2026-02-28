"""
utils/image_utils.py
────────────────────
Görüntü dönüşüm ve ön işleme yardımcıları.
GUI render, kaydetme ve görüntü manipülasyonu için ortak fonksiyonlar.
"""

from __future__ import annotations

from typing import Tuple, Optional

import cv2
import numpy as np
from PIL import Image, ImageTk


def bgr_to_photoimage(
    frame_bgr: np.ndarray,
    max_size: Tuple[int, int] = (820, 520),
) -> Tuple["ImageTk.PhotoImage", Tuple[int, int]]:
    """
    BGR NumPy dizisini Tkinter PhotoImage'e dönüştürür.
    Aspect-ratio korunarak max_size sınırına ölçeklenir.

    Args:
        frame_bgr: OpenCV BGR görüntüsü.
        max_size : (genişlik, yükseklik) maksimum hedef boyutu.

    Returns:
        (PhotoImage, (yeni_genişlik, yeni_yükseklik)) çifti.
    """
    h, w = frame_bgr.shape[:2]
    mw, mh = max_size
    scale = min(mw / w, mh / h, 1.0)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))

    resized = cv2.resize(frame_bgr, (nw, nh), interpolation=cv2.INTER_AREA)
    rgb     = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    photo   = ImageTk.PhotoImage(Image.fromarray(rgb))
    return photo, (nw, nh)


def resize_keep_aspect(
    frame: np.ndarray,
    max_w: int,
    max_h: int,
) -> np.ndarray:
    """
    Görüntüyü en-boy oranını koruyarak yeniden boyutlandırır.

    Args:
        frame: Kaynak görüntü.
        max_w: Maksimum genişlik.
        max_h: Maksimum yükseklik.

    Returns:
        Yeniden boyutlandırılmış görüntü.
    """
    h, w = frame.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)
    if scale == 1.0:
        return frame
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    return cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)


def draw_overlay_text(
    frame: np.ndarray,
    lines: list[str],
    origin: Tuple[int, int] = (10, 20),
    font_scale: float = 0.55,
    color: Tuple[int, int, int] = (255, 255, 255),
    bg_color: Optional[Tuple[int, int, int]] = (0, 0, 0),
    thickness: int = 1,
    line_gap: int = 22,
) -> np.ndarray:
    """
    Frame üzerine çok satırlı metin overlay ekler.

    Args:
        frame     : Hedef görüntü (kopyası döner).
        lines     : Metin satırları.
        origin    : İlk satırın sol-üst koordinatı.
        font_scale: Yazı boyutu.
        color     : Yazı rengi (BGR).
        bg_color  : Arka plan rengi; None ise arka plan çizilmez.
        thickness : Yazı kalınlığı.
        line_gap  : Satırlar arası piksel boşluğu.

    Returns:
        Metin eklenmiş frame kopyası.
    """
    out = frame.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    x, y = origin

    for i, line in enumerate(lines):
        ly = y + i * line_gap
        if bg_color is not None:
            (tw, th), _ = cv2.getTextSize(line, font, font_scale, thickness)
            cv2.rectangle(out, (x - 2, ly - th - 2), (x + tw + 2, ly + 4), bg_color, -1)
        cv2.putText(out, line, (x, ly), font, font_scale, color, thickness, cv2.LINE_AA)

    return out


def save_image(frame_bgr: np.ndarray, path: str) -> bool:
    """
    BGR görüntüyü diske kaydeder.

    Args:
        frame_bgr: Kaydedilecek görüntü.
        path     : Hedef dosya yolu.

    Returns:
        Başarı durumu.
    """
    try:
        ok = cv2.imwrite(path, frame_bgr)
        return bool(ok)
    except Exception as exc:
        print(f"[image_utils] Kayıt hatası: {exc}")
        return False


def stack_frames_horizontal(frames: list[np.ndarray]) -> np.ndarray:
    """
    Farklı yüksekliklere sahip frame'leri aynı yüksekliğe normalize edip yan yana birleştirir.

    Args:
        frames: BGR görüntü listesi.

    Returns:
        Yatay birleştirilmiş tek görüntü.
    """
    if not frames:
        raise ValueError("En az bir frame gerekli.")
    if len(frames) == 1:
        return frames[0]

    max_h = max(f.shape[0] for f in frames)
    normalized = []
    for f in frames:
        h, w = f.shape[:2]
        if h < max_h:
            pad = np.zeros((max_h - h, w, 3), dtype=f.dtype)
            f = np.vstack([f, pad])
        normalized.append(f)

    return np.hstack(normalized)