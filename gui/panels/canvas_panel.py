"""
gui/panels/canvas_panel.py
──────────────────────────
Merkez görüntü gösterim alanı.
OpenCV BGR frame'lerini Tkinter Canvas üzerinde render eder.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Optional, Tuple

import numpy as np

from utils.image_utils import bgr_to_photoimage

_BG_CANVAS = "#0f0f1a"
_FG_DIM    = "#94a3b8"
_BORDER    = "#3f3f5a"

_PLACEHOLDER_TEXT = (
    "Görüntü / Video / Webcam seçin\n\n"
    "🖼  Fotoğraf    🎬  Video    📷  Webcam"
)


class CanvasPanel(ttk.Frame):
    """
    Tespit sonucu görüntülerini gösteren panel.

    Özellikler:
      • BGR frame'i aspect-ratio korumalı olarak ölçekler.
      • Placeholder metin gösterir (kaynak seçilmemişse).
      • Frame boyutunu dinamik olarak hesaplar.
    """

    def __init__(self, parent: tk.Widget, **kwargs) -> None:
        super().__init__(parent, **kwargs)

        self._photo_ref: Optional[object] = None   # GC koruması

        self._canvas = tk.Canvas(
            self,
            bg=_BG_CANVAS,
            highlightthickness=1,
            highlightbackground=_BORDER,
            cursor="crosshair",
        )
        self._canvas.pack(fill="both", expand=True)

        # Placeholder
        self._placeholder_id = self._canvas.create_text(
            400, 260,
            text=_PLACEHOLDER_TEXT,
            fill=_FG_DIM,
            font=("Segoe UI", 13),
            justify="center",
        )

    # ── Render ───────────────────────────────────

    def render_frame(self, frame_bgr: np.ndarray) -> None:
        """
        BGR frame'i canvas boyutuna ölçekler ve çizer.

        Args:
            frame_bgr: Çizilecek BGR NumPy dizisi.
        """
        if frame_bgr is None:
            return

        cw = self._canvas.winfo_width()  or 820
        ch = self._canvas.winfo_height() or 520

        photo, (nw, nh) = bgr_to_photoimage(frame_bgr, max_size=(cw, ch))

        self._canvas.delete("all")
        x_off = (cw - nw) // 2
        y_off = (ch - nh) // 2
        self._canvas.create_image(x_off, y_off, anchor="nw", image=photo)
        self._photo_ref = photo   # referansı sakla (GC koruması)

    def show_placeholder(self) -> None:
        """Canvas'ı temizler ve placeholder metnini gösterir."""
        self._canvas.delete("all")
        self._photo_ref = None
        cw = self._canvas.winfo_width()  or 820
        ch = self._canvas.winfo_height() or 520
        self._canvas.create_text(
            cw // 2, ch // 2,
            text=_PLACEHOLDER_TEXT,
            fill=_FG_DIM,
            font=("Segoe UI", 13),
            justify="center",
        )

    def clear(self) -> None:
        """Canvas'ı tamamen temizler (placeholder olmadan)."""
        self._canvas.delete("all")
        self._photo_ref = None

    # ── Boyut Bilgisi ────────────────────────────

    @property
    def size(self) -> Tuple[int, int]:
        """(genişlik, yükseklik) çifti."""
        return self._canvas.winfo_width(), self._canvas.winfo_height()