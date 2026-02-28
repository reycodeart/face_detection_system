"""
gui/panels/metrics_panel.py
────────────────────────────
Alt metrik kartları ve performans gösterge paneli.
FPS, yüz sayısı, gecikme ve toplam frame bilgilerini canlı gösterir.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Dict, Optional

_BG    = "#2a2a3e"
_FG    = "#e2e8f0"
_DIM   = "#94a3b8"
_CYAN  = "#06b6d4"
_GREEN = "#22c55e"
_AMBER = "#f59e0b"


class _MetricCard(ttk.Frame):
    """Tek bir metriği gösteren mini kart bileşeni."""

    def __init__(
        self,
        parent: tk.Widget,
        title: str,
        unit: str = "",
        accent: str = _CYAN,
    ) -> None:
        super().__init__(parent, style="Panel.TFrame", padding=(10, 6))
        self._unit   = unit
        self._accent = accent

        tk.Label(self, text=title, bg=_BG, fg=_DIM,
                 font=("Segoe UI", 8)).pack(anchor="w")

        self._var = tk.StringVar(value="—")
        tk.Label(self, textvariable=self._var, bg=_BG, fg=accent,
                 font=("Consolas", 12, "bold")).pack(anchor="w")

    def update(self, value: float | str, fmt: str = "{:.1f}") -> None:
        """
        Kart değerini günceller.

        Args:
            value: Yeni değer (sayı veya metin).
            fmt  : Format dizgisi (sayı için).
        """
        if isinstance(value, (int, float)):
            text = fmt.format(value)
        else:
            text = str(value)
        self._var.set(f"{text} {self._unit}".strip())

    def reset(self) -> None:
        """Değeri tire (—) ile sıfırlar."""
        self._var.set("—")


class MetricsPanel(ttk.Frame):
    """
    Canlı performans metriklerini gösteren yatay kart çubuğu.

    Kartlar:
      • FPS (kayan pencere)
      • Yüz Sayısı
      • Gecikme (ms)
      • Algoritma
      • Toplam Frame
      • Ortalama FPS
    """

    def __init__(self, parent: tk.Widget, **kwargs) -> None:
        super().__init__(parent, style="Panel.TFrame", padding=(6, 4), **kwargs)
        self._cards: Dict[str, _MetricCard] = {}
        self._build()

    def _build(self) -> None:
        """Metrik kartlarını oluşturur."""
        defs = [
            ("fps",       "FPS",           "fps",  _CYAN,  "{:.1f}"),
            ("faces",     "Yüz Sayısı",    "adet", _GREEN, "{:.0f}"),
            ("latency",   "Gecikme",       "ms",   _AMBER, "{:.1f}"),
            ("algo",      "Algoritma",     "",     _CYAN,  "{}"),
            ("frames",    "Toplam Frame",  "",     _DIM,   "{:.0f}"),
            ("avg_fps",   "Ort. FPS",      "fps",  _CYAN,  "{:.1f}"),
        ]

        for key, title, unit, accent, _ in defs:
            card = _MetricCard(self, title=title, unit=unit, accent=accent)
            card.pack(side="left", expand=True, fill="x", padx=3)
            self._cards[key] = card

    def update_all(
        self,
        fps: float = 0.0,
        faces: int = 0,
        latency_ms: float = 0.0,
        algo: str = "—",
        total_frames: int = 0,
        avg_fps: float = 0.0,
    ) -> None:
        """
        Tüm kartları tek seferde günceller.

        Args:
            fps         : Anlık FPS.
            faces       : Mevcut frame'deki yüz sayısı.
            latency_ms  : Ortalama çıkarım süresi (ms).
            algo        : Aktif algoritma adı.
            total_frames: İşlenen toplam frame.
            avg_fps     : Birikimli ortalama FPS.
        """
        self._cards["fps"].update(fps,          "{:.1f}")
        self._cards["faces"].update(faces,       "{:.0f}")
        self._cards["latency"].update(latency_ms,"{:.1f}")
        self._cards["algo"].update(algo,         "{}")
        self._cards["frames"].update(total_frames,"{:.0f}")
        self._cards["avg_fps"].update(avg_fps,   "{:.1f}")

    def reset_all(self) -> None:
        """Tüm kartları sıfırlar."""
        for card in self._cards.values():
            card.reset()