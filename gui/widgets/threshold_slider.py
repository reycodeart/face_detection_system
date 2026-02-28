"""
gui/widgets/threshold_slider.py
────────────────────────────────
Etiketli parametre slider bileşeni.
Haar scale / min_neighbors ve DNN confidence için yeniden kullanılabilir.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Callable, Optional

_BG     = "#2a2a3e"
_FG     = "#e2e8f0"
_FG_DIM = "#94a3b8"
_ACCENT = "#06b6d4"


class LabeledSlider(ttk.Frame):
    """
    Başlık + Scale + anlık değer etiketi içeren bileşik slider.

    Args:
        parent    : Üst widget.
        label     : Slider başlığı.
        from_     : Minimum değer.
        to        : Maksimum değer.
        default   : Başlangıç değeri.
        resolution: Adım büyüklüğü (int için 1, float için 0.01 vb.)
        fmt       : Değer gösterim formatı (ör. "{:.2f}", "{:.0f}").
        on_change : Değer değişince çağrılacak callback(value: float).
        width     : Slider genişliği (px).
    """

    def __init__(
        self,
        parent: tk.Widget,
        label: str,
        from_: float,
        to: float,
        default: float,
        resolution: float = 0.01,
        fmt: str = "{:.2f}",
        on_change: Optional[Callable[[float], None]] = None,
        width: int = 140,
        **kwargs,
    ) -> None:
        super().__init__(parent, style="Panel.TFrame", **kwargs)
        self._fmt = fmt
        self._on_change = on_change
        self._var = tk.DoubleVar(value=default)
        self._disp_var = tk.StringVar(value=fmt.format(default))

        # Başlık etiketi
        tk.Label(
            self, text=label,
            bg=_BG, fg=_FG_DIM,
            font=("Segoe UI", 8),
        ).pack(anchor="w")

        # Slider + değer satırı
        row = ttk.Frame(self, style="Panel.TFrame")
        row.pack(fill="x")

        self._scale = ttk.Scale(
            row,
            from_=from_, to=to,
            orient="horizontal",
            variable=self._var,
            command=self._on_scale_move,
            length=width,
        )
        self._scale.pack(side="left", fill="x", expand=True)

        tk.Label(
            row, textvariable=self._disp_var,
            bg=_BG, fg=_ACCENT,
            font=("Consolas", 9), width=6,
        ).pack(side="left")

    # ── Olay İşleyici ────────────────────────────

    def _on_scale_move(self, raw_value: str) -> None:
        """Scale hareketinde değeri biçimlendirir ve callback'i çağırır."""
        v = float(raw_value)
        self._disp_var.set(self._fmt.format(v))
        if self._on_change:
            self._on_change(v)

    # ── Özellikler ───────────────────────────────

    @property
    def value(self) -> float:
        """Güncel slider değeri."""
        return self._var.get()

    def set_value(self, v: float) -> None:
        """
        Değeri programatik olarak günceller.

        Args:
            v: Yeni değer (from_–to arasında olmalı).
        """
        self._var.set(v)
        self._disp_var.set(self._fmt.format(v))

    def set_state(self, enabled: bool) -> None:
        """
        Slider'ı etkinleştirir veya devre dışı bırakır.

        Args:
            enabled: True = normal, False = disabled.
        """
        state = "normal" if enabled else "disabled"
        self._scale.config(state=state)