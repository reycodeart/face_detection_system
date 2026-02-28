"""
gui/widgets/algorithm_selector.py
──────────────────────────────────
Algoritma seçim widget'ı.
Haar / DNN / Her İkisi seçeneklerini Radiobutton grubu olarak sunar.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Callable, Optional

# Renk sabitleri (gui/app.py'den de import edilebilir; burada bağımsız tutuldu)
_BG     = "#2a2a3e"
_FG     = "#e2e8f0"
_FG_DIM = "#94a3b8"
_ACCENT = "#06b6d4"

ALGO_OPTIONS = ["Haar Cascade", "DNN (SSD+ResNet)", "Her İkisi"]


class AlgorithmSelector(ttk.LabelFrame):
    """
    Algoritma seçim bileşeni.

    Args:
        parent     : Üst widget.
        on_change  : Seçim değiştiğinde çağrılacak callback(algo_name: str).
        default    : Başlangıç seçimi.
    """

    def __init__(
        self,
        parent: tk.Widget,
        on_change: Optional[Callable[[str], None]] = None,
        default: str = "Her İkisi",
        **kwargs,
    ) -> None:
        kwargs.pop("padding", None)          # çakışmayı önle
        super().__init__(parent, text="Algoritma", padding=8, **kwargs)
        self._on_change = on_change
        self._var = tk.StringVar(value=default)

        for option in ALGO_OPTIONS:
            rb = tk.Radiobutton(
                self,
                text=option,
                variable=self._var,
                value=option,
                bg=_BG, fg=_FG,
                activebackground=_BG,
                activeforeground=_ACCENT,
                selectcolor="#1e1e2e",
                font=("Segoe UI", 10),
                command=self._on_radio_change,
            )
            rb.pack(anchor="w", pady=1)

    def _on_radio_change(self) -> None:
        """Radiobutton değişince callback'i tetikler."""
        if self._on_change:
            self._on_change(self._var.get())

    @property
    def selected(self) -> str:
        """Seçili algoritma adını döner."""
        return self._var.get()

    def set(self, value: str) -> None:
        """
        Seçimi programatik olarak değiştirir.

        Args:
            value: Yeni seçim ("Haar Cascade", "DNN (SSD+ResNet)", "Her İkisi").

        Raises:
            ValueError: Geçersiz seçim.
        """
        if value not in ALGO_OPTIONS:
            raise ValueError(f"Geçersiz algoritma: {value}. Seçenekler: {ALGO_OPTIONS}")
        self._var.set(value)