"""
gui/panels/control_panel.py
Sol kontrol paneli - kaynak secimi, algoritma, parametreler, butonlar.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk, filedialog
from typing import Callable, Optional

# Renkler
BG     = "#2a2a3e"
FG     = "#e2e8f0"
DIM    = "#94a3b8"
ACCENT = "#7c3aed"
DANGER = "#ef4444"
GREEN  = "#22c55e"
BORDER = "#3f3f5a"


class ControlPanel(ttk.Frame):
    """Sol kontrol paneli."""

    WIDTH = 250

    def __init__(self, parent, callbacks: dict, **kw):
        """
        Args:
            callbacks: {
                'load_image': fn(path),
                'load_video': fn(path),
                'webcam'    : fn(),
                'start'     : fn(),
                'stop'      : fn(),
                'reset'     : fn(),
                'chart'     : fn(),
                'save'      : fn(),
            }
        """
        super().__init__(parent, style="Panel.TFrame", padding=10, **kw)
        self.configure(width=self.WIDTH)
        self.pack_propagate(False)

        self._cb = callbacks

        # Parametre degiskenleri
        self._haar_scale  = tk.DoubleVar(value=1.1)
        self._haar_neigh  = tk.IntVar(value=5)
        self._dnn_conf    = tk.DoubleVar(value=0.5)
        self._algo_var    = tk.StringVar(value="Her Ikisi")
        self._src_var     = tk.StringVar(value="Kaynak secilmedi")

        self._build()

    # ------------------------------------------------------------------
    # INSAAT
    # ------------------------------------------------------------------

    def _build(self):
        self._sec_kaynak()
        self._sep()
        self._sec_algo()
        self._sep()
        self._sec_params()
        self._sep()
        self._sec_butonlar()
        self._sep()
        self._sec_analiz()

    def _sep(self):
        ttk.Separator(self, orient="horizontal").pack(fill="x", pady=5)

    def _sec_kaynak(self):
        lf = ttk.LabelFrame(self, text="Kaynak", padding=6)
        lf.pack(fill="x", pady=(0, 2))

        ttk.Button(lf, text="Fotograf Yukle", style="Secondary.TButton",
                   command=self._pick_image).pack(fill="x", pady=2)

        ttk.Button(lf, text="Video Yukle", style="Secondary.TButton",
                   command=self._pick_video).pack(fill="x", pady=2)

        ttk.Button(lf, text="Webcam Ac", style="Secondary.TButton",
                   command=lambda: self._cb.get("webcam") and self._cb["webcam"]()
                   ).pack(fill="x", pady=2)

        tk.Label(lf, textvariable=self._src_var,
                 bg=BG, fg=DIM, font=("Segoe UI", 8),
                 wraplength=210, justify="left").pack(fill="x", pady=(4, 0))

    def _sec_algo(self):
        lf = ttk.LabelFrame(self, text="Algoritma", padding=6)
        lf.pack(fill="x", pady=(0, 2))

        for val in ["Haar Cascade", "DNN (SSD+ResNet)", "Her Ikisi"]:
            tk.Radiobutton(
                lf, text=val,
                variable=self._algo_var, value=val,
                bg=BG, fg=FG,
                activebackground=BG, activeforeground=ACCENT,
                selectcolor="#1e1e2e",
                font=("Segoe UI", 10),
            ).pack(anchor="w", pady=1)

    def _sec_params(self):
        lf = ttk.LabelFrame(self, text="Parametreler", padding=6)
        lf.pack(fill="x", pady=(0, 2))

        self._make_slider(lf, "Haar Scale Factor",
                          self._haar_scale, 1.05, 1.5, "{:.2f}")
        self._make_slider(lf, "Haar Min Neighbours",
                          self._haar_neigh, 1, 15, "{:.0f}")
        self._make_slider(lf, "DNN Guven Esigi",
                          self._dnn_conf, 0.1, 0.99, "{:.2f}")

    def _make_slider(self, parent, label, var, frm, to, fmt):
        tk.Label(parent, text=label, bg=BG, fg=DIM,
                 font=("Segoe UI", 8)).pack(anchor="w", pady=(4, 0))

        disp = tk.StringVar(value=fmt.format(var.get()))
        row  = tk.Frame(parent, bg=BG)
        row.pack(fill="x")

        def on_move(v):
            disp.set(fmt.format(float(v)))

        ttk.Scale(row, from_=frm, to=to, orient="horizontal",
                  variable=var, command=on_move).pack(side="left", fill="x", expand=True)

        tk.Label(row, textvariable=disp, bg=BG, fg="#06b6d4",
                 font=("Consolas", 9), width=6).pack(side="left")

    def _sec_butonlar(self):
        lf = tk.Frame(self, bg=BG)
        lf.pack(fill="x", pady=(0, 2))

        self._btn_start = ttk.Button(lf, text="▶  Baslat",
                                     style="Accent.TButton",
                                     command=lambda: self._cb.get("start") and self._cb["start"]())
        self._btn_start.pack(fill="x", pady=2)

        self._btn_stop = ttk.Button(lf, text="⏹  Durdur",
                                    style="Danger.TButton", state="disabled",
                                    command=lambda: self._cb.get("stop") and self._cb["stop"]())
        self._btn_stop.pack(fill="x", pady=2)

        ttk.Button(lf, text="↺  Sifirla",
                   style="Secondary.TButton",
                   command=lambda: self._cb.get("reset") and self._cb["reset"]()
                   ).pack(fill="x", pady=2)

    def _sec_analiz(self):
        lf = ttk.LabelFrame(self, text="Analiz & Disa Aktar", padding=6)
        lf.pack(fill="x", pady=(0, 2))

        ttk.Button(lf, text="Karsilastirma Grafigi",
                   style="Accent.TButton",
                   command=lambda: self._cb.get("chart") and self._cb["chart"]()
                   ).pack(fill="x", pady=3)

        ttk.Button(lf, text="Sonucu Kaydet",
                   style="Secondary.TButton",
                   command=lambda: self._cb.get("save") and self._cb["save"]()
                   ).pack(fill="x", pady=2)

    # ------------------------------------------------------------------
    # DOSYA SECICILER
    # ------------------------------------------------------------------

    def _pick_image(self):
        path = filedialog.askopenfilename(
            title="Fotograf Sec",
            filetypes=[("Goruntu", "*.jpg *.jpeg *.png *.bmp *.webp *.tiff"),
                       ("Tumu", "*.*")])
        if not path:
            return
        name = path.replace("\\", "/").split("/")[-1]
        self._src_var.set(name)
        fn = self._cb.get("load_image")
        if fn:
            fn(path)

    def _pick_video(self):
        path = filedialog.askopenfilename(
            title="Video Sec",
            filetypes=[("Video", "*.mp4 *.avi *.mov *.mkv *.webm"),
                       ("Tumu", "*.*")])
        if not path:
            return
        name = path.replace("\\", "/").split("/")[-1]
        self._src_var.set(name)
        fn = self._cb.get("load_video")
        if fn:
            fn(path)

    # ------------------------------------------------------------------
    # DURUM
    # ------------------------------------------------------------------

    def set_running(self, running: bool):
        self._btn_start.config(state="disabled" if running else "normal")
        self._btn_stop.config(state="normal"   if running else "disabled")

    def set_src_label(self, text: str):
        self._src_var.set(text)

    # ------------------------------------------------------------------
    # PARAMETRELER
    # ------------------------------------------------------------------

    @property
    def algorithm(self) -> str:
        return self._algo_var.get()

    @property
    def haar_scale(self) -> float:
        return self._haar_scale.get()

    @property
    def haar_neighbors(self) -> int:
        return int(self._haar_neigh.get())

    @property
    def dnn_confidence(self) -> float:
        return self._dnn_conf.get()