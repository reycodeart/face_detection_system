"""
gui/app.py
Ana uygulama penceresi.
"""

from __future__ import annotations

import os
import queue
import threading
import time
from typing import List, Optional

import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox

from core.haar_detector        import HaarDetector
from core.dnn_detector         import DNNDetector
from core.performance_profiler import PerformanceProfiler
from core.result_model         import DetectionResult
from utils.image_utils         import save_image
from utils.export_utils        import export_summary_txt

from gui.panels.control_panel  import ControlPanel
from gui.panels.canvas_panel   import CanvasPanel
from gui.panels.metrics_panel  import MetricsPanel

# ── Renkler ───────────────────────────────────────────────────────────────────
BG      = "#1e1e2e"
PANEL   = "#2a2a3e"
ACCENT  = "#7c3aed"
CYAN    = "#06b6d4"
GREEN   = "#22c55e"
AMBER   = "#f59e0b"
RED     = "#ef4444"
TEXT    = "#e2e8f0"
DIM     = "#94a3b8"
BORDER  = "#3f3f5a"

HAAR_CLR = (0, 220, 80)    # BGR yesil
DNN_CLR  = (0, 140, 255)   # BGR mavi
POLL_MS  = 16              # ~60 fps guncelleme


def _theme(root):
    s = ttk.Style(root)
    s.theme_use("clam")
    s.configure(".",               background=BG,    foreground=TEXT,
                                   font=("Segoe UI", 10), borderwidth=0)
    s.configure("TFrame",          background=BG)
    s.configure("Panel.TFrame",    background=PANEL)
    s.configure("TLabel",          background=BG,    foreground=TEXT)
    s.configure("Status.TLabel",   background=BG,    foreground=DIM,
                                   font=("Segoe UI", 9, "italic"))
    s.configure("TLabelframe",     background=PANEL, foreground=DIM,
                                   bordercolor=BORDER, relief="flat")
    s.configure("TLabelframe.Label", background=PANEL, foreground=DIM,
                                   font=("Segoe UI", 9, "bold"))
    s.configure("TSeparator",      background=BORDER)
    s.configure("TScale",          background=PANEL, troughcolor=BORDER,
                                   sliderlength=14)

    for name, bg, hov in [
        ("Accent.TButton",    ACCENT, "#6d28d9"),
        ("Secondary.TButton", PANEL,  BORDER),
        ("Danger.TButton",    RED,    "#b91c1c"),
    ]:
        bold = "bold" if name != "Secondary.TButton" else "normal"
        s.configure(name, background=bg, foreground=TEXT,
                    font=("Segoe UI", 10, bold),
                    padding=(10, 6), relief="flat")
        s.map(name,
              background=[("active", hov), ("pressed", hov)],
              relief=[("pressed", "flat")])
    return s


# ─────────────────────────────────────────────────────────────────────────────
class AppWindow(tk.Tk):
    """Ana uygulama penceresi."""

    TITLE = "Yuz Tespit Sistemi  |  Haar vs DNN"

    def __init__(self):
        super().__init__()
        self.title(self.TITLE)
        self.configure(bg=BG)
        self.minsize(1100, 680)
        self.resizable(True, True)

        self.update_idletasks()
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        self.geometry(f"1260x760+{(sw-1260)//2}+{(sh-760)//2}")

        _theme(self)

        # ── Uygulama durumu ────────────────────────────────────────────
        self._mode         = "idle"   # idle | image | video | webcam
        self._image_frame  = None     # yuklenmiş ham fotograf
        self._image_path   = None
        self._video_path   = None
        self._last_output  = None     # son islenmis frame (kaydet icin)

        self._stop_evt     = threading.Event()
        self._fq: queue.Queue = queue.Queue(maxsize=2)
        self._worker       = None

        self._profiler     = PerformanceProfiler(fps_window=30)
        self._fidx         = 0        # frame sayaci
        self._results: List[DetectionResult] = []

        # ── GUI ────────────────────────────────────────────────────────
        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self._status("Hazir — Sol panelden kaynak secin.")

    # ──────────────────────────────────────────────────────────────────
    # UI INSAATI
    # ──────────────────────────────────────────────────────────────────

    def _build_ui(self):
        # Baslik
        hdr = tk.Frame(self, bg=PANEL, pady=8)
        hdr.pack(fill="x")
        tk.Label(hdr, text="  Yuz Tespit Sistemi",
                 bg=PANEL, fg=TEXT, font=("Segoe UI", 14, "bold")).pack(side="left", padx=12)
        tk.Label(hdr, text="Haar Cascade  vs  DNN (SSD+ResNet-10)",
                 bg=PANEL, fg=DIM, font=("Segoe UI", 9)).pack(side="left")

        # Ana alan
        body = ttk.Frame(self)
        body.pack(fill="both", expand=True, padx=8, pady=8)

        # Sol panel — callback sozlugu ile
        self._ctrl = ControlPanel(body, callbacks={
            "load_image": self._load_image,
            "load_video": self._load_video,
            "webcam":     self._set_webcam,
            "start":      self._start,
            "stop":       self._stop,
            "reset":      self._reset,
            "chart":      self._show_chart,
            "save":       self._save,
        })
        self._ctrl.pack(side="left", fill="y", padx=(0, 8))

        # Sag: canvas + metrik
        right = ttk.Frame(body)
        right.pack(side="left", fill="both", expand=True)

        self._canvas = CanvasPanel(right)
        self._canvas.pack(fill="both", expand=True)

        self._metrics = MetricsPanel(right)
        self._metrics.pack(fill="x", pady=(6, 0))

        # Durum cubugu
        sb = tk.Frame(self, bg=PANEL, pady=4)
        sb.pack(fill="x", side="bottom")
        self._status_var = tk.StringVar()
        tk.Label(sb, textvariable=self._status_var, bg=PANEL, fg=DIM,
                 font=("Segoe UI", 9, "italic")).pack(side="left", padx=10)
        self._mode_var = tk.StringVar(value="● HAZIR")
        self._mode_lbl = tk.Label(sb, textvariable=self._mode_var,
                                  bg=PANEL, fg=GREEN,
                                  font=("Segoe UI", 9, "bold"))
        self._mode_lbl.pack(side="right", padx=10)

    # ──────────────────────────────────────────────────────────────────
    # DEDEKTORLER
    # ──────────────────────────────────────────────────────────────────

    def _make_detectors(self) -> list:
        algo = self._ctrl.algorithm
        out  = []

        if algo in ("Haar Cascade", "Her Ikisi"):
            try:
                out.append(HaarDetector(
                    scale_factor=self._ctrl.haar_scale,
                    min_neighbors=self._ctrl.haar_neighbors))
            except Exception as e:
                print(f"[App] Haar yuklenemedi: {e}")

        if algo in ("DNN (SSD+ResNet)", "Her Ikisi"):
            try:
                out.append(DNNDetector(
                    confidence_thr=self._ctrl.dnn_confidence,
                    auto_download=True))
            except Exception as e:
                print(f"[App] DNN yuklenemedi: {e}")

        return out

    def _detect(self, frame: np.ndarray, dets: list) -> np.ndarray:
        """Frame uzerinde tespit yap, sonucu ciz, profiler'a kaydet."""
        out = frame.copy()
        for d in dets:
            try:
                res = d.detect(frame, self._fidx)
                self._profiler.record(res)
                self._results.append(res)
                clr = HAAR_CLR if "Haar" in res.algorithm_name else DNN_CLR
                out = d.draw_results(out, res, color=clr)
            except Exception as e:
                print(f"[App] {d.get_name()} hatasi: {e}")
        self._fidx += 1
        return out

    # ──────────────────────────────────────────────────────────────────
    # KAYNAK YUKLEME
    # ──────────────────────────────────────────────────────────────────

    def _load_image(self, path: str):
        """Fotograf yukle — onizleme goster, Baslat bekle."""
        self._stop_workers()
        self._mode        = "image"
        self._image_path  = path
        self._image_frame = cv2.imread(path)
        if self._image_frame is None:
            messagebox.showerror("Hata", f"Goruntu okunamadi:\n{path}")
            return
        self._canvas.render_frame(self._image_frame)
        self._mode_ind("● FOTOGRAF", CYAN)
        self._status(f"Fotograf yuklendi — Baslat'a basin  |  {path}")

    def _load_video(self, path: str):
        """Video yukle — Baslat bekle."""
        self._stop_workers()
        self._mode       = "video"
        self._video_path = path
        # Ilk kareyi onizleme olarak goster
        cap = cv2.VideoCapture(path)
        ok, frame = cap.read()
        cap.release()
        if ok:
            self._canvas.render_frame(frame)
        self._mode_ind("● VIDEO", AMBER)
        self._status(f"Video yuklendi — Baslat'a basin  |  {path}")

    def _set_webcam(self):
        """Webcam hazirla — Baslat bekle."""
        self._stop_workers()
        self._mode = "webcam"
        self._mode_ind("● WEBCAM HAZIR", GREEN)
        self._status("Webcam secildi — Baslat'a basin.")

    # ──────────────────────────────────────────────────────────────────
    # KONTROL BUTONLARI
    # ──────────────────────────────────────────────────────────────────

    def _start(self):
        """Baslat butonu — moda gore islemi baslat."""
        if self._mode == "image":
            if self._image_frame is None:
                self._status("Once fotograf yukleyin.")
                return
            self._profiler.reset()
            self._results.clear()
            self._fidx = 0
            dets   = self._make_detectors()
            output = self._detect(self._image_frame, dets)
            self._last_output = output
            self._canvas.render_frame(output)
            self._update_metrics()
            self._status("Fotograf analizi tamamlandi.")

        elif self._mode == "video":
            if not self._video_path:
                self._status("Once video yukleyin.")
                return
            self._stop_workers()
            self._profiler.reset()
            self._results.clear()
            self._fidx = 0
            self._ctrl.set_running(True)
            self._mode_ind("● VIDEO OYNUYOR", AMBER)
            self._status(f"Video isleniyor: {self._video_path}")
            self._stop_evt.clear()
            dets = self._make_detectors()
            self._worker = threading.Thread(
                target=self._run_video, args=(self._video_path, dets),
                daemon=True)
            self._worker.start()
            self._poll()

        elif self._mode == "webcam":
            self._stop_workers()
            self._profiler.reset()
            self._results.clear()
            self._fidx = 0
            self._ctrl.set_running(True)
            self._mode_ind("● WEBCAM", GREEN)
            self._status("Webcam akisi baslatildi.")
            self._stop_evt.clear()
            dets = self._make_detectors()
            self._worker = threading.Thread(
                target=self._run_webcam, args=(dets,),
                daemon=True)
            self._worker.start()
            self._poll()

        else:
            self._status("Sol panelden kaynak secin (Fotograf / Video / Webcam).")

    def _stop(self):
        self._stop_workers()
        self._mode_ind("● DURDURULDU", RED)
        self._status("Durduruldu.")

    def _reset(self):
        self._stop_workers()
        self._mode         = "idle"
        self._image_frame  = None
        self._image_path   = None
        self._video_path   = None
        self._last_output  = None
        self._profiler.reset()
        self._results.clear()
        self._fidx = 0
        self._canvas.show_placeholder()
        self._metrics.reset_all()
        self._ctrl.set_src_label("Kaynak secilmedi")
        self._mode_ind("● HAZIR", GREEN)
        self._status("Sifirlandi.")

    # ──────────────────────────────────────────────────────────────────
    # VIDEO WORKER
    # ──────────────────────────────────────────────────────────────────

    def _run_video(self, path: str, dets: list):
        """Gercek FPS hizinda video isle."""
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            self.after(0, lambda: messagebox.showerror(
                "Hata", f"Video acilamadi:\n{path}"))
            self._fq.put(None)
            return

        fps      = cap.get(cv2.CAP_PROP_FPS) or 25.0
        duration = 1.0 / fps
        total    = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"[Video] FPS={fps:.1f}  Toplam={total}")

        try:
            while not self._stop_evt.is_set():
                t0  = time.perf_counter()
                ok, frame = cap.read()
                if not ok:
                    break

                output = self._detect(frame, dets)

                # Kuyruga koy — dolu ise eski frame'i at (video modunda gecikme olmaz)
                if self._fq.full():
                    try:
                        self._fq.get_nowait()
                    except queue.Empty:
                        pass
                self._fq.put(output)

                # FPS kontrolu: kalanı bekle
                passed = time.perf_counter() - t0
                wait   = duration - passed
                if wait > 0:
                    time.sleep(wait)
        finally:
            cap.release()
            self._fq.put(None)

    # ──────────────────────────────────────────────────────────────────
    # WEBCAM WORKER
    # ──────────────────────────────────────────────────────────────────

    def _run_webcam(self, dets: list):
        """Webcam akisini isle."""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            self.after(0, lambda: messagebox.showerror(
                "Hata", "Webcam acilamadi."))
            self._fq.put(None)
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        try:
            while not self._stop_evt.is_set():
                ok, frame = cap.read()
                if not ok:
                    break
                output = self._detect(frame, dets)
                if self._fq.full():
                    try:
                        self._fq.get_nowait()
                    except queue.Empty:
                        pass
                self._fq.put(output)
        finally:
            cap.release()
            self._fq.put(None)

    # ──────────────────────────────────────────────────────────────────
    # FRAME QUEUE POLLING
    # ──────────────────────────────────────────────────────────────────

    def _poll(self):
        try:
            frame = self._fq.get_nowait()
        except queue.Empty:
            self.after(POLL_MS, self._poll)
            return

        if frame is None:
            # Worker bitti
            self._ctrl.set_running(False)
            self._mode_ind("● HAZIR", GREEN)
            self._status("Isleme tamamlandi.")
            return

        self._last_output = frame
        self._canvas.render_frame(frame)
        self._update_metrics()
        self.after(POLL_MS, self._poll)

    # ──────────────────────────────────────────────────────────────────
    # METRIKLER
    # ──────────────────────────────────────────────────────────────────

    def _update_metrics(self):
        stats = self._profiler.get_statistics()
        if not stats:
            return

        algos   = list(stats.keys())
        fps_now = [self._profiler.get_live_fps(a) for a in algos]
        avg_fps = sum(s.avg_fps     for s in stats.values()) / len(stats)
        lat     = sum(s.avg_time_ms for s in stats.values()) / len(stats)
        live    = sum(fps_now) / len(fps_now) if fps_now else 0

        faces = 0
        for a in algos:
            recs = self._profiler._records.get(a, [])
            if recs:
                faces += recs[-1].num_detections

        self._metrics.update_all(
            fps=live, faces=faces, latency_ms=lat,
            algo=" + ".join(algos),
            total_frames=self._fidx, avg_fps=avg_fps)

    # ──────────────────────────────────────────────────────────────────
    # GRAFIK & KAYDET
    # ──────────────────────────────────────────────────────────────────

    def _show_chart(self):
        stats = self._profiler.get_statistics()
        if not stats:
            messagebox.showinfo("Bilgi", "Henuz veri yok.\nOnce bir kaynak analiz edin.")
            return
        os.makedirs("results", exist_ok=True)
        try:
            path = self._profiler.plot_comparison("results/comparison.png")
            export_summary_txt(stats, "results/summary.txt")
        except Exception as e:
            messagebox.showerror("Grafik Hatasi", str(e))
            return
        self._show_image_popup(path, "Performans Karsilastirma Grafigi")

    def _save(self):
        if self._last_output is None:
            messagebox.showinfo("Bilgi", "Kaydedilecek sonuc yok.")
            return
        from tkinter.filedialog import asksaveasfilename
        p = asksaveasfilename(
            title="Kaydet",
            defaultextension=".jpg",
            filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png"), ("Tumu", "*.*")])
        if not p:
            return
        if save_image(self._last_output, p):
            messagebox.showinfo("Basarili", f"Kaydedildi:\n{p}")
        else:
            messagebox.showerror("Hata", "Kayit basarisiz.")

    def _show_image_popup(self, img_path: str, title: str):
        from PIL import Image, ImageTk
        win = tk.Toplevel(self)
        win.title(title)
        win.configure(bg=BG)
        try:
            img = Image.open(img_path)
            img.thumbnail((1400, 650), Image.LANCZOS)
            ph = ImageTk.PhotoImage(img)
            lbl = tk.Label(win, image=ph, bg=BG)
            lbl.image = ph
            lbl.pack(padx=10, pady=10)
        except Exception as e:
            tk.Label(win, text=str(e), bg=BG, fg=TEXT).pack(padx=20, pady=20)
        ttk.Button(win, text="Kapat", style="Secondary.TButton",
                   command=win.destroy).pack(pady=(0, 10))

    # ──────────────────────────────────────────────────────────────────
    # YARDIMCILAR
    # ──────────────────────────────────────────────────────────────────

    def _stop_workers(self):
        self._stop_evt.set()
        if self._worker and self._worker.is_alive():
            self._worker.join(timeout=2.0)
        self._stop_evt.clear()
        self._ctrl.set_running(False)

    def _status(self, txt: str):
        self._status_var.set(txt)

    def _mode_ind(self, txt: str, color: str):
        self._mode_var.set(txt)
        self._mode_lbl.config(fg=color)

    def _on_close(self):
        self._stop_evt.set()
        self.destroy()