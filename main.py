"""
main.py
───────
Yüz Tespit Sistemi — Giriş Noktası

GUI modu (varsayılan):
  python main.py

CLI / Benchmark modu:
  python main.py --cli --mode image   --source test.jpg
  python main.py --cli --mode video   --source video.mp4
  python main.py --cli --mode benchmark --source images/

CLI parametreleri:
  --algo      : haar | dnn | both  (varsayılan: both)
  --output    : Çıktı klasörü      (varsayılan: results/)
  --max_frames: Video modu max frame sayısı
  --dnn_conf  : DNN güven eşiği    (varsayılan: 0.5)
  --haar_scale: Haar ölçek faktörü (varsayılan: 1.1)
  --haar_neigh: Haar min komşu     (varsayılan: 5)
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import cv2
import numpy as np

# Proje kökünü path'e ekle
sys.path.insert(0, os.path.dirname(__file__))


# ──────────────────────────────────────────────
# GUI Modu
# ──────────────────────────────────────────────

def run_gui() -> None:
    """Tkinter tabanlı GUI'yi başlatır."""
    from gui.app import AppWindow
    app = AppWindow()
    app.mainloop()


# ──────────────────────────────────────────────
# CLI Modu
# ──────────────────────────────────────────────

def run_cli(args: argparse.Namespace) -> None:
    """
    GUI olmadan komut satırından çalışır.

    Args:
        args: Argparse namespace nesnesi.
    """
    from core.haar_detector       import HaarDetector
    from core.dnn_detector        import DNNDetector
    from core.performance_profiler import PerformanceProfiler
    from core.detector_base       import DetectorBase
    from utils.export_utils       import export_summary_txt

    # Dedektörleri oluştur
    detectors = []
    if args.algo in ("haar", "both"):
        detectors.append(HaarDetector(
            scale_factor=args.haar_scale,
            min_neighbors=args.haar_neigh,
        ))
    if args.algo in ("dnn", "both"):
        detectors.append(DNNDetector(
            confidence_thr=args.dnn_conf,
            auto_download=True,
        ))

    profiler = PerformanceProfiler()
    os.makedirs(args.output, exist_ok=True)

    def process_frame(frame: np.ndarray, idx: int) -> np.ndarray:
        output = frame.copy()
        for det in detectors:
            result = det.detect(frame, idx)
            profiler.record(result)
            color = (0, 220, 80) if "Haar" in result.algorithm_name else (0, 140, 255)
            output = det.draw_results(output, result, color=color)
        return output

    # ── Mod seçimi ──────────────────────────────
    if args.mode == "image":
        frame = cv2.imread(args.source)
        if frame is None:
            print(f"[HATA] Görüntü okunamadı: {args.source}")
            sys.exit(1)
        output = process_frame(frame, 0)
        out_path = os.path.join(args.output, "result.jpg")
        cv2.imwrite(out_path, output)
        print(f"Sonuç: {out_path}")

    elif args.mode == "video":
        cap = cv2.VideoCapture(args.source)
        if not cap.isOpened():
            print(f"[HATA] Video açılamadı: {args.source}")
            sys.exit(1)
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret or (args.max_frames and idx >= args.max_frames):
                break
            process_frame(frame, idx)
            idx += 1
            if idx % 30 == 0:
                print(f"  İşlendi: {idx} frame")
        cap.release()
        print(f"Toplam frame: {idx}")

    elif args.mode == "benchmark":
        paths = []
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
            paths.extend(glob.glob(os.path.join(args.source, ext)))
        paths.sort()
        print(f"Benchmark: {len(paths)} görüntü")
        for i, p in enumerate(paths):
            frame = cv2.imread(p)
            if frame is None:
                continue
            process_frame(frame, i)
        print("Tamamlandı.")

    # Rapor ve grafik
    stats = profiler.get_statistics()
    _print_summary(stats)
    try:
        profiler.plot_comparison(os.path.join(args.output, "comparison.png"))
        profiler.plot_fps_series(os.path.join(args.output, "fps_series.png"))
        export_summary_txt(stats, os.path.join(args.output, "summary.txt"))
        profiler.export_csv(os.path.join(args.output, "raw_metrics.csv"))
    except Exception as e:
        print(f"[Uyarı] Rapor oluşturulamadı: {e}")


def _print_summary(stats: dict) -> None:
    """Terminal özet tablosunu yazdırır."""
    print("\n" + "=" * 55)
    print("  PERFORMANS ÖZET")
    print("=" * 55)
    for name, s in stats.items():
        d = s.to_dict()
        print(f"\n  [{name}]")
        for k, v in d.items():
            if v is not None:
                print(f"    {k:<20}: {v}")
    print("=" * 55 + "\n")


# ──────────────────────────────────────────────
# Argparse
# ──────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Yüz Tespit Sistemi",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--cli",        action="store_true",   help="CLI modunda çalış (GUI açma).")
    p.add_argument("--mode",       default="image",
                   choices=["image", "video", "benchmark"],
                   help="CLI çalışma modu.")
    p.add_argument("--source",     default=None,          help="Kaynak dosya veya klasör.")
    p.add_argument("--output",     default="results",     help="Çıktı klasörü.")
    p.add_argument("--algo",       default="both",
                   choices=["haar", "dnn", "both"],       help="Kullanılacak algoritma.")
    p.add_argument("--max_frames", type=int, default=None,help="Maksimum frame (video).")
    p.add_argument("--dnn_conf",   type=float, default=0.5)
    p.add_argument("--haar_scale", type=float, default=1.1)
    p.add_argument("--haar_neigh", type=int,   default=5)
    return p


# ──────────────────────────────────────────────
# Giriş Noktası
# ──────────────────────────────────────────────

if __name__ == "__main__":
    args = _build_parser().parse_args()

    if args.cli:
        if not args.source:
            print("[HATA] CLI modunda --source gereklidir.")
            sys.exit(1)
        run_cli(args)
    else:
        run_gui()