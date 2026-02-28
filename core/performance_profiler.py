"""
core/performance_profiler.py
────────────────────────────
Performans ölçüm motoru.

Hesaplanan metrikler:
  • FPS (kayan pencere + ortalama)
  • Çıkarım süresi: ort., min, max, std sapma
  • Precision / Recall / F1  (IoU tabanlı, GT gerektirir)

Görselleştirme:
  • Karşılaştırma bar chart (5 panel)
  • Kayan pencere FPS zaman serisi
"""

from __future__ import annotations

import csv
import os
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from core.result_model import DetectionResult


# ──────────────────────────────────────────────
# Veri Sınıfları
# ──────────────────────────────────────────────

@dataclass
class FrameRecord:
    """Tek frame'e ait ham profil kaydı."""
    algorithm: str
    frame_index: int
    inference_time_ms: float
    num_detections: int
    timestamp: float = field(default_factory=time.time)


@dataclass
class AlgorithmStats:
    """Bir algoritmanın tüm oturumu için istatistik özeti."""
    algorithm: str
    num_frames: int
    avg_time_ms: float
    min_time_ms: float
    max_time_ms: float
    std_time_ms: float
    avg_fps: float
    avg_detections: float
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None

    def to_dict(self) -> dict:
        return {k: (round(v, 4) if isinstance(v, float) else v)
                for k, v in self.__dict__.items()}


# ──────────────────────────────────────────────
# IoU Yardımcısı
# ──────────────────────────────────────────────

def compute_iou(
    a: Tuple[int, int, int, int],
    b: Tuple[int, int, int, int],
) -> float:
    """
    (x, y, w, h) formatındaki iki kutu arasında IoU hesaplar.

    Returns:
        0.0–1.0 arası IoU.
    """
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


# ──────────────────────────────────────────────
# Ana Profil Sınıfı
# ──────────────────────────────────────────────

class PerformanceProfiler:
    """
    Tüm dedektörler için merkezi performans takip motoru.

    Kullanım:
        profiler = PerformanceProfiler()
        profiler.record(result)
        stats = profiler.get_statistics()
        profiler.plot_comparison("results/chart.png")
    """

    def __init__(self, fps_window: int = 30) -> None:
        """
        Args:
            fps_window: Kayan FPS penceresi (frame sayısı).
        """
        self.fps_window = fps_window
        self._records: Dict[str, List[FrameRecord]] = {}
        self._fps_ts: Dict[str, deque] = {}           # zaman damgaları
        self._gt: Dict[str, Dict[str, int]] = {}      # TP/FP/FN sayaçları

    # ── Kayıt ────────────────────────────────────

    def record(self, result: DetectionResult) -> None:
        """
        DetectionResult'ı profilleyiciye kaydeder.

        Args:
            result: Kaydedilecek tespit sonucu.
        """
        name = result.algorithm_name
        self._records.setdefault(name, []).append(
            FrameRecord(
                algorithm=name,
                frame_index=result.frame_index,
                inference_time_ms=result.inference_time_ms,
                num_detections=result.num_detections,
            )
        )
        ts = self._fps_ts.setdefault(name, deque(maxlen=self.fps_window))
        ts.append(time.perf_counter())

    def record_gt_match(
        self,
        algorithm: str,
        predicted: List[Tuple[int, int, int, int]],
        ground_truth: List[Tuple[int, int, int, int]],
        iou_thr: float = 0.5,
    ) -> None:
        """
        Prediksiyon ve GT kutularını IoU ile eşleştirerek TP/FP/FN sayar.

        Args:
            algorithm   : Algoritma adı.
            predicted   : Model çıktısı bounding box'lar.
            ground_truth: Gerçek yüz kutular.
            iou_thr     : Eşleşme kabul eşiği.
        """
        cnt = self._gt.setdefault(algorithm, {"tp": 0, "fp": 0, "fn": 0})
        matched_gt: set = set()

        for pred in predicted:
            best_iou, best_gi = 0.0, -1
            for gi, gt in enumerate(ground_truth):
                if gi in matched_gt:
                    continue
                iou = compute_iou(pred, gt)
                if iou > best_iou:
                    best_iou, best_gi = iou, gi
            if best_iou >= iou_thr and best_gi >= 0:
                matched_gt.add(best_gi)
                cnt["tp"] += 1
            else:
                cnt["fp"] += 1

        cnt["fn"] += len(ground_truth) - len(matched_gt)

    # ── Anlık FPS ────────────────────────────────

    def get_live_fps(self, algorithm: str) -> float:
        """
        Kayan penceredeki frame'lere göre anlık FPS hesaplar.

        Args:
            algorithm: Algoritma adı.

        Returns:
            FPS değeri; yeterli kayıt yoksa 0.0.
        """
        ts = self._fps_ts.get(algorithm)
        if not ts or len(ts) < 2:
            return 0.0
        elapsed = ts[-1] - ts[0]
        return (len(ts) - 1) / elapsed if elapsed > 0 else 0.0

    # ── İstatistikler ────────────────────────────

    def get_statistics(self) -> Dict[str, AlgorithmStats]:
        """
        Tüm algoritmalar için AlgorithmStats sözlüğü döner.

        Returns:
            {algorithm_name: AlgorithmStats}
        """
        result: Dict[str, AlgorithmStats] = {}
        for name, records in self._records.items():
            times = np.array([r.inference_time_ms for r in records])
            dets  = np.array([r.num_detections    for r in records])
            avg_ms = float(np.mean(times))

            p, r, f1 = self._precision_recall(name)
            result[name] = AlgorithmStats(
                algorithm=name,
                num_frames=len(records),
                avg_time_ms=avg_ms,
                min_time_ms=float(np.min(times)),
                max_time_ms=float(np.max(times)),
                std_time_ms=float(np.std(times)),
                avg_fps=1000.0 / avg_ms if avg_ms > 0 else 0.0,
                avg_detections=float(np.mean(dets)),
                precision=p, recall=r, f1_score=f1,
            )
        return result

    def get_time_series(self, algorithm: str) -> Tuple[List[int], List[float]]:
        """Frame bazlı çıkarım süresini döner."""
        recs = self._records.get(algorithm, [])
        return [r.frame_index for r in recs], [r.inference_time_ms for r in recs]

    def _precision_recall(
        self, algorithm: str
    ) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """Birikmiş sayaçlardan P/R/F1 hesaplar."""
        cnt = self._gt.get(algorithm)
        if not cnt:
            return None, None, None
        tp, fp, fn = cnt["tp"], cnt["fp"], cnt["fn"]
        p  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        return p, r, f1

    # ── Dışa Aktarma ─────────────────────────────

    def export_csv(self, path: str) -> None:
        """Ham kayıtları CSV'ye yazar."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        fields = ["algorithm", "frame_index", "inference_time_ms",
                  "num_detections", "timestamp"]
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for recs in self._records.values():
                for r in recs:
                    w.writerow({
                        "algorithm": r.algorithm,
                        "frame_index": r.frame_index,
                        "inference_time_ms": round(r.inference_time_ms, 4),
                        "num_detections": r.num_detections,
                        "timestamp": round(r.timestamp, 6),
                    })
        print(f"[Profiler] CSV → {path}")

    # ── Görselleştirme ───────────────────────────

    def plot_comparison(
        self,
        save_path: str = "results/comparison.png",
        show: bool = False,
    ) -> str:
        """
        Çok panelli karşılaştırma grafiği üretir (PNG).

        Paneller: Süre | FPS | Tespit | Zaman Serisi | (P/R/F1)

        Returns:
            Kaydedilen dosyanın mutlak yolu.
        """
        stats = self.get_statistics()
        if not stats:
            raise RuntimeError("Grafik için kayıt yok.")

        algos  = list(stats.keys())
        colors = plt.cm.Set2(np.linspace(0, 0.8, len(algos)))
        has_gt = any(s.precision is not None for s in stats.values())
        n_col  = 5 if has_gt else 4

        fig, axes = plt.subplots(1, n_col, figsize=(5 * n_col, 5))
        fig.suptitle("Yüz Tespit — Performans Karşılaştırması",
                     fontsize=13, fontweight="bold", y=1.02)

        def bar(ax, vals, title, ylabel, fmt):
            bars = ax.bar(algos, vals, color=colors, edgecolor="white")
            ax.set_title(title, fontsize=10)
            ax.set_ylabel(ylabel)
            ax.set_ylim(0, max(vals) * 1.3 if max(vals) > 0 else 1)
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, v + ax.get_ylim()[1] * 0.01,
                        fmt.format(v), ha="center", va="bottom", fontsize=9)
            ax.grid(axis="y", alpha=0.4)

        bar(axes[0], [stats[a].avg_time_ms    for a in algos], "Ort. Süre (ms)", "ms",    "{:.1f}")
        bar(axes[1], [stats[a].avg_fps         for a in algos], "Ort. FPS",       "fps",   "{:.1f}")
        bar(axes[2], [stats[a].avg_detections  for a in algos], "Ort. Tespit",    "adet",  "{:.2f}")

        ax = axes[3]
        for alg, col in zip(algos, colors):
            idx, times = self.get_time_series(alg)
            if idx:
                ax.plot(idx, times, label=alg, color=col, linewidth=1.6, alpha=0.85)
        ax.set_title("Süre / Frame", fontsize=10)
        ax.set_xlabel("Frame")
        ax.set_ylabel("ms")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.4)

        if has_gt:
            ax = axes[4]
            x = np.arange(len(algos))
            w = 0.25
            ax.bar(x - w, [stats[a].precision or 0 for a in algos], w, label="Precision", color="#4FC3F7")
            ax.bar(x,     [stats[a].recall    or 0 for a in algos], w, label="Recall",    color="#AED581")
            ax.bar(x + w, [stats[a].f1_score  or 0 for a in algos], w, label="F1",        color="#FFB74D")
            ax.set_title("P / R / F1", fontsize=10)
            ax.set_xticks(x); ax.set_xticklabels(algos)
            ax.set_ylim(0, 1.2)
            ax.legend(fontsize=8)
            ax.grid(axis="y", alpha=0.4)

        plt.tight_layout()
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)
        print(f"[Profiler] Grafik → {save_path}")
        return os.path.abspath(save_path)

    def plot_fps_series(
        self,
        save_path: str = "results/fps_series.png",
        window: int = 10,
        show: bool = False,
    ) -> str:
        """Kayan FPS zaman serisini çizer ve kaydeder."""
        fig, ax = plt.subplots(figsize=(10, 4))
        colors = plt.cm.Set2(np.linspace(0, 0.8, len(self._records)))

        for (alg, recs), col in zip(self._records.items(), colors):
            times = [r.inference_time_ms for r in recs]
            fps_s = []
            for i in range(len(times)):
                chunk = times[max(0, i - window + 1): i + 1]
                avg = sum(chunk) / len(chunk)
                fps_s.append(1000.0 / avg if avg > 0 else 0)
            ax.plot(fps_s, label=alg, color=col, linewidth=1.8)

        ax.set_title(f"Kayan Pencere FPS (window={window})")
        ax.set_xlabel("Frame")
        ax.set_ylabel("FPS")
        ax.legend()
        ax.grid(alpha=0.4)
        plt.tight_layout()

        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)
        print(f"[Profiler] FPS serisi → {save_path}")
        return os.path.abspath(save_path)

    def reset(self) -> None:
        """Tüm kayıtları ve sayaçları sıfırlar."""
        self._records.clear()
        self._fps_ts.clear()
        self._gt.clear()

    def __repr__(self) -> str:
        total = sum(len(v) for v in self._records.values())
        return f"PerformanceProfiler(algos={list(self._records)}, records={total})"