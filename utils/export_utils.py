"""
utils/export_utils.py
─────────────────────
Tespit sonuçları ve metriklerin CSV / PNG / JSON formatında dışa aktarılması.
"""

from __future__ import annotations

import csv
import json
import os
from typing import Dict, List

import cv2
import numpy as np

from core.result_model import DetectionResult


def export_results_csv(
    results: List[DetectionResult],
    path: str,
) -> str:
    """
    DetectionResult listesini CSV dosyasına yazar.

    Args:
        results: Dışa aktarılacak sonuç listesi.
        path   : Hedef .csv dosyası yolu.

    Returns:
        Kaydedilen dosyanın mutlak yolu.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fields = [
        "algorithm", "frame_index", "num_detections",
        "avg_confidence", "inference_time_ms", "timestamp",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in results:
            writer.writerow({
                "algorithm"       : r.algorithm_name,
                "frame_index"     : r.frame_index,
                "num_detections"  : r.num_detections,
                "avg_confidence"  : round(r.avg_confidence, 4),
                "inference_time_ms": round(r.inference_time_ms, 4),
                "timestamp"       : round(r.timestamp, 6),
            })
    print(f"[export] CSV → {path}")
    return os.path.abspath(path)


def export_results_json(
    results: List[DetectionResult],
    path: str,
) -> str:
    """
    DetectionResult listesini JSON dosyasına yazar.

    Args:
        results: Dışa aktarılacak sonuç listesi.
        path   : Hedef .json dosyası yolu.

    Returns:
        Kaydedilen dosyanın mutlak yolu.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    data = [r.to_dict() for r in results]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"[export] JSON → {path}")
    return os.path.abspath(path)


def export_annotated_image(
    frame_bgr: np.ndarray,
    path: str,
) -> bool:
    """
    Açıklamalar (bounding box) çizilmiş frame'i PNG/JPG olarak kaydeder.

    Args:
        frame_bgr: Kaydedilecek BGR görüntü.
        path     : Hedef dosya yolu.

    Returns:
        True = başarılı, False = hata.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    ok = cv2.imwrite(path, frame_bgr)
    if ok:
        print(f"[export] Görüntü → {path}")
    else:
        print(f"[export] HATA: görüntü kaydedilemedi → {path}")
    return bool(ok)


def export_summary_txt(
    stats: Dict,
    path: str,
) -> str:
    """
    AlgorithmStats sözlüğünü okunabilir metin raporuna yazar.

    Args:
        stats: {algo_name: AlgorithmStats} sözlüğü.
        path : Hedef .txt dosyası yolu.

    Returns:
        Kaydedilen dosyanın mutlak yolu.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    lines = ["=" * 55, "  YÜZ TESPİT SİSTEMİ — PERFORMANS RAPORU", "=" * 55, ""]

    for name, s in stats.items():
        d = s.to_dict()
        lines += [
            f"  [{name}]",
            f"    Frame Sayısı    : {d['num_frames']}",
            f"    Ort. Süre (ms)  : {d['avg_time_ms']}",
            f"    Min / Max (ms)  : {d['min_time_ms']} / {d['max_time_ms']}",
            f"    Std Sapma (ms)  : {d['std_time_ms']}",
            f"    Ort. FPS        : {d['avg_fps']}",
            f"    Ort. Tespit     : {d['avg_detections']}",
        ]
        if d.get("precision") is not None:
            lines += [
                f"    Precision       : {d['precision']}",
                f"    Recall          : {d['recall']}",
                f"    F1-Score        : {d['f1_score']}",
            ]
        lines.append("")

    lines.append("=" * 55)
    text = "\n".join(lines)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"[export] Rapor → {path}")
    return os.path.abspath(path)