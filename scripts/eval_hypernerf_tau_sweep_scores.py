#!/usr/bin/env python3
"""Score rendered HyperNeRF objectness long-range tau sweep outputs."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.eval_maskbenchmark_trase_style import SCENE_ORDER, SCENE_SPLITS, evaluate_scene, load_run_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", default="output/final_hypernerf_1/maskbenchmark_full10_res003_global_longrange_input.csv")
    parser.add_argument("--out-dir", default="output/hypernerf_tau_lr_sweep_eval")
    parser.add_argument("--data-root", default="data/HyperNeRF")
    parser.add_argument("--taus", nargs="*", default=["020", "030", "040", "050", "060", "070", "075", "085", "090"])
    args = parser.parse_args()

    source_rows = load_run_rows(ROOT / args.input_csv)
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    per_scene = []
    summary = []
    for tau in args.taus:
        tag = f"objectness_lr_tau{tau}"
        rows = []
        for scene in SCENE_ORDER:
            source = source_rows[scene]
            pred_dir = ROOT / source["run_dir"] / tag / "cluster_ids_train"
            if not pred_dir.exists() or not any(pred_dir.glob("*.png")):
                per_scene.append({"tau": tau, "scene": scene, "status": "missing", "trase_style_miou": "", "trase_style_macc": "", "best_k": "", "matched_frames": "", "K_seen": "", "pred_dir": str(pred_dir.relative_to(ROOT))})
                continue
            gt_dir = ROOT / args.data_root / SCENE_SPLITS[scene] / scene / "gt_masks"
            try:
                result = evaluate_scene(pred_dir, gt_dir)
            except Exception as exc:  # keep the sweep moving and preserve provenance
                per_scene.append({"tau": tau, "scene": scene, "status": f"error: {exc}", "trase_style_miou": "", "trase_style_macc": "", "best_k": "", "matched_frames": "", "K_seen": "", "pred_dir": str(pred_dir.relative_to(ROOT))})
                continue
            row = {"tau": tau, "scene": scene, "status": "done", "pred_dir": str(pred_dir.relative_to(ROOT)), **result}
            rows.append(row)
            per_scene.append(row)
            print(f"tau={tau} {scene}: mIoU={result['trase_style_miou']:.4f} mAcc={result['trase_style_macc']:.4f}")
        if rows:
            summary.append(
                {
                    "tau": tau,
                    "scenes_done": len(rows),
                    "mean_miou": float(np.mean([float(row["trase_style_miou"]) for row in rows])),
                    "mean_macc": float(np.mean([float(row["trase_style_macc"]) for row in rows])),
                }
            )

    with (out_dir / "tau_sweep_per_scene.csv").open("w", newline="") as f:
        fieldnames = ["tau", "scene", "status", "trase_style_miou", "trase_style_macc", "best_k", "matched_frames", "K_seen", "alignment", "pred_dir"]
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(per_scene)
    with (out_dir / "tau_sweep_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["tau", "scenes_done", "mean_miou", "mean_macc"])
        writer.writeheader()
        writer.writerows(summary)
    print(f"Wrote {out_dir / 'tau_sweep_per_scene.csv'}")
    print(f"Wrote {out_dir / 'tau_sweep_summary.csv'}")


if __name__ == "__main__":
    main()
