"""
Compute mean per-frame pixel accuracy (mAcc) for a self-supervised spectral run.

mAcc convention matches metrics_segmentation.py in this repo (TRASE):
  per-frame:  acc_f = (pred_binary == gt_binary).sum() / total_pixels
  scene:      mAcc  = mean_f(acc_f)

The selected cluster(s) defining `pred_binary` come from a companion
miou_results JSON — `selected_clusters` when present (greedy_union path),
else `best_cluster` as a singleton (best_cluster path / pre-selection_mode
JSONs). They can also be passed explicitly via --selected_clusters.

Usage:
    python self_supervised_scripts/compute_macc.py \\
        --pred_dir   <run>/cluster_ids_train \\
        --gt_dir     Mask-Benchmark/HyperNeRF-Mask/<scene>/gt_masks \\
        --miou_json  <run>/miou_results.json \\
        --output_json <run>/macc_results.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from self_supervised_scripts.compute_miou import (  # noqa: E402
    align_single_object_gt_paths,
    index_pngs,
    load_binary_mask,
    load_cluster_id_map,
    load_gt_layout,
)


def selected_clusters_from_miou_json(miou_json_path):
    """Read selected_clusters from a miou_results JSON, with backwards compat.

    Pre-selection_mode JSONs only carry `best_cluster`; treat it as a
    singleton. Newer JSONs may have an explicit `selected_clusters` list
    (greedy_union always sets this).
    """
    data = json.loads(Path(miou_json_path).read_text())
    results = data.get("results", data)
    selected = results.get("selected_clusters")
    if selected:
        return [int(k) for k in selected]
    best = results.get("best_cluster")
    return [int(best)] if best is not None else None


def compute_macc(pred_paths, gt_paths, selected_clusters):
    """Compute per-frame pixel accuracy + mAcc for a cluster union."""
    selected = np.array(selected_clusters, dtype=np.int32)
    per_frame_acc = {}
    skipped = 0
    matched = sorted(set(pred_paths) & set(gt_paths))
    for fid in matched:
        pred = load_cluster_id_map(pred_paths[fid])
        gt = load_binary_mask(gt_paths[fid])
        if pred.shape != gt.shape:
            skipped += 1
            continue
        pred_b = np.isin(pred, selected)
        per_frame_acc[fid] = float((pred_b == gt).sum() / gt.size)
    mAcc = float(np.mean(list(per_frame_acc.values()))) if per_frame_acc else 0.0
    return {
        "mAcc": mAcc,
        "n_frames": len(per_frame_acc),
        "skipped_shape_mismatch": skipped,
        "selected_clusters": [int(k) for k in selected_clusters],
        "per_frame_acc": {str(fid): v for fid, v in per_frame_acc.items()},
    }


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pred_dir", required=True,
                   help="Directory of cluster-ID PNGs (cluster_ids_train or _test).")
    p.add_argument("--gt_dir", required=True,
                   help="Directory of binary GT mask PNGs (flat single-object layout).")
    p.add_argument("--miou_json", default=None,
                   help="miou_results JSON to read selected_clusters / best_cluster "
                        "from. Skip when passing --selected_clusters directly.")
    p.add_argument("--selected_clusters", type=int, nargs="+", default=None,
                   help="Cluster IDs to union into the foreground. Overrides "
                        "the values read from --miou_json.")
    p.add_argument("--output_json", required=True,
                   help="Where to write the macc_results JSON.")
    return p


def main():
    args = build_parser().parse_args()

    selected = args.selected_clusters
    if not selected:
        if not args.miou_json:
            raise SystemExit(
                "Need either --selected_clusters or --miou_json to decide which "
                "cluster IDs form the foreground."
            )
        selected = selected_clusters_from_miou_json(args.miou_json)
        if not selected:
            raise SystemExit(
                f"Could not read selected_clusters or best_cluster from "
                f"{args.miou_json}."
            )

    pred_paths = index_pngs(args.pred_dir)
    gt_layout, gt_payload = load_gt_layout(args.gt_dir)
    if gt_layout != "single":
        raise SystemExit(
            f"compute_macc.py currently supports single-object GT only "
            f"(detected layout={gt_layout}). Use compute_miou.py's multi-object "
            f"path for nested GT."
        )
    gt_paths, frame_alignment = align_single_object_gt_paths(
        pred_paths, gt_payload, args.gt_dir,
    )

    result = compute_macc(pred_paths, gt_paths, selected)

    out = {
        "pred_dir": str(Path(args.pred_dir).resolve()),
        "gt_dir": str(Path(args.gt_dir).resolve()),
        "source_miou_json": (str(Path(args.miou_json).resolve())
                             if args.miou_json else None),
        "frame_alignment": frame_alignment,
        "results": result,
    }
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(out, indent=2))
    print(f"mAcc = {result['mAcc']:.4f}  ({result['n_frames']} frames, "
          f"selected={result['selected_clusters']}) → {args.output_json}")


if __name__ == "__main__":
    main()
