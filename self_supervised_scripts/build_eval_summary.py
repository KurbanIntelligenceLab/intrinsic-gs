"""Aggregate every miou_results*.json under outputs/ into a single CSV.

Scans for run dirs produced by every clusterer (kmeans → spectral_*,
leiden → leiden_*, hdbscan → hdbscan_*) and emits a row per
(run, selection_mode) pair.

Usage:
    python self_supervised_scripts/build_eval_summary.py \\
        [--outputs_dir outputs] \\
        [--csv_path outputs/evaluation_summary.csv]
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
from pathlib import Path

# Run-dir prefixes per clusterer — must stay in sync with
# pipeline_common.CLUSTERER_PREFIX.
CLUSTERER_PREFIXES = ["spectral", "leiden", "hdbscan"]


def detect_clusterer_from_dir(run_dir: str) -> str:
    """Map a run directory name back to its clusterer.

    Falls back to 'unknown' if the prefix doesn't match any known
    clusterer (defensive: shouldn't happen for runs we produced).
    """
    base = os.path.basename(run_dir)
    if base.startswith("spectral_"):
        return "kmeans"
    if base.startswith("leiden_"):
        return "leiden"
    if base.startswith("hdbscan_"):
        return "hdbscan"
    return "unknown"


def parse_diagnostics_from_report(report_md_path: str) -> dict:
    """Re-implement the algorithm-aware regexes used by compute_miou.parse_report_md.

    Extracted here so this script does not import compute_miou (which has
    GPU/scene dependencies). Keep regex-equivalent.
    """
    out: dict = {}
    if not os.path.exists(report_md_path):
        return out
    text = Path(report_md_path).read_text()

    m = re.search(r"\|\s*n_clusters\s*\|\s*(\d+)\s*\|", text)
    if m:
        out["K_used"] = int(m.group(1))

    m = re.search(r"\|\s*rgb_edge_method\s*\|\s*([\w]+)\s*\|", text)
    if m and m.group(1) != "—":
        out["rgb_edge_method"] = m.group(1)
    m = re.search(r"\|\s*pidinet_variant\s*\|\s*([\w]+)\s*\|", text)
    if m and m.group(1) != "—":
        out["pidinet_variant"] = m.group(1)

    if re.search(r"^##\s*Spectral Analysis\s*$", text, flags=re.MULTILINE):
        out["clusterer"] = "kmeans"
        m = re.search(r"Eigengap suggested k:\s*\*\*(\d+)\*\*", text)
        if m:
            out["K_auto_eigengap"] = int(m.group(1))
        m = re.search(r"ρ\s*=.*?\*\*([\d.]+)\*\*", text)
        if m:
            out["rho"] = float(m.group(1))
    elif re.search(r"^##\s*Leiden Community Detection\s*$", text, flags=re.MULTILINE):
        out["clusterer"] = "leiden"
        m = re.search(r"Modularity Q:\s*\*\*([\d.\-]+)\*\*", text)
        if m:
            out["modularity_q"] = float(m.group(1))
    elif re.search(r"^##\s*HDBSCAN", text, flags=re.MULTILINE):
        out["clusterer"] = "hdbscan"
        m = re.search(
            r"Noise points reassigned to nearest cluster:\s*\*\*([\d,]+)\*\*", text
        )
        if m:
            out["n_noise"] = int(m.group(1).replace(",", ""))
    return out


def _glob_miou_files(outputs_dir: str) -> list[str]:
    files: list[str] = []
    for prefix in CLUSTERER_PREFIXES:
        files += glob.glob(f"{outputs_dir}/*/{prefix}_*/miou_results*.json")
        # Also handle grouped sweep outputs: outputs/multiple-N/<scene>/<prefix>_*
        files += glob.glob(f"{outputs_dir}/*/*/{prefix}_*/miou_results*.json")
    return sorted(set(files))


def _parse_k_from_run_name(run_name: str) -> int | None:
    """Extract the numeric K from a kmeans run dir like 'spectral_k14_...'."""
    m = re.match(r"spectral_k(\d+)_", run_name)
    return int(m.group(1)) if m else None


def collect_rows(outputs_dir: str) -> list[dict]:
    files = _glob_miou_files(outputs_dir)
    rows: list[dict] = []
    for f in files:
        d = json.load(open(f))
        fa = d.get("frame_alignment") or {}
        diag = d.get("diagnostics") or {}
        res = d.get("results") or {}
        parts = f.split(os.sep)
        run = parts[-2]
        scene = parts[-3]  # may be 'multiple-N' fallback handled below
        # If the parent of run dir is a "multiple-N" group, use its parent's name.
        if scene.startswith("multiple-") and len(parts) >= 4:
            scene = parts[-4]  # but this is now 'outputs' not the scene; fix below
            # Actually for grouped layout: outputs/multiple-N/<scene>/<run>/
            # parts[-1]=miou_results.json, [-2]=run, [-3]=scene_dir, [-4]='multiple-N'
            scene = parts[-3]
            # Re-extract: in grouped layout, parts[-3] IS the scene dir.

        fname = os.path.basename(f)
        mode = res.get("selection_mode")
        if not mode:
            mode = "greedy_union" if "greedy" in fname else "best_cluster"

        clusterer = detect_clusterer_from_dir(run)
        # Augment diagnostics from report.md if missing in JSON.
        report_path = os.path.join(os.path.dirname(f), "report.md")
        report_diag = parse_diagnostics_from_report(report_path)
        merged = {**report_diag, **diag}  # JSON wins over report

        k_req = _parse_k_from_run_name(run)
        pf = res.get("per_frame_iou") or {}
        pf_vals = list(pf.values()) if pf else []
        selected = res.get("selected_clusters") or []

        tc = d.get("temporal_consistency") or {}
        t = d.get("timings") or {}
        spectral_t = t.get("spectral") or {}
        render_t = t.get("render") or {}

        macc_suffix = "_greedy" if mode == "greedy_union" else ""
        macc_path = os.path.join(
            os.path.dirname(f), f"macc_results{macc_suffix}.json"
        )
        macc_scene = None
        if os.path.exists(macc_path):
            macc_scene = (json.load(open(macc_path)).get("results") or {}).get("mAcc")

        rows.append({
            "scene": scene,
            "clusterer": clusterer,
            "run_id": run.split("_")[-1],
            "k_requested": k_req,
            "selection_mode": mode,
            "n_frames_eval": res.get("matched_frames"),
            "gt_frames_total": fa.get("gt_frames"),
            "alignment_method": fa.get("method"),
            "mIoU_scene": res.get("mIoU"),
            "mAcc_scene": macc_scene,
            "mean_per_frame": (sum(pf_vals) / len(pf_vals)) if pf_vals else None,
            "min_per_frame": min(pf_vals) if pf_vals else None,
            "max_per_frame": max(pf_vals) if pf_vals else None,
            "tc": tc.get("tc"),
            "tc_n_pairs": tc.get("n_pairs"),
            "t_graph_s": spectral_t.get("graph_build"),
            "t_eig_s": spectral_t.get("clusterer"),
            "t_render_s": render_t.get("render_main"),
            "t_spectral_total_s": t.get("spectral_total_s"),
            "t_render_total_s": t.get("render_total_s"),
            "device": t.get("device"),
            "best_cluster": res.get("best_cluster"),
            "selected_clusters": ",".join(map(str, selected)) if selected else None,
            "num_selected": len(selected) if selected else None,
            "K_used": merged.get("K_used"),
            "K_auto_eigengap": merged.get("K_auto_eigengap") or merged.get("K_auto"),
            "rho": merged.get("rho"),
            "modularity_q": merged.get("modularity_q"),
            "n_noise": merged.get("n_noise"),
            "rgb_edge_method": merged.get("rgb_edge_method"),
            "pidinet_variant": merged.get("pidinet_variant"),
        })

    # greedy_gain = greedy_union mIoU − best_cluster mIoU per (scene, clusterer, run_id).
    by_pair: dict = {}
    for r in rows:
        key = (r["scene"], r["clusterer"], r["run_id"])
        by_pair.setdefault(key, {})[r["selection_mode"]] = r["mIoU_scene"]
    for r in rows:
        key = (r["scene"], r["clusterer"], r["run_id"])
        pair = by_pair[key]
        if (
            r["selection_mode"] == "greedy_union"
            and pair.get("best_cluster") is not None
        ):
            r["greedy_gain"] = r["mIoU_scene"] - pair["best_cluster"]
        else:
            r["greedy_gain"] = None
    return rows


COLUMNS = [
    "scene", "clusterer", "k_requested", "run_id", "selection_mode",
    "n_frames_eval", "gt_frames_total", "alignment_method",
    "mIoU_scene", "mAcc_scene",
    "mean_per_frame", "min_per_frame", "max_per_frame",
    "tc", "tc_n_pairs",
    "t_graph_s", "t_eig_s", "t_render_s",
    "t_spectral_total_s", "t_render_total_s", "device",
    "greedy_gain", "best_cluster", "selected_clusters", "num_selected",
    "K_used", "K_auto_eigengap", "rho", "modularity_q", "n_noise",
    "rgb_edge_method", "pidinet_variant",
]


def write_csv(rows: list[dict], csv_path: str) -> None:
    Path(os.path.dirname(csv_path) or ".").mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=COLUMNS)
        w.writeheader()
        for r in sorted(
            rows,
            key=lambda x: (
                x["scene"] or "",
                x["clusterer"] or "",
                x["k_requested"] or 0,
                x["selection_mode"] or "",
            ),
        ):
            out = {k: r.get(k) for k in COLUMNS}
            for k in (
                "mIoU_scene", "mAcc_scene", "mean_per_frame", "min_per_frame",
                "max_per_frame", "greedy_gain", "rho", "modularity_q",
            ):
                if isinstance(out[k], float):
                    out[k] = round(out[k], 4)
            w.writerow(out)


def main_with_args(outputs_dir: str, csv_path: str) -> list[dict]:
    rows = collect_rows(outputs_dir)
    write_csv(rows, csv_path)
    print(f"Wrote {len(rows)} rows to {csv_path}")
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--outputs_dir", default="outputs")
    p.add_argument("--csv_path", default="outputs/evaluation_summary.csv")
    args = p.parse_args()
    main_with_args(args.outputs_dir, args.csv_path)


if __name__ == "__main__":
    main()
