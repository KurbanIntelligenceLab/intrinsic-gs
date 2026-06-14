#!/usr/bin/env python3
"""Build a seed-0 TeX report from completed Neu3D/HyperNeRF artifacts."""

from __future__ import annotations

import csv
import json
import re
import subprocess
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_TEX = ROOT / "docs" / "seed0_all_results.tex"


def read_csv(path: str) -> list[dict[str, str]]:
    with (ROOT / path).open(newline="") as f:
        return list(csv.DictReader(f))


def esc(value: object) -> str:
    text = "" if value is None else str(value)
    repl = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(repl.get(ch, ch) for ch in text)


def f4(value: object) -> str:
    if value in ("", None):
        return "--"
    return f"{float(value):.4f}"


def f1(value: object) -> str:
    if value in ("", None):
        return "--"
    return f"{float(value):.1f}"


def mean(rows: list[dict[str, object]], key: str) -> float:
    vals = [float(row[key]) for row in rows if row.get(key) not in ("", None)]
    return sum(vals) / len(vals) if vals else float("nan")


def json_metric(run_dir: Path, name: str) -> float | None:
    path = run_dir / f"{name}_results.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    key = "mIoU" if name == "miou" else "mAcc"
    return float(data["results"][key])


def json_metric_greedy(run_dir: Path, name: str) -> float | None:
    path = run_dir / f"{name}_results_greedy.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    key = "mIoU" if name == "miou" else "mAcc"
    return float(data["results"][key])


def json_details(run_dir: Path) -> tuple[int | None, int | None, str]:
    path = run_dir / "miou_results.json"
    if not path.exists():
        return None, None, ""
    data = json.loads(path.read_text())
    best = data.get("results", {}).get("best_cluster")
    k_seen = data.get("K_seen")
    diag = data.get("diagnostics", {})
    method = diag.get("rgb_edge_method", "")
    return best, k_seen, method


def parse_report(run_dir: Path) -> dict[str, object]:
    report = run_dir / "report.md"
    timing = run_dir / "timings.json"
    out: dict[str, object] = {
        "device": "--",
        "n_valid": "--",
        "params": {},
        "graph": {},
        "resolution": "--",
        "communities": "--",
        "modularity_q": "--",
        "timing": {},
    }
    if timing.exists():
        data = json.loads(timing.read_text())
        out["device"] = data.get("device", "--")
        if data.get("n_valid_gaussians") is not None:
            out["n_valid"] = data.get("n_valid_gaussians")
        timing_bits = {}
        for group in ("spectral", "render", "eval"):
            for key, value in data.get(group, {}).items():
                timing_bits[f"{group}.{key}"] = value
        for key in ("spectral_total_s", "render_total_s", "eval_total_s"):
            if key in data:
                timing_bits[key] = data[key]
        out["timing"] = timing_bits

    if not report.exists():
        return out
    text = report.read_text(errors="replace")
    section = None
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("## Parameters"):
            section = "params"
            continue
        if line.startswith("## Graph Stats"):
            section = "graph"
            continue
        if line.startswith("## ") and not line.startswith(("## Parameters", "## Graph Stats")):
            section = None
        if line.startswith("|") and section in ("params", "graph"):
            cells = [cell.strip() for cell in line.strip("|").split("|")]
            if len(cells) >= 2 and cells[0] not in ("param", "stat") and not set(cells[0]) <= {"-"}:
                target = out[section]
                assert isinstance(target, dict)
                target[cells[0]] = cells[1]
    m = re.search(r"Resolution parameter:\s+\*\*([^*]+)\*\*", text)
    if m:
        out["resolution"] = m.group(1)
    m = re.search(r"Communities found:\s+\*\*([^*]+)\*\*", text)
    if m:
        out["communities"] = m.group(1)
    m = re.search(r"Modularity Q:\s+\*\*([^*]+)\*\*", text)
    if m:
        out["modularity_q"] = m.group(1)
    return out


def kv_compact(values: dict[str, object], keys: list[str] | None = None) -> str:
    if not values:
        return "--"
    items = values.items() if keys is None else [(key, values.get(key, "--")) for key in keys]
    return "; ".join(f"{key}={value}" for key, value in items if value not in ("", None))


def timing_compact(values: dict[str, object]) -> str:
    if not values:
        return "--"
    wanted = [
        "spectral.graph_build",
        "spectral.symmetrize_laplacian",
        "spectral.clusterer",
        "spectral_total_s",
        "render.render_main",
        "render_total_s",
        "eval.eval_single_object",
        "eval_total_s",
    ]
    parts = []
    for key in wanted:
        if key in values:
            parts.append(f"{key}={float(values[key]):.3f}s")
    return "; ".join(parts) if parts else "--"


def base_run_from_pred(pred_dir: str) -> Path:
    path = ROOT / pred_dir
    if path.name == "cluster_ids_train":
        path = path.parent
    if path.name.startswith("objectness_"):
        return path.parent
    return path


def provenance_row(dataset: str, family: str, variant: str, scene: str, run_dir: Path) -> list[object]:
    info = parse_report(run_dir)
    params = info["params"] if isinstance(info["params"], dict) else {}
    graph = info["graph"] if isinstance(info["graph"], dict) else {}
    timing = info["timing"] if isinstance(info["timing"], dict) else {}
    if "use_geometry" not in params:
        params["use_geometry"] = "False" if "nogeo" in str(run_dir) or variant == "no_geo" else "True"
    param_keys = [
        "n_clusters",
        "k (kNN)",
        "sigma_color",
        "sigma_scale",
        "sigma_pos",
        "power",
        "opacity_thresh",
        "use_geometry",
        "use_motion",
        "n_time_steps",
        "static_thresh",
        "motion_floor",
        "use_boundary",
        "boundary_views",
        "alpha_depth",
        "beta_rgb",
        "gamma",
        "presmooth_σ",
        "rgb_edge_method",
        "pidinet_variant",
        "pidinet_bin_thr",
        "solver",
        "load_iteration",
    ]
    graph_keys = ["total gaussians", "valid (opacity)", "edges", "nnz (sym)"]
    return [
        dataset,
        family,
        variant,
        scene,
        info["device"],
        info["n_valid"],
        info["resolution"],
        info["communities"],
        info["modularity_q"],
        kv_compact(params, param_keys),
        kv_compact(graph, graph_keys),
        timing_compact(timing),
        str(run_dir.relative_to(ROOT)) if run_dir.is_absolute() or str(run_dir).startswith(str(ROOT)) else str(run_dir),
    ]


def build_provenance_rows(
    hyp_global: list[dict[str, str]],
    hyp_ab_scene: list[dict[str, str]],
    neu_ab_scene: list[dict[str, str]],
    hyp_pid: list[dict[str, object]],
    neu_pid: list[dict[str, object]],
) -> list[list[object]]:
    rows: list[list[object]] = []
    for row in hyp_global:
        rows.append(
            provenance_row(
                "hypernerf",
                "single_global_base",
                "global_objectness_longrange",
                row["scene"],
                base_run_from_pred(row["pred_dir"]),
            )
        )
        rows.append(
            provenance_row(
                "hypernerf",
                "longrange_render",
                "objectness_longrange_default",
                row["scene"],
                (ROOT / row["pred_dir"]).parent,
            )
        )
    for row in hyp_ab_scene:
        rows.append(provenance_row("hypernerf", "component_ablation", row["variant"], row["scene"], ROOT / row["run_dir"]))
    for row in neu_ab_scene:
        rows.append(provenance_row("neu3d", "component_ablation", row["variant"], row["scene"], ROOT / row["run_dir"]))
    for row in hyp_pid:
        rows.append(provenance_row("hypernerf", "boundary_operator", "pidinet_full", str(row["scene"]), ROOT / str(row["run_dir"])))
    for row in neu_pid:
        rows.append(provenance_row("neu3d", "boundary_operator", "pidinet_full", str(row["scene"]), ROOT / str(row["run_dir"])))
    return rows


def longrange_defaults_rows() -> list[list[object]]:
    return [
        ["tag", "objectness_longrange_default / objectness_lr_tauXX"],
        ["pair_tau", "0.70 for the scored global row; sweep rows vary tau as listed"],
        ["traj_min", "0.995"],
        ["visibility_corr_max", "0.25"],
        ["max_components", "6"],
        ["max_component_clusters", "5"],
        ["max_component_area_frac", "0.55"],
        ["w_traj", "0.7"],
        ["w_color", "0.3"],
        ["top_neighbors", "8"],
        ["objectness_weight", "0.15"],
        ["visibility_weight", "0.10"],
        ["load_iteration", "30000"],
        ["source", "scripts/run_objectness_longrange_res003.py defaults"],
    ]


def gpu_inventory_rows() -> list[list[object]]:
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,driver_version",
                "--format=csv,noheader",
            ],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except FileNotFoundError:
        return [["--", "nvidia-smi unavailable", "--", "--"]]
    rows = []
    for line in proc.stdout.splitlines():
        cells = [cell.strip() for cell in line.split(",")]
        if len(cells) >= 4:
            rows.append([cells[0], cells[1], cells[2], cells[3]])
    return rows or [["--", "no GPU inventory captured", "--", "--"]]


def gpu_monitor_rows() -> list[list[object]]:
    paths = [
        ROOT / "output/neu3d_pidinet_boundary_operator_logs/gpu_usage_by_worker.csv",
    ]
    rows: list[list[object]] = []
    for path in paths:
        if not path.exists():
            continue
        samples = list(csv.DictReader(path.open(newline="")))
        grouped: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
        for row in samples:
            grouped[(row.get("gpu_index", "--"), row.get("gpu_name", "--"), row.get("scene", "--"))].append(row)
        for (gpu_idx, gpu_name, scene), vals in sorted(grouped.items()):
            util = [float(v["gpu_utilization_pct"]) for v in vals if v.get("gpu_utilization_pct")]
            mem = [float(v["gpu_memory_used_mb"]) for v in vals if v.get("gpu_memory_used_mb")]
            pid_mem = [float(v["pid_used_memory_mb"]) for v in vals if v.get("pid_used_memory_mb")]
            rows.append(
                [
                    path.relative_to(ROOT),
                    gpu_idx,
                    gpu_name,
                    scene,
                    len(vals),
                    f"{max(util):.1f}" if util else "--",
                    f"{max(mem)/1024:.2f}" if mem else "--",
                    f"{max(pid_mem)/1024:.2f}" if pid_mem else "--",
                ]
            )
    return rows or [["--", "--", "--", "--", "--", "--", "--", "--"]]


def add_table(lines: list[str], spec: str, headers: list[str], rows: list[list[object]]) -> None:
    lines.append(r"\small")
    lines.append(rf"\begin{{longtable}}{{{spec}}}")
    lines.append(r"\toprule")
    lines.append(" & ".join(esc(h) for h in headers) + r" \\")
    lines.append(r"\midrule")
    lines.append(r"\endfirsthead")
    lines.append(r"\toprule")
    lines.append(" & ".join(esc(h) for h in headers) + r" \\")
    lines.append(r"\midrule")
    lines.append(r"\endhead")
    for row in rows:
        lines.append(" & ".join(esc(cell) for cell in row) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{longtable}")
    lines.append(r"\normalsize")
    lines.append("")


def pidinet_rows() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    neu_rows: list[dict[str, object]] = []
    for run_dir in sorted((ROOT / "output/neu3d_pidinet_boundary_operator/runs/pidinet_full").glob("*/*")):
        if not (run_dir / "miou_results.json").exists():
            continue
        scene = run_dir.parent.name
        best, k_seen, method = json_details(run_dir)
        neu_rows.append(
            {
                "dataset": "neu3d",
                "scene": scene,
                "miou": json_metric(run_dir, "miou"),
                "greedy_miou": json_metric_greedy(run_dir, "miou"),
                "macc": json_metric(run_dir, "macc"),
                "greedy_macc": json_metric_greedy(run_dir, "macc"),
                "best_cluster": best,
                "K_seen": k_seen,
                "method": method or "pidinet",
                "run_dir": str(run_dir.relative_to(ROOT)),
            }
        )

    hyp_rows: list[dict[str, object]] = []
    for path in sorted((ROOT / "output").glob("*_30k_gs/30_05/*B-pdF-v12a5.0b2.0g2.0*/miou_results.json")):
        run_dir = path.parent
        scene = run_dir.parents[1].name.removesuffix("_30k_gs")
        best, k_seen, method = json_details(run_dir)
        hyp_rows.append(
            {
                "dataset": "hypernerf",
                "scene": scene,
                "miou": json_metric(run_dir, "miou"),
                "greedy_miou": json_metric_greedy(run_dir, "miou"),
                "macc": json_metric(run_dir, "macc"),
                "greedy_macc": json_metric_greedy(run_dir, "macc"),
                "best_cluster": best,
                "K_seen": k_seen,
                "method": method or "pidinet",
                "run_dir": str(run_dir.relative_to(ROOT)),
            }
        )
    return hyp_rows, neu_rows


def tau_status_rows() -> list[list[object]]:
    rendered: dict[str, set[str]] = defaultdict(set)
    for cluster_dir in (ROOT / "output").glob("*_30k_gs/*/*/objectness_lr_tau*/cluster_ids_train"):
        if not any(cluster_dir.glob("*.png")):
            continue
        tag = cluster_dir.parent.name.replace("objectness_lr_tau", "")
        scene = cluster_dir.parts[-5].removesuffix("_30k_gs")
        rendered[tag].add(scene)

    done: dict[str, int] = defaultdict(int)
    err: dict[str, int] = defaultdict(int)
    log_dirs = [
        ROOT / "output/hypernerf_tau_lr_sweep_logs",
        ROOT / "output/hypernerf_tau_lr_extra_sweep_logs",
        ROOT / "output/hypernerf_tau_lr_gpu1_sweep_logs",
        ROOT / "output/hypernerf_tau_lr_tau030_sweep_logs",
        ROOT / "output/hypernerf_tau_lr_tau020_sweep_logs",
    ]
    err_re = re.compile(r"Traceback|Error|No space|Killed|RuntimeError|CalledProcessError")
    for log_dir in log_dirs:
        for log_path in log_dir.glob("*.log"):
            match = re.search(r"tau(\d+)", log_path.name)
            if not match:
                continue
            tag = match.group(1)
            text = log_path.read_text(errors="replace")
            if "[done]" in text:
                done[tag] += 1
            if err_re.search(text):
                err[tag] += 1

    rows = []
    for tag in sorted(set(rendered) | set(done) | set(err), key=lambda x: int(x)):
        tau = int(tag) / 100.0
        rows.append(
            [
                f"{tau:.2f}",
                len(rendered.get(tag, set())),
                done.get(tag, 0),
                err.get(tag, 0),
                ", ".join(sorted(rendered.get(tag, set()))) if rendered.get(tag) else "--",
            ]
        )
    return rows


def timing_summary(rows: list[dict[str, str]]) -> list[list[object]]:
    grouped: dict[str, list[dict[str, float]]] = defaultdict(list)
    for row in rows:
        run_dir = ROOT / row.get("run_dir", "")
        timing_path = run_dir / "timings.json"
        if not timing_path.exists():
            continue
        data = json.loads(timing_path.read_text())
        grouped[row["variant"]].append(
            {
                "n": float(data.get("n_valid_gaussians", 0)),
                "graph": float(data.get("spectral", {}).get("graph_build", 0)),
                "cluster": float(data.get("spectral", {}).get("clusterer", 0)),
                "spectral": float(data.get("spectral_total_s", 0)),
                "render": float(data.get("render_total_s", 0)),
                "eval": float(data.get("eval_total_s", 0)),
            }
        )
    out = []
    for variant, vals in sorted(grouped.items()):
        out.append(
            [
                variant,
                len(vals),
                f"{mean(vals, 'n')/1000:.1f}k",
                f"{mean(vals, 'graph'):.1f}",
                f"{mean(vals, 'cluster'):.1f}",
                f"{mean(vals, 'spectral'):.1f}",
                f"{mean(vals, 'render'):.1f}",
                f"{mean(vals, 'eval'):.1f}",
            ]
        )
    return out


def main() -> None:
    hyp_global = read_csv("output/final_hypernerf_1/maskbenchmark_full10_res003_global_longrange_trase_style.csv")
    hyp_methods = read_csv("output/final_hypernerf_1/ablations/method_summary.csv")
    hyp_method_scene = read_csv("output/final_hypernerf_1/ablations/per_scene_method_ablation.csv")
    hyp_ab_sum = read_csv("output/final_hypernerf_component_ablation_logs/component_ablation_summary.csv")
    hyp_ab_scene = read_csv("output/final_hypernerf_component_ablation_logs/component_ablation_per_scene.csv")
    neu_ab_sum = read_csv("output/final_neu3d_leiden018_component_ablations/component_ablation_summary.csv")
    neu_ab_scene = read_csv("output/final_neu3d_leiden018_component_ablations/component_ablation_per_scene.csv")
    hung_sum = read_csv("output/non_oracle_metrics/hungarian_overseg_summary.csv")
    hung_scene = read_csv("output/non_oracle_metrics/hungarian_overseg_per_scene.csv")
    hyp_pid, neu_pid = pidinet_rows()

    lines: list[str] = [
        r"\documentclass[10pt]{article}",
        r"\usepackage[margin=0.55in,landscape]{geometry}",
        r"\usepackage{booktabs,longtable,array}",
        r"\usepackage[T1]{fontenc}",
        r"\usepackage{lmodern}",
        r"\setlength{\parindent}{0pt}",
        r"\setlength{\parskip}{0.35em}",
        r"\renewcommand{\arraystretch}{1.05}",
        r"\begin{document}",
        r"\title{Seed-0 Neu3D and HyperNeRF Results}",
        r"\author{Generated from local result artifacts}",
        r"\date{May 30, 2026}",
        r"\maketitle",
        "",
        r"\section{Scope}",
        "This document consolidates the seed-0 artifacts currently available for Neu3D and HyperNeRF. "
        "All numbers are single-seed unless explicitly labeled otherwise; seed variance and mean\\,$\\pm$\\,std are still pending. "
        "The report includes single-cluster metrics, greedy-union metrics, non-oracle Hungarian/over-segmentation metrics, available timing fields, boundary-operator runs, and tau-sweep completion status.",
        "",
        r"\section{Headline Seed-0 Means}",
    ]

    hyp_global_mean_iou = mean(hyp_global, "trase_style_miou")
    hyp_global_mean_acc = mean(hyp_global, "trase_style_macc")
    hyp_targeted = next(row for row in hyp_methods if row["method"] == "targeted_default")
    neu_best = max(neu_ab_sum, key=lambda r: float(r["mean_miou_best"]))
    neu_greedy_best = max(neu_ab_sum, key=lambda r: float(r["mean_miou_greedy"]))
    add_table(
        lines,
        "lrrrrp{9cm}",
        ["Result", "Scenes", "mIoU", "mAcc", "Greedy mIoU", "Notes"],
        [
            [
                "HyperNeRF single-global long-range",
                len(hyp_global),
                f4(hyp_global_mean_iou),
                f4(hyp_global_mean_acc),
                "--",
                "Leiden res0.03 with fixed global long-range/objectness merge; tau_lr=0.70 artifact.",
            ],
            [
                "HyperNeRF targeted/per-scene upper-bound preset",
                10,
                f4(hyp_targeted["mean_miou"]),
                f4(hyp_targeted["mean_macc"]),
                "--",
                hyp_targeted["notes"],
            ],
            [
                "Neu3D best single-cluster ablation",
                neu_best["num_done"],
                f4(neu_best["mean_miou_best"]),
                f4(neu_best["mean_macc_best"]),
                f4(neu_best["mean_miou_greedy"]),
                f"variant={neu_best['variant']}; best by single-cluster mIoU",
            ],
            [
                "Neu3D best greedy-union ablation",
                neu_greedy_best["num_done"],
                f4(neu_greedy_best["mean_miou_best"]),
                f4(neu_greedy_best["mean_macc_best"]),
                f4(neu_greedy_best["mean_miou_greedy"]),
                f"variant={neu_greedy_best['variant']}; best by greedy mIoU",
            ],
        ],
    )

    lines.append(r"\section{HyperNeRF Single-Global Long-Range}")
    add_table(
        lines,
        "lrrrrrp{12cm}",
        ["Scene", "mIoU", "mAcc", "Best k", "Frames", "K seen", "Prediction directory"],
        [
            [
                row["scene"],
                f4(row["trase_style_miou"]),
                f4(row["trase_style_macc"]),
                row["best_k"],
                row["matched_frames"],
                row["K_seen"],
                row["pred_dir"],
            ]
            for row in hyp_global
        ]
        + [["Mean", f4(hyp_global_mean_iou), f4(hyp_global_mean_acc), "--", "--", "--", "--"]],
    )

    lines.append(r"\section{Long-Range Merge Defaults}")
    lines.append("These are the objectness/long-range merge settings used by the scored global HyperNeRF row unless overridden by the tau sweep.")
    add_table(
        lines,
        "lp{18cm}",
        ["Parameter", "Value"],
        longrange_defaults_rows(),
    )

    lines.append(r"\section{GPU Hardware and Sampled Usage}")
    add_table(
        lines,
        "llll",
        ["GPU index", "GPU name", "Memory total", "Driver"],
        gpu_inventory_rows(),
    )
    lines.append("The sampled worker telemetry below is from the GPU monitor CSV that was active for the seed-0 batch. It is sampled usage, not a full per-stage peak-memory profiler.")
    add_table(
        lines,
        "p{7cm}lllrlll",
        ["Source", "GPU", "Name", "Scene/owner", "Samples", "Max util \\%", "Max GPU GB", "Max PID GB"],
        gpu_monitor_rows(),
    )

    lines.append(r"\section{HyperNeRF Method Variants}")
    add_table(
        lines,
        "lrrp{12cm}",
        ["Method", "Mean mIoU", "Mean mAcc", "Notes"],
        [[row["method"], f4(row["mean_miou"]), f4(row["mean_macc"]), row["notes"]] for row in hyp_methods],
    )
    add_table(
        lines,
        "lrrrrrrrp{6cm}",
        ["Scene", "Baseline mIoU", "Global LR mIoU", "Targeted mIoU", "Baseline mAcc", "Global LR mAcc", "Targeted mAcc", "Targeted delta vs global", "Targeted variant"],
        [
            [
                row["scene"],
                f4(row["baseline_miou"]),
                f4(row["global_longrange_miou"]),
                f4(row["targeted_default_miou"]),
                f4(row["baseline_macc"]),
                f4(row["global_longrange_macc"]),
                f4(row["targeted_default_macc"]),
                f4(row["targeted_delta_vs_global_longrange"]),
                row["targeted_variant"],
            ]
            for row in hyp_method_scene
        ],
    )

    lines.append(r"\section{HyperNeRF Component Ablations}")
    add_table(
        lines,
        "lrrrrr",
        ["Variant", "Scenes", "mIoU", "Greedy mIoU", "mAcc", "Greedy mAcc"],
        [
            [
                row["variant"],
                row["num_done"],
                f4(row["mean_miou_best"]),
                f4(row["mean_miou_greedy"]),
                f4(row["mean_macc_best"]),
                f4(row["mean_macc_greedy"]),
            ]
            for row in hyp_ab_sum
        ],
    )
    add_table(
        lines,
        "llrrrrp{9cm}",
        ["Variant", "Scene", "mIoU", "Greedy mIoU", "mAcc", "Greedy mAcc", "Run directory"],
        [
            [
                row["variant"],
                row["scene"],
                f4(row["miou_best"]),
                f4(row["miou_greedy"]),
                f4(row["macc_best"]),
                f4(row["macc_greedy"]),
                row["run_dir"],
            ]
            for row in hyp_ab_scene
        ],
    )

    lines.append(r"\section{Neu3D Component Ablations}")
    add_table(
        lines,
        "lrrrrr",
        ["Variant", "Scenes", "mIoU", "Greedy mIoU", "mAcc", "Greedy mAcc"],
        [
            [
                row["variant"],
                row["num_done"],
                f4(row["mean_miou_best"]),
                f4(row["mean_miou_greedy"]),
                f4(row["mean_macc_best"]),
                f4(row["mean_macc_greedy"]),
            ]
            for row in neu_ab_sum
        ],
    )
    add_table(
        lines,
        "lllrrrrp{9cm}",
        ["Variant", "Scene", "Status", "mIoU", "Greedy mIoU", "mAcc", "Greedy mAcc", "Run directory"],
        [
            [
                row["variant"],
                row["scene"],
                row["status"],
                f4(row["miou_best"]),
                f4(row["miou_greedy"]),
                f4(row["macc_best"]),
                f4(row["macc_greedy"]),
                row["run_dir"],
            ]
            for row in neu_ab_scene
        ],
    )

    lines.append(r"\section{Neu3D Timing Fields Available}")
    lines.append("Timing is currently available for the Neu3D component ablation runs through per-run \\texttt{timings.json}. Memory and FPS telemetry were not present in these artifacts.")
    add_table(
        lines,
        "lrrrrrrr",
        ["Variant", "Runs", "Gaussians", "Graph build s", "Clusterer s", "Spectral total s", "Render s", "Eval s"],
        timing_summary(neu_ab_scene),
    )

    lines.append(r"\section{Per-Run Settings, Graph Stats, Timings, and GPU}")
    lines.append("This table is intentionally verbose: it preserves the per-run parameter table, graph statistics, timing fields, GPU/device label, Leiden resolution, community count, and modularity from the run reports. Long-range render rows contain render-only timing because their report is produced by the merge/render stage after the base Leiden run.")
    add_table(
        lines,
        "lllllrrrrp{18cm}p{9cm}p{14cm}p{12cm}",
        [
            "Dataset",
            "Family",
            "Variant",
            "Scene",
            "GPU/device",
            "N valid",
            "Leiden rho",
            "Communities",
            "Q",
            "Parameters",
            "Graph stats",
            "Timings",
            "Run directory",
        ],
        build_provenance_rows(hyp_global, hyp_ab_scene, neu_ab_scene, hyp_pid, neu_pid),
    )

    lines.append(r"\section{Boundary Operator: PiDiNet}")
    add_table(
        lines,
        "lrrrrrrp{9cm}",
        ["HyperNeRF scene", "mIoU", "Greedy mIoU", "mAcc", "Greedy mAcc", "Best cluster", "K seen", "Run directory"],
        [
            [
                row["scene"],
                f4(row["miou"]),
                f4(row["greedy_miou"]),
                f4(row["macc"]),
                f4(row["greedy_macc"]),
                row["best_cluster"],
                row["K_seen"],
                row["run_dir"],
            ]
            for row in hyp_pid
        ]
        + (
            [["Mean over completed HyperNeRF PiDiNet runs", f4(mean(hyp_pid, "miou")), f4(mean(hyp_pid, "greedy_miou")), f4(mean(hyp_pid, "macc")), f4(mean(hyp_pid, "greedy_macc")), "--", "--", "--"]]
            if hyp_pid
            else []
        ),
    )
    add_table(
        lines,
        "lrrrrrrp{9cm}",
        ["Neu3D scene", "mIoU", "Greedy mIoU", "mAcc", "Greedy mAcc", "Best cluster", "K seen", "Run directory"],
        [
            [
                row["scene"],
                f4(row["miou"]),
                f4(row["greedy_miou"]),
                f4(row["macc"]),
                f4(row["greedy_macc"]),
                row["best_cluster"],
                row["K_seen"],
                row["run_dir"],
            ]
            for row in neu_pid
        ]
        + (
            [["Mean over completed Neu3D PiDiNet runs", f4(mean(neu_pid, "miou")), f4(mean(neu_pid, "greedy_miou")), f4(mean(neu_pid, "macc")), f4(mean(neu_pid, "greedy_macc")), "--", "--", "--"]]
            if neu_pid
            else []
        ),
    )

    lines.append(r"\section{Non-Oracle Hungarian and Over-Segmentation Metrics}")
    add_table(
        lines,
        "llrrrrr",
        ["Dataset", "Variant", "Scenes", "Hungarian mIoU", "Hungarian mAcc", "Mean clusters", "Over-seg ratio"],
        [
            [
                row["dataset"],
                row["variant"],
                row["scenes"],
                f4(row["hungarian_miou_mean"]),
                f4(row["hungarian_macc_mean"]),
                f1(row["cluster_count_mean"]),
                f1(row["overseg_ratio_mean"]),
            ]
            for row in hung_sum
        ],
    )
    add_table(
        lines,
        "lllrrrrrrrp{8cm}",
        ["Dataset", "Variant", "Scene", "Hungarian mIoU", "Hungarian mAcc", "GT objs", "Matched clusters", "Clusters", "Over-seg", "Frames", "Pred dir"],
        [
            [
                row["dataset"],
                row["variant"],
                row["scene"],
                f4(row["hungarian_miou"]),
                f4(row["hungarian_macc"]),
                row["gt_object_count"],
                row["matched_cluster_count"],
                row["cluster_count"],
                f1(row["overseg_ratio"]),
                row["valid_pred_frames"],
                row["pred_dir"],
            ]
            for row in hung_scene
        ],
    )

    lines.append(r"\section{HyperNeRF Tau Sweep Render Status}")
    lines.append("These rows summarize rendered seed-0 long-range tau sweep outputs and logs. They are completion/provenance rows; only the tau=0.70 global objectness long-range configuration above has been scored into the clean Mask-Benchmark CSV in this report.")
    add_table(
        lines,
        "rrrrp{18cm}",
        ["Tau", "Rendered scenes", "Done logs", "Error logs", "Rendered scene names"],
        tau_status_rows(),
    )

    lines.extend(
        [
            r"\section{Pending Items}",
            r"\begin{itemize}",
            r"\item Seed variance and mean\,\(\pm\)\,std: not yet available in seed-0 artifacts.",
            r"\item Full per-stage GPU memory, host RAM, FPS, and edges/sec telemetry: not present in the completed result CSVs/timings.",
            r"\item HyperNeRF tau sweep score table beyond tau=0.70: rendered outputs exist for most tau values, but clean mIoU/mAcc CSVs have not yet been produced for every tau.",
            r"\item TRASE/SAM measured baseline timing on this hardware: not present in seed-0 artifacts.",
            r"\item Failure-mode cue-gap diagnostics: not present in seed-0 artifacts.",
            r"\end{itemize}",
            "",
            r"\section{Source Artifacts}",
        ]
    )
    add_table(
        lines,
        "p{24cm}",
        ["Source file or pattern"],
        [[p] for p in [
            "output/final_hypernerf_1/maskbenchmark_full10_res003_global_longrange_trase_style.csv",
            "output/final_hypernerf_1/ablations/method_summary.csv",
            "output/final_hypernerf_1/ablations/per_scene_method_ablation.csv",
            "output/final_hypernerf_component_ablation_logs/component_ablation_summary.csv",
            "output/final_hypernerf_component_ablation_logs/component_ablation_per_scene.csv",
            "output/final_neu3d_leiden018_component_ablations/component_ablation_summary.csv",
            "output/final_neu3d_leiden018_component_ablations/component_ablation_per_scene.csv",
            "output/non_oracle_metrics/hungarian_overseg_summary.csv",
            "output/non_oracle_metrics/hungarian_overseg_per_scene.csv",
            "output/neu3d_pidinet_boundary_operator/runs/pidinet_full/*/*/{miou,macc}_results*.json",
            "output/*_30k_gs/30_05/*B-pdF-v12a5.0b2.0g2.0*/{miou,macc}_results*.json",
            "output/hypernerf_tau_lr*_logs/*.log",
            "output/*_30k_gs/*/*/objectness_lr_tau*/cluster_ids_train/*.png",
        ]],
    )

    lines.append(r"\end{document}")
    OUT_TEX.write_text("\n".join(lines) + "\n")
    print(f"Wrote {OUT_TEX}")


if __name__ == "__main__":
    main()
