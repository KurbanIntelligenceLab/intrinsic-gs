#!/usr/bin/env python3
"""Summarize Neu3D component-ablation runs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


SCENES = [
    "coffee_martini",
    "cook_spinach",
    "cut_roasted_beef",
    "flame_steak",
    "sear_steak",
]

VARIANTS = [
    "default",
    "no_geo",
    "no_motion",
    "no_boundary",
    "no_motion_no_boundary",
]


def latest_done(root: Path, variant: str, scene: str) -> Path | None:
    scene_root = root / "runs" / variant / scene
    if not scene_root.exists():
        return None
    candidates = sorted(
        (
            path
            for path in scene_root.iterdir()
            if path.is_dir()
            and (path / "miou_results_greedy.json").exists()
            and (path / "macc_results_greedy.json").exists()
        ),
        key=lambda path: (path.stat().st_mtime, path.name),
        reverse=True,
    )
    return candidates[0] if candidates else None


def load_metric(run: Path, stem: str) -> float:
    data = json.loads((run / stem).read_text())
    return float(data["results"]["mIoU" if stem.startswith("miou") else "mAcc"])


def write_csv(path: Path, rows: list[dict[str, object]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="output/final_neu3d_component_ablations")
    args = parser.parse_args()

    root = Path(args.root)
    rows: list[dict[str, object]] = []
    for variant in VARIANTS:
        for scene in SCENES:
            run = latest_done(root, variant, scene)
            if run is None:
                rows.append(
                    {
                        "variant": variant,
                        "scene": scene,
                        "status": "missing",
                        "run_dir": "",
                        "miou_best": "",
                        "miou_greedy": "",
                        "macc_best": "",
                        "macc_greedy": "",
                    }
                )
                continue
            rows.append(
                {
                    "variant": variant,
                    "scene": scene,
                    "status": "done",
                    "run_dir": str(run),
                    "miou_best": load_metric(run, "miou_results.json"),
                    "miou_greedy": load_metric(run, "miou_results_greedy.json"),
                    "macc_best": load_metric(run, "macc_results.json"),
                    "macc_greedy": load_metric(run, "macc_results_greedy.json"),
                }
            )

    columns = [
        "variant",
        "scene",
        "status",
        "miou_best",
        "miou_greedy",
        "macc_best",
        "macc_greedy",
        "run_dir",
    ]
    write_csv(root / "component_ablation_per_scene.csv", rows, columns)

    summary_rows = []
    for variant in VARIANTS:
        done = [row for row in rows if row["variant"] == variant and row["status"] == "done"]
        summary_rows.append(
            {
                "variant": variant,
                "num_done": len(done),
                "mean_miou_best": np.mean([float(row["miou_best"]) for row in done]) if done else "",
                "mean_miou_greedy": np.mean([float(row["miou_greedy"]) for row in done]) if done else "",
                "mean_macc_best": np.mean([float(row["macc_best"]) for row in done]) if done else "",
                "mean_macc_greedy": np.mean([float(row["macc_greedy"]) for row in done]) if done else "",
            }
        )
    write_csv(
        root / "component_ablation_summary.csv",
        summary_rows,
        [
            "variant",
            "num_done",
            "mean_miou_best",
            "mean_miou_greedy",
            "mean_macc_best",
            "mean_macc_greedy",
        ],
    )

    default = next((row for row in summary_rows if row["variant"] == "default"), None)
    default_miou = float(default["mean_miou_greedy"]) if default and default["num_done"] else None
    lines = [
        "# Neu3D Component Ablations",
        "",
        "| Variant | Done | Mean mIoU best | Mean mIoU greedy | Delta greedy vs default | Mean mAcc greedy |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        greedy = row["mean_miou_greedy"]
        delta = ""
        if default_miou is not None and greedy != "":
            delta = f"{float(greedy) - default_miou:+.4f}"
        lines.append(
            f"| {row['variant']} | {row['num_done']} | "
            f"{float(row['mean_miou_best']):.4f}" if row["mean_miou_best"] != "" else f"| {row['variant']} | {row['num_done']} | "
        )
        if row["mean_miou_best"] != "":
            lines[-1] += (
                f" | {float(row['mean_miou_greedy']):.4f} | "
                f"{delta} | {float(row['mean_macc_greedy']):.4f} |"
            )
        else:
            lines[-1] += " |  |  |  |"

    (root / "summary.md").write_text("\n".join(lines) + "\n")
    print(f"Wrote {root / 'component_ablation_per_scene.csv'}")
    print(f"Wrote {root / 'component_ablation_summary.csv'}")
    print(f"Wrote {root / 'summary.md'}")


if __name__ == "__main__":
    main()
