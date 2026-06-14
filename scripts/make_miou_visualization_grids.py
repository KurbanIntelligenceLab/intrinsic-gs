#!/usr/bin/env python3
"""Build two 2x5 matplotlib figures: mIoU (best cluster) and g_mIoU (greedy), as separate PNGs."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
OUT_ROOT = REPO / "outputs"
RUN_PREFIX = "spectral_k14_sc0.8_ss1.0_p2.0_k20_mot20_B-v12a5.0b2.0g2.0_"
PREFER_RUN_SUFFIX = {"americano": "05-43"}
PATH_MIOU_GRID = OUT_ROOT / "default_config_miou_grid.png"
PATH_GREEDY_GRID = OUT_ROOT / "default_config_greedy_miou_grid.png"


def is_ablation(p: Path) -> bool:
    return any("ablation" in part for part in p.parts)


def greedy_miou_path(run_dir: Path) -> Path | None:
    for name in ("miou_results_greedy.json", "miou_results_greedy_union.json"):
        p = run_dir / name
        if p.is_file():
            return p
    return None


def greedy_vis_dir(run_dir: Path) -> Path:
    for name in ("visualizations_greedy_union", "visualizations_greedy"):
        p = run_dir / name
        if p.is_dir() and any(p.glob("*.png")):
            return p
    return run_dir / "visualizations"


def first_png(folder: Path) -> Path:
    files = sorted(folder.glob("*.png"))
    if not files:
        raise FileNotFoundError(f"No PNG in {folder}")
    return files[0]


def pick_run_dir(scene_dir: Path) -> Path:
    candidates: list[Path] = []
    for p in scene_dir.rglob("miou_results.json"):
        run_dir = p.parent
        if is_ablation(run_dir):
            continue
        if run_dir.name.startswith(RUN_PREFIX):
            candidates.append(run_dir.resolve())
    if not candidates:
        raise FileNotFoundError(f"No default run under {scene_dir}")
    candidates = list(dict.fromkeys(candidates))
    if len(candidates) == 1:
        return candidates[0]
    pref = PREFER_RUN_SUFFIX.get(scene_dir.name)
    if pref:
        for c in candidates:
            if c.name.endswith("_" + pref):
                return c
    return sorted(candidates, key=lambda x: x.name)[-1]


def collect_rows() -> list[tuple[str, Path, Path, float, float]]:
    rows: list[tuple[str, Path, Path, float, float]] = []
    for scene_dir in sorted(OUT_ROOT.iterdir()):
        if not scene_dir.is_dir():
            continue
        if scene_dir.name.startswith(".") or scene_dir.name == "evaluation_summary.csv":
            continue
        try:
            run_dir = pick_run_dir(scene_dir)
        except FileNotFoundError:
            continue
        miou_j = json.loads((run_dir / "miou_results.json").read_text())
        m = float(miou_j["results"]["mIoU"])
        gp = greedy_miou_path(run_dir)
        if gp:
            g = float(json.loads(gp.read_text())["results"]["mIoU"])
        else:
            g = m
        vis_best = run_dir / "visualizations"
        vis_g = greedy_vis_dir(run_dir)
        rows.append(
            (
                scene_dir.name,
                first_png(vis_best),
                first_png(vis_g),
                m,
                g,
            )
        )
    rows.sort(key=lambda x: x[0].lower())
    return rows


def save_grid_2x5(
    rows: list[tuple[str, Path, Path, float, float]],
    *,
    image_pick: int,
    value_index: int,
    metric_label: str,
    suptitle: str,
    out_path: Path,
) -> None:
    fig = plt.figure(figsize=(22, 5.2), dpi=150)
    fig.patch.set_facecolor("#fafafa")
    gs = GridSpec(2, 5, figure=fig, hspace=0.28, wspace=0.06)
    fig.suptitle(suptitle, fontsize=12, y=0.98)

    for i in range(10):
        r, c = divmod(i, 5)
        ax = fig.add_subplot(gs[r, c])
        img_path = rows[i][image_pick]
        ax.imshow(Image.open(img_path))
        ax.axis("off")
        ax.set_title(f"{rows[i][0]}\n{metric_label}={rows[i][value_index]:.3f}", fontsize=10)

    fig.subplots_adjust(left=0.015, right=0.985, top=0.88, bottom=0.06)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> None:
    rows = collect_rows()
    if len(rows) != 10:
        raise SystemExit(f"Expected 10 scenes, got {len(rows)}: {[r[0] for r in rows]}")

    save_grid_2x5(
        rows,
        image_pick=1,
        value_index=3,
        metric_label="mIoU",
        suptitle="Default config (spectral k=14) — best cluster — first PNG in visualizations/",
        out_path=PATH_MIOU_GRID,
    )

    save_grid_2x5(
        rows,
        image_pick=2,
        value_index=4,
        metric_label="g_mIoU",
        suptitle=(
            "Default config (spectral k=14) — greedy union — "
            "first PNG in visualizations_greedy / visualizations_greedy_union"
        ),
        out_path=PATH_GREEDY_GRID,
    )

    print(f"Wrote {PATH_MIOU_GRID}")
    print(f"Wrote {PATH_GREEDY_GRID}")


if __name__ == "__main__":
    main()
