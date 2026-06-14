#!/usr/bin/env python
"""Render and evaluate an existing Neu3D labels.npy run."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_neu3d_leiden_scene import eval_metrics, prepare_scene  # noqa: E402
from self_supervised_scripts.pipeline_common import run_render  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--dataset_root", default="data/Neu3D")
    parser.add_argument("--iteration", type=int, default=30000)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--eval", action="store_true")
    args = parser.parse_args()

    repo = PROJECT_ROOT
    os.chdir(repo)
    run_dir = Path(args.run_dir)
    if not (run_dir / "labels.npy").exists():
        raise FileNotFoundError(run_dir / "labels.npy")
    if not (run_dir / "report.md").exists():
        raise FileNotFoundError(run_dir / "report.md")

    paths = prepare_scene(args.scene, repo, repo / args.dataset_root)
    if args.render:
        run_render(paths, str(run_dir), args.iteration, use_test_cameras=True)
    if args.eval:
        eval_metrics(paths, str(run_dir), args.iteration)
    print(f"FINISHED_EXISTING_RUN {args.scene} {run_dir}", flush=True)


if __name__ == "__main__":
    main()
