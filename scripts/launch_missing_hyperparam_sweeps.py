#!/usr/bin/env python3
"""Launch missing one-at-a-time hyperparameter sweeps for HyperNeRF and Neu3D."""

from __future__ import annotations

import argparse
import csv
import os
import queue
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

HYPERNERF_SCENES = [
    "chickchicken",
    "cut-lemon1",
    "hand1-dense-v2",
    "slice-banana",
    "torchocolate",
    "americano",
    "espresso",
    "keyboard",
    "oven-mitts",
    "split-cookie",
]
NEU3D_SCENES = [
    "coffee_martini",
    "cook_spinach",
    "cut_roasted_beef",
    "flame_steak",
    "sear_steak",
]


@dataclass(frozen=True)
class Job:
    dataset: str
    scene: str
    sweep: str
    value: str
    gpu: int
    cmd: list[str]
    log: Path


def value_tag(value: object) -> str:
    text = str(value).replace("-", "m").replace(".", "p")
    return text.rstrip("0").rstrip("p") if "p" in text else text


def hyper_cmd(py: str, scene: str, sweep: str, value: object) -> list[str]:
    cmd = [
        py,
        "scripts/run_hypernerf_leiden_scene.py",
        "--scene",
        scene,
        "--iteration",
        "30000",
        "--clusterer",
        "leiden",
        "--leiden_resolution",
        "0.03",
        "--use_motion",
        "--use_boundary",
    ]
    if sweep == "rho":
        cmd[cmd.index("--leiden_resolution") + 1] = str(value)
    elif sweep == "knn_k":
        cmd.extend(["--knn_k", str(value)])
    elif sweep == "power":
        cmd.extend(["--power", str(value)])
    elif sweep == "n_time_steps":
        cmd.extend(["--n_time_steps", str(value)])
    elif sweep == "boundary_views":
        cmd.extend(["--boundary_views", str(value)])
    elif sweep == "sigma_color":
        cmd.extend(["--sigma_color", str(value)])
    elif sweep == "sigma_scale":
        cmd.extend(["--sigma_scale", str(value)])
    elif sweep == "motion_floor":
        cmd.extend(["--motion_floor", str(value)])
    else:
        raise ValueError(sweep)
    return cmd


def neu_cmd(py: str, scene: str, sweep: str, value: object, output_root: str) -> list[str]:
    variant = f"sensitivity_{sweep}_{value_tag(value)}"
    cmd = [
        py,
        "scripts/run_neu3d_leiden_scene.py",
        "--scene",
        scene,
        "--variant",
        variant,
        "--iteration",
        "30000",
        "--leiden_resolution",
        "0.018",
        "--output_root",
        output_root,
        "--use_motion",
        "--use_boundary",
        "--skip_done",
    ]
    if sweep == "rho":
        cmd[cmd.index("--leiden_resolution") + 1] = str(value)
    elif sweep == "knn_k":
        cmd.extend(["--knn_k", str(value)])
    elif sweep == "power":
        cmd.extend(["--power", str(value)])
    elif sweep == "n_time_steps":
        cmd.extend(["--n_time_steps", str(value)])
    elif sweep == "boundary_views":
        cmd.extend(["--boundary_views", str(value)])
    elif sweep == "sigma_color":
        cmd.extend(["--sigma_color", str(value)])
    elif sweep == "sigma_scale":
        cmd.extend(["--sigma_scale", str(value)])
    elif sweep == "motion_floor":
        cmd.extend(["--motion_floor", str(value)])
    else:
        raise ValueError(sweep)
    return cmd


def build_jobs(py: str, log_root: Path, output_root: str) -> list[Job]:
    sweeps = [
        ("rho", [0.018, 0.05, 0.08], [0.03, 0.05]),
        ("knn_k", [10, 30, 40], [10, 30, 40]),
        ("power", [1.0, 3.0], [1.0, 3.0]),
        ("n_time_steps", [10, 40], [10, 40]),
        ("boundary_views", [6, 24], [6, 24]),
        ("sigma_color", [0.5, 1.1], [0.5, 1.1]),
        ("sigma_scale", [0.5, 1.5], [0.5, 1.5]),
        ("motion_floor", [0.0, 0.5], [0.0, 0.5]),
    ]
    jobs: list[Job] = []
    next_gpu = 0
    for sweep, hyper_values, neu_values in sweeps:
        for value in hyper_values:
            for scene in HYPERNERF_SCENES:
                tag = value_tag(value)
                log = log_root / "hypernerf" / sweep / f"{scene}_{tag}.log"
                jobs.append(Job("hypernerf", scene, sweep, str(value), next_gpu, hyper_cmd(py, scene, sweep, value), log))
                next_gpu = 1 - next_gpu
        for value in neu_values:
            for scene in NEU3D_SCENES:
                tag = value_tag(value)
                log = log_root / "neu3d" / sweep / f"{scene}_{tag}.log"
                jobs.append(Job("neu3d", scene, sweep, str(value), next_gpu, neu_cmd(py, scene, sweep, value, output_root), log))
                next_gpu = 1 - next_gpu
    return jobs


def write_manifest(jobs: list[Job], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["dataset", "scene", "sweep", "value", "gpu", "log", "cmd"],
        )
        writer.writeheader()
        for job in jobs:
            writer.writerow(
                {
                    "dataset": job.dataset,
                    "scene": job.scene,
                    "sweep": job.sweep,
                    "value": job.value,
                    "gpu": job.gpu,
                    "log": str(job.log.relative_to(ROOT)),
                    "cmd": " ".join(job.cmd),
                }
            )


def worker(worker_id: int, jobs: "queue.Queue[Job]", status_path: Path) -> None:
    while True:
        try:
            job = jobs.get_nowait()
        except queue.Empty:
            return
        job.log.parent.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(job.gpu)
        env["PATH"] = f"/workspace/.envs/trase/bin:{env.get('PATH', '')}"
        start = time.time()
        with job.log.open("w") as f:
            f.write(
                f"START worker={worker_id} gpu={job.gpu} dataset={job.dataset} "
                f"scene={job.scene} sweep={job.sweep} value={job.value}\n"
            )
            f.write("CMD " + " ".join(job.cmd) + "\n")
            f.flush()
            proc = subprocess.run(job.cmd, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT)
            f.write(f"\nEXIT status={proc.returncode} elapsed_s={time.time() - start:.1f}\n")
        with status_path.open("a") as f:
            f.write(
                f"{time.strftime('%Y-%m-%d %H:%M:%S')},{worker_id},{job.gpu},"
                f"{job.dataset},{job.scene},{job.sweep},{job.value},{proc.returncode},"
                f"{time.time() - start:.1f},{job.log.relative_to(ROOT)}\n"
            )
        jobs.task_done()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", default="/workspace/.envs/trase/bin/python")
    parser.add_argument("--log-root", default="output/missing_hyperparam_sweeps/logs")
    parser.add_argument("--output-root", default="output/missing_hyperparam_sweeps/neu3d")
    parser.add_argument("--per-gpu", type=int, default=2)
    args = parser.parse_args()

    log_root = ROOT / args.log_root
    jobs = build_jobs(args.python, log_root, args.output_root)
    manifest = ROOT / "output/missing_hyperparam_sweeps/manifest.csv"
    status = ROOT / "output/missing_hyperparam_sweeps/status.csv"
    write_manifest(jobs, manifest)
    status.parent.mkdir(parents=True, exist_ok=True)
    status.write_text("timestamp,worker,gpu,dataset,scene,sweep,value,status,elapsed_s,log\n")

    q: queue.Queue[Job] = queue.Queue()
    for job in jobs:
        q.put(job)
    threads = []
    for worker_id in range(args.per_gpu * 2):
        thread = threading.Thread(target=worker, args=(worker_id, q, status), daemon=False)
        thread.start()
        threads.append(thread)
    print(f"Launched {len(jobs)} jobs with {len(threads)} workers")
    print(f"Manifest: {manifest}")
    print(f"Status: {status}")
    for thread in threads:
        thread.join()
    print("All sweep jobs finished")


if __name__ == "__main__":
    main()
