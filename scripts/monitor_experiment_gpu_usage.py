#!/usr/bin/env python3
"""Sample GPU usage and attribute CUDA child processes to experiment workers."""

from __future__ import annotations

import argparse
import csv
import subprocess
import time
from pathlib import Path


def run(cmd: list[str]) -> str:
    return subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL)


def parse_ps() -> dict[int, dict[str, object]]:
    rows: dict[int, dict[str, object]] = {}
    out = run(["ps", "-eo", "pid=,ppid=,args="])
    for line in out.splitlines():
        parts = line.strip().split(None, 2)
        if len(parts) < 3:
            continue
        pid, ppid, cmd = parts
        rows[int(pid)] = {"ppid": int(ppid), "cmd": cmd}
    return rows


def parse_gpu_apps() -> list[dict[str, object]]:
    try:
        out = run(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid,used_memory",
                "--format=csv,noheader,nounits",
            ]
        )
    except subprocess.CalledProcessError:
        return []
    rows = []
    for line in out.splitlines():
        if not line.strip():
            continue
        gpu_uuid, pid, used_mb = [x.strip() for x in line.split(",")]
        rows.append({"gpu_uuid": gpu_uuid, "pid": int(pid), "used_memory_mb": float(used_mb)})
    return rows


def parse_gpu_cards() -> list[dict[str, object]]:
    out = run(
        [
            "nvidia-smi",
            "--query-gpu=uuid,index,name,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
    )
    rows = []
    for line in out.splitlines():
        uuid, index, name, used, total, util = [x.strip() for x in line.split(",")]
        rows.append(
            {
                "gpu_uuid": uuid,
                "gpu_index": index,
                "gpu_name": name,
                "gpu_memory_used_mb": float(used),
                "gpu_memory_total_mb": float(total),
                "gpu_utilization_pct": float(util),
            }
        )
    return rows


def find_owner(pid: int, ps_rows: dict[int, dict[str, object]]) -> tuple[str, str, str]:
    cur = pid
    for _ in range(12):
        row = ps_rows.get(cur)
        if not row:
            break
        cmd = str(row["cmd"])
        if "run_neu3d_leiden_scene.py" in cmd or "run_hypernerf_leiden_scene.py" in cmd:
            toks = cmd.split()
            scene = value_after(toks, "--scene") or "unknown_scene"
            variant = value_after(toks, "--variant") or "unknown_variant"
            return scene, variant, cmd
        cur = int(row["ppid"])
    cmd = str(ps_rows.get(pid, {}).get("cmd", ""))
    return "unattributed", "unattributed", cmd


def value_after(tokens: list[str], key: str) -> str | None:
    try:
        idx = tokens.index(key)
    except ValueError:
        return None
    if idx + 1 >= len(tokens):
        return None
    return tokens[idx + 1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--interval", type=float, default=5.0)
    parser.add_argument("--duration", type=float, default=0.0, help="Seconds; 0 means until interrupted.")
    args = parser.parse_args()

    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "timestamp",
        "gpu_index",
        "gpu_name",
        "gpu_utilization_pct",
        "gpu_memory_used_mb",
        "gpu_memory_total_mb",
        "pid",
        "pid_used_memory_mb",
        "scene",
        "variant",
        "owner_cmd",
    ]
    start = time.time()
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        while True:
            now = time.time()
            ps_rows = parse_ps()
            cards = {row["gpu_uuid"]: row for row in parse_gpu_cards()}
            apps = parse_gpu_apps()
            for app in apps:
                card = cards.get(app["gpu_uuid"], {})
                scene, variant, owner_cmd = find_owner(int(app["pid"]), ps_rows)
                writer.writerow(
                    {
                        "timestamp": f"{now:.3f}",
                        "gpu_index": card.get("gpu_index", ""),
                        "gpu_name": card.get("gpu_name", ""),
                        "gpu_utilization_pct": card.get("gpu_utilization_pct", ""),
                        "gpu_memory_used_mb": card.get("gpu_memory_used_mb", ""),
                        "gpu_memory_total_mb": card.get("gpu_memory_total_mb", ""),
                        "pid": app["pid"],
                        "pid_used_memory_mb": app["used_memory_mb"],
                        "scene": scene,
                        "variant": variant,
                        "owner_cmd": owner_cmd,
                    }
                )
            f.flush()
            if args.duration > 0 and now - start >= args.duration:
                break
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
