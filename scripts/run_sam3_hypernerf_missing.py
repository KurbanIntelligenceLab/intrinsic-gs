#!/usr/bin/env python3
"""Run SAM3 video concept segmentation for HyperNeRF scenes lacking masks."""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
import zipfile
from pathlib import Path

import numpy as np


DEFAULT_SCENES = ("cut-lemon1", "hand1-dense-v2", "oven-mitts", "slice-banana")
DEFAULT_PROMPTS = {
    "cut-lemon1": ["lemon", "knife", "hand", "cutting board", "wine bottle", "countertop"],
    "hand1-dense-v2": ["hand", "arm", "floor", "rug"],
    "oven-mitts": ["oven mitt", "hand", "arm", "shirt", "table"],
    "slice-banana": ["banana", "banana slicer", "hand", "cutting board", "towel"],
}


def scene_rgb_dir(data_root: Path, scene: str) -> Path:
    for family in ("misc", "interp"):
        rgb = data_root / family / scene / "rgb" / "2x"
        if rgb.exists():
            return rgb
    raise FileNotFoundError(f"No HyperNeRF rgb/2x directory for {scene}")


def frame_count(rgb_dir: Path) -> int:
    return len([p for p in rgb_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])


def as_numpy(value):
    if value is None:
        return None
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def metadata_for_frame(frame_idx: int, outputs: dict) -> dict:
    obj_ids = as_numpy(outputs.get("out_obj_ids"))
    scores = as_numpy(outputs.get("out_probs"))
    boxes = as_numpy(outputs.get("out_boxes_xywh"))
    if obj_ids is None:
        obj_ids = np.arange(0, len(scores) if scores is not None else 0)
    if scores is None:
        scores = np.zeros((len(obj_ids),), dtype=np.float32)
    if boxes is None:
        boxes = np.zeros((len(obj_ids), 4), dtype=np.float32)
    return {
        "frame_index": int(frame_idx),
        "num_objects": int(len(obj_ids)),
        "object_ids": [int(x) for x in obj_ids.tolist()],
        "scores": [float(x) for x in scores.tolist()],
        "boxes": [[float(v) for v in row] for row in boxes.tolist()],
    }


def masks_for_output(outputs: dict) -> np.ndarray:
    masks = as_numpy(outputs.get("out_binary_masks"))
    if masks is None:
        raise RuntimeError("SAM3 output did not include out_binary_masks")
    masks = np.asarray(masks).astype(bool)
    if masks.ndim == 2:
        masks = masks[None, ...]
    if masks.ndim != 3:
        raise RuntimeError(f"Unexpected mask shape: {masks.shape}")
    return masks


def write_npz(path: Path, frame_masks: dict[int, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {f"frame_{idx:06d}": frame_masks[idx] for idx in sorted(frame_masks)}
    np.savez_compressed(path, **arrays)


def slug(text: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in text.lower()).strip("_")


def dedupe_concat(arrays: list[np.ndarray]) -> np.ndarray:
    original = arrays
    arrays = [a for a in arrays if a.size and a.shape[0] > 0]
    if not arrays:
        if original:
            h, w = original[0].shape[-2:]
            return np.zeros((0, h, w), dtype=bool)
        return np.zeros((0, 0, 0), dtype=bool)
    out = []
    seen = set()
    for arr in arrays:
        for mask in arr.astype(bool):
            key = np.packbits(mask.reshape(-1)).tobytes()
            if key in seen:
                continue
            seen.add(key)
            out.append(mask)
    if not out:
        h, w = arrays[0].shape[-2:]
        return np.zeros((0, h, w), dtype=bool)
    return np.stack(out, axis=0)


def zip_output(root: Path, zip_path: Path) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for path in sorted(root.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(root.parent))


def run_prompt(predictor, session_id: str, scene_out: Path, prompt: str, n_frames: int, args) -> tuple[dict, dict[int, np.ndarray]]:
    prompt_slug = slug(prompt)
    timings: dict[str, float] = {}
    frame_masks: dict[int, np.ndarray] = {}
    frames_meta: dict[str, dict] = {}

    t_prompt = time.perf_counter()
    t0 = time.perf_counter()
    predictor.handle_request({"type": "reset_session", "session_id": session_id})
    timings["reset_session_sec"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    response = predictor.handle_request(
        {
            "type": "add_prompt",
            "session_id": session_id,
            "frame_index": args.prompt_frame,
            "text": prompt,
            "output_prob_thresh": args.output_prob_thresh,
        }
    )
    timings["add_prompt_sec"] = time.perf_counter() - t0
    frame_idx = int(response["frame_index"])
    frame_masks[frame_idx] = masks_for_output(response["outputs"])
    frames_meta[str(frame_idx)] = metadata_for_frame(frame_idx, response["outputs"])

    t0 = time.perf_counter()
    for response in predictor.handle_stream_request(
        {
            "type": "propagate_in_video",
            "session_id": session_id,
            "propagation_direction": args.propagation_direction,
            "start_frame_index": args.prompt_frame,
            "output_prob_thresh": args.output_prob_thresh,
        }
    ):
        frame_idx = int(response["frame_index"])
        frame_masks[frame_idx] = masks_for_output(response["outputs"])
        frames_meta[str(frame_idx)] = metadata_for_frame(frame_idx, response["outputs"])
    timings["propagate_sec"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    write_npz(scene_out / f"masks_{prompt_slug}.npz", frame_masks)
    metadata = {
        "text_prompt": prompt,
        "num_frames": n_frames,
        "num_output_frames": len(frame_masks),
        "frames": frames_meta,
    }
    (scene_out / f"metadata_{prompt_slug}.json").write_text(json.dumps(metadata, indent=2))
    timings["save_prompt_sec"] = time.perf_counter() - t0
    timings["prompt_total_sec"] = time.perf_counter() - t_prompt

    mask_counts = [m.shape[0] for m in frame_masks.values()]
    row = {
        "prompt": prompt,
        "prompt_slug": prompt_slug,
        "output_frames": len(frame_masks),
        "mean_masks_per_frame": float(np.mean(mask_counts)) if mask_counts else 0.0,
        "max_masks_per_frame": int(max(mask_counts)) if mask_counts else 0,
        **timings,
    }
    row["sec_per_output_frame"] = row["propagate_sec"] / max(row["output_frames"], 1)
    return row, frame_masks


def run_scene(predictor, scene: str, rgb_dir: Path, out_root: Path, prompts: list[str], args) -> tuple[dict, list[dict]]:
    scene_out = out_root / scene
    scene_out.mkdir(parents=True, exist_ok=True)
    n_frames = frame_count(rgb_dir)
    t_scene = time.perf_counter()
    timings: dict[str, float] = {}

    t0 = time.perf_counter()
    response = predictor.handle_request(
        {
            "type": "start_session",
            "resource_path": str(rgb_dir),
            "offload_video_to_cpu": args.offload_video_to_cpu,
            "offload_state_to_cpu": args.offload_state_to_cpu,
        }
    )
    timings["start_session_sec"] = time.perf_counter() - t0
    session_id = response["session_id"]
    scene_prompt_rows = []
    prompt_outputs: list[dict[int, np.ndarray]] = []

    try:
        for prompt in prompts:
            print(f"[SAM3]   prompt={prompt!r}", flush=True)
            prompt_row, frame_masks = run_prompt(predictor, session_id, scene_out, prompt, n_frames, args)
            prompt_row.update(scene=scene, rgb_dir=str(rgb_dir), frames=n_frames)
            scene_prompt_rows.append(prompt_row)
            prompt_outputs.append(frame_masks)
    finally:
        t0 = time.perf_counter()
        predictor.handle_request({"type": "close_session", "session_id": session_id})
        close_session_sec = time.perf_counter() - t0

    t0 = time.perf_counter()
    aggregate: dict[int, np.ndarray] = {}
    for idx in range(n_frames):
        parts = [out[idx] for out in prompt_outputs if idx in out]
        if parts:
            aggregate[idx] = dedupe_concat(parts)
        else:
            aggregate[idx] = np.zeros((0, 0, 0), dtype=bool)
    write_npz(scene_out / "masks.npz", aggregate)
    metadata = {
        "text_prompt": " + ".join(prompts),
        "text_prompts": prompts,
        "num_frames": n_frames,
        "num_output_frames": len(aggregate),
        "source_rgb_dir": str(rgb_dir),
        "prompt_metadata_files": [f"metadata_{slug(prompt)}.json" for prompt in prompts],
    }
    (scene_out / "metadata.json").write_text(json.dumps(metadata, indent=2))
    save_aggregate_sec = time.perf_counter() - t0
    total_sec = time.perf_counter() - t_scene

    mask_counts = [m.shape[0] for m in aggregate.values()]
    row = {
        "scene": scene,
        "prompt": " + ".join(prompts),
        "rgb_dir": str(rgb_dir),
        "frames": n_frames,
        "output_frames": len(aggregate),
        "mean_masks_per_frame": float(np.mean(mask_counts)) if mask_counts else 0.0,
        "max_masks_per_frame": int(max(mask_counts)) if mask_counts else 0,
        "start_session_sec": timings["start_session_sec"],
        "close_session_sec": close_session_sec,
        "save_aggregate_sec": save_aggregate_sec,
        "total_sec": total_sec,
    }
    row["prompt_count"] = len(prompts)
    (scene_out / "timing.json").write_text(json.dumps(row, indent=2))
    return row, scene_prompt_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data/HyperNeRF")
    parser.add_argument("--out-root", default="data/sam3_hypernerf_missing")
    parser.add_argument("--zip-out", default="/workspace/sam3_masks_hypernerf_missing.zip")
    parser.add_argument("--scenes", nargs="*", default=list(DEFAULT_SCENES))
    parser.add_argument("--prompt", default="object")
    parser.add_argument(
        "--prompts-json",
        default="",
        help="Optional JSON dict mapping scene names to prompt lists. Defaults to built-in HyperNeRF prompts.",
    )
    parser.add_argument("--prompt-frame", type=int, default=0)
    parser.add_argument("--output-prob-thresh", type=float, default=0.5)
    parser.add_argument("--propagation-direction", choices=("forward", "backward", "both"), default="forward")
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--offload-video-to-cpu", action="store_true")
    parser.add_argument("--offload-state-to-cpu", action="store_true")
    parser.add_argument("--no-zip", action="store_true")
    args = parser.parse_args()

    from sam3.model_builder import build_sam3_video_predictor

    gpus = [int(x) for x in args.gpus.split(",") if x.strip()]
    t0 = time.perf_counter()
    predictor = build_sam3_video_predictor(gpus_to_use=gpus)
    model_load_sec = time.perf_counter() - t0

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    data_root = Path(args.data_root)
    rows = []
    prompt_rows = []
    prompts_by_scene = DEFAULT_PROMPTS
    if args.prompts_json:
        prompts_by_scene = json.loads(Path(args.prompts_json).read_text())
    for scene in args.scenes:
        rgb_dir = scene_rgb_dir(data_root, scene)
        prompts = list(prompts_by_scene.get(scene, [args.prompt]))
        print(f"[SAM3] scene={scene} frames={frame_count(rgb_dir)} prompts={prompts}", flush=True)
        row, scene_prompt_rows = run_scene(predictor, scene, rgb_dir, out_root, prompts, args)
        row["model_load_sec"] = model_load_sec
        rows.append(row)
        prompt_rows.extend(scene_prompt_rows)
        print(
            f"[SAM3] done scene={scene} total={row['total_sec']:.1f}s "
            f"prompts={row['prompt_count']} masks/frame={row['mean_masks_per_frame']:.2f}",
            flush=True,
        )

    timing_csv = out_root / "sam3_timing.csv"
    if rows:
        cols = list(rows[0].keys())
        with timing_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=cols)
            writer.writeheader()
            writer.writerows(rows)
    prompt_timing_csv = out_root / "sam3_prompt_timing.csv"
    if prompt_rows:
        cols = list(prompt_rows[0].keys())
        with prompt_timing_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=cols)
            writer.writeheader()
            writer.writerows(prompt_rows)
    summary = {
        "scenes": args.scenes,
        "prompts_by_scene": {scene: prompts_by_scene.get(scene, [args.prompt]) for scene in args.scenes},
        "gpus": gpus,
        "model_load_sec": model_load_sec,
        "rows": rows,
        "prompt_rows": prompt_rows,
    }
    (out_root / "sam3_run_summary.json").write_text(json.dumps(summary, indent=2))

    if not args.no_zip:
        zip_output(out_root, Path(args.zip_out))
        print(f"[SAM3] wrote zip {args.zip_out}", flush=True)
    print(f"[SAM3] wrote timing {timing_csv}", flush=True)


if __name__ == "__main__":
    main()
