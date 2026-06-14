#!/usr/bin/env python3
"""Convert SAM3 aggregate NPZ masks into TRASE training .pt masks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


SCENES = ("slice-banana", "cut-lemon1", "hand1-dense-v2", "oven-mitts")


def scene_dir(data_root: Path, scene: str) -> Path:
    for split in ("misc", "interp"):
        path = data_root / split / scene
        if path.exists():
            return path
    raise FileNotFoundError(f"Could not find HyperNeRF scene: {scene}")


def normalize_masks(masks: np.ndarray) -> np.ndarray:
    masks = masks.astype(bool, copy=False)
    if masks.ndim == 2:
        masks = masks[None, ...]
    if masks.ndim != 3:
        raise ValueError(f"Unexpected mask shape: {masks.shape}")
    return np.ascontiguousarray(masks)


def convert_scene(mask_root: Path, data_root: Path, scene: str, overwrite: bool) -> int:
    src = mask_root / scene / "masks.npz"
    if not src.exists():
        raise FileNotFoundError(src)

    root = scene_dir(data_root, scene)
    ids = json.loads((root / "dataset.json").read_text())["ids"]
    masks_dir = root / "masks"
    masks_dir.mkdir(exist_ok=True)

    npz = np.load(src, allow_pickle=False)
    written = 0
    for idx, image_id in enumerate(tqdm(ids, desc=f"{scene} masks")):
        candidates = (f"frame_{idx:06d}", f"frame_{idx + 1:06d}", f"frame_{idx - 1:06d}")
        key = next((name for name in candidates if name in npz.files), None)
        if key is None:
            continue

        out_path = masks_dir / f"{image_id}.pt"
        if out_path.exists() and not overwrite:
            continue

        tensor = torch.from_numpy(normalize_masks(npz[key]))
        torch.save(
            {
                "masks": tensor,
                "N": int(tensor.shape[0]),
                "H": int(tensor.shape[1]),
                "W": int(tensor.shape[2]),
            },
            out_path,
        )
        written += 1
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mask-root", default="data/sam3_hypernerf_missing")
    parser.add_argument("--data-root", default="data/HyperNeRF")
    parser.add_argument("--scenes", nargs="*", default=list(SCENES))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    total = 0
    for scene in args.scenes:
        n = convert_scene(Path(args.mask_root), Path(args.data_root), scene, args.overwrite)
        print(f"{scene}: wrote {n} .pt masks")
        total += n
    print(f"Total written: {total}")


if __name__ == "__main__":
    main()
