#!/usr/bin/env python3
"""Convert sam_masks_hypernerf.zip NPZ masks into TRASE training .pt files.

TRASE's HyperNeRF loader reads:

    <scene>/masks/<image_name>.pt

and train.py expects torch.load(...) to be a dict with:

    masks: [N, H, W] bool tensor
    N, H, W: shape metadata

The SAM zip stores sequential keys frame_000000, frame_000001, ... while the
HyperNeRF dataset ids are image names 000001, 000002, ... . We map zip frame
position i to dataset.json["ids"][i].
"""

from __future__ import annotations

import argparse
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


SCENES = (
    "americano",
    "chickchicken",
    "espresso",
    "keyboard",
    "split-cookie",
    "torchocolate",
)


def scene_dir(data_root: Path, scene: str) -> Path:
    for split in ("misc", "interp"):
        path = data_root / split / scene
        if path.exists():
            return path
    raise FileNotFoundError(f"Could not find HyperNeRF scene: {scene}")


def convert_scene(zip_file: zipfile.ZipFile, data_root: Path, scene: str, overwrite: bool) -> int:
    src_name = f"sam_masks_hypernerf/{scene}/masks.npz"
    if src_name not in zip_file.namelist():
        raise FileNotFoundError(f"{src_name} missing from zip")

    root = scene_dir(data_root, scene)
    ids = json.loads((root / "dataset.json").read_text())["ids"]
    masks_dir = root / "masks"
    masks_dir.mkdir(exist_ok=True)

    npz = np.load(io.BytesIO(zip_file.read(src_name)), allow_pickle=False)
    written = 0
    for idx, image_id in enumerate(tqdm(ids, desc=f"{scene} masks")):
        key = f"frame_{idx:06d}"
        if key not in npz.files:
            continue
        out_path = masks_dir / f"{image_id}.pt"
        if out_path.exists() and not overwrite:
            continue

        masks = npz[key].astype(bool)
        if masks.ndim == 2:
            masks = masks[None, ...]
        if masks.ndim != 3:
            raise ValueError(f"{src_name}:{key} has unexpected shape {masks.shape}")

        tensor = torch.from_numpy(np.ascontiguousarray(masks))
        payload = {
            "masks": tensor,
            "N": int(tensor.shape[0]),
            "H": int(tensor.shape[1]),
            "W": int(tensor.shape[2]),
        }
        torch.save(payload, out_path)
        written += 1
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", default="/workspace/sam_masks_hypernerf.zip")
    parser.add_argument("--data_root", default="data/HyperNeRF")
    parser.add_argument("--scenes", nargs="*", default=list(SCENES))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    total = 0
    with zipfile.ZipFile(args.zip) as zip_file:
        for scene in args.scenes:
            n = convert_scene(zip_file, data_root, scene, overwrite=args.overwrite)
            print(f"{scene}: wrote {n} .pt masks")
            total += n
    print(f"Total written: {total}")


if __name__ == "__main__":
    main()
