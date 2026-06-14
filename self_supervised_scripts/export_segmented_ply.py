"""
Build a segmented .ply from a Stage-1 Gaussian checkpoint and labels.npy.

Adds / refreshes:
  - cls (float32): cluster id per vertex (0 = filtered / background in pipeline)
  - red, green, blue (uint8): RGB from the same palette as spectral_cluster.py
  - By default, f_dc_0..2 and f_rest_* are overwritten so viewers that color by
    3D Gaussian SH (macOS Quick Look, many .splat-style previews) show segments.
    Pass --keep_original_sh to leave appearance SH unchanged (then use MeshLab
    and color by vertex RGB).

Does not run clustering — only merges existing labels into a PLY you can inspect.

Usage (repo root):

    python self_supervised_scripts/export_segmented_ply.py \\
        --ply_in output/americano_run/point_cloud/iteration_20000/point_cloud.ply \\
        --labels_file outputs/americano/ablation_leiden20/.../labels.npy \\
        --output outputs/americano/segmented_viewer.ply

    python self_supervised_scripts/export_segmented_ply.py \\
        --model_path output/americano_run \\
        --load_iteration 20000 \\
        --labels_file path/to/labels.npy \\
        --output segmented.ply

    # Smaller file: only x,y,z + normals + cls + RGB (drops f_dc, SH, feats, etc.)
    python self_supervised_scripts/export_segmented_ply.py \\
        --ply_in .../point_cloud.ply \\
        --labels_file .../labels.npy \\
        --output lite.ply \\
        --minimal
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
from plyfile import PlyData, PlyElement

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# Same colors as self_supervised_scripts/spectral_cluster.py (index 0 = invalid / black)
_PALETTE_FLOAT = np.array(
    [
        [0, 0, 0],
        [230, 25, 75],
        [60, 180, 75],
        [67, 99, 216],
        [255, 225, 25],
        [245, 130, 49],
        [145, 30, 180],
        [66, 212, 244],
        [240, 50, 230],
        [188, 246, 12],
        [250, 190, 212],
        [0, 128, 128],
        [220, 190, 255],
        [154, 99, 36],
        [255, 250, 200],
        [128, 0, 0],
        [170, 255, 195],
    ],
    dtype=np.float32,
)
PALETTE_U8 = np.clip(_PALETTE_FLOAT, 0, 255).astype(np.uint8)

_STRIP_NAMES = frozenset(
    {"cls", "red", "green", "blue", "seg_r", "seg_g", "seg_b"}
)

# 3DGS SH DC convention: RGB2SH from utils/sh_utils.py
_C0 = 0.28209479177387814


def _rgb01_to_f_dc(rgb: np.ndarray) -> np.ndarray:
    """rgb: (N, 3) float in [0, 1] → (N, 3) f_dc coefficients."""
    return (rgb.astype(np.float32) - 0.5) / _C0


def _labels_to_rgb(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """labels: int array, shape [N]. Returns red, green, blue uint8 each [N]."""
    li = labels.astype(np.int64)
    idx = li % PALETTE_U8.shape[0]
    rgb = PALETTE_U8[idx]
    return rgb[:, 0], rgb[:, 1], rgb[:, 2]


def export_full(
    ply_in: str,
    labels: np.ndarray,
    ply_out: str,
    *,
    keep_original_sh: bool,
) -> None:
    plydata = PlyData.read(ply_in)
    vertex = plydata.elements[0]
    data = vertex.data
    n = len(data)
    if labels.shape != (n,):
        raise ValueError(
            f"labels.npy has shape {labels.shape}, expected ({n},) to match PLY vertices"
        )

    old_names = data.dtype.names
    if not old_names:
        raise ValueError("PLY vertex element has no fields")
    base_names = [n for n in old_names if n not in _STRIP_NAMES]

    red, green, blue = _labels_to_rgb(labels)
    new_dtype = [(n, data.dtype[n]) for n in base_names]
    new_dtype += [
        ("cls", "f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
    ]

    new_data = np.empty(n, dtype=new_dtype)
    for name in base_names:
        new_data[name] = data[name]
    new_data["cls"] = labels.astype(np.float32)
    new_data["red"] = red
    new_data["green"] = green
    new_data["blue"] = blue

    if keep_original_sh:
        sh_note = ", original SH kept (--keep_original_sh)"
    elif all(f"f_dc_{i}" in new_data.dtype.names for i in range(3)):
        li = labels.astype(np.int64) % PALETTE_U8.shape[0]
        rgb01 = PALETTE_U8[li].astype(np.float32) / 255.0
        sh = _rgb01_to_f_dc(rgb01)
        new_data["f_dc_0"] = sh[:, 0]
        new_data["f_dc_1"] = sh[:, 1]
        new_data["f_dc_2"] = sh[:, 2]
        for name in new_data.dtype.names:
            if name.startswith("f_rest_"):
                new_data[name] = 0.0
        sh_note = ", SH DC baked for preview"
    else:
        sh_note = " (warn: f_dc_* missing, could not bake SH)"

    out_el = PlyElement.describe(new_data, "vertex")
    os.makedirs(os.path.dirname(os.path.abspath(ply_out)) or ".", exist_ok=True)
    PlyData([out_el], text=False).write(ply_out)
    print(
        f"Wrote {ply_out} ({n:,} vertices, full attributes + cls + RGB{sh_note})"
    )


def export_minimal(ply_in: str, labels: np.ndarray, ply_out: str) -> None:
    plydata = PlyData.read(ply_in)
    data = plydata.elements[0].data
    n = len(data)
    if labels.shape != (n,):
        raise ValueError(
            f"labels.npy has shape {labels.shape}, expected ({n},) to match PLY vertices"
        )

    x = np.asarray(data["x"], dtype=np.float32)
    y = np.asarray(data["y"], dtype=np.float32)
    z = np.asarray(data["z"], dtype=np.float32)
    red, green, blue = _labels_to_rgb(labels)

    new_dtype = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("nx", "f4"),
        ("ny", "f4"),
        ("nz", "f4"),
        ("cls", "f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
    ]
    new_data = np.empty(n, dtype=new_dtype)
    new_data["x"] = x
    new_data["y"] = y
    new_data["z"] = z
    new_data["nx"] = 0.0
    new_data["ny"] = 0.0
    new_data["nz"] = 0.0
    new_data["cls"] = labels.astype(np.float32)
    new_data["red"] = red
    new_data["green"] = green
    new_data["blue"] = blue

    out_el = PlyElement.describe(new_data, "vertex")
    os.makedirs(os.path.dirname(os.path.abspath(ply_out)) or ".", exist_ok=True)
    PlyData([out_el], text=False).write(ply_out)
    print(f"Wrote {ply_out} ({n:,} vertices, minimal xyz + cls + RGB)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--ply_in",
        type=str,
        help="Path to source point_cloud.ply (e.g. .../iteration_20000/point_cloud.ply)",
    )
    src.add_argument(
        "--model_path",
        type=str,
        help="Model dir; uses <model_path>/point_cloud/iteration_<iter>/point_cloud.ply",
    )
    parser.add_argument("--load_iteration", type=int, default=20000)
    parser.add_argument("--labels_file", type=str, required=True)
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output .ply path",
    )
    parser.add_argument(
        "--minimal",
        action="store_true",
        help="Only x,y,z + zero normals + cls + RGB (much smaller; good for MeshLab)",
    )
    parser.add_argument(
        "--keep_original_sh",
        action="store_true",
        help="Do not overwrite f_dc / f_rest (preview stays RGB-like; use MeshLab vertex RGB)",
    )
    args = parser.parse_args()

    if args.model_path:
        ply_in = os.path.join(
            args.model_path,
            "point_cloud",
            f"iteration_{args.load_iteration}",
            "point_cloud.ply",
        )
    else:
        ply_in = args.ply_in

    if not os.path.isfile(ply_in):
        raise SystemExit(f"Missing PLY: {ply_in}")
    if not os.path.isfile(args.labels_file):
        raise SystemExit(f"Missing labels: {args.labels_file}")

    labels = np.load(args.labels_file)
    if labels.ndim != 1:
        labels = labels.reshape(-1)

    if args.minimal:
        export_minimal(ply_in, labels, args.output)
    else:
        export_full(
            ply_in,
            labels,
            args.output,
            keep_original_sh=args.keep_original_sh,
        )


if __name__ == "__main__":
    main()
