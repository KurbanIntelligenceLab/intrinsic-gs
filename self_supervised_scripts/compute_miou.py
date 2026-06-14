"""
Evaluate a self-supervised cluster-ID segmentation against ground-truth masks.

Reads:
  --pred_dir   per-frame uint8 PNGs (pixel value = cluster ID, 0 = no cluster)
               produced by render_clusters.py with --save_cluster_ids
  --gt_dir     ground-truth masks. Two layouts auto-detected:
                 (a) Flat:   gt_dir/<frame>.png      (single-object binary)
                 (b) Nested: gt_dir/<obj>/<frame>.png (multi-object binary)

Outputs (--output_json):
  K_used, K_auto, rho diagnostics, per-cluster IoU vector, mIoU, per-frame IoU.

Single-object: scene-wide best-matching cluster (k* = argmax_k IoU(cluster_k, gt));
               mIoU = IoU at k*.
Multi-object:  Hungarian matching on the [K x num_objects] IoU matrix;
               mIoU = mean of matched IoUs.

Usage:
    python self_supervised_scripts/compute_miou.py \\
        --pred_dir outputs/spectral_*/cluster_ids_test \\
        --gt_dir   gt_masks \\
        --report_md outputs/spectral_*/report.md \\
        --output_json outputs/spectral_*/miou_results.json
"""

import os
import re
import sys
import json
from argparse import ArgumentParser
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
from PIL import Image

from self_supervised_scripts.timing import TimingRecorder


def parse_frame_index(filename):
    """Extract the frame index from a filename stem.

    Returns the trailing digit run so both HyperNeRF-style stems
    (`00001` -> 1) and Neu3D-style `cam<NN>_<FFFF>` stems
    (`cam00_0000` -> 0, `cam00_0299` -> 299) parse correctly.
    """
    stem = os.path.splitext(os.path.basename(filename))[0]
    matches = re.findall(r"\d+", stem)
    return int(matches[-1]) if matches else None


def index_pngs(directory):
    """Return {frame_index: file_path} for all PNGs directly inside `directory`."""
    out = {}
    for fname in os.listdir(directory):
        if not fname.lower().endswith('.png'):
            continue
        idx = parse_frame_index(fname)
        if idx is not None:
            out[idx] = os.path.join(directory, fname)
    return out


def load_gt_layout(gt_dir):
    """Detect single-object (flat) vs multi-object (subdirs) layout.

    Returns (kind, payload):
      ('single', {frame_idx: path})
      ('multi',  {object_name: {frame_idx: path}})
    """
    entries = sorted(os.listdir(gt_dir))
    subdirs = [e for e in entries if os.path.isdir(os.path.join(gt_dir, e))]
    pngs = [e for e in entries if e.lower().endswith('.png')]

    if subdirs and not pngs:
        per_obj = {}
        for obj in subdirs:
            per_obj[obj] = index_pngs(os.path.join(gt_dir, obj))
        return 'multi', per_obj
    return 'single', index_pngs(gt_dir)


def find_hypernerf_dataset_json(gt_dir):
    """Find the scene dataset.json for root-level or Mask-Benchmark masks."""
    gt_path = Path(gt_dir)
    # Keep both the logical path and the resolved target. `gt_masks` is often a
    # symlink from the HyperNeRF scene into Mask-Benchmark; resolving first
    # loses the neighboring scene dataset.json needed for val-id alignment.
    roots = [gt_path.parent, gt_path.resolve().parent]
    candidates = []
    for root in roots:
        direct = root / "dataset.json"
        if direct.exists():
            candidates.append(direct)
        candidates.extend(sorted(root.glob("*/dataset.json")))
        scene_name = root.name
        for parent in root.parents:
            sibling_scene = parent / scene_name / "dataset.json"
            if sibling_scene.exists():
                candidates.append(sibling_scene)
                break
    # Standard HyperNeRF layout finds the same dataset.json twice (direct + parent traversal).
    seen = set()
    unique = []
    for c in candidates:
        rp = c.resolve()
        if rp not in seen:
            seen.add(rp)
            unique.append(c)
    return unique[0] if len(unique) == 1 else None


def is_hypernerf_mask_benchmark(gt_dir):
    return "HyperNeRF-Mask" in Path(gt_dir).resolve().parts


def remap_gt_paths_to_hypernerf_split(gt_paths, split_ids):
    """Map sequential GT mask indices to original HyperNeRF image ids."""
    gt_items = sorted(gt_paths.items())
    if len(split_ids) < len(gt_items):
        return {}
    return {
        int(split_ids[pos]): path
        for pos, (_, path) in enumerate(gt_items)
    }


def remap_indexed_gt_paths_to_hypernerf_split(gt_paths, split_ids):
    """Map GT mask filename indices to split ids.

    The filename IS the val-split position, regardless of whether numbering
    starts at 0 (americano) or 1 (chickchicken): GT K names the annotation
    for val_ids[K]. This matches TRASE's convention — render.py writes pred
    files as enumerate(getTestCameras()) with shuffle disabled, so pred
    file '{0:05d}'.format(idx) pairs with val_ids[idx], and the published
    GT masks are named to match.
    """
    if not gt_paths:
        return {}
    mapped = {}
    for gt_idx, path in sorted(gt_paths.items()):
        if 0 <= gt_idx < len(split_ids):
            mapped[int(split_ids[gt_idx])] = path
    return mapped


def has_sparse_or_one_based_gt_indices(gt_paths):
    keys = sorted(gt_paths)
    if not keys:
        return False
    dense_zero = keys == list(range(0, len(keys)))
    return not dense_zero


def align_single_object_gt_paths(pred_paths, gt_paths, gt_dir):
    """Return GT paths keyed by the prediction frame ids they should evaluate."""
    is_hypernerf_benchmark = is_hypernerf_mask_benchmark(gt_dir)
    direct_matched = len(set(pred_paths) & set(gt_paths))
    direct_info = {
        'method': 'direct',
        'matched': direct_matched,
        'gt_frames': len(gt_paths),
    }
    if direct_matched == len(gt_paths) and not is_hypernerf_benchmark:
        return gt_paths, direct_info

    dataset_json = find_hypernerf_dataset_json(gt_dir)
    if not dataset_json:
        return gt_paths, direct_info

    with open(dataset_json) as f:
        dataset = json.load(f)
    ids = dataset.get('ids')
    if not ids:
        return gt_paths, direct_info

    # HyperNeRF-Mask annotations are indexed against the validation split used
    # by TRASE's dataset loader: val_img = ids[2::4].
    val_ids = ids[2::4]
    candidates = []
    if has_sparse_or_one_based_gt_indices(gt_paths):
        candidates.append(
            ('hypernerf_val_indexed',
             remap_indexed_gt_paths_to_hypernerf_split(gt_paths, val_ids))
        )
    candidates.append(
        ('hypernerf_val', remap_gt_paths_to_hypernerf_split(gt_paths, val_ids))
    )
    best_paths = gt_paths
    best_info = direct_info
    best_matched = direct_matched
    for method, mapped in candidates:
        matched = len(set(pred_paths) & set(mapped))
        if matched > best_matched:
            best_paths = mapped
            best_matched = matched
            best_info = {
                'method': method,
                'dataset_json': str(dataset_json),
                'original_dir': str(dataset_json.parent / "rgb" / "1x"),
                'matched': matched,
                'gt_frames': len(gt_paths),
            }
        if matched == len(gt_paths):
            break

    return best_paths, best_info


def select_original_dir(cli_original_dir, frame_alignment):
    """Prefer an explicit RGB directory over the dataset.json-derived default."""
    return cli_original_dir or frame_alignment.get('original_dir')


def load_binary_mask(path):
    """Load a binary mask. Handles 1-bit, grayscale, RGB, RGBA — all → bool[H,W]."""
    img = Image.open(path)
    # Convert mode='1' (1-bit) and 'P' (palette) to 'L' first so np.asarray gives uint8.
    if img.mode != 'L' and img.mode != 'RGB' and img.mode != 'RGBA':
        img = img.convert('L')
    arr = np.asarray(img)
    if arr.dtype == bool:
        return arr
    if arr.ndim == 3:
        arr = arr[..., :3].mean(axis=-1)
    return arr > 127


def load_cluster_id_map(path):
    """Load a uint8 cluster-ID PNG. Returns int array of shape [H, W]."""
    arr = np.asarray(Image.open(path))
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr.astype(np.int32)


def parse_report_md(report_path):
    """Pull algorithm-aware diagnostics from a spectral_cluster.py report.md.

    Always tries to read K_used (n_clusters row in the params table). Then
    detects which clusterer produced the report — by section heading — and
    extracts the corresponding diagnostics:

      - kmeans:  K_auto (eigengap suggestion), rho (eigengap decisiveness)
      - leiden:  modularity_q, resolution
      - hdbscan: n_noise, min_cluster_size

    Returns a dict with whatever could be parsed; missing keys absent."""
    out = {}
    if not report_path or not os.path.exists(report_path):
        return out
    text = Path(report_path).read_text()

    m = re.search(r"\|\s*n_clusters\s*\|\s*(\d+)\s*\|", text)
    if m:
        out['K_used'] = int(m.group(1))

    # rgb_edge_method / pidinet_variant rows in the params table (added when
    # use_boundary is on). Rows holding '—' indicate the field doesn't apply
    # to this run and stay absent.
    m = re.search(r"\|\s*rgb_edge_method\s*\|\s*([\w]+)\s*\|", text)
    if m and m.group(1) != '—':
        out['rgb_edge_method'] = m.group(1)
    m = re.search(r"\|\s*pidinet_variant\s*\|\s*([\w]+)\s*\|", text)
    if m and m.group(1) != '—':
        out['pidinet_variant'] = m.group(1)

    # Detect clusterer from algorithm-specific section heading.
    if re.search(r"^##\s*Spectral Analysis\s*$", text, flags=re.MULTILINE):
        out['clusterer'] = 'kmeans'
        m = re.search(r"Eigengap suggested k:\s*\*\*(\d+)\*\*", text)
        if m:
            out['K_auto'] = int(m.group(1))
        m = re.search(r"ρ\s*=.*?\*\*([\d.]+)\*\*", text)
        if m:
            out['rho'] = float(m.group(1))
    elif re.search(r"^##\s*Leiden Community Detection\s*$", text, flags=re.MULTILINE):
        out['clusterer'] = 'leiden'
        m = re.search(r"Modularity Q:\s*\*\*([\d.\-]+)\*\*", text)
        if m:
            out['modularity_q'] = float(m.group(1))
        m = re.search(r"Resolution parameter:\s*\*\*([\d.]+)\*\*", text)
        if m:
            out['resolution'] = float(m.group(1))
    elif re.search(r"^##\s*HDBSCAN", text, flags=re.MULTILINE):
        out['clusterer'] = 'hdbscan'
        m = re.search(r"Noise points reassigned to nearest cluster:\s*\*\*([\d,]+)\*\*", text)
        if m:
            out['n_noise'] = int(m.group(1).replace(',', ''))
        m = re.search(r"min_cluster_size_frac:\s*\*\*([\d.]+)\*\*", text)
        if m:
            out['min_cluster_size_frac'] = float(m.group(1))

    return out


def accumulate_iou_stats(pred_paths, gt_paths, K):
    """For each cluster k in 1..K, sum (intersection, union) across all matched frames.

    Returns:
      stats dict with global and per-frame intersection / prediction / GT counts.
    """
    inter = np.zeros(K + 1, dtype=np.int64)
    pred_count = np.zeros(K + 1, dtype=np.int64)
    gt_count = 0
    per_frame = {}  # frame_idx -> [inter_per_k, pred_count_per_k, gt_count]

    matched_frames = sorted(set(pred_paths) & set(gt_paths))
    for fid in matched_frames:
        pred = load_cluster_id_map(pred_paths[fid])
        gt = load_binary_mask(gt_paths[fid])
        if pred.shape != gt.shape:
            print(f"  WARN: shape mismatch at frame {fid}: "
                  f"pred {pred.shape} vs gt {gt.shape}; skipping")
            continue
        frame_inter = np.zeros(K + 1, dtype=np.int64)
        frame_pred_count = np.zeros(K + 1, dtype=np.int64)
        frame_gt_count = int(gt.sum())
        for k in range(1, K + 1):
            cluster_mask = (pred == k)
            i = int(np.logical_and(cluster_mask, gt).sum())
            p = int(cluster_mask.sum())
            frame_inter[k] = i
            frame_pred_count[k] = p
        inter += frame_inter
        pred_count += frame_pred_count
        gt_count += frame_gt_count
        per_frame[fid] = (frame_inter, frame_pred_count, frame_gt_count)

    return {
        'inter': inter,
        'pred_count': pred_count,
        'gt_count': gt_count,
        'per_frame': per_frame,
        'n_matched': len(matched_frames),
    }


def iou_per_cluster_from_stats(stats, K):
    inter = stats['inter']
    pred_count = stats['pred_count']
    gt_count = stats['gt_count']
    union = pred_count + gt_count - inter
    iou_per_cluster = np.zeros(K + 1, dtype=np.float64)
    for k in range(1, K + 1):
        iou_per_cluster[k] = inter[k] / union[k] if union[k] > 0 else 0.0
    return iou_per_cluster


def accumulate_iou(pred_paths, gt_paths, K):
    stats = accumulate_iou_stats(pred_paths, gt_paths, K)
    iou_per_cluster = iou_per_cluster_from_stats(stats, K)
    per_frame = {}
    for fid, (frame_inter, frame_pred_count, frame_gt_count) in stats['per_frame'].items():
        frame_union = frame_pred_count + frame_gt_count - frame_inter
        per_frame[fid] = (frame_inter, frame_union)
    return iou_per_cluster, per_frame, stats['n_matched']


def compute_selected_clusters_iou_from_stats(stats, selected_clusters):
    selected = np.array(selected_clusters, dtype=np.int32)
    inter = int(stats['inter'][selected].sum())
    pred_count = int(stats['pred_count'][selected].sum())
    union = pred_count + int(stats['gt_count']) - inter
    per_frame_iou = {}
    for fid, (frame_inter, frame_pred_count, frame_gt_count) in stats['per_frame'].items():
        fi = int(frame_inter[selected].sum())
        fp = int(frame_pred_count[selected].sum())
        fu = fp + int(frame_gt_count) - fi
        per_frame_iou[fid] = float(fi / fu) if fu > 0 else 0.0
    return float(inter / union) if union > 0 else 0.0, per_frame_iou


def compute_selected_clusters_iou(pred_paths, gt_paths, selected_clusters):
    """Compute scene-wide and per-frame IoU for a union of cluster IDs."""
    selected = np.array(selected_clusters, dtype=np.int32)
    total_inter = 0
    total_union = 0
    per_frame_iou = {}
    matched_frames = sorted(set(pred_paths) & set(gt_paths))
    for fid in matched_frames:
        pred = load_cluster_id_map(pred_paths[fid])
        gt = load_binary_mask(gt_paths[fid])
        if pred.shape != gt.shape:
            continue
        pred_b = np.isin(pred, selected)
        inter = int(np.logical_and(pred_b, gt).sum())
        union = int(np.logical_or(pred_b, gt).sum())
        total_inter += inter
        total_union += union
        per_frame_iou[fid] = float(inter / union) if union > 0 else 0.0

    return (
        float(total_inter / total_union) if total_union > 0 else 0.0,
        per_frame_iou,
    )


def select_greedy_union_clusters(pred_paths, gt_paths, K, seed_cluster):
    """Greedily add clusters while scene-wide IoU improves."""
    selected = [seed_cluster]
    selected_set = {seed_cluster}
    best_iou, _ = compute_selected_clusters_iou(pred_paths, gt_paths, selected)

    while True:
        best_candidate = None
        best_candidate_iou = best_iou
        for k in range(1, K + 1):
            if k in selected_set:
                continue
            candidate_iou, _ = compute_selected_clusters_iou(
                pred_paths, gt_paths, selected + [k]
            )
            if candidate_iou > best_candidate_iou:
                best_candidate = k
                best_candidate_iou = candidate_iou

        if best_candidate is None:
            break
        selected.append(best_candidate)
        selected_set.add(best_candidate)
        best_iou = best_candidate_iou

    return selected


def select_greedy_union_clusters_from_stats(stats, K, seed_cluster):
    selected = [seed_cluster]
    selected_set = {seed_cluster}
    best_iou, _ = compute_selected_clusters_iou_from_stats(stats, selected)

    while True:
        best_candidate = None
        best_candidate_iou = best_iou
        for k in range(1, K + 1):
            if k in selected_set:
                continue
            candidate_iou, _ = compute_selected_clusters_iou_from_stats(
                stats, selected + [k]
            )
            if candidate_iou > best_candidate_iou:
                best_candidate = k
                best_candidate_iou = candidate_iou

        if best_candidate is None:
            break
        selected.append(best_candidate)
        selected_set.add(best_candidate)
        best_iou = best_candidate_iou

    return selected


def evaluate_single_object(pred_paths, gt_paths, K, selection_mode="best_cluster"):
    """Pick the cluster k* with highest scene-wide IoU; report per-frame IoU at k*."""
    stats = accumulate_iou_stats(pred_paths, gt_paths, K)
    iou_vec = iou_per_cluster_from_stats(stats, K)
    n_matched = stats['n_matched']
    if n_matched == 0:
        return {'mIoU': 0.0, 'matched_frames': 0,
                'iou_per_cluster': iou_vec.tolist(),
                'best_cluster': None, 'selected_clusters': [],
                'selection_mode': selection_mode, 'per_frame_iou': {}}
    k_star = int(np.argmax(iou_vec[1:])) + 1
    if selection_mode == "best_cluster":
        selected_clusters = [k_star]
        per_frame_iou = {}
        for fid, (fi, fp, fg) in stats['per_frame'].items():
            fu = fp[k_star] + fg - fi[k_star]
            per_frame_iou[fid] = float(fi[k_star] / fu) if fu > 0 else 0.0
        miou = float(iou_vec[k_star])
    elif selection_mode == "greedy_union":
        selected_clusters = select_greedy_union_clusters_from_stats(stats, K, k_star)
        miou, per_frame_iou = compute_selected_clusters_iou_from_stats(
            stats, selected_clusters
        )
    else:
        raise ValueError(f"Unknown selection_mode: {selection_mode}")

    return {
        'mIoU': miou,
        'matched_frames': n_matched,
        'iou_per_cluster': iou_vec.tolist(),
        'best_cluster': k_star,
        'selected_clusters': selected_clusters,
        'selection_mode': selection_mode,
        'per_frame_iou': per_frame_iou,
    }


def save_visualizations(vis_dir, pred_paths, gt_paths, best_cluster=None,
                        selected_clusters=None,
                        palette_dir=None, original_dir=None):
    """For each matched frame, save a side-by-side PNG:
       [original? | palette? | GT | pred==k* | overlay].

    Filename encodes IoU so frames sort by quality:
       f00103_iou0.378.png
    """
    os.makedirs(vis_dir, exist_ok=True)
    for old_panel in Path(vis_dir).glob("f*_iou*.png"):
        old_panel.unlink()
    original_paths = (index_pngs(original_dir)
                      if original_dir and os.path.isdir(original_dir) else {})
    palette_paths = (index_pngs(palette_dir)
                     if palette_dir and os.path.isdir(palette_dir) else {})
    if selected_clusters is None:
        selected_clusters = [best_cluster]
    selection_label = ",".join(f"k{k}" for k in selected_clusters)

    matched = sorted(set(pred_paths) & set(gt_paths))
    rows = []
    skipped = 0
    for fid in matched:
        pred = load_cluster_id_map(pred_paths[fid])
        gt = load_binary_mask(gt_paths[fid])
        if pred.shape != gt.shape:
            skipped += 1
            continue
        pred_b = np.isin(pred, np.array(selected_clusters, dtype=np.int32))

        inter = int(np.logical_and(pred_b, gt).sum())
        union = int(np.logical_or(pred_b, gt).sum())
        iou = inter / union if union > 0 else 0.0

        H, W = gt.shape
        panels = []

        # Optional original RGB frame, keyed by the aligned prediction frame id.
        if fid in original_paths:
            orig = np.asarray(Image.open(original_paths[fid]).convert('RGB'))
            if orig.shape[:2] != (H, W):
                orig = np.asarray(
                    Image.fromarray(orig).resize((W, H), Image.BILINEAR))
            panels.append(orig)

        # Optional palette panel — resize to GT shape if needed
        if fid in palette_paths:
            pal = np.asarray(Image.open(palette_paths[fid]).convert('RGB'))
            if pal.shape[:2] != (H, W):
                pal = np.asarray(
                    Image.fromarray(pal).resize((W, H), Image.NEAREST))
            panels.append(pal)

        # GT mask (white)
        gt_rgb = np.zeros((H, W, 3), dtype=np.uint8)
        gt_rgb[gt] = 255
        panels.append(gt_rgb)

        # Predicted best-cluster mask (white)
        pr_rgb = np.zeros((H, W, 3), dtype=np.uint8)
        pr_rgb[pred_b] = 255
        panels.append(pr_rgb)

        # Diagnostic overlay
        ov = np.zeros((H, W, 3), dtype=np.uint8)
        ov[gt & ~pred_b] = [255, 0, 0]      # red — false negative (GT only)
        ov[~gt & pred_b] = [0, 255, 0]      # green — false positive (pred only)
        ov[gt & pred_b]  = [255, 255, 0]    # yellow — true positive
        panels.append(ov)

        canvas = np.hstack(panels)
        out_name = f"f{fid:05d}_iou{iou:.3f}.png"
        Image.fromarray(canvas).save(os.path.join(vis_dir, out_name))
        rows.append((iou, fid, out_name))

    rows.sort(reverse=True)
    with open(os.path.join(vis_dir, '_index.txt'), 'w') as f:
        f.write(
            "# Visualizations sorted by IoU (best first). "
            f"selected_clusters=[{selection_label}]\n"
        )
        columns = []
        if original_paths:
            columns.append("original")
        if palette_paths:
            columns.append("segmented")
        columns.extend(["GT", "pred in selected clusters", "overlay(R=GT-only, G=pred-only, Y=both)"])
        f.write(f"# Columns: [{' | '.join(columns)}]\n")
        for iou, fid, name in rows:
            f.write(f"{iou:.4f}  frame {fid:5d}  {name}\n")

    print(f"Visualizations: wrote {len(rows)} panels to {vis_dir}/"
          f" (skipped {skipped} for shape mismatch)")
    print(f"Index: {os.path.join(vis_dir, '_index.txt')}")


def compute_temporal_consistency(tc_dir):
    """Cluster-label stability across adjacent frames of a fixed-camera sweep.

    Inputs:
      tc_dir: directory of cluster-ID PNGs from render_clusters.py's TC sweep
              (camera fixed, time swept). Filenames must sort temporally.

    For each adjacent frame pair (t, t+1):
      - Take the union of cluster IDs > 0 present in either frame.
      - For each ID k, compute IoU((pred_t == k), (pred_{t+1} == k)).
      - Skip pairs where union==0 (cluster absent in both frames).
      - Average per-cluster IoUs to get the pair's TC.
    Final TC is the mean over all pairs.

    Returns dict with: 'tc' (float), 'n_pairs' (int), 'per_pair_tc' (list[float]).
    A 3D-native pipeline should achieve TC > 0.95 since labels are fixed in 3D
    and only deformation perturbs the projections.
    """
    paths = index_pngs(tc_dir)
    ordered = [paths[i] for i in sorted(paths)]
    if len(ordered) < 2:
        return {'tc': 0.0, 'n_pairs': 0, 'per_pair_tc': []}

    per_pair = []
    for p_a, p_b in zip(ordered, ordered[1:]):
        a = load_cluster_id_map(p_a)
        b = load_cluster_id_map(p_b)
        if a.shape != b.shape:
            continue
        ks = np.unique(np.concatenate([a.ravel(), b.ravel()]))
        ks = ks[ks > 0]
        per_k = []
        for k in ks:
            ak = (a == k)
            bk = (b == k)
            inter = int(np.logical_and(ak, bk).sum())
            union = int(np.logical_or(ak, bk).sum())
            if union > 0:
                per_k.append(inter / union)
        if per_k:
            per_pair.append(float(np.mean(per_k)))

    tc = float(np.mean(per_pair)) if per_pair else 0.0
    return {'tc': tc, 'n_pairs': len(per_pair), 'per_pair_tc': per_pair}


def evaluate_multi_object(pred_paths, gt_per_obj, K):
    """Hungarian matching on [K x num_objects] IoU matrix; mIoU = mean of matches."""
    from scipy.optimize import linear_sum_assignment

    obj_names = sorted(gt_per_obj.keys())
    iou_matrix = np.zeros((K, len(obj_names)), dtype=np.float64)
    matched_per_obj = {}
    for j, obj in enumerate(obj_names):
        iou_vec, _, n = accumulate_iou(pred_paths, gt_per_obj[obj], K)
        iou_matrix[:, j] = iou_vec[1:]  # drop index 0
        matched_per_obj[obj] = n

    # Hungarian maximizes total → use cost = -iou
    row_ind, col_ind = linear_sum_assignment(-iou_matrix)
    matches = []
    iou_values = []
    for r, c in zip(row_ind, col_ind):
        cluster_id = int(r + 1)
        obj = obj_names[c]
        iou = float(iou_matrix[r, c])
        matches.append({'cluster': cluster_id, 'object': obj, 'iou': iou})
        iou_values.append(iou)

    return {
        'mIoU': float(np.mean(iou_values)) if iou_values else 0.0,
        'matched_frames_per_object': matched_per_obj,
        'iou_matrix': iou_matrix.tolist(),
        'matches': matches,
        'objects': obj_names,
    }


def main():
    parser = ArgumentParser(description="Compute mIoU for self-supervised cluster segmentation.")
    parser.add_argument('--pred_dir', required=True, type=str,
                        help="Directory of cluster-ID uint8 PNGs (from render_clusters.py "
                             "with --save_cluster_ids).")
    parser.add_argument('--gt_dir', required=True, type=str,
                        help="Directory of GT masks. Flat for single-object, "
                             "subdir-per-object for multi-object.")
    parser.add_argument('--report_md', type=str, default="",
                        help="Optional path to report.md to extract K_auto / rho diagnostics.")
    parser.add_argument('--output_json', type=str, default="",
                        help="Where to dump JSON results. If empty, prints summary only.")
    parser.add_argument('--vis_dir', type=str, default="",
                        help="If set, save side-by-side PNG panels (per matched frame) "
                             "showing GT, pred==k*, and an R/G/Y overlay. Filenames are "
                             "sorted by IoU.")
    parser.add_argument('--palette_dir', type=str, default="",
                        help="Optional: directory of palette-colored renders to include "
                             "as the first panel of each visualization.")
    parser.add_argument('--original_dir', type=str, default="",
                        help="Optional: directory of original RGB frames to include "
                             "as the first visualization panel.")
    parser.add_argument('--selection_mode', type=str, default="best_cluster",
                        choices=["best_cluster", "greedy_union"],
                        help="Single-object cluster selection. best_cluster keeps the "
                             "strict legacy one-cluster oracle; greedy_union adds "
                             "clusters while scene-wide IoU improves.")
    parser.add_argument('--tc_dir', type=str, default="",
                        help="Optional: directory of cluster-ID PNGs from a "
                             "fixed-camera time sweep (render_clusters.py "
                             "--tc_camera_idx ...). If set, computes temporal "
                             "consistency (TC) and adds it to the JSON.")
    args = parser.parse_args()

    if not os.path.isdir(args.pred_dir):
        sys.exit(f"pred_dir not found: {args.pred_dir}")
    if not os.path.isdir(args.gt_dir):
        sys.exit(f"gt_dir not found: {args.gt_dir}")

    timer = TimingRecorder()
    pred_paths = index_pngs(args.pred_dir)
    gt_kind, gt_payload = load_gt_layout(args.gt_dir)
    print(f"pred_dir: {len(pred_paths)} frames")
    print(f"gt_dir layout: {gt_kind}")

    # Determine K from the prediction PNGs (max cluster ID seen). Scan all
    # frames: late-appearing clusters are common in dynamic scenes, and using a
    # small prefix can silently drop valid cluster IDs from evaluation.
    sample_paths = list(pred_paths.values())
    K_seen = 0
    for p in sample_paths:
        K_seen = max(K_seen, int(load_cluster_id_map(p).max()))
    if K_seen == 0:
        sys.exit("No non-zero cluster IDs found in pred_dir — wrong directory?")
    print(f"K (max cluster ID observed in pred): {K_seen}")

    diagnostics = parse_report_md(args.report_md)
    if diagnostics:
        print(f"From report.md: {diagnostics}")

    if gt_kind == 'single':
        raw_gt_paths = gt_payload
        gt_paths, frame_alignment = align_single_object_gt_paths(
            pred_paths, raw_gt_paths, args.gt_dir)
        overlap_pred = set(pred_paths) & set(gt_paths)
        only_pred = set(pred_paths) - set(gt_paths)
        only_gt = set(gt_paths) - set(pred_paths)
        print(f"Frame alignment: method={frame_alignment['method']}, "
              f"matched={len(overlap_pred)}, "
              f"pred_only={len(only_pred)}, gt_only={len(only_gt)}")
        if not overlap_pred:
            sys.exit("No overlapping frame indices between pred and gt — filename mismatch?")

        with timer.stage("eval_single_object"):
            results = evaluate_single_object(
                pred_paths, gt_paths, K_seen, selection_mode=args.selection_mode
            )
        print()
        print(f"Best-matching cluster: k* = {results['best_cluster']}")
        if args.selection_mode != "best_cluster":
            print(f"Selected clusters ({args.selection_mode}): "
                  f"{results['selected_clusters']}")
        print(f"Per-cluster IoU: " +
              ", ".join(f"k={k}:{v:.4f}" for k, v in enumerate(results['iou_per_cluster'])
                        if k > 0))
        print(f"mIoU (scene-wide {args.selection_mode}): {results['mIoU']:.4f}")
        print(f"Frames evaluated: {results['matched_frames']}")

        if args.vis_dir:
            print()
            save_visualizations(
                args.vis_dir, pred_paths, gt_paths,
                best_cluster=results['best_cluster'],
                selected_clusters=results['selected_clusters'],
                palette_dir=args.palette_dir or None,
                original_dir=select_original_dir(args.original_dir, frame_alignment),
            )
    else:
        frame_alignment = {'method': 'direct'}
        with timer.stage("eval_multi_object"):
            results = evaluate_multi_object(pred_paths, gt_payload, K_seen)
        print()
        for m in results['matches']:
            print(f"  cluster {m['cluster']:>2} ↔ {m['object']:<20} IoU={m['iou']:.4f}")
        print(f"mIoU (Hungarian): {results['mIoU']:.4f}")

    tc_results = None
    if args.tc_dir:
        if not os.path.isdir(args.tc_dir):
            print(f"WARN: --tc_dir not found, skipping TC: {args.tc_dir}")
        else:
            with timer.stage("eval_tc"):
                tc_results = compute_temporal_consistency(args.tc_dir)
            print()
            print(f"Temporal consistency (TC): {tc_results['tc']:.4f}  "
                  f"(over {tc_results['n_pairs']} adjacent frame pairs)")

    # Pull pipeline-level timings from the run dir (spectral + render),
    # if a sibling timings.json exists alongside pred_dir's parent.
    run_dir = os.path.dirname(os.path.abspath(args.pred_dir))
    timings_path = os.path.join(run_dir, "timings.json")
    timer.merge_into_json(timings_path, step="eval")
    timer.append_to_report_md(
        os.path.join(run_dir, "report.md"), step="eval"
    )
    pipeline_timings = (
        json.load(open(timings_path)) if os.path.exists(timings_path) else None
    )

    summary = {
        'pred_dir': os.path.abspath(args.pred_dir),
        'gt_dir': os.path.abspath(args.gt_dir),
        'gt_layout': gt_kind,
        'frame_alignment': frame_alignment,
        'K_seen': K_seen,
        'diagnostics': diagnostics,
        'results': results,
        'temporal_consistency': tc_results,
        'timings': pipeline_timings,
    }
    if args.output_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_json)) or '.', exist_ok=True)
        with open(args.output_json, 'w') as f:
            json.dump(summary, f, indent=2, default=float)
        print(f"\nResults written to: {args.output_json}")


if __name__ == "__main__":
    main()
