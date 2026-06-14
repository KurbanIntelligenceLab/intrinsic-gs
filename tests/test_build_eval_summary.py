"""Unit tests for build_eval_summary's clusterer-aware schema."""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from self_supervised_scripts.build_eval_summary import (  # noqa: E402
    COLUMNS,
    collect_rows,
    detect_clusterer_from_dir,
    parse_diagnostics_from_report,
    write_csv,
)


def test_detect_clusterer_from_dir_maps_known_prefixes():
    assert detect_clusterer_from_dir("output/foo/28_04/spectral_k14_run") == "kmeans"
    assert detect_clusterer_from_dir("output/foo/28_04/leiden_res1.0_run") == "leiden"
    assert detect_clusterer_from_dir("output/foo/28_04/hdbscan_mcs0.005_run") == "hdbscan"
    assert detect_clusterer_from_dir("output/foo/random_dir") == "unknown"


def test_parse_diagnostics_from_report_kmeans(tmp_path):
    report = tmp_path / "report.md"
    report.write_text(
        """# Spectral Clustering Run Report

## Parameters
| param         | value |
|---------------|-------|
| n_clusters    | 14 |

## Spectral Analysis
- Eigengap suggested k: **14** (requested: 14)
- ρ = δ_max/δ_2nd: **1.103**  (higher ⇒ more decisive cluster count)
"""
    )
    diag = parse_diagnostics_from_report(str(report))
    assert diag["clusterer"] == "kmeans"
    assert diag["K_used"] == 14
    assert diag["K_auto_eigengap"] == 14
    assert diag["rho"] == 1.103


def test_parse_diagnostics_from_report_leiden(tmp_path):
    report = tmp_path / "report.md"
    report.write_text(
        """# Spectral Clustering Run Report

## Parameters
| param         | value |
|---------------|-------|
| n_clusters    | 7 |

## Leiden Community Detection
- Resolution parameter: **1.0**
- Communities found: **7**
- Modularity Q: **0.4231**  (higher ⇒ stronger community structure)
"""
    )
    diag = parse_diagnostics_from_report(str(report))
    assert diag["clusterer"] == "leiden"
    assert diag["K_used"] == 7
    assert diag["modularity_q"] == 0.4231
    assert "K_auto_eigengap" not in diag
    assert "rho" not in diag


def test_parse_diagnostics_from_report_hdbscan(tmp_path):
    report = tmp_path / "report.md"
    report.write_text(
        """# Spectral Clustering Run Report

## Parameters
| param         | value |
|---------------|-------|
| n_clusters    | 9 |

## HDBSCAN Density-Based Clustering
- min_cluster_size_frac: **0.005** (resolved to 5,150 points)
- min_samples: **5**
- Clusters found: **9**
- Noise points reassigned to nearest cluster: **12,345**
"""
    )
    diag = parse_diagnostics_from_report(str(report))
    assert diag["clusterer"] == "hdbscan"
    assert diag["K_used"] == 9
    assert diag["n_noise"] == 12345
    assert "K_auto_eigengap" not in diag
    assert "modularity_q" not in diag


def test_columns_includes_new_clusterer_fields():
    for col in ("clusterer", "modularity_q", "n_noise"):
        assert col in COLUMNS


def _make_run(scene_dir: Path, run_name: str, clusterer: str, k: int,
              miou: float, mode: str = "best_cluster"):
    """Helper to fabricate a run dir with miou_results.json + report.md."""
    run_dir = scene_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_greedy" if mode == "greedy_union" else ""
    payload = {
        "frame_alignment": {"method": "test", "gt_frames": 10},
        "diagnostics": {},
        "results": {
            "selection_mode": mode,
            "matched_frames": 10,
            "mIoU": miou,
            "best_cluster": 1,
            "selected_clusters": [1, 2] if mode == "greedy_union" else [],
            "per_frame_iou": {str(i): miou for i in range(10)},
        },
    }
    (run_dir / f"miou_results{suffix}.json").write_text(json.dumps(payload))
    if clusterer == "kmeans":
        report = (
            f"# Run\n\n## Parameters\n| param | value |\n|---|---|\n| n_clusters | {k} |\n\n"
            f"## Spectral Analysis\n- Eigengap suggested k: **{k}** (requested: {k})\n"
            f"- ρ = δ_max/δ_2nd: **1.5**\n"
        )
    elif clusterer == "leiden":
        report = (
            f"# Run\n\n## Parameters\n| param | value |\n|---|---|\n| n_clusters | {k} |\n\n"
            f"## Leiden Community Detection\n- Resolution parameter: **1.0**\n"
            f"- Communities found: **{k}**\n- Modularity Q: **0.42**\n"
        )
    else:  # hdbscan
        report = (
            f"# Run\n\n## Parameters\n| param | value |\n|---|---|\n| n_clusters | {k} |\n\n"
            f"## HDBSCAN Density-Based Clustering\n- min_cluster_size_frac: **0.005**\n"
            f"- Clusters found: **{k}**\n- Noise points reassigned to nearest cluster: **42**\n"
        )
    (run_dir / "report.md").write_text(report)


def test_collect_rows_aggregates_three_clusterers(tmp_path):
    outputs = tmp_path / "outputs"
    scene_dir = outputs / "fake_scene_run" / "28_04"
    _make_run(scene_dir, "spectral_k5_runA", "kmeans", k=5, miou=0.7, mode="best_cluster")
    _make_run(scene_dir, "leiden_res1.0_runB", "leiden", k=4, miou=0.65, mode="best_cluster")
    _make_run(scene_dir, "hdbscan_mcs0.005_runC", "hdbscan", k=6, miou=0.55, mode="best_cluster")

    rows = collect_rows(str(outputs))
    assert len(rows) == 3
    by_clust = {r["clusterer"]: r for r in rows}
    assert by_clust["kmeans"]["K_used"] == 5
    assert by_clust["kmeans"]["K_auto_eigengap"] == 5
    assert by_clust["kmeans"]["rho"] == 1.5
    assert by_clust["leiden"]["modularity_q"] == 0.42
    assert by_clust["leiden"]["n_noise"] is None
    assert by_clust["hdbscan"]["n_noise"] == 42
    assert by_clust["hdbscan"]["modularity_q"] is None


def test_collect_rows_computes_greedy_gain(tmp_path):
    outputs = tmp_path / "outputs"
    scene_dir = outputs / "fake_scene_run" / "28_04"
    _make_run(scene_dir, "spectral_k5_runX", "kmeans", k=5, miou=0.7, mode="best_cluster")
    _make_run(scene_dir, "spectral_k5_runX", "kmeans", k=5, miou=0.85, mode="greedy_union")

    rows = collect_rows(str(outputs))
    assert len(rows) == 2
    greedy_row = next(r for r in rows if r["selection_mode"] == "greedy_union")
    best_row = next(r for r in rows if r["selection_mode"] == "best_cluster")
    assert greedy_row["greedy_gain"] is not None
    assert abs(greedy_row["greedy_gain"] - 0.15) < 1e-9
    assert best_row["greedy_gain"] is None


def test_write_csv_emits_known_columns(tmp_path):
    rows = [
        {
            "scene": "s1", "clusterer": "leiden", "k_requested": None,
            "run_id": "11-22", "selection_mode": "best_cluster",
            "n_frames_eval": 10, "gt_frames_total": 10,
            "alignment_method": "x", "mIoU_scene": 0.5,
            "mean_per_frame": 0.5, "min_per_frame": 0.4, "max_per_frame": 0.6,
            "greedy_gain": None, "best_cluster": 2,
            "selected_clusters": None, "num_selected": None,
            "K_used": 5, "K_auto_eigengap": None, "rho": None,
            "modularity_q": 0.4, "n_noise": None,
        }
    ]
    csv_path = tmp_path / "out.csv"
    write_csv(rows, str(csv_path))
    text = csv_path.read_text()
    header = text.splitlines()[0]
    for col in COLUMNS:
        assert col in header
