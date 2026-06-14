import argparse
import sys
from pathlib import Path
from unittest.mock import MagicMock
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from self_supervised_scripts.pipeline_common import (
    DEFAULT_DATASET_ROOT,
    DEFAULT_ITERATION,
    SpectralParams,
    build_miou_cmd,
    build_render_cmd,
    build_spectral_cmd,
    build_train_cmd,
    deform_published,
    discover_scenes,
    is_trained,
    newest_spec_run,
    publish_deform_paths,
    run_pipeline_pass,
    scene_paths,
)
from self_supervised_scripts.pipeline_multi_scene import args_to_params
from self_supervised_scripts.pipeline_multi_scene import build_parser
from self_supervised_scripts.pipeline_multi_scene import resolve_scenes


def test_discover_scenes_lists_dataset_directories_sorted(tmp_path):
    root = tmp_path / "hypernerf"
    root.mkdir()
    (root / "cut-lemon1").mkdir()
    (root / "americano").mkdir()
    (root / "notes.txt").write_text("ignore")

    assert discover_scenes(str(root)) == ["americano", "cut-lemon1"]


def test_resolve_scenes_supports_all_keyword(tmp_path):
    root = tmp_path / "hypernerf"
    root.mkdir()
    (root / "espresso").mkdir()
    (root / "chickchicken").mkdir()

    assert resolve_scenes(["all"], str(root)) == ["chickchicken", "espresso"]


def test_resolve_scenes_returns_explicit_names():
    assert resolve_scenes(["americano", "cut-lemon1"], DEFAULT_DATASET_ROOT) == [
        "americano",
        "cut-lemon1",
    ]


def test_scene_paths_use_hypernerf_dataset_root():
    paths = scene_paths("americano")

    assert paths.scene == "americano"
    assert paths.data == f"{DEFAULT_DATASET_ROOT}/americano"
    assert paths.model == "output/americano_run"


def test_checkpoint_detection_helpers(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ply = tmp_path / "output/cut-lemon1_run/point_cloud/iteration_20000/point_cloud.ply"
    ply.parent.mkdir(parents=True)
    ply.write_text("")
    deform = tmp_path / "deform/deform_cut-lemon1.pth"
    deform.parent.mkdir(parents=True)
    deform.write_text("")

    assert is_trained("output/cut-lemon1_run", DEFAULT_ITERATION)
    assert deform_published("cut-lemon1")
    assert not is_trained("output/americano_run", DEFAULT_ITERATION)
    assert not deform_published("americano")


def test_build_train_cmd_uses_stage1_feature_warmup():
    paths = scene_paths("cut-lemon1")

    cmd = build_train_cmd(paths, iteration=20000)

    assert cmd == [
        "python",
        "train.py",
        "-s",
        f"{DEFAULT_DATASET_ROOT}/cut-lemon1",
        "--model_path",
        "output/cut-lemon1_run",
        "--iterations",
        "20000",
        "--warm_up_3d_features",
        "20001",
    ]


def test_build_spectral_cmd_translates_pipeline_params():
    paths = scene_paths("cut-lemon1")
    params = SpectralParams(
        n_clusters=27, eigengap_k=30, knn_k=40, write_annotated_ply=True
    )

    cmd = build_spectral_cmd(paths, params, iteration=20000)

    assert cmd[:2] == ["python", "self_supervised_scripts/spectral_cluster.py"]
    assert "-s" in cmd and f"{DEFAULT_DATASET_ROOT}/cut-lemon1" in cmd
    assert "--n_clusters" in cmd and "27" in cmd
    assert "--eigengap_k" in cmd and "30" in cmd
    assert "--k" in cmd and "40" in cmd
    assert "--use_motion" in cmd
    assert "--use_boundary" in cmd
    assert "--no_annotated_ply" not in cmd


def test_build_spectral_cmd_skips_annotated_ply_by_default():
    cmd = build_spectral_cmd(scene_paths("cut-lemon1"), SpectralParams())
    assert "--no_annotated_ply" in cmd


def test_build_spectral_cmd_emits_opacity_thresh():
    cmd = build_spectral_cmd(
        scene_paths("cut-lemon1"),
        SpectralParams(opacity_thresh=0.1),
    )
    assert "--opacity_thresh" in cmd
    assert cmd[cmd.index("--opacity_thresh") + 1] == "0.1"


def test_build_spectral_cmd_omits_disabled_optional_terms():
    cmd = build_spectral_cmd(
        scene_paths("cut-lemon1"),
        SpectralParams(use_motion=False, use_boundary=False),
    )

    assert "--use_motion" not in cmd
    assert "--use_boundary" not in cmd


def test_build_spectral_cmd_emits_no_geo_ablation_flag():
    cmd = build_spectral_cmd(
        scene_paths("cut-lemon1"),
        SpectralParams(use_geometry=False),
    )

    assert "--no_geo" in cmd


def test_build_spectral_cmd_always_emits_clusterer_flag():
    cmd_kmeans = build_spectral_cmd(scene_paths("cut-lemon1"), SpectralParams())
    assert "--clusterer" in cmd_kmeans
    assert cmd_kmeans[cmd_kmeans.index("--clusterer") + 1] == "kmeans"


def test_build_spectral_cmd_leiden_omits_kmeans_only_args():
    cmd = build_spectral_cmd(
        scene_paths("cut-lemon1"),
        SpectralParams(clusterer="leiden", leiden_resolution=1.5),
    )
    # Leiden does not consume --n_clusters / --eigengap_k / --solver.
    assert "--n_clusters" not in cmd
    assert "--eigengap_k" not in cmd
    assert "--solver" not in cmd
    # Leiden-specific arg present.
    assert "--leiden_resolution" in cmd
    assert cmd[cmd.index("--leiden_resolution") + 1] == "1.5"


def test_build_spectral_cmd_hdbscan_emits_density_args():
    cmd = build_spectral_cmd(
        scene_paths("cut-lemon1"),
        SpectralParams(
            clusterer="hdbscan",
            hdbscan_min_cluster_size_frac=0.01,
            hdbscan_min_samples=3,
        ),
    )
    # HDBSCAN uses spectral embedding, so eigengap_k and solver still flow.
    assert "--eigengap_k" in cmd
    assert "--solver" in cmd
    # n_clusters is kmeans-only.
    assert "--n_clusters" not in cmd
    # HDBSCAN-specific args.
    assert "--hdbscan_min_cluster_size_frac" in cmd
    assert cmd[cmd.index("--hdbscan_min_cluster_size_frac") + 1] == "0.01"
    assert "--hdbscan_min_samples" in cmd
    assert cmd[cmd.index("--hdbscan_min_samples") + 1] == "3"


def test_build_spectral_cmd_default_emits_sobel_edge_method():
    cmd = build_spectral_cmd(scene_paths("cut-lemon1"), SpectralParams())
    assert "--rgb_edge_method" in cmd
    assert cmd[cmd.index("--rgb_edge_method") + 1] == "sobel"
    # pidinet_variant only emits when method == pidinet.
    assert "--pidinet_variant" not in cmd


def test_build_spectral_cmd_pidinet_emits_variant_flag():
    cmd = build_spectral_cmd(
        scene_paths("cut-lemon1"),
        SpectralParams(rgb_edge_method="pidinet", pidinet_variant="small"),
    )
    assert cmd[cmd.index("--rgb_edge_method") + 1] == "pidinet"
    assert "--pidinet_variant" in cmd
    assert cmd[cmd.index("--pidinet_variant") + 1] == "small"


def test_build_spectral_cmd_omits_edge_method_flags_when_no_boundary():
    cmd = build_spectral_cmd(
        scene_paths("cut-lemon1"),
        SpectralParams(use_boundary=False, rgb_edge_method="pidinet"),
    )
    # rgb_edge_method is a no-op without boundary suppression — don't
    # pollute the CLI when --use_boundary is off.
    assert "--rgb_edge_method" not in cmd
    assert "--pidinet_variant" not in cmd


def test_newest_spec_run_returns_most_recent_spectral_dir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    older = tmp_path / "output/cut-lemon1_run/27_04/spectral_k14_old"
    newer = tmp_path / "output/cut-lemon1_run/28_04/spectral_k14_new"
    older.mkdir(parents=True)
    newer.mkdir(parents=True)
    older.touch()
    newer.touch()

    assert newest_spec_run("output/cut-lemon1_run").endswith("spectral_k14_new")


def test_newest_spec_run_supports_leiden_and_hdbscan_prefixes(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    leiden = tmp_path / "output/cut-lemon1_run/28_04/leiden_res1.0_run"
    hdbscan = tmp_path / "output/cut-lemon1_run/28_04/hdbscan_mcs0.005_run"
    spectral = tmp_path / "output/cut-lemon1_run/28_04/spectral_k14_run"
    for d in (leiden, hdbscan, spectral):
        d.mkdir(parents=True)
        d.touch()

    assert newest_spec_run("output/cut-lemon1_run", prefix="leiden").endswith(
        "leiden_res1.0_run"
    )
    assert newest_spec_run("output/cut-lemon1_run", prefix="hdbscan").endswith(
        "hdbscan_mcs0.005_run"
    )
    # Default prefix is 'spectral' (kmeans).
    assert newest_spec_run("output/cut-lemon1_run").endswith("spectral_k14_run")


def test_detect_actual_k_reads_from_renders_dir(tmp_path):
    from self_supervised_scripts.pipeline_common import detect_actual_k

    spec_run = tmp_path / "spec_run"
    (spec_run / "renders_k7").mkdir(parents=True)
    assert detect_actual_k(str(spec_run)) == 7


def test_detect_actual_k_falls_back_to_labels_npy(tmp_path):
    import numpy as np

    from self_supervised_scripts.pipeline_common import detect_actual_k

    spec_run = tmp_path / "spec_run"
    spec_run.mkdir()
    np.save(spec_run / "labels.npy", np.array([0, 1, 2, 3, 0], dtype=np.int32))
    assert detect_actual_k(str(spec_run)) == 3


def test_build_render_and_miou_cmds_point_to_spectral_run_artifacts():
    paths = scene_paths("cut-lemon1")
    spec_run = "output/cut-lemon1_run/28_04/spectral_k27_run"

    render_cmd = build_render_cmd(paths, spec_run, iteration=20000)
    best_cmd = build_miou_cmd(paths, spec_run, n_clusters=27, mode="best_cluster")
    greedy_cmd = build_miou_cmd(paths, spec_run, n_clusters=27, mode="greedy_union")

    assert "--labels_file" in render_cmd
    assert f"{spec_run}/labels.npy" in render_cmd
    assert "--save_cluster_ids" in render_cmd
    assert f"{spec_run}/cluster_ids_train" in best_cmd
    assert f"{DEFAULT_DATASET_ROOT}/cut-lemon1/gt_masks" in best_cmd
    assert f"{spec_run}/miou_results.json" in best_cmd
    assert f"{spec_run}/renders_k27" in best_cmd
    assert "--selection_mode" not in best_cmd
    assert f"{spec_run}/miou_results_greedy.json" in greedy_cmd
    assert "--selection_mode" in greedy_cmd and "greedy_union" in greedy_cmd


def test_build_render_cmd_emits_tc_args_when_enabled():
    paths = scene_paths("cut-lemon1")
    spec_run = "output/cut-lemon1_run/28_04/spectral_k14_run"

    cmd = build_render_cmd(paths, spec_run, iteration=20000,
                           tc_camera_idx=3, tc_n_steps=40)

    assert "--tc_camera_idx" in cmd
    assert cmd[cmd.index("--tc_camera_idx") + 1] == "3"
    assert "--tc_n_steps" in cmd
    assert cmd[cmd.index("--tc_n_steps") + 1] == "40"


def test_build_render_cmd_omits_tc_args_when_disabled():
    cmd = build_render_cmd(scene_paths("cut-lemon1"),
                           "output/cut-lemon1_run/28_04/spectral_k14_run",
                           iteration=20000)
    assert "--tc_camera_idx" not in cmd
    assert "--tc_n_steps" not in cmd


def test_build_miou_cmd_emits_tc_dir_when_enabled():
    spec_run = "output/cut-lemon1_run/28_04/spectral_k14_run"
    cmd = build_miou_cmd(scene_paths("cut-lemon1"), spec_run,
                         n_clusters=14, mode="best_cluster", tc_camera_idx=3)
    assert "--tc_dir" in cmd
    assert cmd[cmd.index("--tc_dir") + 1] == f"{spec_run}/cluster_ids_tc_v3"


def test_build_miou_cmd_omits_tc_dir_when_disabled():
    cmd = build_miou_cmd(scene_paths("cut-lemon1"),
                         "output/cut-lemon1_run/28_04/spectral_k14_run",
                         n_clusters=14, mode="best_cluster")
    assert "--tc_dir" not in cmd


def test_run_pipeline_pass_skips_train_and_deform_when_artifacts_exist(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ply = tmp_path / "output/cut-lemon1_run/point_cloud/iteration_20000/point_cloud.ply"
    ply.parent.mkdir(parents=True)
    ply.write_text("")
    deform = tmp_path / "deform/deform_cut-lemon1.pth"
    deform.parent.mkdir(parents=True)
    deform.write_text("")

    with patch("self_supervised_scripts.pipeline_common.subprocess.run") as run, patch(
        "self_supervised_scripts.pipeline_common.newest_spec_run",
        return_value="output/cut-lemon1_run/28_04/spectral_k14_run",
    ), patch(
        "self_supervised_scripts.pipeline_common.detect_actual_k",
        return_value=14,
    ):
        run_pipeline_pass("cut-lemon1", SpectralParams())

    called_scripts = [call.args[0][1] for call in run.call_args_list]
    assert called_scripts == [
        "self_supervised_scripts/spectral_cluster.py",
        "self_supervised_scripts/render_clusters.py",
        "self_supervised_scripts/compute_miou.py",
        "self_supervised_scripts/compute_macc.py",
        "self_supervised_scripts/compute_miou.py",
        "self_supervised_scripts/compute_macc.py",
    ]


def test_run_pipeline_pass_runs_train_and_publishes_deform_when_missing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    deform_src = tmp_path / "output/cut-lemon1_run/deform/iteration_20000/deform.pth"

    def fake_run(cmd, check=True):
        if cmd[1] == "train.py":
            deform_src.parent.mkdir(parents=True)
            deform_src.write_text("fake")
        return MagicMock(returncode=0)

    with patch("self_supervised_scripts.pipeline_common.subprocess.run", side_effect=fake_run) as run, patch(
        "self_supervised_scripts.pipeline_common.newest_spec_run",
        return_value="output/cut-lemon1_run/28_04/spectral_k14_run",
    ), patch(
        "self_supervised_scripts.pipeline_common.detect_actual_k",
        return_value=14,
    ):
        run_pipeline_pass("cut-lemon1", SpectralParams())

    called_scripts = [call.args[0][1] for call in run.call_args_list]
    assert called_scripts == [
        "train.py",
        "self_supervised_scripts/spectral_cluster.py",
        "self_supervised_scripts/render_clusters.py",
        "self_supervised_scripts/compute_miou.py",
        "self_supervised_scripts/compute_macc.py",
        "self_supervised_scripts/compute_miou.py",
        "self_supervised_scripts/compute_macc.py",
    ]
    assert (tmp_path / "deform/deform_cut-lemon1.pth").exists()


def test_publish_deform_paths_use_scene_specific_names():
    src, dst = publish_deform_paths("cut-lemon1", iteration=20000)

    assert src == "output/cut-lemon1_run/deform/iteration_20000/deform.pth"
    assert dst == "deform/deform_cut-lemon1.pth"


def test_args_to_params_maps_cli_namespace_to_spectral_params():
    ns = argparse.Namespace(
        n_clusters=27,
        eigengap_k=30,
        sigma_color=0.8,
        sigma_scale=1.0,
        power=2.0,
        knn_k=20,
        use_motion=True,
        n_time_steps=20,
        motion_floor=0.2,
        static_motion_thresh=1e-3,
        use_boundary=True,
        boundary_views=12,
        alpha_depth=5.0,
        beta_rgb=2.0,
        gamma=2.0,
        opacity_thresh=0.07,
        presmooth_sigma=0.0,
        solver="cupy",
        clusterer="leiden",
        leiden_resolution=1.5,
        hdbscan_min_cluster_size_frac=0.01,
        hdbscan_min_samples=4,
    )

    params = args_to_params(ns)

    assert params.n_clusters == 27
    assert params.eigengap_k == 30
    assert params.use_motion
    assert params.use_boundary
    assert params.clusterer == "leiden"
    assert params.leiden_resolution == 1.5
    assert params.hdbscan_min_cluster_size_frac == 0.01
    assert params.hdbscan_min_samples == 4
    assert params.opacity_thresh == 0.07


def test_parser_accepts_no_motion_and_no_boundary_flags():
    parser = build_parser()

    args = parser.parse_args(
        ["--scenes", "cut-lemon1", "--ablation_name", "test_ablation",
         "--no_motion", "--no_boundary"]
    )

    assert args.use_motion is False
    assert args.use_boundary is False


def test_param_sweep_zip_mode_broadcasts_singleton_values():
    from self_supervised_scripts.pipeline_param_sweep import expand_config_dicts

    configs = expand_config_dicts(
        {
            "n_clusters": [14],
            "knn_k": [20, 30, 40],
            "beta_rgb": [0.1],
        },
        sweep_mode="zip",
    )

    assert configs == [
        {"n_clusters": 14, "knn_k": 20, "beta_rgb": 0.1},
        {"n_clusters": 14, "knn_k": 30, "beta_rgb": 0.1},
        {"n_clusters": 14, "knn_k": 40, "beta_rgb": 0.1},
    ]


def test_param_sweep_cartesian_mode_expands_all_combinations():
    from self_supervised_scripts.pipeline_param_sweep import expand_config_dicts

    configs = expand_config_dicts(
        {
            "knn_k": [20, 30, 40],
            "beta_rgb": [0.1, 0.2, 0.5],
        },
        sweep_mode="cartesian",
    )

    assert configs == [
        {"knn_k": 20, "beta_rgb": 0.1},
        {"knn_k": 20, "beta_rgb": 0.2},
        {"knn_k": 20, "beta_rgb": 0.5},
        {"knn_k": 30, "beta_rgb": 0.1},
        {"knn_k": 30, "beta_rgb": 0.2},
        {"knn_k": 30, "beta_rgb": 0.5},
        {"knn_k": 40, "beta_rgb": 0.1},
        {"knn_k": 40, "beta_rgb": 0.2},
        {"knn_k": 40, "beta_rgb": 0.5},
    ]


def test_param_sweep_zip_mode_rejects_mismatched_non_singleton_lengths():
    import pytest

    from self_supervised_scripts.pipeline_param_sweep import expand_config_dicts

    with pytest.raises(ValueError, match="same length"):
        expand_config_dicts(
            {
                "knn_k": [20, 30],
                "beta_rgb": [0.1, 0.2, 0.5],
            },
            sweep_mode="zip",
        )


def test_param_sweep_supports_clusterer_as_categorical_axis():
    """The full ablation use case: --clusterer kmeans leiden hdbscan → 3 configs."""
    from self_supervised_scripts.pipeline_param_sweep import (
        build_parser as build_sweep_parser,
        expand_configs,
    )

    parser = build_sweep_parser()
    args = parser.parse_args(
        [
            "--scene", "cut-lemon1",
            "--clusterer", "kmeans", "leiden", "hdbscan",
        ]
    )
    configs = expand_configs(args)
    assert len(configs) == 3
    assert [c.clusterer for c in configs] == ["kmeans", "leiden", "hdbscan"]


def test_param_sweep_rejects_invalid_clusterer():
    from self_supervised_scripts.pipeline_param_sweep import build_parser as build_sweep_parser
    import pytest

    parser = build_sweep_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--scene", "x", "--clusterer", "meanshift"])


def test_next_output_group_path_skips_existing_groups(tmp_path):
    from self_supervised_scripts.pipeline_param_sweep import next_output_group_path

    (tmp_path / "outputs/multiple-1").mkdir(parents=True)
    (tmp_path / "outputs/multiple-2").mkdir()

    assert next_output_group_path(tmp_path / "outputs") == tmp_path / "outputs/multiple-3"


def test_relocate_spec_run_moves_run_under_group_scene(tmp_path, monkeypatch):
    from self_supervised_scripts.pipeline_common import relocate_spec_run

    monkeypatch.chdir(tmp_path)
    spec_run = tmp_path / "output/americano_run/28_04/spectral_k14_run"
    spec_run.mkdir(parents=True)
    (spec_run / "labels.npy").write_text("fake labels")

    relocated = relocate_spec_run(str(spec_run), "outputs/multiple-1", "americano")

    assert relocated == "outputs/multiple-1/americano/spectral_k14_run"
    assert not spec_run.exists()
    assert (tmp_path / relocated / "labels.npy").read_text() == "fake labels"
