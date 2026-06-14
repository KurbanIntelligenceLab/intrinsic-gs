from self_supervised_scripts.render_cluster_io import feature_checkpoint_path


def test_label_file_rendering_does_not_require_feature_checkpoint():
    assert (
        feature_checkpoint_path(
            model_path="output/chickchicken_run",
            load_iteration=20000,
            run_name="",
            labels_file="labels.npy",
        )
        is None
    )


def test_internal_clustering_uses_feature_checkpoint_suffix():
    assert feature_checkpoint_path(
        model_path="output/chickchicken_run",
        load_iteration=20000,
        run_name="sw0_ep50",
        labels_file="",
    ) == (
        "output/chickchicken_run/point_cloud/"
        "iteration_20000_features_sw0_ep50/point_cloud.ply"
    )
