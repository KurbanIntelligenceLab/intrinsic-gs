import os


def feature_checkpoint_path(model_path, load_iteration, run_name="", labels_file=""):
    """Return the feature checkpoint needed for internal clustering.

    Label-file rendering uses labels produced elsewhere, so it can render from the
    already-loaded Gaussian checkpoint and does not require a feature-only PLY.
    """
    if labels_file:
        return None

    run_suffix = f"_{run_name}" if run_name else ""
    feature_path = os.path.join(
        model_path,
        "point_cloud",
        f"iteration_{load_iteration}_features{run_suffix}",
        "point_cloud.ply",
    )
    if os.path.exists(feature_path):
        return feature_path

    if run_name:
        return feature_path

    standard_path = os.path.join(
        model_path,
        "point_cloud",
        f"iteration_{load_iteration}",
        "point_cloud.ply",
    )
    return standard_path if os.path.exists(standard_path) else feature_path
