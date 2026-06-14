import os
import sys
import importlib.util

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

spec = importlib.util.spec_from_file_location(
    "deform_model", os.path.join(REPO_ROOT, "scene", "deform_model.py")
)
deform_model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(deform_model)
resolve_deform_weights_path = deform_model.resolve_deform_weights_path


def test_resolves_flat_scene_specific_deform_path(tmp_path):
    weights = tmp_path / "deform" / "deform_americano.pth"
    weights.parent.mkdir()
    weights.write_text("weights")

    resolved = resolve_deform_weights_path(str(tmp_path), scene_name="americano")

    assert resolved == str(weights)


def test_scene_specific_deform_path_is_strict(tmp_path):
    with pytest.raises(FileNotFoundError, match="deform_chickchicken.pth"):
        resolve_deform_weights_path(str(tmp_path), scene_name="chickchicken")


def test_resolves_legacy_iteration_path_without_scene_name(tmp_path):
    weights = tmp_path / "deform" / "iteration_20000" / "deform.pth"
    weights.parent.mkdir(parents=True)
    weights.write_text("weights")

    resolved = resolve_deform_weights_path(str(tmp_path), iteration=20000)

    assert resolved == str(weights)
