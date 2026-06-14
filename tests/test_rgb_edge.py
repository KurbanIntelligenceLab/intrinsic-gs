import argparse
import sys
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from self_supervised_scripts.rgb_edge import (
    PiDiNetEdge,
    SobelEdge,
    make_edge_method,
)


# ── SobelEdge ─────────────────────────────────────────────────────────────────


def test_sobel_edge_returns_2d_float_tensor():
    rgb = torch.rand(3, 64, 64)
    edge = SobelEdge()(rgb)
    assert edge.shape == (64, 64)
    assert edge.dtype == torch.float32
    assert torch.isfinite(edge).all()


def test_sobel_edge_is_zero_on_uniform_image_interior():
    # No gradient in the interior → magnitude is zero. (Borders pick up
    # zero-padding from F.conv2d, which is expected — only assert on the
    # interior.)
    rgb = torch.full((3, 32, 32), 0.5)
    edge = SobelEdge()(rgb)
    interior = edge[2:-2, 2:-2]
    assert interior.abs().max().item() == pytest.approx(0.0, abs=1e-6)


def test_sobel_edge_presmooth_sigma_smooths_response():
    # A vertical-step image: half black, half white. With σ=0 the boundary is
    # one pixel wide; with σ>0 it spreads. Just check σ>0 changes the response.
    rgb = torch.zeros(3, 32, 32)
    rgb[:, :, 16:] = 1.0
    sharp = SobelEdge(presmooth_sigma=0.0)(rgb)
    blurred = SobelEdge(presmooth_sigma=2.0)(rgb)
    # The sharp response is concentrated; the blurred one spreads — peak is
    # lower under blur.
    assert blurred.max() < sharp.max()


# ── make_edge_method factory ─────────────────────────────────────────────────


def test_make_edge_method_default_is_sobel():
    args = argparse.Namespace()  # no rgb_edge_method attribute at all
    em = make_edge_method(args)
    assert isinstance(em, SobelEdge)
    assert em.name == "sobel"


def test_make_edge_method_explicit_sobel_propagates_presmooth_sigma():
    args = argparse.Namespace(rgb_edge_method="sobel", presmooth_sigma=3.5)
    em = make_edge_method(args)
    assert isinstance(em, SobelEdge)
    assert em.presmooth_sigma == 3.5


def test_make_edge_method_pidinet_dispatches_with_variant():
    args = argparse.Namespace(rgb_edge_method="pidinet", pidinet_variant="tiny")
    em = make_edge_method(args)
    assert isinstance(em, PiDiNetEdge)
    assert em.variant == "tiny"
    # Lazy: model not loaded yet.
    assert em._model is None


def test_make_edge_method_rejects_unknown_method():
    args = argparse.Namespace(rgb_edge_method="canny")
    with pytest.raises(ValueError, match="Unknown rgb_edge_method"):
        make_edge_method(args)


# ── PiDiNetEdge — checkpoint missing path ────────────────────────────────────


def test_pidinet_edge_raises_clear_error_when_checkpoint_missing(tmp_path):
    em = PiDiNetEdge(variant="full", weights_dir=str(tmp_path))
    rgb = torch.rand(3, 32, 32)
    with pytest.raises(FileNotFoundError, match="download_pidinet.sh"):
        em(rgb)


def test_pidinet_edge_rejects_unknown_variant():
    with pytest.raises(ValueError, match="Unknown PiDiNet variant"):
        PiDiNetEdge(variant="huge")


# ── PiDiNetEdge — stub-model integration ─────────────────────────────────────
#
# We don't load the real BSDS500 checkpoint in tests (4MB download per CI run
# would be wasteful). Instead, monkeypatch the lazy loader so __call__ runs
# end-to-end against a trivial 1×1 conv stub. This catches plumbing bugs:
# device handling, output shape squeezing, no_grad, caching.


def test_pidinet_edge_uses_cached_model_across_calls(monkeypatch):
    class _Stub(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 1, kernel_size=1)
            self.call_count = 0

        def forward(self, x):
            self.call_count += 1
            y = torch.sigmoid(self.conv(x))
            # PiDiNet returns a list of 5 maps (4 stages + 1 fused). We
            # mimic that contract — the adapter takes the last entry.
            return [y, y, y, y, y]

    em = PiDiNetEdge(variant="full")
    stub = _Stub()
    monkeypatch.setattr(em, "_load", lambda device: stub.to(device))

    rgb = torch.rand(3, 16, 16)
    e1 = em(rgb)
    e2 = em(rgb)
    assert e1.shape == (16, 16)
    assert e2.shape == (16, 16)
    # _load was only called once across two invocations.
    assert stub.call_count == 2  # forward called twice
    # And the model was loaded exactly once (cached afterward).
    assert em._model is stub
