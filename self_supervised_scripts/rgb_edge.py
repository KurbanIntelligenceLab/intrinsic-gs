"""RGB edge-detector strategies for boundary suppression.

Two implementations behind a common interface:

  - SobelEdge:    RGB → luminance Y → Sobel magnitude (existing baseline).
  - PiDiNetEdge:  RGB → pretrained PiDiNet → sigmoid edge map.

Both produce a `[H, W]` float32 edge map. Downstream percentile-normalization
(q=0.95) is applied by `BoundarySuppression._boundary_maps` so the scale is
comparable across methods — the same `α_depth, β_rgb, γ` defaults work for
either branch.
"""
from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Any, Optional

import torch

from self_supervised_scripts.boundary_suppression import (
    _rgb_to_luminance,
    _sobel_magnitude,
)


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_WEIGHTS_DIR = os.path.join(REPO_ROOT, "weights")


class EdgeMethod(ABC):
    """Common interface for RGB edge detectors."""

    name: str = ""

    @abstractmethod
    def __call__(self, rgb_3hw: torch.Tensor) -> torch.Tensor:
        """RGB [3, H, W] in [0, 1] → edge magnitude [H, W] (float32)."""


# ── Sobel (existing baseline) ─────────────────────────────────────────────────


class SobelEdge(EdgeMethod):
    """RGB → luminance Y (Rec.709) → Sobel magnitude.

    `presmooth_sigma > 0` applies a separable Gaussian blur before Sobel; this
    is the existing knob exposed via `--presmooth_sigma`.
    """

    name = "sobel"

    def __init__(self, presmooth_sigma: float = 0.0):
        self.presmooth_sigma = float(presmooth_sigma)

    def __call__(self, rgb_3hw: torch.Tensor) -> torch.Tensor:
        lum = _rgb_to_luminance(rgb_3hw).unsqueeze(0)  # [1, H, W]
        return _sobel_magnitude(lum, presmooth_sigma=self.presmooth_sigma)


# ── PiDiNet (pretrained boundary detector) ────────────────────────────────────


def _resolve_pidinet_ckpt(variant: str, weights_dir: str = DEFAULT_WEIGHTS_DIR) -> str:
    """Path to `weights/pidinet_<variant>.pth`. Existence not checked here."""
    return os.path.join(weights_dir, f"pidinet_{variant}.pth")


# BSDS500 per-channel mean (BGR, range [0, 255]) used by the upstream PiDiNet
# training/eval pipeline. The released checkpoints expect inputs in this exact
# distribution; deviating shifts the network out of distribution.
_PIDINET_BSDS_MEAN_BGR = (104.00699, 116.66877, 122.67892)


class PiDiNetEdge(EdgeMethod):
    """Density boundary detector using a pretrained PiDiNet.

    Lazy-loads the checkpoint on the first invocation and caches the model on
    the same device as the input tensor. Subsequent calls reuse the cached
    model. Inference always runs under `torch.no_grad()`.

    Output is the final fused sigmoid edge map (last element of PiDiNet's
    multi-scale output list), shape [H, W], values in [0, 1]. If
    `binarize_threshold` is set, the map is hard-thresholded to {0, 1} after
    inference — this makes PiDiNet's dense, smoothly-graded edge probabilities
    behave more like a sparse silhouette mask, so the downstream q=0.95
    percentile normalization in `BoundarySuppression` does not collapse texture
    and silhouettes to the same magnitude.
    """

    name = "pidinet"

    def __init__(self, variant: str = "full",
                 weights_dir: str = DEFAULT_WEIGHTS_DIR,
                 binarize_threshold: Optional[float] = None):
        if variant not in ("full", "small", "tiny"):
            raise ValueError(
                f"Unknown PiDiNet variant '{variant}'. Choose: full | small | tiny"
            )
        if binarize_threshold is not None and not (0.0 < binarize_threshold < 1.0):
            raise ValueError(
                f"binarize_threshold must be in (0, 1) or None, got {binarize_threshold}"
            )
        self.variant = variant
        self.weights_dir = weights_dir
        self.binarize_threshold = binarize_threshold
        self._model: Optional[torch.nn.Module] = None
        self._device: Optional[torch.device] = None
        self._mean_bgr_1331: Optional[torch.Tensor] = None  # [1,3,1,1] cached on device

    def _load(self, device: torch.device) -> torch.nn.Module:
        from self_supervised_scripts.pidinet_model import pidinet, load_pretrained

        ckpt = _resolve_pidinet_ckpt(self.variant, self.weights_dir)
        print(f"  PiDiNet({self.variant}): loading checkpoint {ckpt} ...")
        model = pidinet(self.variant)
        model = load_pretrained(model, ckpt)
        model = model.to(device)
        return model

    def _preprocess(self, rgb_3hw: torch.Tensor) -> torch.Tensor:
        """RGB [0,1] [3,H,W] → BGR [0,255] mean-subtracted [1,3,H,W].

        Matches the BSDS500 training-time preprocessing of the upstream PiDiNet
        repo. Without this, the network is fed inputs ~100× smaller in scale
        and in the wrong channel order, producing degraded edge maps.
        """
        device = rgb_3hw.device
        if self._mean_bgr_1331 is None or self._mean_bgr_1331.device != device:
            self._mean_bgr_1331 = torch.tensor(
                _PIDINET_BSDS_MEAN_BGR, device=device, dtype=torch.float32
            ).view(1, 3, 1, 1)
        bgr = rgb_3hw[[2, 1, 0], :, :].float() * 255.0  # [3,H,W]
        return bgr.unsqueeze(0) - self._mean_bgr_1331   # [1,3,H,W]

    @torch.no_grad()
    def __call__(self, rgb_3hw: torch.Tensor) -> torch.Tensor:
        device = rgb_3hw.device
        if self._model is None or self._device != device:
            self._model = self._load(device)
            self._device = device
        x = self._preprocess(rgb_3hw)
        outputs = self._model(x)
        edge = outputs[-1]                # [1, 1, H, W], fused sigmoid
        edge = edge.squeeze(0).squeeze(0)
        if self.binarize_threshold is not None:
            edge = (edge > self.binarize_threshold).to(edge.dtype)
        return edge


# ── Factory ──────────────────────────────────────────────────────────────────


def make_edge_method(args: Any) -> EdgeMethod:
    """Build an `EdgeMethod` from a CLI args namespace.

    Reads `args.rgb_edge_method` (default 'sobel') and, when 'pidinet',
    `args.pidinet_variant` (default 'full') plus optional
    `args.pidinet_binarize_threshold` (default None). Sobel additionally
    consumes `args.presmooth_sigma` (default 0.0).
    """
    name = getattr(args, "rgb_edge_method", "sobel")
    if name == "sobel":
        return SobelEdge(presmooth_sigma=getattr(args, "presmooth_sigma", 0.0))
    if name == "pidinet":
        variant = getattr(args, "pidinet_variant", "full")
        thr = getattr(args, "pidinet_binarize_threshold", None)
        return PiDiNetEdge(variant=variant, binarize_threshold=thr)
    raise ValueError(f"Unknown rgb_edge_method '{name}'. Choose: sobel | pidinet")
