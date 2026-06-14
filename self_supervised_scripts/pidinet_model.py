"""Vendored PiDiNet model definition.

Source: https://github.com/hellozhuo/pidinet (commit master, MIT license).
Original authors: Zhuo Su, Wenzhe Liu (paper: ICCV 2021, "Pixel Difference
Networks for Efficient Edge Detection").

This file consolidates the upstream `models/ops.py`, `models/config.py`, and
`models/pidinet.py` into a single self-contained module so we don't need to
clone the upstream repo as a submodule. Behavioral changes versus upstream:

  - `pidinet(variant: str)` factory replaces the upstream `args`-based
    factories (`pidinet`, `pidinet_small`, `pidinet_tiny`). Always uses the
    table5 (`carv4` config + sa=True + dil=variant_specific) configuration
    that matches the public BSDS500 checkpoints.
  - Removed the `print('initialization done')` and config-printing side
    effects from upstream — silent construction.
  - Added `load_pretrained(model, ckpt_path)` helper that handles the
    `state_dict` / `module.` prefix conventions of the released checkpoints.

Public API:
    pidinet(variant: str) -> nn.Module
    load_pretrained(model: nn.Module, ckpt_path: str) -> nn.Module
"""
from __future__ import annotations

import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Pixel Difference Convolution operators (from upstream ops.py) ─────────────


def _create_pdc_func(op_type: str):
    """Build the conv functor for a given PDC variant: 'cv', 'cd', 'ad', 'rd'."""
    assert op_type in ("cv", "cd", "ad", "rd"), f"unknown op type: {op_type}"

    if op_type == "cv":
        return F.conv2d

    if op_type == "cd":
        def _cd(x, weights, bias=None, stride=1, padding=0, dilation=1, groups=1):
            assert dilation in (1, 2)
            assert weights.size(2) == 3 and weights.size(3) == 3
            assert padding == dilation
            weights_c = weights.sum(dim=[2, 3], keepdim=True)
            yc = F.conv2d(x, weights_c, stride=stride, padding=0, groups=groups)
            y = F.conv2d(x, weights, bias, stride=stride, padding=padding,
                         dilation=dilation, groups=groups)
            return y - yc
        return _cd

    if op_type == "ad":
        def _ad(x, weights, bias=None, stride=1, padding=0, dilation=1, groups=1):
            assert dilation in (1, 2)
            assert weights.size(2) == 3 and weights.size(3) == 3
            assert padding == dilation
            shape = weights.shape
            weights = weights.view(shape[0], shape[1], -1)
            # clock-wise pixel reordering, see paper Eq. 6
            weights_conv = (
                weights - weights[:, :, [3, 0, 1, 6, 4, 2, 7, 8, 5]]
            ).view(shape)
            return F.conv2d(x, weights_conv, bias, stride=stride, padding=padding,
                            dilation=dilation, groups=groups)
        return _ad

    # op_type == "rd"
    def _rd(x, weights, bias=None, stride=1, padding=0, dilation=1, groups=1):
        assert dilation in (1, 2)
        assert weights.size(2) == 3 and weights.size(3) == 3
        padding = 2 * dilation
        shape = weights.shape
        if weights.is_cuda:
            buffer = torch.cuda.FloatTensor(shape[0], shape[1], 5 * 5).fill_(0)
        else:
            buffer = torch.zeros(shape[0], shape[1], 5 * 5)
        weights = weights.view(shape[0], shape[1], -1)
        buffer[:, :, [0, 2, 4, 10, 14, 20, 22, 24]] = weights[:, :, 1:]
        buffer[:, :, [6, 7, 8, 11, 13, 16, 17, 18]] = -weights[:, :, 1:]
        buffer[:, :, 12] = 0
        buffer = buffer.view(shape[0], shape[1], 5, 5)
        return F.conv2d(x, buffer, bias, stride=stride, padding=padding,
                        dilation=dilation, groups=groups)
    return _rd


class PDCConv2d(nn.Module):
    """Conv2d wrapper that dispatches to a pixel-difference operator (pdc)."""

    def __init__(self, pdc, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=False):
        super().__init__()
        if in_channels % groups != 0:
            raise ValueError("in_channels must be divisible by groups")
        if out_channels % groups != 0:
            raise ValueError("out_channels must be divisible by groups")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.weight = nn.Parameter(torch.empty(
            out_channels, in_channels // groups, kernel_size, kernel_size,
        ))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter("bias", None)
        self._reset_parameters()
        self.pdc = pdc

    def _reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        return self.pdc(x, self.weight, self.bias, self.stride, self.padding,
                        self.dilation, self.groups)


# ── Architecture config (from upstream config.py) ────────────────────────────
#
# `carv4` is the table-5 BSDS500 config — what the public checkpoints were
# trained with. Pattern: cd, ad, rd, cv repeating across 16 layers.

_CARV4 = ["cd", "ad", "rd", "cv"] * 4


def _carv4_pdcs():
    return [_create_pdc_func(op) for op in _CARV4]


# ── Architecture (from upstream pidinet.py) ──────────────────────────────────


class _CSAM(nn.Module):
    """Compact Spatial Attention Module."""

    def __init__(self, channels):
        super().__init__()
        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv2d(channels, 4, kernel_size=1, padding=0)
        self.conv2 = nn.Conv2d(4, 1, kernel_size=3, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()
        nn.init.constant_(self.conv1.bias, 0)

    def forward(self, x):
        y = self.sigmoid(self.conv2(self.conv1(self.relu1(x))))
        return x * y


class _CDCM(nn.Module):
    """Compact Dilation Convolution based Module."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        self.conv2_1 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                                 dilation=5, padding=5, bias=False)
        self.conv2_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                                 dilation=7, padding=7, bias=False)
        self.conv2_3 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                                 dilation=9, padding=9, bias=False)
        self.conv2_4 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                                 dilation=11, padding=11, bias=False)
        nn.init.constant_(self.conv1.bias, 0)

    def forward(self, x):
        x = self.conv1(self.relu1(x))
        return self.conv2_1(x) + self.conv2_2(x) + self.conv2_3(x) + self.conv2_4(x)


class _MapReduce(nn.Module):
    """Reduce feature maps into a single edge map."""

    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, 1, kernel_size=1, padding=0)
        nn.init.constant_(self.conv.bias, 0)

    def forward(self, x):
        return self.conv(x)


class _PDCBlock(nn.Module):
    def __init__(self, pdc, inplane, ouplane, stride=1):
        super().__init__()
        self.stride = stride
        if self.stride > 1:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.shortcut = nn.Conv2d(inplane, ouplane, kernel_size=1, padding=0)
        self.conv1 = PDCConv2d(pdc, inplane, inplane, kernel_size=3, padding=1,
                               groups=inplane, bias=False)
        self.relu2 = nn.ReLU()
        self.conv2 = nn.Conv2d(inplane, ouplane, kernel_size=1, padding=0, bias=False)

    def forward(self, x):
        if self.stride > 1:
            x = self.pool(x)
        y = self.conv2(self.relu2(self.conv1(x)))
        if self.stride > 1:
            x = self.shortcut(x)
        return y + x


class PiDiNet(nn.Module):
    """Pixel Difference Network — table5 BSDS500 architecture (sa=True, dil set)."""

    def __init__(self, inplane: int, pdcs: list, dil: int):
        super().__init__()
        self.dil = dil
        self.fuseplanes = []
        self.inplane = inplane

        self.init_block = PDCConv2d(pdcs[0], 3, self.inplane, kernel_size=3, padding=1)

        self.block1_1 = _PDCBlock(pdcs[1], self.inplane, self.inplane)
        self.block1_2 = _PDCBlock(pdcs[2], self.inplane, self.inplane)
        self.block1_3 = _PDCBlock(pdcs[3], self.inplane, self.inplane)
        self.fuseplanes.append(self.inplane)

        prev = self.inplane
        self.inplane *= 2
        self.block2_1 = _PDCBlock(pdcs[4], prev, self.inplane, stride=2)
        self.block2_2 = _PDCBlock(pdcs[5], self.inplane, self.inplane)
        self.block2_3 = _PDCBlock(pdcs[6], self.inplane, self.inplane)
        self.block2_4 = _PDCBlock(pdcs[7], self.inplane, self.inplane)
        self.fuseplanes.append(self.inplane)

        prev = self.inplane
        self.inplane *= 2
        self.block3_1 = _PDCBlock(pdcs[8], prev, self.inplane, stride=2)
        self.block3_2 = _PDCBlock(pdcs[9], self.inplane, self.inplane)
        self.block3_3 = _PDCBlock(pdcs[10], self.inplane, self.inplane)
        self.block3_4 = _PDCBlock(pdcs[11], self.inplane, self.inplane)
        self.fuseplanes.append(self.inplane)

        self.block4_1 = _PDCBlock(pdcs[12], self.inplane, self.inplane, stride=2)
        self.block4_2 = _PDCBlock(pdcs[13], self.inplane, self.inplane)
        self.block4_3 = _PDCBlock(pdcs[14], self.inplane, self.inplane)
        self.block4_4 = _PDCBlock(pdcs[15], self.inplane, self.inplane)
        self.fuseplanes.append(self.inplane)

        # sa=True + dil set → CDCM dilation + CSAM attention on every stage.
        self.dilations = nn.ModuleList()
        self.attentions = nn.ModuleList()
        self.conv_reduces = nn.ModuleList()
        for fp in self.fuseplanes:
            self.dilations.append(_CDCM(fp, dil))
            self.attentions.append(_CSAM(dil))
            self.conv_reduces.append(_MapReduce(dil))

        self.classifier = nn.Conv2d(4, 1, kernel_size=1)
        nn.init.constant_(self.classifier.weight, 0.25)
        nn.init.constant_(self.classifier.bias, 0)

    def forward(self, x):
        H, W = x.size()[2:]

        x = self.init_block(x)

        x1 = self.block1_3(self.block1_2(self.block1_1(x)))
        x2 = self.block2_4(self.block2_3(self.block2_2(self.block2_1(x1))))
        x3 = self.block3_4(self.block3_3(self.block3_2(self.block3_1(x2))))
        x4 = self.block4_4(self.block4_3(self.block4_2(self.block4_1(x3))))

        x_fuses = [
            self.attentions[i](self.dilations[i](xi))
            for i, xi in enumerate([x1, x2, x3, x4])
        ]

        es = [
            F.interpolate(self.conv_reduces[i](x_fuses[i]), (H, W),
                          mode="bilinear", align_corners=False)
            for i in range(4)
        ]

        fused = self.classifier(torch.cat(es, dim=1))
        outputs = es + [fused]
        return [torch.sigmoid(r) for r in outputs]


# ── Public factory + checkpoint loader ───────────────────────────────────────


_VARIANT_SPECS = {
    "full":  (60, 24),
    "small": (30, 12),
    "tiny":  (20,  8),
}


def pidinet(variant: str) -> nn.Module:
    """Build a PiDiNet matching the public table5 BSDS500 checkpoint.

    variant ∈ {'full', 'small', 'tiny'} maps to (inplane, dil) per the
    upstream `pidinet` / `pidinet_small` / `pidinet_tiny` factories.
    """
    if variant not in _VARIANT_SPECS:
        choices = " | ".join(_VARIANT_SPECS)
        raise ValueError(f"Unknown variant '{variant}'. Choose: {choices}")
    inplane, dil = _VARIANT_SPECS[variant]
    return PiDiNet(inplane, _carv4_pdcs(), dil=dil)


def load_pretrained(model: nn.Module, ckpt_path: str) -> nn.Module:
    """Load BSDS500-trained weights into `model`.

    Handles the upstream checkpoint convention: `{'state_dict': {'module.<key>': ...}}`.
    Returns the model in eval mode.
    """
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"PiDiNet checkpoint not found at '{ckpt_path}'. "
            f"Run `bash scripts/download_pidinet.sh` to fetch all three variants."
        )
    state = torch.load(ckpt_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    state = {k.replace("module.", "", 1): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    return model.eval()
