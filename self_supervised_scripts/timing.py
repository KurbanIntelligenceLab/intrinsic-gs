"""Wall-clock timing for the segmentation pipeline (paper §6.7 Table 7).

Single small dependency-free recorder used by spectral_cluster.py,
render_clusters.py, and compute_miou.py. Each run accumulates per-stage
seconds in a dict that is mirrored into:

  - `report.md` (`## Timings` section, appended)
  - `timings.json` (machine-parseable; merged across pipeline steps)

Stages are GPU-aware: `torch.cuda.synchronize()` is called before/after
each timed block so the measurement reflects actual compute, not async
kernel-launch time. Sync overhead is negligible for diagnostic runs but
serializes the GPU; gate behind --profile if you ever benchmark throughput.
"""
from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager

try:
    import torch
    _HAS_CUDA = torch.cuda.is_available()
except ImportError:
    torch = None
    _HAS_CUDA = False


def _cuda_sync():
    if _HAS_CUDA:
        torch.cuda.synchronize()


class TimingRecorder:
    """Accumulates per-stage seconds. One instance per pipeline-step process."""

    def __init__(self, n_valid_gaussians: int | None = None):
        self.timings: dict[str, float] = {}
        self.n_valid = n_valid_gaussians
        self.device_name = (
            torch.cuda.get_device_name(0) if _HAS_CUDA else "cpu"
        )

    def set_n_valid(self, n: int) -> None:
        self.n_valid = n

    @contextmanager
    def stage(self, name: str):
        _cuda_sync()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            _cuda_sync()
            self.timings[name] = self.timings.get(name, 0.0) + (
                time.perf_counter() - t0
            )

    # ------------------------------------------------------------------
    #  Persistence
    # ------------------------------------------------------------------

    def merge_into_json(self, json_path: str, step: str) -> dict:
        """Append this run's stages under {step: {...}} in timings.json.

        `step` is one of: 'spectral', 'render', 'eval'. Existing keys for
        the same step are overwritten; other steps are preserved.
        """
        existing = {}
        if os.path.exists(json_path):
            try:
                with open(json_path, encoding="utf-8") as f:
                    existing = json.load(f)
            except (json.JSONDecodeError, OSError):
                existing = {}
        existing.setdefault("device", self.device_name)
        if self.n_valid is not None:
            existing["n_valid_gaussians"] = self.n_valid
        existing[step] = dict(self.timings)
        existing[f"{step}_total_s"] = float(sum(self.timings.values()))
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(existing, f, indent=2)
        return existing

    def append_to_report_md(self, report_path: str, step: str) -> None:
        """Append a `## Timings — <step>` section to report.md.

        Idempotent per step: if a section with the same step already exists,
        it is replaced; otherwise the section is appended.
        """
        section = self._format_section(step)
        if not os.path.exists(report_path):
            # report.md may not exist for steps that don't own it (render/eval).
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(section)
            return

        with open(report_path, encoding="utf-8") as f:
            text = f.read()
        marker = f"## Timings — {step}"
        if marker in text:
            # Replace the existing section (up to the next "## " heading or EOF).
            head, _, tail = text.partition(marker)
            after = tail.split("\n## ", 1)
            new_tail = ("\n## " + after[1]) if len(after) > 1 else ""
            text = head + section.rstrip() + "\n" + new_tail
        else:
            text = text.rstrip() + "\n\n" + section
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(text)

    def _format_section(self, step: str) -> str:
        total = sum(self.timings.values())
        lines = [
            f"## Timings — {step}",
            "",
            f"- device: {self.device_name}",
        ]
        if self.n_valid is not None:
            lines.append(f"- N_valid: {self.n_valid:,}")
        lines += [
            "",
            "| stage | time (s) | per-gauss (µs) |",
            "|-------|----------|----------------|",
        ]
        for name, sec in self.timings.items():
            per_g = (
                f"{1e6 * sec / self.n_valid:.3f}"
                if self.n_valid else "—"
            )
            lines.append(f"| {name} | {sec:.3f} | {per_g} |")
        lines.append(f"| **{step}_total** | **{total:.3f}** | — |")
        lines.append("")
        return "\n".join(lines) + "\n"
