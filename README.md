# intrinsic-gs

**Intrinsic 4D Gaussian Segmentation from Scene Cues**

Hasan Yazar\*, Mohamed Rayan Barhdadi\*, Erchin Serpedin, Mehmet Tuncel, Hasan Kurban

<sup>\*</sup> Equal contribution.

Preprint

> Preprint. This repository contains the source code for Intrinsic-GS, built on
> top of the [TRASE](https://github.com/yunjinli/TRASE) codebase. Trained
> checkpoints, large render outputs, and benchmark result dumps are not
> committed; see [Generated artifacts](#generated-artifacts).

<!-- TODO: add our own teaser figure (do NOT reuse TRASE assets) -->

## Overview

**Intrinsic-GS** is a **training-free, mask-free** method for object-level
segmentation of 3D and 4D Gaussian Splatting scenes. Instead of importing 2D
masks from foundation models (e.g. SAM) and lifting or distilling them into the
Gaussian representation, we ask how much object structure can be recovered from
the Gaussians themselves.

We build a sparse **affinity graph** over Gaussian primitives from intrinsic
scene cues — appearance, orientation, scale, deformation-trajectory, and a
non-learned rendered-boundary cue — and partition it with **Leiden** community
detection. No foundation model and no learned feature field are required.

On the standard 4D Gaussian segmentation benchmarks, Intrinsic-GS recovers
substantial object structure without mask supervision: **0.746 mIoU on Neu3D**
and **0.575 on HyperNeRF**; a geometry-only variant reaches **0.902 mIoU on
Neu3D**, matching SAM-supervised TRASE. On HyperNeRF it runs **12.5× faster**
than the mask-generation and feature-rendering stages used by mask-supervised
pipelines.

This implementation reuses the TRASE-style data layout, training pipeline, and
Mask-Benchmark evaluation conventions.

## Installation

The CUDA extensions are pulled in as git submodules, so clone recursively:

```bash
git clone --recursive https://github.com/KurbanIntelligenceLab/intrinsic-gs.git
cd intrinsic-gs
# If you already cloned without --recursive:
git submodule update --init --recursive
```

<details>
<summary><b>Local installation (conda)</b></summary>

```bash
conda create -n intrinsic-gs python=3.8 -y
conda activate intrinsic-gs
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 \
    --extra-index-url https://download.pytorch.org/whl/cu117
pip install "git+https://github.com/facebookresearch/pytorch3d.git@v0.7.6"
pip install opencv-python plyfile tqdm scipy scikit-learn lpips \
    imageio[ffmpeg] kmeans_pytorch hdbscan scikit-image bitarray
python -m pip install submodules/diff-gaussian-rasterization
python -m pip install submodules/simple-knn
```

</details>

<details>
<summary><b>Docker installation</b></summary>

A devcontainer and `docker-compose.yml` are provided. You need the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
on the host so the container can compile the CUDA extensions and access the GPU.

**VS Code (recommended):** open the folder, then run
`Dev Containers: Rebuild and Reopen in Container`.

**Command line:**

```bash
export UID=$(id -u)
export GID=$(id -g)
docker-compose up -d --build
docker exec -it trase-container bash
```

</details>

## Dataset preparation

See [docs/prepare_dataset.md](docs/prepare_dataset.md).

## Training

The entry point is [`train.py`](train.py); see [docs/train.md](docs/train.md)
for the full list of command-line arguments.

## Evaluation

We evaluate on the Mask-Benchmarks. See [docs/evaluation.md](docs/evaluation.md)
for downloading the benchmark and computing mIoU / mAcc with the
`self_supervised_scripts/` tools.

## Generated artifacts

Large generated outputs (`output/`, `outputs/`), ablation runs
(`multiple_ablation/`), and benchmark result dumps (`paper_benchmark_runs/`)
are intentionally **not** committed. Only source code, launch/evaluation
scripts, and compact report artifacts under `docs/` are tracked.

## Citation

```bibtex
@article{yazar2026intrinsicgs,
    title  = {Intrinsic 4D Gaussian Segmentation from Scene Cues},
    author = {Yazar, Hasan and Barhdadi, Mohamed Rayan and Serpedin, Erchin and Tuncel, Mehmet and Kurban, Hasan},
    year   = {2026}
}
```

## Acknowledgement

This work builds on [TRASE: Tracking-free 4D Segmentation and Editing](https://github.com/yunjinli/TRASE)
and the broader Gaussian Splatting ecosystem. We thank the authors of
[3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting),
[Deformable 3D Gaussians](https://github.com/ingra14m/Deformable-3D-Gaussians),
[SC-GS](https://github.com/yihua7/SC-GS),
[Gaussian Grouping](https://github.com/lkeab/gaussian-grouping), and
[SAGA](https://github.com/Jumpat/SegAnyGAussians). Please consider citing their
work as well.

<details>
<summary><b>Upstream BibTeX</b></summary>

```bibtex
@article{li2024trase,
    title   = {TRASE: Tracking-free 4D Segmentation and Editing},
    author  = {Li, Yun-Jin and Gladkova, Mariia and Xia, Yan and Cremers, Daniel},
    journal = {arXiv preprint arXiv:2411.19290},
    year    = {2024}
}

@Article{kerbl3Dgaussians,
    author  = {Kerbl, Bernhard and Kopanas, Georgios and Leimk{\"u}hler, Thomas and Drettakis, George},
    title   = {3D Gaussian Splatting for Real-Time Radiance Field Rendering},
    journal = {ACM Transactions on Graphics},
    number  = {4},
    volume  = {42},
    month   = {July},
    year    = {2023},
    url     = {https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/}
}
```

</details>
