# Dataset Preparation

## Download and Process the Dataset

### NeRF-DS

```
cd data/NeRF-DS
bash download_dataset.bash
```

### HyperNeRF

```
cd data/HyperNeRF
bash download_dataset.bash
```

### Neu3D

```
cd data/Neu3D
bash download_dataset.bash
```

To reproduce the results shown in the paper, use the precomputed `points3d.ply`, `transforms_test.json`, and `transforms_train.json`:

```
bash download_precomputed_poses.bash
```

### Google Immersive

Download the dataset from [here](https://github.com/augmentedperception/deepview_video_dataset). Note that we only use 01_Welder, 02_Flames, 10_Alexa_Meade_Face_Paint_1, and 11_Alexa_Meade_Face_Paint_2.

```
cd data/immersive
bash download_dataset.bash
```

To reproduce the results shown in the paper, use the precomputed `points3d.ply`, `transforms_test.json`, and `transforms_train.json`:

```
bash download_precomputed_poses.bash
```

### Technicolor Light Field

Please contact the author from "Dataset and Pipeline for Multi-View Light-Field Video" for access. We use the undistorted data `Undistorted/*` from Birthday, Fabien, Painter, and Theater.

To reproduce the results shown in the paper, use the precomputed `points3d.ply`, `transforms_test.json`, and `transforms_train.json`:

```
cd data/technicolor
bash download_precomputed_poses.bash
```
