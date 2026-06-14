# Evaluation on Mask-Benchmarks

To download the Mask-Benchmarks:

```bash
wget https://huggingface.co/datasets/yunjinli/Mask-Benchmark/resolve/main/Mask-Benchmark.zip
python -m zipfile -e Mask-Benchmark.zip .
```

Compute mIoU and mAcc against the benchmark using the affinity-segmentation evaluation scripts:

```bash
python self_supervised_scripts/compute_miou.py \
    --pred_dir output/<DATASET>/<NAME>/<spec_run>/cluster_ids_test \
    --gt_dir <path/to/Mask-Benchmark/dataset/scene>

python self_supervised_scripts/compute_macc.py \
    --pred_dir output/<DATASET>/<NAME>/<spec_run>/cluster_ids_test \
    --gt_dir <path/to/Mask-Benchmark/dataset/scene>
```

For multi-scene runs, `self_supervised_scripts/pipeline_multi_scene.py` chains train → cluster → render → mIoU/mAcc end-to-end.
