# Track B: zero-shot feed-forward models (MapAnything, VGGT, Pi3, MASt3R)

All four models run with their pretrained weights through the benchmark harness of the official
MapAnything repository and receive RGB images only. The patch adds the UAV3DCrop loaders and
configs.

## Setup

```bash
git clone https://github.com/facebookresearch/map-anything.git && cd map-anything
git checkout f065d13383ae84a7f80f0e0104a4d12afc7c7c82
git apply /path/to/map-anything-trackb.patch

conda env create -f /path/to/environment.yml && conda activate mapanything   # Python 3.10, PyTorch 2.3.1, CUDA 11.8
pip install -e . --no-deps
```

Checkpoints:

```bash
python scripts/convert_hf_to_benchmark_checkpoint.py --output_path <CKPT>/map-anything/model.pth   # MapAnything
wget -P <CKPT> https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth
# VGGT (facebook/VGGT-1B) and Pi3 (yyfz233/Pi3) are downloaded automatically.
```

## Data

```bash
huggingface-cli download Link-Dev/UAV3DCrop --repo-type dataset --local-dir <HF>/UAV3DCrop
huggingface-cli download Link-Dev/UAV3DCrop_depth --repo-type dataset --local-dir <HF>/UAV3DCrop_depth
pip install tifffile
python hf_to_wai.py --rgb_root <HF>/UAV3DCrop --depth_root <HF>/UAV3DCrop_depth --out_root <WAI> --workers 16
```

This writes `<WAI>/<sequence>/Day<NNN>_<Crop>/` (660 x 494 undistorted images, depth, `scene_meta.json`).
Set `root_data_dir`, `mapanything_dataset_metadata_dir` and `root_experiments_dir` in
`configs/machine/uav3dcrop.yaml`. Scene lists (`val`, `train`) are in
`mapanything_dataset_metadata_benchmark/`; both are evaluated with the pretrained weights.

## Run (one A100 40 GB)

Each command evaluates 130 sets of 36 views per sequence. For the `train` list, append `$SPLIT_TRAIN`
and write to a separate `hydra.run.dir` (for example `.../dense_36_view_train`); with the same
folder, the second run overwrites the `*_per_scene_results.json` files of the first:

```bash
export HYDRA_FULL_ERROR=1
KEYS="umn2023_corn umn2023_soy umn2024_corn1 umn2024_corn2 umn2025_wheat umn2025_oat umn2025_corn umn2025_soybean"
SPLIT_TRAIN=""; for k in $KEYS; do SPLIT_TRAIN="$SPLIT_TRAIN dataset.${k}_benchmark_wai.val.split=train"; done
COMMON="machine=uav3dcrop dataset.num_views=36 batch_size=1"

# MapAnything (images only)
python benchmarking/dense_n_view/benchmark.py $COMMON dataset=benchmark_eval dataset.num_workers=8 \
    model=mapanything model/task=images_only model.pretrained=<CKPT>/map-anything/model.pth \
    hydra.run.dir='${root_experiments_dir}/benchmarking/mapanything_benchmark/dense_36_view'

# VGGT / Pi3
for m in vggt pi3; do
  python benchmarking/dense_n_view/benchmark.py $COMMON dataset=benchmark_eval dataset.num_workers=12 model=$m \
      hydra.run.dir='${root_experiments_dir}/benchmarking/'"$m"'_benchmark/dense_36_view'
done

# MASt3R (fp32; 3 chunks of 45 sets per sequence, seed 777 + 45 * chunk)
for key in $KEYS; do for chunk in 0 1 2; do
  python benchmarking/dense_n_view/benchmark.py $COMMON dataset=benchmark_eval_512 dataset.num_workers=12 \
      'dataset.test_dataset=+ 45 @ ${dataset.'"$key"'_benchmark_wai.val.dataset_str}' \
      "dataset.${key}_benchmark_wai.val.seed=$((777 + chunk * 45))" \
      model=mast3r model.model_config.ckpt_path=<CKPT>/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth \
      model.model_config.cache_dir=<CACHE>/$key/chunk_$chunk amp=0 \
      hydra.run.dir='${root_experiments_dir}/benchmarking/mast3r_benchmark_chunked/'"$key"'/chunk_'"$chunk"
done; done
```

For MASt3R on the `train` list, add `dataset.${key}_benchmark_wai.val.split=train` and use other
output and cache folders. Loading MapAnything prints 76 `missing_keys`; these are aliases of
loaded weights and can be ignored.

## Metrics

The benchmark computes the metrics for each set of 36 views:

- Predicted and reference geometry are each expressed in the frame of the first view and divided
  by the mean distance of their valid points from its origin. Point-map and z-depth AbsRel and
  inlier rate (δ < 1.03) are computed after this scale normalization.
- `metric_scale_abs_rel` is the relative error of the mean point distance of the unnormalized
  prediction against that of the reference.
- `pose_ate_rmse` (the upstream name) is the mean camera-position error after a rigid alignment of
  the normalized camera centers; it is not a root-mean-square value.
- `pose_auc_5` uses the relative rotation and translation-direction errors of all camera pairs.

## Aggregation

Results are written per sequence to `*_per_scene_results.json`. The reported table pools `val`
and `train`, averages the sets of each scene, then the scenes of each sequence, then the 8
sequences. `2023/Day088_Corn` and `2023/Day088_Soy` are evaluated but not reported.
