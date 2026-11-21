# CityGaussian (Track A)

Official code: https://github.com/Linketic/CityGaussian @ `e84c7c8774dd11d3f4189be3488e1220afa20a86`.
The patch adds the two training configs, `split.json` support, a fix so that `block_id` selects
the images of that block, FPS logging, and native / revised depth export.

## Setup

```bash
git clone https://github.com/Linketic/CityGaussian.git && cd CityGaussian
git checkout e84c7c8774dd11d3f4189be3488e1220afa20a86
git apply /path/to/citygaussian.patch

conda env create -f /path/to/environment.yml && conda activate gspl
export CUDA_HOME=$CONDA_PREFIX TORCH_CUDA_ARCH_LIST="8.0"   # 8.0 = A100
pip install --no-build-isolation \
  git+https://github.com/graphdeco-inria/diff-gaussian-rasterization.git@59f5f77e3ddbac3ed9db93ec2cfe99ed6c5d121d \
  git+https://github.com/yzslab/simple-knn.git@44f764299fa305faf6ec5ebd99939e0508331503 \
  git+https://github.com/DekuLiuTesla/diff-gaussian-rasterization.git@82c31d7ad780f7ef74c4f9485985895e9da2c93a \
  git+https://github.com/DekuLiuTesla/diff-surfel-rasterization.git@9eefc03858a30bb3e5f98eccc56f077420ee2aaf
```

## Data

Build the scenes from the Hugging Face release (`hf_to_colmap.py` is in `track_a/` and takes the
held-out images from the repository's `splits/test/`; needs `colmap`):

```bash
huggingface-cli download Link-Dev/UAV3DCrop --repo-type dataset --local-dir <HF>
python hf_to_colmap.py --rgb_root <HF> --out_root <SCENES>
```

Each scene is written to `<SCENES>/<year>/Day<NNN>_<Crop>/` (`images_undistort/`, `sparse/0/`,
`split.json`, `transform_undistort.json`). Below, `<DATA>` = `<SCENES>/<year>` and
`<scene>` = `Day<NNN>_<Crop>`. Scene names repeat across years, so use one `<OUT>` per year.

## Pipeline

1. Coarse model: 30k steps on all training images.
2. Partition into a 2 x 2 grid. A training image goes to a block if its camera lies in the block
   or if hiding the block's Gaussians changes the coarse render by 1 - SSIM > 0.05.
3. Fine-tune each block from the coarse model for 30k steps on its images.
4. Merge: each block keeps the Gaussians inside its cell.

## Commands (one A100 40 GB)

```bash
export WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=0
S=<scene>; OUT=<OUT>/CityGaussian/$S; M=$OUT/final

# 1. coarse
python main.py fit --config configs/uav3dcrop_coarse.yaml -n coarse --output $OUT --data.path <DATA>/$S

# 2. partition
CKPT=$(ls $OUT/coarse/checkpoints/*step=30000*.ckpt)
CFG=$OUT/finetune.yaml
sed "s|COARSE_CKPT|$CKPT|" configs/uav3dcrop_finetune.yaml > $CFG
python utils/partition_citygs.py --config_path $CFG --force

# 3. fine-tune blocks (independent, can run in parallel)
for b in 0 1 2 3; do
  python main.py fit --config $CFG -n final --output $OUT --data.path <DATA>/$S --data.parser.block_id $b
done

# 4. merge
python utils/merge_citygs_ckpts.py $M

# 5. test RGB, NVS metrics, FPS -> $M/results.txt
python main.py test --config $CFG -n final --output $OUT --data.path <DATA>/$S --save_val

# 6. native depth -> $M/val_depth/
SAVE_VAL_DEPTH=1 SAVE_VAL_DEPTH_DIR=$M/val_depth \
  python main.py test --config $CFG -n final --output $OUT --data.path <DATA>/$S

# 7. revised depth -> $M/val_depth_S3/
UAV3DCROP_ROOT=<DATA> QC_MODE=both QC_SCENE=$S SAVE_VAL_DEPTH=1 SAVE_VAL_DEPTH_DIR=$M/val_depth_S3 \
  python main.py test --config $CFG -n final --output $OUT --data.path <DATA>/$S \
    --trainer.callbacks+=qc_callback.QCFilterCallback
```

Steps 6 and 7 overwrite `results.txt`; copy the step-5 results aside first.

## Notes

- Depth: expected camera-space z per pixel (0 where opacity < 1e-4).
- Revised depth hides Gaussians whose center is outside 1.25x the 0.5-99.5 percentile box of all
  centers or whose largest scale exceeds 2 m. The checkpoint is not changed.
- Poses and depth are in metres.
