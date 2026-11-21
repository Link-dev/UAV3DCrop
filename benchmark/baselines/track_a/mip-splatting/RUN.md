# Mip-Splatting (Track A)

Official code: https://github.com/autonomousvision/mip-splatting @ `dda02ab5ecf45d6edb8c540d9bb65c7e451345a9`.
The patch adds `split.json` support, full-resolution LPIPS, FPS logging, and native / revised
depth export.

## Setup

```bash
git clone https://github.com/autonomousvision/mip-splatting.git && cd mip-splatting
git checkout dda02ab5ecf45d6edb8c540d9bb65c7e451345a9
git apply /path/to/mip-splatting.patch

conda env create -f /path/to/environment.yml && conda activate mip-splatting
export CUDA_HOME=$CONDA_PREFIX TORCH_CUDA_ARCH_LIST="8.0"   # 8.0 = A100
pip install --no-build-isolation submodules/diff-gaussian-rasterization submodules/simple-knn
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

## Commands (one A100 40 GB)

```bash
S=<scene>; M=<OUT>/mip-splatting/$S/${S}_colmap   # keep this folder layout

python train.py -s <DATA>/$S -m $M -i images_undistort -r 1 --eval \
    --data_device cpu --image_uint8   # 30k iterations
python render.py -m $M --skip_train                    # test RGB + FPS
python metrics.py -m $M -r 1                           # PSNR / SSIM / LPIPS
python render.py -m $M --skip_train --render_depth     # native depth -> test/ours_30000/depth_1/
UAV3DCROP_ROOT=<DATA> python render_depth_qc.py -m $M --qc_mode both   # revised depth -> depth_1_S3/
```

## Notes

- Depth: expected camera-space z per pixel (0 where opacity < 1e-4).
- Revised depth hides Gaussians whose center is outside 1.25x the 0.5-99.5 percentile box of all
  centers or whose largest scale exceeds 2 m. The checkpoint is not changed.
- Poses and depth are in metres.
