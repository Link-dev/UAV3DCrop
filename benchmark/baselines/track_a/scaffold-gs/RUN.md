# Scaffold-GS (Track A)

Official code: https://github.com/city-super/Scaffold-GS @ `59c833b56bbbf510f3f64d40f81721995caced66`.
The patch adds `split.json` support, full-resolution LPIPS, FPS logging, and native / revised
depth export.

## Setup

```bash
git clone https://github.com/city-super/Scaffold-GS.git && cd Scaffold-GS
git checkout 59c833b56bbbf510f3f64d40f81721995caced66
git apply /path/to/scaffold-gs.patch

conda env create -f /path/to/environment.yml && conda activate scaffold_gs
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
export WANDB_MODE=disabled
Y=<year>; S=<scene>; M=<OUT>/Scaffold-GS/$S/${S}_colmap   # keep this folder layout
VOX=$(python -c "import json,sys; print(json.load(open(sys.argv[1])).get(sys.argv[2], 0.001))" \
      /path/to/voxel_size.json $Y/$S)

python train.py -s <DATA>/$S -m $M -i images_undistort -r 1 --eval --voxel_size $VOX \
    --data_device cpu --image_uint8   # 30k iterations
python render.py -m $M --skip_train                    # test RGB + FPS
python metrics.py -m $M                                # PSNR / SSIM / LPIPS
python render.py -m $M --skip_train --render_depth     # native depth -> test/ours_30000/depth/
UAV3DCROP_ROOT=<DATA> python render_depth_qc.py -m $M --qc_mode both   # revised depth -> depth_S3/
```

## Notes

- Depth: expected camera-space z per pixel (0 where opacity < 1e-4).
- Revised depth hides decoded Gaussians whose center is outside 1.25x the 0.5-99.5 percentile box
  of the anchors or whose largest scale exceeds 2 m. The checkpoint is not changed.
- Poses and depth are in metres.
- `voxel_size.json` gives the anchor voxel size (metres) for the 2023 scenes, whose benchmark runs
  used the default 0.001 in normalized units; all other scenes use the default 0.001.
