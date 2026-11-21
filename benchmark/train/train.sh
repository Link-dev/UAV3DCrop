#!/usr/bin/env bash
# Track A training and novel-view evaluation for one UAV3DCrop scene (nerfstudio 1.1.5).
#
#   bash train.sh <method> <DATA_ROOT> <SCENE_ID> [OUTPUT_DIR]
#   e.g. bash train.sh splatfacto /data/UAV3DCrop 2025/Day028_Oat outputs
#
# <method> is one of: nerfacto, instant-ngp, splatfacto, splatfacto-big.
# All four methods use the same fixed 90/10 split (held-out images listed in ../splits/test/, written
# into transforms_split.json by ../splits/apply_split.py), full-resolution images (5280 x 3956),
# 30,000 iterations and seed 42. The resulting run settings are recorded in configs/<method>.yml.
set -euo pipefail

METHOD=${1:?method}
DATA_ROOT=${2:?data root}
SCENE=${3:?scene id, e.g. 2025/Day028_Oat}
OUT=${4:-outputs}
HERE=$(cd "$(dirname "$0")" && pwd)

# Fixed split: nerfstudio reads train/val/test_filenames from this copy of transforms.json.
python "$HERE/../splits/apply_split.py" "$DATA_ROOT/$SCENE" "$HERE/../splits/test/$SCENE.txt"
DATA="$DATA_ROOT/$SCENE/transforms_split.json"

case "$METHOD" in
  nerfacto)
    python "$HERE/train_nerfacto.py" \
      --data "$DATA" --output-dir "$OUT" --experiment-name "$SCENE" \
      --max-num-iterations 30000
    ;;
  instant-ngp)
    ns-train instant-ngp \
      --machine.num-devices 1 --output-dir "$OUT" --experiment-name "$SCENE" --vis tensorboard \
      --steps-per-eval-batch 1000000 --steps-per-eval-image 1000000 --steps-per-eval-all-images 1000000 \
      --pipeline.datamanager.train-num-rays-per-batch 2048 \
      --pipeline.datamanager.train-num-images-to-sample-from 64 \
      --pipeline.datamanager.train-num-times-to-repeat-images 1 \
      --pipeline.model.eval-num-rays-per-chunk 4096 \
      --max-num-iterations 30000 \
      nerfstudio-data --data "$DATA" --downscale-factor 1
    ;;
  splatfacto|splatfacto-big)
    # gsplat compiles its CUDA kernels on first use and links against the environment's libcudart.
    export LIBRARY_PATH="${CONDA_PREFIX:?activate the Gaussian environment}/lib:${LIBRARY_PATH:-}"
    export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
    # Splatfacto-big used more conservative densification on five scenes to fit in GPU memory.
    EXTRA=()
    if [[ "$METHOD" == splatfacto-big ]]; then
      case "$SCENE" in
        2024/Day036_Corn2|2024/Day074_Corn2) DENSIFY=(0.00075 0.008 150 10000) ;;
        2024/Day099_Corn2|2024/Day120_Corn2) DENSIFY=(0.001 0.01 200 8000) ;;
        2025/Day001_Wheat)                   DENSIFY=(0.00065 0.0075 150 12000) ;;
        *)                                   DENSIFY=() ;;
      esac
      if (( ${#DENSIFY[@]} )); then
        EXTRA=(--pipeline.model.densify-grad-thresh "${DENSIFY[0]}"
               --pipeline.model.cull-alpha-thresh "${DENSIFY[1]}"
               --pipeline.model.refine-every "${DENSIFY[2]}"
               --pipeline.model.stop-split-at "${DENSIFY[3]}")
      fi
    fi
    ns-train "$METHOD" \
      --machine.num-devices 1 --output-dir "$OUT" --experiment-name "$SCENE" --vis tensorboard \
      --max-num-iterations 30000 ${EXTRA[@]+"${EXTRA[@]}"} \
      nerfstudio-data --data "$DATA" --downscale-factor 1
    ;;
  *) echo "unknown method: $METHOD" >&2; exit 2 ;;
esac

# Novel-view evaluation on the held-out 10% (PSNR / SSIM / LPIPS -> evaluation.json).
RUN_DIR=$(ls -td "$OUT/$SCENE/$METHOD"/*/ | head -1)
case "$METHOD" in
  nerfacto|instant-ngp)
    python "$HERE/eval_nerf.py" --load-config "$RUN_DIR/config.yml" --output-path "$RUN_DIR/evaluation.json"
    ;;
  *)
    ns-eval --load-config "$RUN_DIR/config.yml" --output-path "$RUN_DIR/evaluation.json"
    ;;
esac
