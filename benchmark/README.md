# UAV3DCrop benchmark code

Code for **UAV3DCrop: Benchmarking 3D Reconstruction in Repeated Multi-Angle UAV Crop Surveys**.

Data: [UAV3DCrop](https://huggingface.co/datasets/Link-Dev/UAV3DCrop) and [UAV3DCrop_depth](https://huggingface.co/datasets/Link-Dev/UAV3DCrop_depth) · Project page: https://link-dev.github.io/UAV3DCrop/

| Folder | Contents | Paper |
|---|---|---|
| `splits/` | Held-out images of the 91 scenes | Sec. 3.4 |
| `train/` | Training and novel-view evaluation; run configuration of each method | Sec. 3.4, Table 4 |
| `export/` | Native and revised z-depth export | Sec. 3.4, A.2 |
| `plant_height/` | Canopy-height estimation | Sec. 3.4, A.3, Table 6 |
| `environment/` | Package versions | — |
| `results/` | Per-scene results of every method | Tables 4–12 |
| `baselines/` | Mip-Splatting, Scaffold-GS, CityGaussian (Track A) and the four feed-forward models (Track B) | Sec. 3.4, Tables 4–7 |

Sections 1–4 cover Nerfacto, Instant-NGP, Splatfacto and Splatfacto-big (nerfstudio). Run the commands
from this folder. Environments: `environment/requirements_nerf.txt` (Nerfacto, Instant-NGP) and
`environment/requirements_gaussian.txt` (Splatfacto, Splatfacto-big); every run used one NVIDIA A100.

## 1. Split

About every tenth image along the acquisition sequence is held out (8,806 of 88,413 posed images);
`splits/test/<year>/<scene>.txt` lists them. `splits/apply_split.py` writes the split into
`transforms_split.json` next to the scene's `transforms.json`, and `train/train.sh` runs it before
training. nerfstudio's `--train-split-fraction` selects different images on the released file names,
so do not use it.

## 2. Training and novel-view evaluation

```bash
bash train/train.sh <method> <DATA_ROOT> <SCENE_ID> [OUTPUT_DIR]   # e.g. splatfacto /data/UAV3DCrop 2025/Day028_Oat
```

All methods train for 30,000 iterations at full resolution (5280 × 3956) with seed 42; `train/configs/`
has the complete run configuration of each method. Nerfacto and Instant-NGP sample 2,048 rays per
batch from 64 training images redrawn every iteration. Splatfacto-big uses more conservative
densification on five scenes to fit in GPU memory (settings in `train/train.sh`). `train.sh` then
writes `evaluation.json` with PSNR, SSIM and LPIPS on the held-out images.

## 3. Geometry export

```bash
python export/export_depth.py --run-dir <RUN_DIR> --scene-dir <DATA_ROOT>/<SCENE_ID> \
    --frames test --downscale 1 --mode {native,revised} --output-dir <OUT>
```

Depth is camera-frame z-depth in metres. The native export has no geometric validity control. The
revised export reuses the same checkpoint:
* NeRF: pixels whose 3D point lies outside the scene bounding box expanded 1.25× are invalid;
* Gaussians: Gaussians whose center lies outside the 1.25× box or whose largest scale exceeds 2 m are
  hidden while rendering depth (checkpoints are not edited);
* both: pixels with accumulated opacity below 0.05 (`--min-accumulation`) are invalid.

Coverage (Sec. 3.4) is the fraction of pixels with positive, finite depth in the revised export. RGB
renderings are never filtered.

Depth metrics (Tables 5, 8, 9) compare each export with the depth reference of the same held-out
image, over pixels where both are positive and finite (no depth clipping). Per image: RMSE (m),
AbsRel = mean(|p − g| / g), SILog = sqrt(mean(Δ²) − mean(Δ)²) with Δ = ln p − ln g, and Pearson r.
Scene values weight images by their valid pixels; reported values average the 91 scenes equally.

## 4. Canopy height

1. Export depth of the 31 crop-date scenes in `plant_height/config.json`:
   ```bash
   python export/export_depth.py --run-dir <RUN_DIR> --scene-dir <DATA_ROOT>/<SCENE_ID> \
       --frames-file plant_height/frames/<SCENE_ID>.txt --downscale 4 --mode {native,revised} \
       --min-accumulation 0 --output-dir depth/<mode>/<method>/<SCENE_ID>
   ```
2. Estimate heights at the 210 sampling points:
   ```bash
   python plant_height/plant_height.py --depth-root depth/<mode>/<method> \
       --bare-ground-points <DATA>/ground_measurements/bare_ground_points/<mode>/<method> \
       --ground-dir <DATA>/ground_measurements --out heights.csv
   ```

Within 0.4 m of each sampling point, canopy height is the 85th (corn, soybean, wheat) or 90th (oat)
percentile of crop-date point heights minus the 50th percentile of bare-ground point heights; both
need at least 20 points. Point clouds use 50 of the 100 listed frames, depths up to 40 m, and points
from 2 m below to 3 m above the plot's ground level, capped at 2 million points. The released
bare-ground point clouds are built the same way from each method's own reconstruction and cropped to
within 2 m of the plots. For corn and soybean, whose bare-ground scenes `2025/Day007_*` are in the
dataset, `plant_height.py` builds them from `<depth-root>` when `--bare-ground-points` is omitted.

## 5. Per-scene results

`results/` holds the scene-level values behind the paper's tables. Reported values are equal-weight
means over the 91 scenes, or over the 31 scenes for canopy height (pooled statistics use the 210
points). Track B averages the scenes of each sequence, then the eight sequences.

| File | Contents | Paper |
|---|---|---|
| `nvs_per_scene.csv` | PSNR, SSIM, LPIPS and rendering FPS | Tables 4, 8 |
| `nvs_filtered_rgb_per_scene.csv` | RGB metrics of the five Gaussian methods before and after hiding the Gaussians removed by the revised filter | — |
| `depth_per_scene.csv` | RMSE, AbsRel, SILog and Pearson r for both exports; coverage (revised export, four nerfstudio methods) | Tables 5, 8, 9 |
| `height_per_scene.csv` | Mean predicted and field canopy height, bias, MAE and RMSE | Tables 6, 10–12 |
| `height_per_point.csv` | Predicted and field canopy height at each sampling point | Tables 6, 12 |
| `feedforward_per_scene.csv` | Track B metrics, mean over the subsets of each scene (definitions in `baselines/track_b/RUN.md`) | Table 7 |

## 6. Official-code baselines and Track B

See `baselines/README.md` and the `RUN.md` of each method. `baselines/track_a/hf_to_colmap.py` builds
the COLMAP input of Mip-Splatting, Scaffold-GS and CityGaussian with the released cameras held fixed.
Track B runs MASt3R, VGGT, Pi3 and MapAnything with their pretrained weights through the benchmark of
the official MapAnything repository, on RGB images only; each scene receives 9 to 69 random 36-image
subsets.

## License

MIT License (`LICENSE`). The patches in `baselines/` modify upstream code, which keeps its own license.
