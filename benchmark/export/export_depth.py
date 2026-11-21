#!/usr/bin/env python3
"""Export camera-frame z-depth (metres) from a trained nerfstudio run, native or revised.

Used for both Track A geometry outputs:
  * depth benchmark:  --frames test        --downscale 1   (held-out images, full resolution)
  * canopy height:    --frames-file <list> --downscale 4 --min-accumulation 0
                      (100 training cameras per scene, lists in ../plant_height/frames/; the paper's
                      canopy-height exports used no opacity threshold)

Depth definition
  NeRF methods (Nerfacto: `expected_depth`, Instant-NGP: `depth`) return distance along the unit
  ray; it is converted to z-depth with cos(theta) = 1 / sqrt(x_n^2 + y_n^2 + 1). gsplat already
  renders z-depth, so Gaussian depth is used as is. Depth is converted from nerfstudio scene units
  to metres with the run's dataparser scale.

Export modes (the checkpoint is never modified)
  native   no geometric validity control.
  revised  NeRF: pixels whose back-projected point falls outside the scene AABB expanded 1.25x are
           set invalid. Gaussians: while rendering depth, Gaussians whose centre lies outside the
           1.25x AABB or whose largest scale exceeds 2 m are made transparent; their opacity is
           restored after each render. In both cases pixels whose accumulated opacity is below 0.05
           (nothing rendered there, e.g. holes left by the hidden Gaussians) are also invalid; without
           this, splatfacto fills empty pixels with the image's maximum depth. RGB renderings are never
           filtered.

Outputs in --output-dir
  <image stem>.npy  float32 depth in metres, NaN = invalid
  transforms.json   pinhole intrinsics at the export resolution and OpenCV camera-to-world poses
                    in the dataset frame (metres), so the maps can be back-projected directly
  meta.json         settings and per-frame / Gaussian-filter statistics

    python export_depth.py --run-dir outputs/2025/Day028_Oat/splatfacto/<timestamp> \
        --scene-dir <DATA_ROOT>/2025/Day028_Oat --frames test --mode revised --output-dir depth/...
"""
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


# --------------------------------------------------------------------------- loading

def load_dataparser_scale(run_dir: Path) -> float:
    """nerfstudio scene units per metre (dataparser_transforms.json 'scale')."""
    obj = json.loads((run_dir / "dataparser_transforms.json").read_text())
    scale = float(obj["scale"])
    if scale <= 0:
        raise ValueError(f"invalid dataparser scale {scale}")
    return scale


def load_pipeline(run_dir: Path):
    import yaml
    from nerfstudio.engine.trainer import TrainerConfig
    from nerfstudio.pipelines.base_pipeline import Pipeline

    config = yaml.load((run_dir / "config.yml").read_text(), Loader=yaml.Loader)
    assert isinstance(config, TrainerConfig)
    config.load_dir = run_dir / "nerfstudio_models"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline = config.pipeline.setup(device=device, test_mode="inference")
    assert isinstance(pipeline, Pipeline)
    pipeline.eval()

    steps = sorted(int(p.name.split("-")[1].split(".")[0]) for p in config.load_dir.glob("step-*.ckpt"))
    if not steps:
        raise FileNotFoundError(f"no checkpoint in {config.load_dir}")
    state = torch.load(config.load_dir / f"step-{steps[-1]:09d}.ckpt", map_location="cpu")
    pipeline.load_pipeline(state["pipeline"], state["step"])
    return pipeline, steps[-1]


def dataparser_cameras(pipeline) -> Tuple[Dict[str, np.ndarray], List[str], List[str]]:
    """Image stem -> nerfstudio camera-to-world (OpenGL), plus train and test stems."""
    stem_to_c2w: Dict[str, np.ndarray] = {}
    stems = {}
    for attr in ("train_dataset", "eval_dataset"):
        ds = getattr(pipeline.datamanager, attr)
        stems[attr] = [Path(f).stem for f in ds.image_filenames]
        for k, stem in enumerate(stems[attr]):
            stem_to_c2w[stem] = ds.cameras.camera_to_worlds[k].cpu().numpy().astype(np.float32)
    return stem_to_c2w, stems["train_dataset"], stems["eval_dataset"]


# --------------------------------------------------------------------------- geometry

def gl_to_cv(c2w_gl: np.ndarray) -> np.ndarray:
    c2w = c2w_gl.copy()
    c2w[:3, 1] *= -1.0
    c2w[:3, 2] *= -1.0
    return c2w


def make_camera(c2w_gl, fx, fy, cx, cy, width, height, device):
    from nerfstudio.cameras.cameras import CameraType, Cameras

    camera_type = getattr(CameraType.PERSPECTIVE, "value", CameraType.PERSPECTIVE)
    cam = Cameras(
        camera_to_worlds=torch.from_numpy(c2w_gl[:3, :4]).unsqueeze(0).float(),
        fx=torch.tensor([fx], dtype=torch.float32),
        fy=torch.tensor([fy], dtype=torch.float32),
        cx=torch.tensor([cx], dtype=torch.float32),
        cy=torch.tensor([cy], dtype=torch.float32),
        width=torch.tensor([width], dtype=torch.int32),
        height=torch.tensor([height], dtype=torch.int32),
        camera_type=torch.tensor([int(camera_type)], dtype=torch.int32),
    )
    return cam.to(device)


def render_depth(pipeline, camera, is_gaussian: bool, gaussian_excluded: Optional[torch.Tensor]):
    saved = None
    try:
        with torch.no_grad():
            if is_gaussian:
                if gaussian_excluded is not None:
                    saved = pipeline.model.opacities[gaussian_excluded].clone()
                    pipeline.model.opacities[gaussian_excluded] = -100.0
                outputs = pipeline.model.get_outputs(camera.to(pipeline.device))
            else:
                outputs = pipeline.model.get_outputs_for_camera_ray_bundle(camera.generate_rays(camera_indices=0))
    finally:
        if saved is not None:
            with torch.no_grad():
                pipeline.model.opacities[gaussian_excluded] = saved
    key = "depth" if is_gaussian else ("expected_depth" if "expected_depth" in outputs else "depth")
    depth = np.squeeze(outputs[key].detach().cpu().numpy()).astype(np.float32)
    acc = outputs.get("accumulation")
    acc = None if acc is None else np.squeeze(acc.detach().cpu().numpy()).astype(np.float32)
    return depth, acc, key


def gaussian_filter_mask(model, metres_per_scene_unit: float, aabb_multiplier: float, max_scale_m: float):
    """Revised export: Gaussians to hide (centre outside expanded AABB, or largest scale > max)."""
    with torch.no_grad():
        means = model.means.detach()
        aabb = model.scene_box.aabb.to(device=means.device, dtype=means.dtype)
        center, half = aabb.mean(dim=0), (aabb[1] - aabb[0]) / 2
        inside = torch.all(torch.abs(means - center) <= half * aabb_multiplier, dim=-1)
        max_axis_m = torch.exp(model.scales.detach().float()).max(dim=-1).values * metres_per_scene_unit
        small = max_axis_m <= max_scale_m
        excluded = ~(inside & small)
    total = int(excluded.numel())
    return excluded, {
        "gaussian_count": total,
        "excluded_count": int(excluded.sum().item()),
        "outside_expanded_aabb": int((~inside).sum().item()),
        "oversized": int((~small).sum().item()),
    }


def nerf_aabb_mask(z_scene, c2w_gl, fx, fy, cx, cy, aabb: np.ndarray, aabb_multiplier: float):
    """Revised export: keep z-depth pixels whose point lies inside the expanded scene AABB."""
    h, w = z_scene.shape
    uu, vv = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    valid = np.isfinite(z_scene) & (z_scene > 0)
    pts_cam = np.stack((((uu - cx) / fx) * z_scene, ((vv - cy) / fy) * z_scene, z_scene), axis=-1)
    c2w = gl_to_cv(c2w_gl)
    pts = pts_cam @ c2w[:3, :3].T + c2w[:3, 3]
    center, half = aabb.mean(axis=0), (aabb[1] - aabb[0]) / 2
    keep = valid & np.all(np.abs(pts - center) <= half * aabb_multiplier, axis=-1)
    return np.where(keep, z_scene, np.nan).astype(np.float32), int(valid.sum() - keep.sum())


# --------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True, type=Path, help="nerfstudio run directory (has config.yml)")
    ap.add_argument("--scene-dir", required=True, type=Path, help="scene directory with transforms.json")
    ap.add_argument("--output-dir", required=True, type=Path)
    sel = ap.add_mutually_exclusive_group(required=True)
    sel.add_argument("--frames", choices=("test", "train"), help="all held-out or all training images")
    sel.add_argument("--frames-file", type=Path, help="text file of image paths, exported in this order")
    ap.add_argument("--mode", choices=("native", "revised"), required=True)
    ap.add_argument("--downscale", type=int, default=1)
    ap.add_argument("--aabb-multiplier", type=float, default=1.25)
    ap.add_argument("--max-gaussian-scale-m", type=float, default=2.0)
    ap.add_argument("--min-accumulation", type=float, default=0.05,
                    help="revised mode only: pixels whose accumulated opacity is below this are invalid "
                         "(0.05 as in the paper; native mode applies no threshold)")
    args = ap.parse_args()

    run_dir = args.run_dir.resolve()
    meta = json.loads((args.scene_dir / "transforms.json").read_text())
    metres_per_scene_unit = 1.0 / load_dataparser_scale(run_dir)   # dataset poses are in metres

    pipeline, step = load_pipeline(run_dir)
    model_name = type(pipeline.model).__name__
    is_gaussian = "Splat" in model_name or "Gaussian" in model_name
    stem_to_c2w, train_stems, test_stems = dataparser_cameras(pipeline)

    by_stem = {Path(f["file_path"]).stem: f for f in meta["frames"]}
    if args.frames_file:
        wanted = [Path(ln.strip()).stem for ln in args.frames_file.read_text().splitlines() if ln.strip()]
    else:
        wanted = test_stems if args.frames == "test" else train_stems
    missing = [s for s in wanted if s not in by_stem or s not in stem_to_c2w]
    if missing:
        raise RuntimeError(f"{len(missing)} requested images are not cameras of this run (first: {missing[0]})")

    s = float(args.downscale)
    fx, fy = float(meta["fl_x"]) / s, float(meta.get("fl_y", meta["fl_x"])) / s
    cx, cy = float(meta["cx"]) / s, float(meta["cy"]) / s
    w, h = int(round(int(meta["w"]) / s)), int(round(int(meta["h"]) / s))
    uu, vv = np.meshgrid(np.arange(w), np.arange(h))
    ray_to_z = (1.0 / np.sqrt(((uu - cx) / fx) ** 2 + ((vv - cy) / fy) ** 2 + 1.0)).astype(np.float32)

    revised = args.mode == "revised"
    excluded, gauss_stats, aabb = None, None, None
    if revised and is_gaussian:
        excluded, gauss_stats = gaussian_filter_mask(
            pipeline.model, metres_per_scene_unit, args.aabb_multiplier, args.max_gaussian_scale_m)
    if revised and not is_gaussian:
        aabb = pipeline.model.scene_box.aabb.detach().cpu().numpy().astype(np.float32)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_frames, frame_stats = [], []
    for i, stem in enumerate(wanted):
        c2w_gl = stem_to_c2w[stem]
        depth, acc, key = render_depth(pipeline, make_camera(c2w_gl, fx, fy, cx, cy, w, h, pipeline.device),
                                       is_gaussian, excluded)
        removed = None
        if not is_gaussian:
            depth = depth * ray_to_z
            if aabb is not None:
                depth, removed = nerf_aabb_mask(depth, c2w_gl, fx, fy, cx, cy, aabb, args.aabb_multiplier)
        depth_m = (depth * metres_per_scene_unit).astype(np.float32)
        low_acc = None
        if revised and acc is not None and args.min_accumulation > 0:
            keep = acc >= args.min_accumulation
            low_acc = int((~keep & np.isfinite(depth_m)).sum())
            depth_m = np.where(keep, depth_m, np.nan).astype(np.float32)
        np.save(args.output_dir / f"{stem}.npy", depth_m)

        raw = np.asarray(by_stem[stem]["transform_matrix"], dtype=np.float32)
        out_frames.append({"file_path": by_stem[stem]["file_path"], "transform_matrix": gl_to_cv(raw).tolist()})
        valid = np.isfinite(depth_m) & (depth_m > 0)
        frame_stats.append({"stem": stem, "depth_key": key, "valid_fraction": float(valid.mean()),
                            "nerf_aabb_removed_pixels": removed, "low_accumulation_pixels": low_acc})
        if (i + 1) % 10 == 0 or i == len(wanted) - 1:
            print(f"[{i + 1}/{len(wanted)}] {stem} valid={valid.mean() * 100:.1f}%")

    (args.output_dir / "transforms.json").write_text(json.dumps(
        {"fl_x": fx, "fl_y": fy, "cx": cx, "cy": cy, "w": w, "h": h, "frames": out_frames}, indent=1))
    (args.output_dir / "meta.json").write_text(json.dumps({
        "model": model_name, "checkpoint_step": step, "mode": args.mode, "downscale": args.downscale,
        "depth": "camera-frame z-depth in metres; NaN = invalid",
        "metres_per_scene_unit": metres_per_scene_unit,
        "aabb_multiplier": args.aabb_multiplier if revised else None,
        "max_gaussian_scale_m": args.max_gaussian_scale_m if revised and is_gaussian else None,
        "gaussian_filter": gauss_stats, "min_accumulation": args.min_accumulation if revised else None,
        "frames": frame_stats,
    }, indent=1))
    print(f"[done] {len(wanted)} depth maps -> {args.output_dir}")


if __name__ == "__main__":
    main()
