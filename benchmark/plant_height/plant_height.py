#!/usr/bin/env python3
"""Canopy height at the field sampling points from exported depth (paper Sec. 3.4 and A.3).

For each crop-date scene and its plot's bare-ground survey, reconstructed by the SAME method:
  1. take the 100 depth maps written by export_depth.py (--frames-file frames/<scene>.txt
     --downscale 4) and use 50 of them, uniformly spaced in list order;
  2. back-project every valid pixel with depth <= 40 m into the dataset frame (metres) and keep
     points with ground_z - 2 m < z < ground_z + 3 m; if more than 2,000,000 points remain, keep a
     random 2,000,000 (numpy RandomState(0));
  3. at every sampling point, collect points within a 0.4 m horizontal radius:
        ground height = 50th percentile of bare-ground point z
        canopy height = 85th (corn, soybean, wheat) or 90th (oat) percentile of crop-date point z
        predicted height = canopy height - ground height
     Both sets need at least 20 points; otherwise the prediction is missing.

The bare-ground points are part of the dataset:
ground_measurements/bare_ground_points/<mode>/<method>/<bare-ground scene>.ply holds the result of
steps 1-2 for that method's bare-ground reconstruction, cropped to within 2 m (horizontal) of the
plot's sampling points. With --bare-ground-points only the 31 crop-date scenes need exporting:

    python plant_height.py --depth-root depth/revised/splatfacto \
        --bare-ground-points <DATA>/ground_measurements/bare_ground_points/revised/splatfacto \
        --ground-dir <DATA>/ground_measurements --out heights_splatfacto_revised.csv

--depth-root holds one export_depth.py output directory per scene, at <depth-root>/<scene id>.
Without --bare-ground-points the bare ground is built from <depth-root>/<bare-ground scene> in the
same way as the crop-date scenes; this needs your own bare-ground reconstruction and is possible for
corn and soybean only (2025/Day007_Corn and 2025/Day007_soybean are dataset scenes; the wheat and
oat bare-ground images are not released).
--ground-dir is the dataset's ground_measurements/ folder: sampling-point positions come from
ground_plots.csv and field canopy heights (mean of five readings) from ground_measurements.csv,
paired with the scene given in its matched_scene column.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


def scene_points(export_dir: Path, cfg: dict, ground_z: float) -> np.ndarray:
    tf = json.loads((export_dir / "transforms.json").read_text())
    fx, fy, cx, cy = (np.float32(tf[k]) for k in ("fl_x", "fl_y", "cx", "cy"))
    frames = tf["frames"]
    n = cfg["frames_used"]
    if len(frames) > n:
        frames = [frames[int(round(i * (len(frames) - 1) / (n - 1)))] for i in range(n)]
    chunks = []
    for fr in frames:
        depth = np.load(export_dir / f"{Path(fr['file_path']).stem}.npy")
        valid = np.isfinite(depth) & (depth > 0) & (depth <= cfg["max_depth_m"])
        v, u = np.nonzero(valid)
        z = depth[valid].astype(np.float32)
        cam = np.stack([(u - cx) * z / fx, (v - cy) * z / fy, z], axis=-1)
        c2w = np.asarray(fr["transform_matrix"], dtype=np.float32)       # OpenCV, metres
        chunks.append((cam @ c2w[:3, :3].T + c2w[:3, 3]).astype(np.float32))
    pts = np.concatenate(chunks)
    lo, hi = ground_z + cfg["z_band_m"][0], ground_z + cfg["z_band_m"][1]
    pts = pts[(pts[:, 2] > lo) & (pts[:, 2] < hi)]
    if len(pts) > cfg["max_points"]:
        pts = pts[np.random.RandomState(0).choice(len(pts), cfg["max_points"], replace=False)]
    return pts


def read_points(path: Path) -> np.ndarray:
    """x, y, z (float32, metres) of a released bare-ground point cloud (binary little-endian PLY)."""
    raw = path.read_bytes()
    end = raw.index(b"end_header\n") + len(b"end_header\n")
    head = raw[:end].decode("ascii").split("\n")
    assert head[1] == "format binary_little_endian 1.0", path
    assert head[3:6] == ["property float x", "property float y", "property float z"], path
    n = int(head[2].split()[-1])
    return np.frombuffer(raw, dtype="<f4", count=3 * n, offset=end).reshape(n, 3)


def near(pts: np.ndarray, xy, radius: float) -> np.ndarray:
    """z of the points within `radius` (horizontal) of xy."""
    return pts[np.hypot(pts[:, 0] - xy[0], pts[:, 1] - xy[1]) < radius, 2]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=Path(__file__).with_name("config.json"))
    ap.add_argument("--depth-root", type=Path, required=True)
    ap.add_argument("--bare-ground-points", type=Path,
                    help="ground_measurements/bare_ground_points/<mode>/<method> of the dataset "
                         "(default: build the bare ground from --depth-root)")
    ap.add_argument("--ground-dir", type=Path, required=True, help="the dataset's ground_measurements/ folder")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    CFG = json.loads(args.config.read_text())
    if args.bare_ground_points:
        assert CFG["radius_m"] <= CFG["bare_ground_points_crop_m"], "released bare-ground points stop 2 m from each plot"

    sites = {}                                   # group -> plot_id -> (x, y)
    for r in csv.DictReader(open(args.ground_dir / "ground_plots.csv")):
        sites.setdefault(r["group"], {})[r["plot_id"]] = (float(r["x_m"]), float(r["y_m"]))
    field = {}                                   # (scene, plot_id) -> canopy height in metres
    for r in csv.DictReader(open(args.ground_dir / "ground_measurements.csv")):
        if r["variable"] == "canopy_height":
            assert r["unit"] == "cm", r
            field[(r["matched_scene"], r["plot_id"])] = float(r["value"]) / 100.0

    rows = []
    for crop, p in CFG["plots"].items():
        if args.bare_ground_points:
            ground = read_points(args.bare_ground_points / f"{p['bare_ground_scene']}.ply")
        else:
            ground = scene_points(args.depth_root / p["bare_ground_scene"], CFG, p["ground_z_m"])
        for scene in p["scenes"]:
            canopy = scene_points(args.depth_root / scene, CFG, p["ground_z_m"])
            for plot_id, xy in sites[p["group"]].items():
                zc, zg = near(canopy, xy, CFG["radius_m"]), near(ground, xy, CFG["radius_m"])
                ok = len(zc) >= CFG["min_points"] and len(zg) >= CFG["min_points"]
                h = float(np.percentile(zc, p["canopy_percentile"]) - np.percentile(zg, CFG["ground_percentile"])) \
                    if ok else float("nan")
                rows.append({"crop": crop, "scene_id": scene, "plot_id": plot_id,
                             "pred_height_m": round(h, 4), "field_height_m": field.get((scene, plot_id), "")})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # Scene-macro errors as in the paper's Table 6 (errors averaged within scene, then over scenes).
    paired = [r for r in rows if r["field_height_m"] != "" and np.isfinite(r["pred_height_m"])]
    if paired:
        by_scene = {}
        for r in paired:
            by_scene.setdefault(r["scene_id"], []).append(r["pred_height_m"] - r["field_height_m"])
        mae = np.mean([np.mean(np.abs(e)) for e in by_scene.values()])
        rmse = np.mean([np.sqrt(np.mean(np.square(e))) for e in by_scene.values()])
        print(f"{len(paired)} points in {len(by_scene)} scenes: scene-macro MAE {mae:.3f} m, RMSE {rmse:.3f} m")
    print(f"wrote {len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
