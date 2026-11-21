#!/usr/bin/env python3
"""Convert the UAV3DCrop release on Hugging Face into the WAI layout read by the Track B
benchmark (map-anything-trackb.patch).

Inputs (local copies of the two Hugging Face datasets):
    <rgb_root>/<year>/Day<NNN>_<Crop>/transforms.json, images/<n>.JPG       Link-Dev/UAV3DCrop
    <depth_root>/<year>/Day<NNN>_<Crop>_depth/<n>_depth.tif                 Link-Dev/UAV3DCrop_depth

Output (one folder per field sequence, as expected by configs/dataset/*_benchmark_wai):
    <out_root>/<sequence>/Day<NNN>_<Crop>/
        images/<n>.jpg        undistorted, downsampled 8x (660 x 494), JPEG quality 95
        depth/<n>.exr         reference z-depth in metres, same 660 x 494 pixel grid
        scene_meta.json       PINHOLE intrinsics (scaled by 1/8) and per-frame camera-to-world poses

Per frame:
    image  = INTER_AREA downsample (8x) of cv2.undistort(<n>.JPG, K, [k1, k2, p1, p2, k3])
    depth  = every 8th pixel of <n>_depth.tif (the depth maps are on the undistorted pixel grid)
    pose   = diag(-1, 1, -1, 1) @ transform_matrix @ diag(1, -1, -1, 1)
             (camera-to-world with OpenCV camera axes, as read by the loader)
Frames without a depth map are skipped. The remaining frames keep the order of the image
index n, which fixes the multi-view sets drawn by the benchmark.

Usage:
    huggingface-cli download Link-Dev/UAV3DCrop --repo-type dataset --local-dir <rgb_root>
    huggingface-cli download Link-Dev/UAV3DCrop_depth --repo-type dataset --local-dir <depth_root>
    python hf_to_wai.py --rgb_root <rgb_root> --depth_root <depth_root> --out_root <out_root> [--workers 16]
    # or a subset of scenes:  --scenes 2025/Day007_Wheat 2023/Day006_Corn
Requires numpy, opencv-python (OpenEXR enabled through OPENCV_IO_ENABLE_OPENEXR) and tifffile.
"""
import os
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime

import cv2
import numpy as np
import tifffile

DOWN = 8
WORLD_FLIP = np.diag([-1.0, 1.0, -1.0, 1.0])
OPENGL_TO_OPENCV = np.diag([1.0, -1.0, -1.0, 1.0])

# (year, crop) of a Hugging Face scene -> sequence folder used by the benchmark configs
SEQUENCE = {
    ("2023", "Corn"): "2023corn", ("2023", "Soy"): "2023soy",
    ("2024", "Corn1"): "2024corn1", ("2024", "Corn2"): "2024corn2",
    ("2025", "Wheat"): "2025wheat", ("2025", "Oat"): "2025oat",
    ("2025", "Corn"): "2025corn", ("2025", "soybean"): "2025soybean",
}


def frame_index(frame):
    return int(os.path.splitext(os.path.basename(frame["file_path"]))[0])


def convert_frame(job):
    rgb_dir, depth_dir, out_dir, n, K, dist, size = job
    w8, h8 = size
    img = cv2.imread(os.path.join(rgb_dir, "images", f"{n}.JPG"), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(os.path.join(rgb_dir, "images", f"{n}.JPG"))
    und = cv2.undistort(img, K, dist)
    small = cv2.resize(und, (w8, h8), interpolation=cv2.INTER_AREA)
    cv2.imwrite(os.path.join(out_dir, "images", f"{n}.jpg"), small, [cv2.IMWRITE_JPEG_QUALITY, 95])

    d = tifffile.imread(os.path.join(depth_dir, f"{n}_depth.tif")).astype(np.float32)
    cv2.imwrite(os.path.join(out_dir, "depth", f"{n}.exr"), d[::DOWN, ::DOWN][:h8, :w8])
    return n


def convert_scene(rgb_root, depth_root, out_root, scene, workers):
    year, name = scene.split("/")
    crop = name.split("_", 1)[1]
    seq = SEQUENCE[(year, crop)]
    rgb_dir = os.path.join(rgb_root, year, name)
    depth_dir = os.path.join(depth_root, year, f"{name}_depth")
    out_dir = os.path.join(out_root, seq, name)
    os.makedirs(os.path.join(out_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "depth"), exist_ok=True)

    tf = json.load(open(os.path.join(rgb_dir, "transforms.json")))
    K = np.array([[tf["fl_x"], 0, tf["cx"]], [0, tf["fl_y"], tf["cy"]], [0, 0, 1]], np.float64)
    dist = np.array([tf.get(k, 0.0) for k in ("k1", "k2", "p1", "p2", "k3")], np.float64)
    w8, h8 = int(tf["w"]) // DOWN, int(tf["h"]) // DOWN
    all_frames = sorted(tf["frames"], key=frame_index)
    frames = [f for f in all_frames
              if os.path.exists(os.path.join(depth_dir, f"{frame_index(f)}_depth.tif"))]
    no_depth = [frame_index(f) for f in all_frames if f not in frames]

    jobs = [(rgb_dir, depth_dir, out_dir, frame_index(f), K, dist, (w8, h8)) for f in frames]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        list(ex.map(convert_frame, jobs, chunksize=4))

    meta = {
        "scene_name": name,
        "dataset_name": "uav3dcrop",
        "version": "1.0",
        "last_modified": datetime.now().isoformat(),
        "camera_model": "PINHOLE",
        "camera_convention": "opencv",
        "fl_x": tf["fl_x"] / DOWN, "fl_y": tf["fl_y"] / DOWN,
        "cx": tf["cx"] / DOWN, "cy": tf["cy"] / DOWN,
        "w": w8, "h": h8,
        "scale": 1.0,
        "depth_source": "uav3dcrop_depth",
        "frames": [],
        "scene_modalities": {},
        "frame_modalities": {
            "depth": {"frame_key": "depth", "format": "depth"},
            "image": {"frame_key": "image", "format": "image"},
        },
        "_applied_transformation": np.eye(4).tolist(),
    }
    for f in frames:
        n = frame_index(f)
        meta["frames"].append({
            "frame_name": str(n),
            "image": f"images/{n}.jpg",
            "depth": f"depth/{n}.exr",
            "transform_matrix": (WORLD_FLIP @ np.array(f["transform_matrix"], np.float64)
                                 @ OPENGL_TO_OPENCV).tolist(),
        })
    with open(os.path.join(out_dir, "scene_meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"{scene} -> {out_dir}: {len(frames)} frames"
          + (f" ({len(no_depth)} without depth skipped)" if no_depth else ""))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rgb_root", required=True, help="local copy of Link-Dev/UAV3DCrop")
    ap.add_argument("--depth_root", required=True, help="local copy of Link-Dev/UAV3DCrop_depth")
    ap.add_argument("--out_root", required=True, help="WAI output root (machine config root_data_dir)")
    ap.add_argument("--scenes", nargs="*", default=None, help="e.g. 2025/Day007_Wheat; default: all scenes")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    scenes = args.scenes
    if not scenes:
        scenes = sorted(f"{y}/{s}" for y in os.listdir(args.rgb_root) if y.isdigit()
                        for s in os.listdir(os.path.join(args.rgb_root, y))
                        if os.path.isfile(os.path.join(args.rgb_root, y, s, "transforms.json")))
    for s in scenes:
        convert_scene(args.rgb_root, args.depth_root, args.out_root, s, args.workers)


if __name__ == "__main__":
    main()
