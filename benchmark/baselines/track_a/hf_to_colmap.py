#!/usr/bin/env python3
"""Convert the UAV3DCrop release on Hugging Face (Link-Dev/UAV3DCrop) into the scene layout read
by the Track A patches (Mip-Splatting, Scaffold-GS, CityGaussian).

Input:   <rgb_root>/<year>/Day<NNN>_<Crop>/transforms.json, images/<n>.JPG
Output:  <out_root>/<year>/Day<NNN>_<Crop>/
             images_undistort/<n>.JPG   undistorted full-resolution images
             sparse/0/                  COLMAP model: PINHOLE camera, the released poses, triangulated points
             split.json                 {"train": [...], "val": [...]}; val = the benchmark held-out images
             transform_undistort.json   {"scale": 1.0} (poses are in metres)

The COLMAP model keeps the released cameras fixed: features are extracted and matched
(sequential matcher), and points are triangulated with `colmap point_triangulator`.
Held-out images are read from the repository's splits/test/<year>/<scene>.txt.

Usage:
    huggingface-cli download Link-Dev/UAV3DCrop --repo-type dataset --local-dir <rgb_root>
    python hf_to_colmap.py --rgb_root <rgb_root> --out_root <out_root> [--scenes 2024/Day036_Corn2 ...]
Then use <out_root>/<year> as <DATA> and Day<NNN>_<Crop> as <scene> in RUN.md.
Requires numpy, opencv-python and the colmap executable (--colmap).
"""
import argparse
import json
import os
import shutil
import sqlite3
import subprocess
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SPLITS = os.path.join(HERE, "..", "..", "splits", "test")   # <repo>/splits/test/<year>/<scene>.txt


def frame_index(frame):
    return int(os.path.splitext(os.path.basename(frame["file_path"]))[0])


def undistort(job):
    src, dst, K, dist = job
    if not os.path.exists(dst):
        img = cv2.imread(src, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(src)
        tmp = dst[:-4] + ".part.JPG"   # rename when complete, so an interrupted run is redone
        cv2.imwrite(tmp, cv2.undistort(img, K, dist), [cv2.IMWRITE_JPEG_QUALITY, 95])
        os.replace(tmp, dst)


def rotmat_to_qvec(R):
    # as rotmat2qvec in COLMAP's scripts/python/read_write_model.py (stable near 180 degrees)
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    Km = np.array([[Rxx - Ryy - Rzz, 0, 0, 0],
                   [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
                   [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
                   [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
    vals, vecs = np.linalg.eigh(Km)
    q = vecs[[3, 0, 1, 2], np.argmax(vals)]
    return -q if q[0] < 0 else q


def run(cmd):
    print("  $", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def convert_scene(args, scene):
    src_dir = os.path.join(args.rgb_root, scene)
    out_dir = os.path.join(args.out_root, scene)
    img_dir = os.path.join(out_dir, "images_undistort")
    work = os.path.join(out_dir, "colmap_work")
    os.makedirs(img_dir, exist_ok=True)

    tf = json.load(open(os.path.join(src_dir, "transforms.json")))
    w, h = int(tf["w"]), int(tf["h"])
    fx, fy, cx, cy = tf["fl_x"], tf["fl_y"], tf["cx"], tf["cy"]
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], np.float64)
    dist = np.array([tf.get(k, 0.0) for k in ("k1", "k2", "p1", "p2", "k3")], np.float64)
    frames = sorted(tf["frames"], key=frame_index)
    names = [f"{frame_index(f)}.JPG" for f in frames]

    # 1. undistorted images (the released camera model becomes PINHOLE with the same K)
    jobs = [(os.path.join(src_dir, "images", n), os.path.join(img_dir, n), K, dist) for n in names]
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(undistort, jobs, chunksize=4))

    # 2. features + sequential matching
    if os.path.exists(work):
        shutil.rmtree(work)
    os.makedirs(os.path.join(work, "known"))
    db = os.path.join(work, "database.db")
    run([args.colmap, "feature_extractor", "--database_path", db, "--image_path", img_dir,
         "--ImageReader.camera_model", "PINHOLE", "--ImageReader.single_camera", "1",
         "--ImageReader.camera_params", f"{fx},{fy},{cx},{cy}"])
    run([args.colmap, "sequential_matcher", "--database_path", db])

    # 3. known-pose model, written with the image / camera ids assigned in the database
    con = sqlite3.connect(db)
    ids = {name: (iid, cid) for iid, name, cid in con.execute("SELECT image_id, name, camera_id FROM images")}
    con.close()
    cam_ids = {cid for _, cid in ids.values()}
    assert len(cam_ids) == 1 and len(ids) == len(names), "expected one camera and one database entry per image"
    with open(os.path.join(work, "known", "cameras.txt"), "w") as f:
        f.write(f"{cam_ids.pop()} PINHOLE {w} {h} {fx} {fy} {cx} {cy}\n")
    with open(os.path.join(work, "known", "images.txt"), "w") as f:
        for fr, name in zip(frames, names):
            c2w = np.array(fr["transform_matrix"], np.float64)
            c2w[:3, 1:3] *= -1                      # OpenGL -> OpenCV camera axes
            w2c = np.linalg.inv(c2w)
            q, t = rotmat_to_qvec(w2c[:3, :3]), w2c[:3, 3]
            iid, cid = ids[name]
            f.write(f"{iid} {q[0]} {q[1]} {q[2]} {q[3]} {t[0]} {t[1]} {t[2]} {cid} {name}\n\n")
    open(os.path.join(work, "known", "points3D.txt"), "w").close()

    # 4. triangulate with the cameras held fixed
    sparse = os.path.join(out_dir, "sparse", "0")
    os.makedirs(sparse, exist_ok=True)
    run([args.colmap, "point_triangulator", "--database_path", db, "--image_path", img_dir,
         "--input_path", os.path.join(work, "known"), "--output_path", sparse])
    if not args.keep_work:
        shutil.rmtree(work)

    # 5. split + scale
    with open(os.path.join(args.splits_dir, scene + ".txt")) as f:
        val = {os.path.basename(ln.strip()) for ln in f if ln.strip()}
    assert val <= set(names), f"{scene}: held-out images missing from transforms.json"
    with open(os.path.join(out_dir, "split.json"), "w") as f:
        json.dump({"train": [n for n in names if n not in val], "val": [n for n in names if n in val]}, f, indent=1)
    with open(os.path.join(out_dir, "transform_undistort.json"), "w") as f:
        json.dump({"scale": 1.0}, f)
    print(f"{scene}: {len(names)} images ({len(names) - len(val)} train, {len(val)} test) -> {out_dir}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rgb_root", required=True, help="local copy of Link-Dev/UAV3DCrop")
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--scenes", nargs="*", default=None, help="e.g. 2024/Day036_Corn2; default: the 91 benchmark scenes")
    ap.add_argument("--splits_dir", default=SPLITS, help="splits/test/ of this repository")
    ap.add_argument("--colmap", default="colmap", help="colmap executable")
    ap.add_argument("--workers", type=int, default=8, help="processes for undistortion")
    ap.add_argument("--keep_work", action="store_true", help="keep the COLMAP database and known-pose model")
    args = ap.parse_args()

    scenes = args.scenes or sorted(f"{y}/{os.path.splitext(n)[0]}" for y in os.listdir(args.splits_dir)
                                   for n in os.listdir(os.path.join(args.splits_dir, y)) if n.endswith(".txt"))
    for s in scenes:
        convert_scene(args, s)


if __name__ == "__main__":
    main()
