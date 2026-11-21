#!/usr/bin/env python3
"""Write the benchmark's fixed 90/10 split into a copy of a scene's transforms.json.

nerfstudio's default `--train-split-fraction 0.9` rule depends on how image files are named and
sorted, so it does not reproduce the benchmark split on the released file names. This script adds
explicit `train_filenames`, `val_filenames` and `test_filenames` (nerfstudio reads these instead of
the fraction rule) and writes them next to the original as `transforms_split.json`:

    python apply_split.py <DATA_ROOT>/<SCENE_ID> splits/test/<SCENE_ID>.txt
    ns-train <method> ... nerfstudio-data --data <DATA_ROOT>/<SCENE_ID>/transforms_split.json

Every image listed in transforms.json that is not in the test list is used for optimization.
"""
import json
import sys
from pathlib import Path


def main(scene_dir: Path, test_list: Path) -> None:
    meta = json.loads((scene_dir / "transforms.json").read_text())
    images = [f["file_path"] for f in meta["frames"]]
    test = [ln.strip() for ln in test_list.read_text().splitlines() if ln.strip()]
    unknown = sorted(set(test) - set(images))
    if unknown:
        raise SystemExit(f"{len(unknown)} test images are not in transforms.json (first: {unknown[0]})")
    test_set = set(test)
    meta["train_filenames"] = [p for p in images if p not in test_set]
    meta["val_filenames"] = test
    meta["test_filenames"] = test
    out = scene_dir / "transforms_split.json"
    out.write_text(json.dumps(meta, indent=1))
    print(f"{scene_dir}: {len(images) - len(test)} train / {len(test)} test -> {out}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    main(Path(sys.argv[1]), Path(sys.argv[2]))
