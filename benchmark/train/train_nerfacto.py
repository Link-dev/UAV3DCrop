#!/usr/bin/env python
"""Train Nerfacto on one UAV3DCrop scene with the benchmark settings (nerfstudio 1.1.5).

nerfstudio's `nerfacto` method uses a ParallelDataManager that cannot be swapped for the
VanillaDataManager from the command line, so this wrapper builds the config in Python.
Changes from `method_configs["nerfacto"]`: VanillaDataManager, 2,048 rays per batch drawn from
64 training images re-sampled every iteration, full-resolution images, 90/10 split.

    python train_nerfacto.py --data <DATA_ROOT>/<SCENE_ID> --output-dir outputs --experiment-name <SCENE_ID>
"""
import argparse
from copy import deepcopy
from pathlib import Path

from nerfstudio.configs.method_configs import method_configs
from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManagerConfig
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.scripts.train import main as train_main


def build_config(args: argparse.Namespace):
    config = deepcopy(method_configs["nerfacto"])
    config.output_dir = args.output_dir
    config.experiment_name = args.experiment_name
    config.vis = "tensorboard"
    config.max_num_iterations = args.max_num_iterations
    config.machine.num_devices = 1
    config.steps_per_eval_batch = 1000000
    config.steps_per_eval_image = 1000000
    config.steps_per_eval_all_images = 1000000
    config.viewer.quit_on_train_completion = True
    config.pipeline.model.eval_num_rays_per_chunk = 4096

    config.pipeline.datamanager = VanillaDataManagerConfig(
        dataparser=NerfstudioDataParserConfig(
            data=args.data,
            train_split_fraction=0.9,
            downscale_factor=1,
        ),
        train_num_rays_per_batch=2048,
        train_num_images_to_sample_from=64,
        train_num_times_to_repeat_images=1,
        eval_num_rays_per_batch=4096,
        eval_num_images_to_sample_from=1,
        eval_num_times_to_repeat_images=1,
        eval_image_indices=(0,),
        camera_res_scale_factor=1.0,
        patch_size=1,
    )
    return config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--max-num-iterations", type=int, default=30000)
    return parser.parse_args()


if __name__ == "__main__":
    train_main(build_config(parse_args()))
