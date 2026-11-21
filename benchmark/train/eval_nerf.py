#!/usr/bin/env python
"""Novel-view evaluation for Nerfacto and Instant-NGP runs (writes evaluation.json).

Equivalent to `ns-eval`, except that the run is evaluated with the VanillaDataManager it was
trained with (ns-eval would rebuild the method's default datamanager). Metrics come from
`pipeline.get_average_eval_image_metrics()`: PSNR, SSIM and LPIPS on every held-out image at
full resolution, averaged over images.

    python eval_nerf.py --load-config <RUN_DIR>/config.yml --output-path <RUN_DIR>/evaluation.json
"""
import argparse
import json
from pathlib import Path

import torch
import yaml
from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManager
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.pipelines.base_pipeline import Pipeline
from nerfstudio.utils.eval_utils import eval_load_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--load-config", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = yaml.load(args.load_config.read_text(), Loader=yaml.Loader)
    assert isinstance(config, TrainerConfig)

    config.pipeline.datamanager._target = VanillaDataManager
    config.load_dir = config.get_checkpoint_dir()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline = config.pipeline.setup(device=device, test_mode="test")
    assert isinstance(pipeline, Pipeline)
    pipeline.eval()

    checkpoint_path, _ = eval_load_checkpoint(config, pipeline)

    metrics_dict = pipeline.get_average_eval_image_metrics(get_std=True)

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    benchmark_info = {
        "experiment_name": config.experiment_name,
        "method_name": config.method_name,
        "checkpoint": str(checkpoint_path),
        "results": metrics_dict,
    }
    args.output_path.write_text(json.dumps(benchmark_info, indent=2), "utf8")
    print(f"Saved results to: {args.output_path}")


if __name__ == "__main__":
    main()
