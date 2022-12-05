# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import os
import random

import hydra
import numpy as np
import pytorch_lightning as zeus
import torch
from omegaconf import DictConfig, OmegaConf, open_dict

from syncmatch.datasets import build_loader
from syncmatch.nnutils.tester import MultiviewRegistrationTest

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


@hydra.main(config_name="test", config_path="syncmatch/configs", version_base=None)
def test(cfg: DictConfig) -> None:
    # --- Reproducibility | https://pytorch.org/docs/stable/notes/randomness.html
    torch.manual_seed(cfg.system.random_seed)
    random.seed(cfg.system.random_seed)
    np.random.seed(cfg.system.random_seed)

    # --- load checkpoint ---
    ckpt_cfg = cfg.test.checkpoint

    if ckpt_cfg.path != "":
        ckpt_path = ckpt_cfg.path
        ckpt_file = ckpt_path.split("/")[-1]
        model = MultiviewRegistrationTest.load_from_checkpoint(ckpt_path, strict=False)
    elif ckpt_cfg.name != "":
        if ckpt_cfg.time == "":
            exp_name = ckpt_cfg.name
        else:
            exp_name = f"{ckpt_cfg.name}_{ckpt_cfg.time}"
        ckpt_dir = os.path.join(cfg.paths.experiments_dir, exp_name)

        # pick last file by default -- most recent checkpoint
        if ckpt_cfg.step == -1:
            ckpts = os.listdir(ckpt_dir)
            ckpts.sort()
            ckpt_file = ckpts[-1]
        else:
            epoch = ckpt_cfg.epoch
            step = ckpt_cfg.step
            ckpt_file = f"checkpoint-epoch={epoch:03d}-step={step:07d}.ckpt"

        ckpt_path = os.path.join(ckpt_dir, ckpt_file)
        model = MultiviewRegistrationTest.load_from_checkpoint(ckpt_path, strict=False)
    else:
        ckpt_path = "N/A"
        ckpt_file = cfg.model.name
        model = MultiviewRegistrationTest(cfg)

    # -- Get Dataset --
    loader = build_loader(cfg.dataset, split=cfg.test.split)
    # get first item, useful when debugging
    loader.dataset.__getitem__(0)

    # --- Update model configs ---
    model.cfg.paths = cfg.paths
    model.cfg.dataset = cfg.dataset
    model.cfg.dataset.num_views = loader.dataset.num_views
    with open_dict(model.model.cfg):
        model.visualize_test = cfg.test.visualize_test
        model.model.cfg.light_first_run = cfg.test.model_cfg.light_first_run
        model.model.cfg.confidence_min = cfg.test.model_cfg.confidence_min
        model.model.cfg.sync_algorithm = cfg.test.model_cfg.sync_algorithm
        model.model.cfg.alignment = cfg.test.model_cfg.alignment
        model.model.cfg.refinement = cfg.test.model_cfg.refinement
        model.model.cfg.correspondence = cfg.test.model_cfg.correspondence

    # -- test model --
    trainer = zeus.Trainer(accelerator="gpu", devices=1, max_epochs=-1)

    print(f"==== {ckpt_file} ====")
    trainer.test(model, loader, verbose=False)


if __name__ == "__main__":
    test()
