# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import os
import random
from datetime import datetime

import hydra
import numpy as np
import pytorch_lightning as zeus
import torch
from omegaconf import DictConfig, OmegaConf

from syncmatch.datasets import build_loader
from syncmatch.nnutils.trainer import MultiviewRegistration
from syncmatch.utils.io import makedir

# set CUDA_LAUNCH_BLOCKING -- dunno why it's an issue
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


@hydra.main(config_name="config", config_path="syncmatch/configs", version_base=None)
def train(cfg: DictConfig) -> None:

    # Reproducibility - refer https://pytorch.org/docs/stable/notes/randomness.html
    torch.manual_seed(cfg.system.random_seed)
    random.seed(cfg.system.random_seed)
    np.random.seed(cfg.system.random_seed)

    if "resume" in cfg:
        ckpt_epoch = cfg.resume.epoch
        ckpt_step = cfg.resume.step
        ckpt_exp = cfg.resume.experiment

        checkpoint = os.path.join(
            cfg.paths.experiments_dir,
            ckpt_exp,
            f"checkpoint-epoch={ckpt_epoch:03d}-step={ckpt_step:07d}.ckpt",
        )
        exp_version = cfg.resume.experiment.split("_")[-1] + "-resume"
    else:
        assert cfg.experiment.name != "", "Experiment name is not defined."
        exp_version = datetime.today().strftime("%m%d-%H%M")
        checkpoint = None

    full_exp_name = f"{cfg.experiment.name}_{exp_version}"
    OmegaConf.set_struct(cfg, False)
    cfg.experiment.full_name = full_exp_name
    OmegaConf.set_struct(cfg, True)

    print("=====================================")
    print(f"Experiment name: {full_exp_name}")
    print()
    print(OmegaConf.to_yaml(cfg))
    print("=====================================")

    # setup checkpoint directory
    exp_dir = os.path.join(cfg.paths.experiments_dir, full_exp_name)
    makedir(exp_dir)

    # Datasets
    train_loader = build_loader(cfg.dataset, split="train")
    valid_loader = build_loader(cfg.dataset, split="valid")
    train_loader.dataset.__getitem__(0)

    # Trainer Plugins
    checkpoint_callback = zeus.callbacks.ModelCheckpoint(
        dirpath=exp_dir,
        filename="checkpoint-{epoch:03d}-{step:07d}",
        save_top_k=-1,
        every_n_train_steps=cfg.train.checkpoint_step,
    )
    logger = zeus.loggers.TensorBoardLogger(
        save_dir=cfg.paths.tensorboard_dir,
        name=cfg.experiment.name,
        version=exp_version,
    )
    lr_monitor = zeus.callbacks.LearningRateMonitor(logging_interval="step")

    # Set up Trainer
    model = MultiviewRegistration(cfg)
    trainer = zeus.Trainer(
        accelerator="gpu",
        devices=1,
        num_sanity_val_steps=0,
        benchmark=True,
        logger=logger,
        val_check_interval=cfg.train.eval_step,
        detect_anomaly=cfg.system.detect_anomaly,
        max_steps=cfg.train.max_steps,
        callbacks=[checkpoint_callback, lr_monitor],
    )

    if checkpoint is None:
        trainer.validate(model, valid_loader, verbose=False)

    trainer.fit(model, train_loader, valid_loader, ckpt_path=checkpoint)


if __name__ == "__main__":
    train()
