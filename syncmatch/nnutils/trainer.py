# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import pathlib
import resource

import numpy as np
import pytorch_lightning as zeus
import torch
from hydra.utils import instantiate

from ..models import build_model
from ..models.synchronization import camera_chaining, camera_synchronization
from ..utils.transformations import SE3_inverse, get_relative_Rt, transform_points_Rt
from ..utils.util import detach_dictionary, modify_keys


class MultiviewRegistration(zeus.LightningModule):
    def __init__(self, cfg):
        super().__init__()

        # define hyperparameters
        self.cfg = cfg
        self.save_hyperparameters()

        # define model
        self.model = build_model(cfg.model)

        # set debug
        self.debug = False
        self.bad_grads = 0

    def process_batch(self, batch):
        rgb = [batch[f"rgb_{i}"] for i in range(self.cfg.dataset.num_views)]
        dep = [batch[f"depth_{i}"] for i in range(self.cfg.dataset.num_views)]
        if "Rt_0" in batch:
            Rts = [batch[f"Rt_{i}"] for i in range(self.cfg.dataset.num_views)]
        else:
            Rts = None
        K = batch["K"]
        K_full = batch["K_full"]
        return rgb, dep, Rts, K, K_full

    def training_step(self, batch, batch_idx):
        batch = self.process_batch(batch)
        gt_rgb, gt_dep, gt_Rt, K = batch[:4]
        output = self.model(rgbs=gt_rgb, deps=gt_dep, K=K, gt_Rts=gt_Rt)
        loss = output["loss"].mean()

        mem_use = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        mem_use = torch.ones(1) * mem_use / 2 ** 20
        self.log("mem_use/val", mem_use, prog_bar=True)
        self.log("loss/train", loss)

        # debug
        if self.debug:
            saved_stuff = {}
            for out_key in output:

                if "pw_Rts" in out_key or "corr_loss" in out_key:
                    saved_stuff[out_key] = {}
                    for ij in output[out_key]:
                        saved_stuff[out_key][ij] = output[out_key][ij]
                        saved_stuff[out_key][ij].retain_grad()
                elif "Rts" in out_key:
                    saved_stuff[out_key] = output[out_key]
                    saved_stuff[out_key].retain_grad()

                if "pw_corr" in out_key:
                    saved_stuff[out_key] = {}
                    for ij in output[out_key]:
                        saved_stuff[out_key][ij] = output[out_key][ij][2]
                        saved_stuff[out_key][ij].retain_grad()

            self.debug_dict = saved_stuff

        return loss

    def validation_step(self, batch, batch_idx):
        batch = self.process_batch(batch)
        gt_rgb, gt_dep, gt_Rt, K = batch[:4]
        output = self.model(rgbs=gt_rgb, deps=gt_dep, K=K, gt_Rts=gt_Rt)
        loss = output["loss"].mean()
        self.log("loss/valid", loss)
        return loss

    def configure_optimizers(self):
        params = self.model.parameters()
        optimizer = instantiate(self.cfg.train.optimizer, params=params)
        output = {"optimizer": optimizer}

        if "scheduler" in self.cfg.train:
            scheduler = instantiate(self.cfg.train.scheduler, optimizer=optimizer)
            output["lr_scheduler"] = {"scheduler": scheduler, "interval": "step"}

        return output

    def on_before_backward(self, loss):
        if self.debug:
            if not loss.isfinite():
                print("Something is off with the loss")
                breakpoint()

    def on_after_backward(self):
        grad_exploded = False
        for p in self.parameters():
            if p.grad is not None:
                if not p.grad.isfinite().all():
                    grad_exploded = True
                    if self.debug:
                        print("gradient is not finite | debug through breakpoint")
                        breakpoint()
                    p.grad.zero_()

        if grad_exploded:
            self.bad_grads += 1
            print(f"Zero-gradients: {self.bad_grads}")
