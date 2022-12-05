# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.i
import torch
from pytorch3d.ops.knn import knn_points as pt3d_knn
from torch import nn as nn
from torch.nn.functional import normalize
from tqdm import tqdm

from ..utils.transformations import get_relative_Rt, split_Rt, transform_points_Rt
from ..utils.util import corr_dict_to_3dcorr, nn_gather
from .alignment import align_cpa, align_cpa_ransac, align_o3d
from .correspondence import (
    get_correspondences_ratio_test,
    get_geometry_weighted_correspondences,
)
from .multiview_utils import MultiviewScreen
from .synchronization import camera_chaining, camera_synchronization


class SyncMatch(nn.Module):
    def __init__(self, cfg):
        """Create the SyncMatch model based on config

        Args:
            cfg (DictConfig): A DictConfig from hydra that defines the model's hyperparameters
        """
        super().__init__()

        self.cfg = cfg
        self.multiview_screen = MultiviewScreen(cfg)
        self.pw_loss_normalization = self.cfg.get("pw_loss_normalization", "none")

    def generate_keypoints(self, rgb_0, rgb_1):
        """Generate a feature pointcloud for a view pair for easy integration
        into pairwise evaluations

        Args:
            rgb_0 (FloatTensor): RGB for first view (batch, 3, H, W)
            rgb_1 (FloatTensor): RGB for second view (batch, 3, H, W)

        Returns:
            tuple:
                (kps0, kps1): pointclouds for the view pair (batch, N, 3)
                (des0, des1): features for pointclouds (batch, N, F)
                (None, None): to match API
                minkps: the minimum number of keypoints for the two views
        """
        kps0, des0 = self.multiview_screen.get_keypoints(rgb_0)
        kps1, des1 = self.multiview_screen.get_keypoints(rgb_1)

        minkps = min(len(kps0), len(kps1))

        return (kps0, kps1), (des0, des1), (None, None), minkps

    def forward(self, rgbs, K, gt_Rts=None, deps=None):
        """Main forward function for training

        Args:
            rgbs (tuple(FloatTensor)): tuple of images, each is (batch, 3, H, W)
            K (FloatTensor): Camera intriniscs (batch, 3, 3)
            gt_Rts (tuple(FloatTensor), optional): tuple of extrinsics (batch, 4, 3)
            deps (tuple(FloatTensor), optional): depth images, each is (batch, 3, H, W)

        Raises:
            ValueError: raised if synchronization algorithm is not known

        Returns:
            dict: dictionary that contains outputs, variable depending on model.
        """
        output = {}
        losses = []

        # ==== Do once ====
        batch_size, num_views = rgbs[0].shape[0], len(rgbs)
        screen, pc = self.multiview_screen(rgbs, deps, K)

        output["pc_feats"] = pc.feats
        output["pc_xyz"] = pc.xyz_world
        output["pc_uv"] = pc.uv_screen
        # ==== and now we iterate
        for step_i in range(self.cfg.refinement.num_steps):
            # Extract correspondences between features (only use corr when alpha = 0)
            alpha = self.cfg.refinement.alpha
            if screen.num_views > 2 and self.cfg.refinement.num_steps > 1:
                alpha *= max(0, step_i - 1) / (self.cfg.refinement.num_steps - 1)
            else:
                alpha *= step_i / self.cfg.refinement.num_steps

            # compute pairwise correspondence and rotations
            pw_corr = self.extract_correspondences(pc, alpha=alpha, step=step_i)
            pw_Rts = self.multiview_pw_align(pc, pw_corr)

            # compute pairwise conf
            pw_conf = {}
            for (i, j), corr_ij in pw_corr.items():
                conf_ij = corr_ij[2].mean(dim=1)

                if abs(i - j) > 1:
                    conf_ij = (conf_ij - self.cfg.confidence_min).relu()
                    conf_ij = conf_ij / (1 - self.cfg.confidence_min)
                pw_conf[(i, j)] = conf_ij

            output[f"pw_conf_{step_i}"] = pw_conf
            output[f"pw_corr_{step_i}"] = corr_dict_to_3dcorr(pw_corr, pc.xyz_cam)
            output[f"pw_Rts_{step_i}"] = pw_Rts

            # log pariwise rotations and correspondences
            if not self.cfg.use_gt_Rt:
                pw_loss = self.get_corr_loss(pc, pw_corr, pw_conf, pw_Rts=pw_Rts)
                output[f"pw-corr_loss_{step_i}"] = pw_loss
                pw_losses = sum(pw_loss[ij] for ij in pw_loss) / len(pw_loss)
                losses.append(self.cfg.loss.weights.pairwise * pw_losses)

            # synchronize cameras
            if self.cfg.sync_algorithm == "adjacent":
                abs_Rts = camera_chaining(pw_Rts, pw_conf, num_views)
            elif self.cfg.sync_algorithm == "all":
                abs_Rts = camera_synchronization(pw_Rts, pw_conf, num_views)
            else:
                raise ValueError("Unknown sync. algorithm: ", self.cfg.sync_algorithm)
            output[f"Rts_{step_i}"] = abs_Rts

            # apply sync loss -- use gt_Rts if self.use_gt_Rt
            Rts = torch.stack(gt_Rts, dim=1) if self.cfg.use_gt_Rt else abs_Rts
            sync_loss = self.get_corr_loss(pc, pw_corr, pw_conf, abs_Rts=Rts)
            output[f"sync-corr_loss_{step_i}"] = sync_loss
            sync_losses = sum(sync_loss[ij] for ij in sync_loss) / len(sync_loss)

            losses.append(self.cfg.loss.weights.sync * sync_losses)

            # update xyz_world for refinement
            pc.xyz_world = transform_points_Rt(pc.xyz_cam, Rts, inverse=True)

        b_num_corr = torch.ones(batch_size).to(K) * self.cfg.correspondence.num_corr
        output["num_corr"] = b_num_corr
        output["loss"] = sum(losses)
        return output

    def multiview_pw_align(self, pc, corr, update_corr=False):
        """Given a set of pointclouds and a correspondence dictonary that indexes
        into the pointclouds, extract the pairwise transformation using Ummeyama's
        algorithm

        Args:
            pc (EasyDict): dictionary with all pointclouds, defined in multiview_utils
            corr (dict): dictionary, corr[(i, j)] is correspondence i -> j
            update_corr (bool, optional): do we update correspondence weights for loss?

        Raises:
            ValueError: Raise error if alignment algorithm is not known

        Returns:
            dict: Rt_out[(i, j)] estimated camera alignment for view i -> j
        """
        Rt_out = {}

        for i, j in corr:
            corr_ij = [_corr.flatten(1) for _corr in corr[(i, j)]]
            xyz_i = pc.xyz_cam[:, i]
            xyz_j = pc.xyz_cam[:, j]
            if self.cfg.alignment.algorithm == "cpa":
                Rt_out[(i, j)] = align_cpa(corr_ij, xyz_i, xyz_j)
            elif self.cfg.alignment.algorithm == "o3d":
                Rt_out[(i, j)] = align_o3d(corr_ij, xyz_i, xyz_j)
            elif self.cfg.alignment.algorithm == "cpa_ransac":
                corr_i_id, corr_j_id, corr_w = corr_ij[:3]

                corr_i = nn_gather(xyz_i, corr_i_id)
                corr_j = nn_gather(xyz_j, corr_j_id)

                Rt_out[(i, j)], new_weights = align_cpa_ransac(
                    corr_i,
                    corr_j,
                    corr_w,
                    schedule=self.cfg.alignment.ransac.schedule,
                    threshold=self.cfg.alignment.ransac.threshold,
                    return_new_weights=True,
                )
            else:
                raise ValueError(f"Unknown algorithm {self.cfg.alignment.algorithm}")

            if update_corr:
                corr[(i, j)] = (corr[(i, j)][0], corr[(i, j)][1], new_weights)

        return Rt_out

    def get_corr_loss(self, pc, corr, conf, pw_Rts=None, abs_Rts=None):
        """Compute the correspondence loss

        Args:
            pc (EasyDict): All the pointcloud information as defned in multiview_util
            corr (dict): corr[(i, j)] is correspondence view i -> view j
            conf (dict): conf[(i, j)] is the confidence is i->j pairwise estimate
            pw_Rts (dict, optional): pw_Rts[(i, j)] is pairwise transformation i -> j
            abs_Rts (FloatDict, optional): camera parameters (batch, num_view, 3, 4)

        Raises:
            ValueError: Unknown loss type

        Returns:
            dict: corr_loss[(i, j)] is correspondence loss for i -> j
        """
        assert (pw_Rts is None) ^ (abs_Rts is None), "only one should be defined"
        corr_loss = {}
        xyz = pc.xyz_cam

        for i, j in corr:
            id_i, id_j, w_ij = corr[(i, j)]
            corr_i = nn_gather(xyz[:, i], id_i)
            corr_j = nn_gather(xyz[:, j], id_j)

            # get Rt_ij
            if pw_Rts is None:
                Rt_ij = get_relative_Rt(abs_Rts[:, i], abs_Rts[:, j])
            else:
                Rt_ij = pw_Rts[(i, j)]

            corr_i = transform_points_Rt(corr_i, Rt_ij)

            # loss is weighted sum over residuals; weights are L1 normalized first
            w_ij_n = normalize(w_ij, p=1, dim=-1)

            loss_type = getattr(self.cfg.loss, "type", "rmse")
            if "robust" in loss_type:
                delta = float(loss_type.split("_")[1])
                corr_d2 = (corr_i - corr_j).pow(2).sum(dim=-1)
                corr_d1 = (corr_i - corr_j).abs().sum(dim=-1)
                corr_diff = torch.where(
                    corr_d1 < delta, 0.5 * corr_d2, delta * (corr_d1 - 0.5 * delta)
                )
            elif "gm" in loss_type:
                mu = float(loss_type.split("_")[1])
                corr_d2 = (corr_i - corr_j).pow(2).sum(dim=-1)
                corr_diff = (mu * corr_d2) / (mu + corr_d2)
            elif loss_type == "rmse":
                corr_diff = (corr_i - corr_j).norm(p=2, dim=-1)
            elif loss_type == "mse":
                corr_diff = (corr_i - corr_j).pow(2).sum(dim=-1)
            else:
                raise ValueError()

            loss = (w_ij_n * corr_diff).sum(dim=-1)

            # weighted with the detached mean weight wo help with non-overlapping pairs
            try:
                conf_weighted = self.cfg.loss.confidence_weighted
                detached = self.cfg.loss.detached_loss
            except:
                conf_weighted = False
                detached = False
            if conf_weighted and detached:
                loss = loss * conf[(i, j)].detach()
            elif conf_weighted:
                loss = loss * conf[(i, j)]

            corr_loss[(i, j)] = loss

        return corr_loss

    def extract_correspondences(self, pc, alpha=0.0, step=0):
        """Extract all pairwise correspondence i, j for j > i

        Args:
            pc (EasyDict): All the pointclouds as defined in multiview_utils
            alpha (float, optional): Weighing for geometric proximity in estimation
            step (int, optional): iteration step for correspondence refinement

        Returns:
            dict: corr[(i, j)] is the correspondence between view_i and view_j
        """
        IJs = pc.IJs
        feats = pc.feats
        xyz = pc.xyz_world.detach()
        valid = pc.xyz_cam[:, :, :, 2:3] > 0
        bidirectional = getattr(self.cfg.correspondence, "bidirectional", False)

        corr = {}

        for i, j in tqdm(IJs, disable=not self.cfg.light_first_run):
            if self.cfg.light_first_run and abs(i - j) > 1 and step == 0:
                continue

            valid_i = valid[:, i, :].float().contiguous().clamp(min=1e-6)
            valid_j = valid[:, j, :].float().contiguous().clamp(min=1e-6)

            if alpha == 0:
                corr[(i, j)] = get_correspondences_ratio_test(
                    feats[:, i],
                    feats[:, j],
                    num_corres=self.cfg.correspondence.num_corr,
                    W1=valid_i,
                    W2=valid_j,
                    normed_features=True,
                    bidirectional=bidirectional,
                )
            else:
                corr[(i, j)] = get_geometry_weighted_correspondences(
                    xyz[:, i],
                    xyz[:, j],
                    feats[:, i],
                    feats[:, j],
                    num_corres=self.cfg.correspondence.num_corr,
                    alpha=alpha,
                    normed_features=True,
                    P1_W=valid_i,
                    P2_W=valid_j,
                    bidirectional=bidirectional,
                )

        return corr
