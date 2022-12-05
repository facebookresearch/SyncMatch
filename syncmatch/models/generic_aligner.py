# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import cv2 as cv
import numpy as np
import torch
from torch import nn as nn
from torch.nn.functional import grid_sample

from ..utils.ransac import o3d_3d_correspondence_registration
from ..utils.transformations import transform_points_Rt
from ..utils.util import pixel_to_ndc
from .alignment import align_cpa_ransac
from .correspondence import (
    get_correspondences_ratio_test,
    get_geometry_weighted_correspondences,
)

try:
    from .pairwise_superglue import PairwiseSuperGlue
except:
    print("Unable to import SuperGlue. Please check docs/evaluation for instructions.")

try:
    from .pairwise_loftr import PairwiseLoFTR
except:
    print("Unable to import LoFTR. Please check docs/evaluation for instructions.")


class GenericAligner(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.failed = 0
        self.num_fail = 0
        self.return_corr2d = cfg.get("return_corr2d", False)

        print(cfg.baseline)
        self.aligner = cfg.baseline.aligner

        if self.cfg.baseline.feature in ["rootsift", "sift"]:
            self.get_descriptors = self.get_opencv_feature
        elif self.cfg.baseline.feature == "superpoint":
            model = PairwiseSuperGlue(cfg)
            self.superpoint = model.matcher.superpoint
            self.get_descriptors = self.get_superpoint
        elif self.cfg.baseline.feature == "loftr_fine":
            self.loftr = PairwiseLoFTR(cfg, fine=True)
            self.get_descriptors = self.loftr.generate_keypoints
        elif self.cfg.baseline.feature == "loftr_coarse":
            self.loftr = PairwiseLoFTR(cfg, fine=False)
            self.get_descriptors = self.loftr.generate_keypoints
        elif self.cfg.baseline.feature == "superglue":
            self.superglue = PairwiseSuperGlue(cfg)
            self.get_descriptors = self.superglue.generate_keypoints
        elif self.cfg.baseline.feature == "syncmatch":
            self.syncmatch = None
        else:
            raise ValueError(f"Unknown feature descriptor: {self.cfg.baseline.feature}")

        if self.aligner == "cpa_ransac":
            self.align_3d = self.cpa_ransac_align
        elif self.aligner == "o3d_ransac":
            self.align_3d = self.o3d_ransac_align
        else:
            raise ValueError(f"Unknown aligner: {self.aligner}")

    def forward(self, rgbs, K, gt_Rts=None, deps=None):

        batch_size = rgbs[0].shape[0]
        # undo image normalization
        rgbs_0 = rgbs[0] * 0.5 + 0.5
        rgbs_1 = rgbs[1] * 0.5 + 0.5
        Rts = []

        corr2d_i = []
        corr2d_j = []
        corr3d_i = []
        corr3d_j = []
        corr2d_w = []
        num_corr = torch.zeros(batch_size).to(rgbs_0)
        valid_out = torch.zeros(batch_size).to(rgbs_0)

        if self.cfg.baseline.feature == "syncmatch":
            model_out = self.syncmatch(rgbs, K, gt_Rts, deps)

        for i in range(batch_size):
            img_0 = rgbs_0[i : i + 1]
            img_1 = rgbs_1[i : i + 1]
            dep_0 = deps[0][i]
            dep_1 = deps[1][i]

            # get feature descriptors
            if self.cfg.baseline.feature == "syncmatch":
                kpts_2d = (model_out["pc_uv"][i][0], model_out["pc_uv"][i][1])
                desc = (model_out["pc_feats"][i][0], model_out["pc_feats"][i][1])
                kps_sz = (None, None)
                min_kps = desc[0].shape[0]
            else:
                kpts_2d, desc, kps_sz, min_kps = self.get_descriptors(img_0, img_1)

            kpts_2d_0, desc_0 = self.filter_invalid_depth(kpts_2d[0], desc[0], dep_0)
            kpts_2d_1, desc_1 = self.filter_invalid_depth(kpts_2d[1], desc[1], dep_1)

            kpts_2d = (kpts_2d_0, kpts_2d_1)
            desc = (desc_0, desc_1)
            min_kps = min(len(kpts_2d_0), len(kpts_2d_1))

            if min_kps > 5:
                # get matches/correspondences
                mkpts_0, mkpts_1, mconf = self.get_matches(
                    kpts_2d, desc, kps_sz, (dep_0, dep_1), K[i]
                )

                enough_matches = mconf.shape[0] > 5
            else:
                print("not enough")
                enough_matches = False

            if enough_matches:
                # convert kpts into 3D pts
                xyz_0, xyz_1, mconf = self.keypoints_to_3d(
                    mkpts_0, mkpts_1, dep_0, dep_1, K[i], mconf
                )
                Rt = self.align_3d(xyz_0, xyz_1, mconf)

                # make sure it's valid
                num_corr_i = mkpts_0.shape[0]
                valid_i = 1.0
            else:
                Rt = torch.eye(4).to(rgbs_0)
                xyz_0 = xyz_1 = torch.zeros(1, 3).to(rgbs_0)
                mkpts_0 = mkpts_1 = torch.zeros(1, 2).to(rgbs_0)
                mconf = torch.zeros(1).to(rgbs_0)
                num_corr_i = 0
                valid_i = 1.0

            Rts.append(torch.stack((torch.eye(4).to(Rt), Rt)))
            corr3d_i.append(xyz_0)
            corr3d_j.append(xyz_1)
            corr2d_i.append(mkpts_0)
            corr2d_j.append(mkpts_1)
            corr2d_w.append(mconf)
            num_corr[i] = num_corr_i
            valid_out[i] = valid_i

        output = {
            "Rts_0": torch.stack(Rts, dim=0),
            "pw_corr_0": {(0, 1): (corr3d_i, corr3d_j, corr2d_w)},
            "num_corr": num_corr,
            "loss": num_corr.float(),
        }

        if self.cfg.refinement.num_steps == 2:
            output["Rts_1"] = torch.stack(Rts, dim=0)
            output["pw_corr_1"] = {(0, 1): (corr3d_i, corr3d_j, corr2d_w)}

        if self.return_corr2d:
            output["corr2d"] = {(0, 1): (corr2d_i, corr2d_j, corr2d_w)}

        return output

    def filter_invalid_depth(self, kpts, desc, dep):
        _, H, W = dep.shape
        kpts_0_ndc = pixel_to_ndc(kpts, H=H, W=W)[None, None, :, :]

        kdep = grid_sample(
            dep.unsqueeze(0), kpts_0_ndc, mode="nearest", align_corners=True
        )
        kdep = kdep[0, 0, 0, :]

        valid = kdep > 0

        return kpts[valid], desc[valid]

    def keypoints_to_3d(self, mkpts_0, mkpts_1, dep_0, dep_1, K, mconf):

        _, H, W = dep_0.shape
        mkpts_0_ndc = pixel_to_ndc(mkpts_0, H=H, W=W)[None, None, :, :]
        mkpts_1_ndc = pixel_to_ndc(mkpts_1, H=H, W=W)[None, None, :, :]

        mdep_0 = grid_sample(
            dep_0.unsqueeze(0), mkpts_0_ndc, mode="nearest", align_corners=False
        )
        mdep_1 = grid_sample(
            dep_1.unsqueeze(0), mkpts_1_ndc, mode="nearest", align_corners=False
        )
        mdep_0 = mdep_0[0, 0, 0, :, None]
        mdep_1 = mdep_1[0, 0, 0, :, None]

        # num_pts (x 2, 1, nothing)
        h = torch.ones_like(mdep_0)
        xyh0 = torch.cat((mkpts_0, h), dim=1)
        xyh1 = torch.cat((mkpts_1, h), dim=1)
        # filter 0 dep points

        valid = ((mdep_0 > 0) & (mdep_1 > 0)).squeeze(1)
        xyh0 = xyh0[valid]
        xyh1 = xyh1[valid]
        mconf = mconf[valid] if mconf is not None else None

        mdep_0 = mdep_0[valid]
        mdep_1 = mdep_1[valid]

        # homogenous to 3D
        xyz_0 = K.inverse() @ (xyh0 * mdep_0).T
        xyz_1 = K.inverse() @ (xyh1 * mdep_1).T
        xyz_0 = xyz_0.T
        xyz_1 = xyz_1.T

        return xyz_0, xyz_1, mconf

    def get_opencv_feature(self, rgb_0, rgb_1):
        kps0, des0, kps0_sz = opencv_descriptor(
            rgb_0[0].cpu(), self.cfg.baseline.feature
        )
        kps1, des1, kps1_sz = opencv_descriptor(
            rgb_1[0].cpu(), self.cfg.baseline.feature
        )

        kps0 = torch.tensor(kps0).to(rgb_0)
        kps1 = torch.tensor(kps1).to(rgb_0)

        if kps0_sz is not None:
            kps0_sz = torch.tensor(kps0_sz).to(rgb_0)
            kps1_sz = torch.tensor(kps1_sz).to(rgb_1)

        des0 = kps0 if len(kps0) == 0 else torch.tensor(des0).to(rgb_0)
        des1 = kps1 if len(kps1) == 0 else torch.tensor(des1).to(rgb_1)

        min_kps = min(len(kps0), len(kps1))
        return (kps0, kps1), (des0, des1), (kps0_sz, kps1_sz), min_kps

    def get_superpoint(self, rgb_0, rgb_1):
        rgb_0 = rgb_0.mean(dim=1, keepdim=True)
        rgb_1 = rgb_1.mean(dim=1, keepdim=True)

        pred0 = self.superpoint({"image": rgb_0})
        pred1 = self.superpoint({"image": rgb_1})

        # Note, ignoring saliency score from Superpoint for now
        kps0 = pred0["keypoints"][0]
        kps1 = pred1["keypoints"][0]
        des0 = pred0["descriptors"][0].T
        des1 = pred1["descriptors"][0].T

        kps0_sz = None
        kps1_sz = None

        min_kps = min(len(kps0), len(kps1))
        return (kps0, kps1), (des0, des1), (kps0_sz, kps1_sz), min_kps

    def get_matches(self, kpts, descriptors, kp_szs=None, deps=None, K=None):
        kpts_0, kpts_1 = kpts
        desc_0, desc_1 = descriptors
        dep_0, dep_1 = deps

        # use faiss to get get nn
        kpts_0 = kpts_0.contiguous()
        kpts_1 = kpts_1.contiguous()
        desc_0 = desc_0.contiguous()
        desc_1 = desc_1.contiguous()

        # form feautre distance matrix
        c_id_0, c_id_1, mconf = get_correspondences_ratio_test(
            desc_0[None, :],
            desc_1[None, :],
            500,
            metric=self.cfg.baseline.distance,
            bidirectional=self.cfg.correspondence.bidirectional,
        )

        mkpts_0 = kpts_0[c_id_0[0]]
        mkpts_1 = kpts_1[c_id_1[0]]
        mconf = mconf[0]

        if self.cfg.refinement.num_steps == 1:
            return mkpts_0, mkpts_1, mconf
        elif self.cfg.refinement.num_steps == 2:
            # -- align --
            # convert kpts into 3D pts
            xyz_0, xyz_1, mconf = self.keypoints_to_3d(
                mkpts_0, mkpts_1, dep_0, dep_1, K, mconf
            )
            Rt = self.align_3d(xyz_0, xyz_1, mconf)
            xyz_0, xyz_1, val_0, val_1 = keypoints_to_unfiltered3d(
                kpts_0, kpts_1, dep_0, dep_1, K
            )
            xyz_0 = xyz_0.T
            xyz_1 = xyz_1.T

            c_id_0, c_id_1, mconf = get_geometry_weighted_correspondences(
                transform_points_Rt(xyz_0, Rt)[None, :],
                xyz_1[None, :],
                desc_0[None, :],
                desc_1[None, :],
                500,
                self.cfg.refinement.alpha,
                bidirectional=self.cfg.correspondence.bidirectional,
            )

            mkpts_0 = kpts_0[c_id_0[0]]
            mkpts_1 = kpts_1[c_id_1[0]]
            mconf = mconf[0]
            return mkpts_0, mkpts_1, mconf
        else:
            raise ValueError("Either num_steps is 1 or 2")

    def o3d_ransac_align(self, xyz_0, xyz_1, mconf):
        out = o3d_3d_correspondence_registration(xyz_0, xyz_1)
        Rt = torch.tensor(out.transformation).to(xyz_0)
        return Rt

    def cpa_ransac_align(self, xyz_0, xyz_1, mconf):
        # expand a batch dimensions
        xyz_0 = xyz_0[None, :]
        xyz_1 = xyz_1[None, :]
        mconf = mconf[None, :]

        # align
        Rt = align_cpa_ransac(
            xyz_0, xyz_1, mconf, schedule=self.cfg.alignment.ransac.schedule
        )
        return Rt[0]


def opencv_descriptor(img, feature):
    """
    Computes keypoints and feature descriptors of a given image using SIFT.
    """
    img = img.permute(1, 2, 0).numpy()
    if img.dtype != np.dtype("uint8"):
        # Convert to image to uint8 if necessary.
        img = cv.normalize(img, None, 0, 255, cv.NORM_MINMAX).astype("uint8")
    gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)

    if feature in ["sift", "rootsift"]:
        sift = cv.SIFT_create(nfeatures=5000, contrastThreshold=0.00, edgeThreshold=100)
        # sift = cv.SIFT_create()
        kps, des = sift.detectAndCompute(gray, None)
        if feature == "rootsift" and len(kps) > 0:
            des = des / des.sum(axis=1, keepdims=True).clip(min=1e-5)
            des = np.sqrt(des)
    else:
        raise ValueError(f"Unknown OpenCV descriptor: {feature}")

    kps_xy = [kp.pt for kp in kps]
    kps_sz = [kp.size for kp in kps]
    return kps_xy, des, kps_sz


def keypoints_to_unfiltered3d(mkpts_0, mkpts_1, dep_0, dep_1, K):
    _, H, W = dep_0.shape
    mkpts_0_ndc = pixel_to_ndc(mkpts_0, H=H, W=W)[None, None, :]
    mkpts_1_ndc = pixel_to_ndc(mkpts_1, H=H, W=W)[None, None, :]
    mdep_0 = grid_sample(
        dep_0.unsqueeze(0), mkpts_0_ndc, mode="nearest", align_corners=False
    )
    mdep_1 = grid_sample(
        dep_1.unsqueeze(0), mkpts_1_ndc, mode="nearest", align_corners=False
    )

    # num_pts x 1
    mdep_0 = mdep_0[0, 0, 0, :, None]
    mdep_1 = mdep_1[0, 0, 0, :, None]
    h0 = torch.ones_like(mdep_0)
    h1 = torch.ones_like(mdep_1)

    xyh0 = torch.cat((mkpts_0, h0), dim=1)
    xyh1 = torch.cat((mkpts_1, h1), dim=1)

    valid_0 = (mdep_0 > 0).float()
    valid_1 = (mdep_1 > 0).float()

    xyz_0 = K.inverse() @ (xyh0 * mdep_0).T
    xyz_1 = K.inverse() @ (xyh1 * mdep_1).T
    return xyz_0, xyz_1, valid_0, valid_1
