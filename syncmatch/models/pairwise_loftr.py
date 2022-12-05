# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from pathlib import Path

import torch
from torch.nn.functional import grid_sample

from ..utils.ransac import o3d_3d_correspondence_registration
from ..utils.transformations import transform_points_Rt
from ..utils.util import get_grid, pixel_to_ndc
from .alignment import align_cpa_ransac
from .correspondence import get_geometry_weighted_correspondences

try:
    from .loftr import LoFTR, default_cfg
except:
    print("Unable to import LoFTR. Please check docs/evaluation for instructions.")


class PairwiseLoFTR(torch.nn.Module):
    def __init__(self, model_cfg, fine=True):
        super().__init__()
        self.cfg = model_cfg
        assert self.cfg.refinement.num_steps in [1, 2], "Only handle 1 or 2 steps"

        # Initialize LoFTR
        ckpt_path = Path(__file__).parent / "pretrained_weights/loftr_ds.ckpt"
        ckpt = torch.load(ckpt_path)

        # update default cfg due to using new ckpt
        # https://github.com/zju3dv/LoFTR/issues/64
        default_cfg["coarse"]["temp_bug_fix"] = True
        # set threshold to 0 to output as many correspondences as possible
        default_cfg["match_coarse"]["thr"] = 0.0

        self.matcher = LoFTR(config=default_cfg)
        self.matcher.load_state_dict(ckpt["state_dict"])

        self.num_fail = 0
        self.return_corr2d = model_cfg.get("return_corr2d", False)
        self.fine = fine

    def generate_keypoints(self, rgb0, rgb1):
        batch_size = rgb0.shape[0]
        assert batch_size == 1
        rgb0 = (0.5 * rgb0 + 0.5).mean(dim=1, keepdim=True)
        rgb1 = (0.5 * rgb1 + 0.5).mean(dim=1, keepdim=True)

        if self.fine:
            keys = ["feat_f0", "feat_f1"]
            grid = get_grid(240, 320)[0:2].contiguous() * 2
            feat_dim = 128
        else:
            keys = ["feat_c0_t", "feat_c1_t"]
            grid = get_grid(60, 80)[0:2].contiguous() * 8
            feat_dim = 256

        kps = grid.view(2, -1).transpose(1, 0).contiguous().to(rgb0)

        pred = {"image0": rgb0, "image1": rgb1}
        self.matcher(pred)

        des0 = pred[keys[0]][0]
        des1 = pred[keys[1]][0]

        if len(des1.shape) == 3:
            des0 = des0.view(feat_dim, -1).transpose(1, 0).contiguous()
            des1 = des1.view(feat_dim, -1).transpose(1, 0).contiguous()

        return (kps, kps), (des0, des1), (None, None), kps.shape[0]

    def forward(self, rgbs, K, gt_Rts=None, deps=None):

        if hasattr(self, "depth_network"):
            print("Using predicted depth")
            self.depth_network.eval()
            deps = [self.depth_network(x)[0] for x in rgbs]

        batch_size = rgbs[0].shape[0]
        rgbs_0 = (0.5 * rgbs[0] + 0.5).mean(dim=1, keepdim=True)
        rgbs_1 = (0.5 * rgbs[1] + 0.5).mean(dim=1, keepdim=True)
        Rts = []

        corr2d_i = []
        corr2d_j = []
        corr3d_i = []
        corr3d_j = []
        corr2d_w = []

        if self.cfg.refinement.num_steps == 2:
            Rts_1 = []
            corr3d_i_1 = []
            corr3d_j_1 = []
            corr3d_w_1 = []

        num_corr = torch.zeros(batch_size).to(rgbs_0)
        _, _, H, W = rgbs_0.shape

        for i in range(batch_size):
            inp0 = rgbs_0[i : i + 1]
            inp1 = rgbs_1[i : i + 1]

            pred = {"image0": inp0, "image1": inp1}

            self.matcher(pred)

            if self.fine:
                mkpts0 = pred["mkpts0_f"]
                mkpts1 = pred["mkpts1_f"]
                f0_all = pred["feat_f0"]
                f1_all = pred["feat_f1"]
            else:
                mkpts0 = pred["mkpts0_c"]
                mkpts1 = pred["mkpts1_c"]
                f0_all = pred["feat_c0"]
                f1_all = pred["feat_c1"]

            mkpts0_ndc = pixel_to_ndc(mkpts0, H=H, W=W)[None, None, :, :]
            mkpts1_ndc = pixel_to_ndc(mkpts1, H=H, W=W)[None, None, :, :]
            feats0 = grid_sample(f0_all, mkpts0_ndc, align_corners=True)
            feats1 = grid_sample(f1_all, mkpts1_ndc, align_corners=True)
            feats0 = feats0[0, :, 0].transpose(0, 1).contiguous()
            feats1 = feats1[0, :, 0].transpose(0, 1).contiguous()

            mconf = pred["mconf"]

            mdep0 = deps[0][i, 0, mkpts0[:, 1].long(), mkpts0[:, 0].long()]
            mdep1 = deps[1][i, 0, mkpts1[:, 1].long(), mkpts1[:, 0].long()]

            h = torch.ones_like(mdep0[:, None])

            xyh0 = torch.cat((mkpts0 + 0.5, h), dim=1)
            xyh1 = torch.cat((mkpts1 + 0.5, h), dim=1)

            # filter 0 dep points
            filter_zero = True
            if filter_zero:
                valid = (mdep0 > 0) & (mdep1 > 0)
                xyh0 = xyh0[valid]
                xyh1 = xyh1[valid]
                mconf = mconf[valid]
                mdep0 = mdep0[valid]
                mdep1 = mdep1[valid]
                feats0 = feats0[valid]
                feats1 = feats1[valid]

            xyz0 = K[i].inverse() @ (xyh0 * mdep0[:, None]).T
            xyz1 = K[i].inverse() @ (xyh1 * mdep1[:, None]).T
            xyz0 = xyz0.T
            xyz1 = xyz1.T

            # filter to num matches
            n_match = 500
            if n_match < len(mconf):
                mconf_m, indices = torch.topk(torch.tensor(mconf), n_match, dim=0)
                mkpts0_m = mkpts0[indices]
                mkpts1_m = mkpts1[indices]
                mxyz0 = xyz0[indices]
                mxyz1 = xyz1[indices]
            else:
                print(f"Total number was {len(mconf)}")
                mconf_m = mconf
                mkpts0_m = mkpts0
                mkpts1_m = mkpts1
                mxyz0 = xyz0
                mxyz1 = xyz1

            num_corr[i] = len(mkpts0_m)
            mxyz0 = mxyz0[None, :, :]
            mxyz1 = mxyz1[None, :, :]
            mconf_m = mconf_m[None, :]

            if self.cfg.alignment.algorithm == "cpa_ransac":
                Rt = align_cpa_ransac(
                    mxyz0, mxyz1, mconf_m, schedule=self.cfg.alignment.ransac.schedule
                )
            elif self.cfg.alignment.algorithm == "o3d":
                out = o3d_3d_correspondence_registration(mxyz0[0], mxyz1[0])
                Rt = torch.tensor(out.transformation).to(mxyz0)[None, :]

            Rts.append(torch.stack((torch.eye(4).to(Rt), Rt[0])))
            corr3d_i.append(mxyz0[0])
            corr3d_j.append(mxyz1[0])
            corr2d_i.append(mkpts0_m)
            corr2d_j.append(mkpts1_m)
            corr2d_w.append(mconf_m)

            if self.cfg.refinement.num_steps == 2:
                c_id_0, c_id_1, mconf = get_geometry_weighted_correspondences(
                    transform_points_Rt(xyz0[None, :], Rt),
                    xyz1[None, :],
                    feats0[None, :],
                    feats1[None, :],
                    min(500, len(xyz0), len(xyz1)),
                    self.cfg.refinement.alpha,
                    bidirectional=True,
                )

                mxyz0 = xyz0[c_id_0[0]][None, :]
                mxyz1 = xyz1[c_id_1[0]][None, :]
                mkpts0_m = mkpts0[c_id_0[0]][None, :]
                mkpts1_m = mkpts1[c_id_1[0]][None, :]
                mconf_m = mconf.clamp(min=0)
                Rt = align_cpa_ransac(
                    mxyz0, mxyz1, mconf, schedule=self.cfg.alignment.ransac.schedule
                )

                Rts_1.append(torch.stack((torch.eye(4).to(Rt), Rt[0])))
                corr3d_i_1.append(mxyz0[0])
                corr3d_j_1.append(mxyz1[0])
                corr3d_w_1.append(mconf[0])

        output = {
            "loss": torch.zeros(batch_size),  # placeholder
            "num_corr": num_corr,
            "Rts_0": torch.stack(Rts, dim=0),
            "pw_corr_0": {(0, 1): (corr3d_i, corr3d_j, corr2d_w)},
        }

        if self.cfg.refinement.num_steps == 2:
            output["Rts_1"] = torch.stack(Rts_1, dim=0)
            output["pw_corr_1"] = {(0, 1): (corr3d_i_1, corr3d_j_1, corr3d_w_1)}

        return output
