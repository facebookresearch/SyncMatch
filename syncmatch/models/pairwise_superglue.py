# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import torch

from ..utils.ransac import o3d_3d_correspondence_registration
from ..utils.transformations import transform_points_Rt
from .alignment import align_cpa_ransac
from .correspondence import get_geometry_weighted_correspondences

try:
    from .superglue.matching import Matching
except:
    print("Unable to import SuperGlue. Please check docs/evaluation for instructions.")


class PairwiseSuperGlue(torch.nn.Module):
    def __init__(self, model_cfg):
        super().__init__()
        self.cfg = model_cfg
        cfg = {
            "superpoint": {
                "nms_radius": 4,  # opt.nms_radius,
                "keypoint_threshold": 0.0,  # opt.keypoint_threshold,
                "max_keypoints": 5000,  # opt.max_keypoints (-1 keeps all)
            },
            "superglue": {
                "weights": "indoor",  # opt.superglue,
                "sinkhorn_iterations": 20,  # opt.sinkhorn_iterations,
                "match_threshold": 0.0,  # opt.match_threshold,
            },
        }

        assert self.cfg.refinement.num_steps in [1, 2], "Only handle 1 or 2 steps"
        self.matcher = Matching(cfg)
        self.num_fail = 0

    def generate_keypoints(self, rgb0, rgb1):
        batch_size = rgb0.shape[0]
        assert batch_size == 1
        rgb0 = (0.5 * rgb0 + 0.5).mean(dim=1, keepdim=True)
        rgb1 = (0.5 * rgb1 + 0.5).mean(dim=1, keepdim=True)

        pred = self.matcher({"image0": rgb0, "image1": rgb1})

        kps0 = pred["keypoints0"][0]
        kps1 = pred["keypoints1"][0]
        minkps = min(len(kps0), len(kps1))

        des0 = pred["mdesc0"][0].transpose(1, 0).contiguous()
        des1 = pred["mdesc1"][0].transpose(1, 0).contiguous()

        return (kps0, kps1), (des0, des1), (None, None), minkps

    def forward(self, rgbs, K, gt_Rts=None, deps=None):

        batch_size = rgbs[0].shape[0]
        rgbs_0 = (rgbs[0] * 0.5 + 0.5).mean(dim=1, keepdim=True)
        rgbs_1 = (rgbs[1] * 0.5 + 0.5).mean(dim=1, keepdim=True)

        Rts = []
        corr3d_i = []
        corr3d_j = []
        corr3d_w = []

        if self.cfg.refinement.num_steps == 2:
            Rts_1 = []
            corr3d_i_1 = []
            corr3d_j_1 = []
            corr3d_w_1 = []

        num_corr = torch.zeros(batch_size).to(rgbs_0)

        for i in range(batch_size):
            inp0 = rgbs_0[i : i + 1]
            inp1 = rgbs_1[i : i + 1]

            pred = self.matcher({"image0": inp0, "image1": inp1})
            pred = {k: v[0].cpu().numpy() for k, v in pred.items()}
            kpts0, kpts1 = pred["keypoints0"], pred["keypoints1"]
            matches, conf = pred["matches0"], pred["matching_scores0"]
            feat0, feat1 = pred["descriptors0"].T, pred["descriptors1"].T

            # move to gpu
            kpts0 = torch.tensor(kpts0).to(deps[0])
            kpts1 = torch.tensor(kpts1).to(deps[1])
            feat0 = torch.tensor(feat0).to(deps[0])
            feat1 = torch.tensor(feat1).to(deps[1])
            conf = torch.tensor(conf).to(deps[1])
            matches = torch.tensor(matches).to(deps[0])

            # get depths
            dep0 = deps[0][i, 0, kpts0[:, 1].long(), kpts0[:, 0].long()]
            dep1 = deps[1][i, 0, kpts1[:, 1].long(), kpts1[:, 0].long()]

            # filter valid depth 0
            valid_dep0 = dep0 > 0

            kpts0 = kpts0[valid_dep0]
            feat0 = feat0[valid_dep0]
            dep0 = dep0[valid_dep0]
            conf = conf[valid_dep0]
            matches = matches[valid_dep0]

            # set matches to invalid depth to no matches
            matches[matches > -1][dep1[matches[matches > -1].long()] <= 0] = -1

            # convert keypoints to xyz
            h0 = torch.ones_like(dep0[:, None])
            h1 = torch.ones_like(dep1[:, None])
            xyh0 = torch.cat((kpts0 + 0.5, h0), dim=1)
            xyh1 = torch.cat((kpts1 + 0.5, h1), dim=1)
            xyz0 = (K[i].inverse() @ (xyh0 * dep0[:, None]).T).T
            xyz1 = (K[i].inverse() @ (xyh1 * dep1[:, None]).T).T

            # filter based on superglue and num matches
            mconf = conf[matches > -1]
            mxyz0 = xyz0[matches > -1]
            matches = matches[matches > -1]
            mxyz1 = xyz1[matches.long()]

            n_match = 500
            if n_match < len(mconf):
                _, indices = torch.topk(mconf.clone().detach(), n_match, dim=0)
                mconf = mconf[indices][None, :]
                mxyz0 = mxyz0[indices][None, :]
                mxyz1 = mxyz1[indices][None, :]
            else:
                mconf = mconf[None, :]
                mxyz0 = mxyz0[None, :]
                mxyz1 = mxyz1[None, :]

            if self.cfg.alignment.algorithm == "cpa_ransac":
                Rt = align_cpa_ransac(
                    mxyz0, mxyz1, mconf, schedule=self.cfg.alignment.ransac.schedule
                )
            elif self.cfg.alignment.algorithm == "o3d":
                out = o3d_3d_correspondence_registration(mxyz0[0], mxyz1[0])
                Rt = torch.tensor(out.transformation).to(mxyz0)[None, :]

            Rts.append(torch.stack((torch.eye(4).to(Rt), Rt[0])))
            corr3d_i.append(mxyz0[0])
            corr3d_j.append(mxyz1[0])
            corr3d_w.append(mconf[0])
            num_corr[i] = mconf.shape[1]

            if self.cfg.refinement.num_steps == 2:
                # filter valid_dep 1
                valid_dep1 = dep1 > 0
                xyz1 = xyz1[valid_dep1]
                feat1 = feat1[valid_dep1]

                # compute new correspondences
                c_id_0, c_id_1, mconf = get_geometry_weighted_correspondences(
                    transform_points_Rt(xyz0[None, :], Rt),
                    xyz1[None, :],
                    feat0[None, :],
                    feat1[None, :],
                    min(500, len(xyz0), len(xyz1)),
                    self.cfg.refinement.alpha,
                    bidirectional=True,
                )

                mxyz0 = xyz0[c_id_0[0]][None, :]
                mxyz1 = xyz1[c_id_1[0]][None, :]
                mconf = mconf.clamp(min=0)
                Rt = align_cpa_ransac(
                    mxyz0, mxyz1, mconf, schedule=self.cfg.alignment.ransac.schedule
                )

                Rts_1.append(torch.stack((torch.eye(4).to(Rt), Rt[0])))
                corr3d_i_1.append(mxyz0[0])
                corr3d_j_1.append(mxyz1[0])
                corr3d_w_1.append(mconf[0])

        output = {
            "Rts_0": torch.stack(Rts, dim=0),
            "pw_corr_0": {(0, 1): (corr3d_i, corr3d_j, corr3d_w)},
            "num_corr": num_corr,
            "loss": torch.zeros_like(num_corr).float(),
        }

        if self.cfg.refinement.num_steps == 2:
            output["Rts_1"] = torch.stack(Rts_1, dim=0)
            output["pw_corr_1"] = {(0, 1): (corr3d_i_1, corr3d_j_1, corr3d_w_1)}

        return output
