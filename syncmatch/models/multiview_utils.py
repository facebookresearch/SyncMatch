# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import torch
from easydict import EasyDict
from torch.nn.functional import interpolate, normalize

from ..utils.util import get_grid
from .backbones import get_visual_backbone
from .depth import DepthPredictionNetwork


class MultiviewScreen(torch.nn.Module):
    def __init__(self, cfg):
        """Generates a backend object that keeps track of images, depth maps, and their
        associated pointclouds and features.

        Args:
            cfg (DictConfig): parameters defining models and point cloud options
        """
        super().__init__()

        self.cfg = cfg
        self.visual_backbone = get_visual_backbone(cfg.features)

        # get depth network
        if not cfg.use_gt_depth:
            self.depth_network = DepthPredictionNetwork(cfg.depth)

            # load depth weights
            depth_weights = torch.load(cfg.depth.path)[1]
            print(f"loaded network for path {cfg.depth.path}")
            self.depth_network.load_state_dict(depth_weights)

            for p in self.depth_network.parameters():
                p.requires_grad = False

    def forward(self, rgbs, deps, K):
        """Given an image and its depth, extract the features and project the depth
        into 3D.

        Args:
            rgbs (list): list of RGB images, each is FloatTensor(batch, 3, H, W)
            deps (list): list of depth maps, each is FloatTensor(batch, H, W)
            K (FloatTensor): camera intrinsics

        Returns:
            tuple:
                screen (EasyDict) contains 2D data; eg, rgb, depth map, keypoints
                pc (EasyDict) contains 3D point cloud information
        """
        screen = self.setup_screen(rgbs, deps, K)
        pc = self.setup_pc(screen, 0)
        return screen, pc

    def get_keypoints(self, rgb):
        feats = self.get_features(rgb.unsqueeze(1))[0][0, 0]

        _, _, H, W = rgb.shape
        feat_dim, fH, fW = feats.shape
        grid = get_grid(fH, fW).to(rgb) * H / fH

        grid = grid[:2].reshape(2, -1).transpose(1, 0).contiguous()
        feats = feats.reshape(feat_dim, -1).transpose(1, 0).contiguous()

        return grid, feats

    def setup_screen(self, rgbs, deps, K):
        """Creates an object that has the 2D information as well as the sampled
        view pairs.

        Args:
            rgbs (list): RGB images, each is a FloatTensor (batch, 3, H, W)
            deps (list): Depth maps, each is FloatTensor (batch, H, W)
            K (FloatTensor): camera intrinsics

        Returns:
            _type_: _description_
        """
        screen = EasyDict()

        screen.rgbs = torch.stack(rgbs, dim=1)
        screen.batch, screen.num_views, _, screen.H, screen.W = screen.rgbs.shape

        # sample (i,j) edges for consideration
        screen.IJs = []  # edges to compare
        for i in range(screen.num_views - 1):
            screen.IJs.append((i, i + 1))

            if self.cfg.correspondence.sampling == "all":
                for j in range(i + 2, screen.num_views):
                    screen.IJs.append((i, j))
            elif self.cfg.correspondence.sampling[:11] == "localsparse":
                _, window, p = self.cfg.correspondence.sampling.split("_")
                window = int(window)
                p = float(p)
                for j in range(i + 2, screen.num_views):
                    if j <= i + window:
                        screen.IJs.append((i, j))
                    else:
                        if torch.rand(()) < p:
                            screen.IJs.append((i, j))
            elif self.cfg.correspondence.sampling[:9] == "sparse_0.":
                p = float(self.cfg.correspondence.sampling[7:])
                for j in range(i + 2, screen.num_views):
                    if torch.rand(()) < p:
                        screen.IJs.append((i, j))
            elif self.cfg.correspondence.sampling[:7] == "sparse_":
                n = int(self.cfg.correspondence.sampling[7:])
                p = 2 * (n - 1) / (screen.num_views - 1)
                for j in range(i + 2, screen.num_views):
                    if torch.rand(()) < p:
                        screen.IJs.append((i, j))

        # get grid
        # - multiview, homogeoneous coordinates (batch, 3, H, W)
        # - assume K is the same across views
        grid = get_grid(screen.H, screen.W).unsqueeze(0)
        xyh_uncal = grid.to(screen.rgbs).expand(screen.batch, 3, -1, -1)

        xyh_cal = K.inverse() @ xyh_uncal.flatten(2, 3)
        xyh_cal = xyh_cal.view(screen.batch, 1, 3, screen.H, screen.W)

        screen.xyh_uncal = xyh_uncal
        screen.xyh_cal = xyh_cal.expand_as(screen.rgbs)

        # get depth
        screen.depth = torch.stack(deps, dim=1)

        # Extract features and depth; multiscale, but for now, there's one scale
        screen.feats = self.get_features(screen.rgbs)

        return screen

    def get_features(self, rgbs):
        """Extracts features. Could do more if we had multi-scale features,
        but currently we stick to just one scale

        Args:
            rgbs (FloatTensor): rgbs (batch, num_views, 3, H, W)

        Returns:
            tuple:
                feats (batch, num_views, F, H, W) keep normalized and unnormalized features
        """
        batch, num_views = rgbs.shape[0:2]

        # for a very large number of views (and a potentially large backbone),
        # it might make more sense to only process a subset of views at a time
        if num_views > 128:
            feat_slices = []
            for start in range(0, num_views, 128):
                end = min(start + 128, num_views)
                feats_i = self.visual_backbone(rgbs[:, start:end].flatten(0, 1))
                feats_i = feats_i.view(batch, end - start, *feats_i.shape[1:])
                feat_slices.append(feats_i)

            features = [torch.cat(feat_slices, dim=1).flatten(0, 1)]
        else:
            features = [self.visual_backbone(rgbs.flatten(0, 1))]

        feats = []

        for scale_i, f in enumerate(features):
            assert f.isfinite().all()
            f = f.view(batch, num_views, *f.shape[1:])
            feats.append(normalize(f, dim=2))

        return feats

    # TODO | need better name
    def img_scale_pc(self, pc, x):
        if x is None:
            return None

        if x.shape[3:] != (pc.H, pc.W):
            x = interpolate(x.flatten(0, 1), (pc.H, pc.W))
            x = x.view(pc.batch, pc.num_views, -1, pc.H, pc.W)

        # flatten spatial dimensions and tranpose: (F x H x W) -> (N x F)
        x = x.flatten(3, 4).transpose(2, 3).contiguous()
        return x

    def setup_pc(self, screen, screen_scale=0):
        pc = EasyDict()  # .feats .depth .screen .cam .world .beta .depth_i
        feats = screen.feats[screen_scale]
        pc.batch, pc.num_views, pc.feat_dim, pc.H, pc.W = feats.shape
        pc.IJs = screen.IJs

        uv_multi = screen.xyh_uncal[:, None, :2].repeat(1, pc.num_views, 1, 1, 1)
        uv_multi = uv_multi.contiguous()

        # crop features/depth to fit pointcloud
        pc.feats = self.img_scale_pc(pc, feats)
        pc.xy1 = self.img_scale_pc(pc, screen.xyh_cal)
        pc.depth = self.img_scale_pc(pc, screen.depth)
        pc.uv_screen = self.img_scale_pc(pc, uv_multi)

        pc.xyz_cam = pc.xy1 * pc.depth

        with torch.no_grad():
            # noisy version of pc.xyz_cam for judging overalap of frames
            i = (
                torch.arange(pc.H // 20, pc.H, pc.H // 10)[:, None] * pc.W
                + torch.arange(pc.W // 20, pc.W, pc.W // 10)
            ).view(-1)
            pc.xyz_cam_blur = pc.xyz_cam[:, :, i]
            pc.xyz_cam_blur.mul_(torch.rand_like(pc.xyz_cam_blur[:, :, :, :1]) + 0.5)

        # initialize xyz_world with xyz_cam (basically assuming identity)
        # could perform step_0 alignment with alpha>0 for neighbor frames IJ=(i,i+1)
        pc.xyz_world = pc.xyz_cam

        return pc
