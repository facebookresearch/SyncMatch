# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import os
from typing import Optional

import numpy as np
import pytorch3d
import torch
from scipy import interpolate as scipy_interpolate


def pixel_to_ndc(kpts, H, W):
    x, y = kpts[..., 0:1], kpts[..., 1:2]
    x = (x / W) * 2 - 1
    y = (y / H) * 2 - 1
    return torch.cat((x, y), dim=-1)


# @torch.jit.script
def get_grid(H: int, W: int):
    # Generate a grid that's equally spaced based on image & embed size
    grid_x = torch.linspace(0.5, W - 0.5, W)
    grid_y = torch.linspace(0.5, H - 0.5, H)

    xs = grid_x.view(1, W).repeat(H, 1)
    ys = grid_y.view(H, 1).repeat(1, W)
    zs = torch.ones_like(xs)

    # Camera coordinate frame is +xyz (right, down, into-camera)
    # Dims: 3 x H x W
    grid_xyz = torch.stack((xs, ys, zs), dim=0)
    return grid_xyz


# @torch.jit.script
def grid_to_pointcloud(K_inv, depth, grid: Optional[torch.Tensor]):
    _, H, W = depth.shape

    if grid is None:
        grid = get_grid(H, W)

    # Apply inverse projection
    points = depth * grid

    # Invert intriniscs
    points = points.view(3, H * W)
    points = K_inv @ points
    points = points.permute(1, 0)

    return points


def depth_to_pointclouds(depth, K):
    B, _, H, W = depth.shape
    K_inv = K.inverse()
    grid = get_grid(H, W).to(depth)
    pointclouds = []

    for i in range(B):
        pc = grid_to_pointcloud(K_inv[i], depth[i], grid)

        # filter out invalid points
        pc = pc[pc[:, 2] > 0]
        pointclouds.append(pc)

    return pytorch3d.structures.Pointclouds(points=pointclouds)


def homocoord_for_depth(depth, multiview=False):
    if multiview:
        assert len(depth.shape) == 5, f"{depth.shape} != (batch, views, 1, H, W)"
    else:
        assert len(depth.shape) == 4, f"{depth.shape} != (batch, 1, H, W)"
        depth = depth.unsqueeze(1)

    batch, views, _, H, W = depth.shape

    # 3 x H x W
    grid = get_grid(H, W).to(depth)

    # add batch and views dimensions
    grid = grid[None, None, :].repeat(batch, views, 1, 1, 1).contiguous()
    return grid


def mv_grid_to_points(mv_map):
    batch, views, feat, H, W = mv_map.shape
    mv_map = mv_map.view(batch, views, feat, H * W)
    mv_points = mv_map.permute(0, 1, 3, 2).contiguous()
    return mv_points


def xyz_to_camera(xyz, K):
    uvd = xyz @ K.transpose(-2, -1)
    _d = uvd[..., :, 2:3]
    _d[_d == 0] = 1e-9
    return uvd[..., :, :2] / _d


def corr_dict_to_3dcorr(corr, xyz):
    corr_output = {}
    for ij in corr:
        corr_output[ij] = (
            nn_gather(xyz[:, ij[0]], corr[ij][0].squeeze(-1)),
            nn_gather(xyz[:, ij[1]], corr[ij][1].squeeze(-1)),
            corr[ij][2],
        )
    return corr_output


def corr_dict_to_2dcorr(corr, xyh):
    corr_output = {}
    for ij in corr:
        corr_output[ij] = (
            nn_gather(xyh[:, ij[0]], corr[ij][0].squeeze(-1)),
            nn_gather(xyh[:, ij[1]], corr[ij][1].squeeze(-1)),
            corr[ij][2],
        )
    return corr_output


def fill_depth(depth):
    """Fill depth function from Zach Teed's V2D: an approximate way to densify
    already pretty dense depth.
    """
    x, y = np.meshgrid(
        np.arange(depth.shape[1]).astype("float32"),
        np.arange(depth.shape[0]).astype("float32"),
    )
    xx = x[depth > 0]
    yy = y[depth > 0]
    zz = depth[depth > 0]

    grid = scipy_interpolate.griddata((xx, yy), zz.ravel(), (x, y), method="nearest")
    return grid


def nn_gather(points, indices):
    # expand indices to same dimensions as points
    indices = indices[:, :, None]
    indices = indices.expand(-1, -1, points.shape[2])
    return points.gather(1, indices)


def knn_gather(x, idxs):
    # x NMU
    # idxs NLK
    N, L, K = idxs.shape
    M = x.size(1)
    idxs = (
        idxs.flatten(1, 2)
        .add(torch.arange(N, device=x.device)[:, None] * M)
        .flatten(0, 1)
    )
    return x.flatten(0, 1)[idxs].view(N, L, K, -1)


def list_knn_gather(xs, idxs):
    # x[0] NMU
    # idxs NLK
    N, L, K = idxs.shape
    M = xs[0].size(1)
    idxs = (
        idxs.flatten(1, 2)
        .add(torch.arange(N, device=xs[0].device)[:, None] * M)
        .flatten(0, 1)
    )
    return [x.flatten(0, 1)[idxs].view(N, L, K, -1) for x in xs]


def modify_keys(old_dict, prefix="", suffix=""):
    new_dict = {}
    for key in old_dict:
        new_dict[f"{prefix}{key}{suffix}"] = old_dict[key]

    return new_dict


def full_detach(x):
    x_type = type(x)
    if x_type == dict:
        return detach_dictionary(x)
    elif x_type == tuple:
        return tuple(full_detach(x_el) for x_el in x)
    elif x_type == list:
        return [full_detach(x_el) for x_el in x]
    elif x_type == torch.Tensor:
        return x.detach().cpu()
    else:
        raise ValueError(f"Unable to detach input of type {x_type}")


def detach_dictionary(gpu_dict):
    for k in gpu_dict:
        old_val = gpu_dict[k]
        gpu_dict[k] = full_detach(old_val)

    return gpu_dict
