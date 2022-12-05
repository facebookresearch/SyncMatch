# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import pytorch3d
import torch
from torch.nn.functional import normalize

convention = "Rt_{w2c} @ World[3,n] = Camera[3,n]"

# Pointclouds are 3xN (with invisible 4-th row of 1s
# Left-multiplication maps world to camera: P @ X_world = X_cam, P is 4x4
# P_ij = P_j @ P_i^{-1}.
# P_ij @ X_i = X_j

# P = [[ R t ]
#      [ 0 1 ]]


def make_Rt(R, t):
    """
    Encode the transformation X -> X @ R + t where X has shape [n,3]
    """
    Rt = torch.cat([R.transpose(-2, -1), t[..., None]], dim=-1)
    pad = torch.zeros_like(Rt[..., 2:3, :])
    pad[..., -1] = 1.0
    Rt = torch.cat((Rt, pad), dim=-2)
    return Rt


def split_Rt(Rt):
    """
    Split SE(3) into SO(3) and R^3
    """
    return Rt[..., :3, :3], Rt[..., :3, 3]


def SE3_inverse(P):
    R_inv = P[..., :3, :3].transpose(-2, -1)
    t_inv = -1 * R_inv @ P[..., :3, 3:4]
    bottom_row = P[..., 3:4, :]
    Rt_inv = torch.cat((R_inv, t_inv), dim=-1)
    P_inv = torch.cat((Rt_inv, bottom_row), dim=-2)
    return P_inv


# @torch.jit.script
def transform_points_Rt(
    points: torch.Tensor, viewpoint: torch.Tensor, inverse: bool = False
):
    R = viewpoint[..., :3, :3]
    t = viewpoint[..., None, :3, 3]
    # N.B. points is (..., n, 3) not (..., 3, n)
    if inverse:
        return (points - t) @ R
    else:
        return points @ R.transpose(-2, -1) + t


# @torch.jit.script
def transform_points_R(
    points: torch.Tensor, viewpoint: torch.Tensor, inverse: bool = False
):
    R = viewpoint[..., :3, :3]
    if inverse:
        return points @ R
    else:
        return points @ R.transpose(-2, -1)


def get_relative_Rt(Rt_i, Rt_j):
    """Generates the relative Rt assuming that we have two world
    to camera Rts. Hence, Rt_ij = inverse(Rt_i) @ Rt_j.

    Args:
        Rt_i (FloatTensor): world_to_camera for camera i (batch, 4, 4)
        Rt_j (FloatTensor): world_to_camera for camera j (batch, 4, 4)

    Returns:
        Rt_ij (FloatTensor): transformation from i to j (batch, 4, 4)
    """
    assert Rt_i.shape == Rt_j.shape, "Shape mismatch"
    assert Rt_i.size(-2) == 4
    assert Rt_i.size(-1) == 4

    return Rt_j @ SE3_inverse(Rt_i)


def random_Rt(batch_size, r_mag, t_mag):
    """
    Generate a random Rt matrix based on a rotation and translation magnitude
    """
    noise_t = t_mag * normalize(torch.randn(batch_size, 3), p=2, dim=1)
    noise_r = r_mag * normalize(torch.randn(batch_size, 3), p=2, dim=1)
    noise_R = pytorch3d.transforms.euler_angles_to_matrix(noise_r * 3.14 / 180, "XYZ")
    return make_Rt(noise_R, noise_t)
