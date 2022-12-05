# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import pykeops
import torch
from pykeops.torch import LazyTensor

pykeops.set_build_folder("/tmp/mbanani/keops")


def pykeops_knn(X_f, Y_f, X_n=None, Y_n=None, K=1):
    assert X_f.norm(dim=-1).max() < 1e5, "Need better batching"
    assert Y_f.norm(dim=-1).max() < 1e5, "Need better batching"

    X = LazyTensor(X_f[:, :, None, :].contiguous())
    Y = LazyTensor(Y_f[:, None, :, :].contiguous())

    D = ((X - Y) ** 2).sum(dim=-1, keepdims=True)
    D = D.sum(dim=3)

    if X_n is not None:
        # create valid tensors to handle heterogenous batch
        X_valid = torch.zeros_like(X_f[:, :, 0])
        Y_valid = torch.zeros_like(Y_f[:, :, 0])

        # set invalid to 0
        for i in range(X_f.shape[0]):
            X_valid[i, 0 : X_n[i]] = 1.0
            Y_valid[i, 0 : Y_n[i]] = 1.0

        # All valid pairs are assigned 0; invalid to 1
        X_v = LazyTensor(X_valid[:, :, None])
        Y_v = LazyTensor(Y_valid[:, None, :])
        D_invalid = 1 - (X_v * Y_v)

        # make invalid pairwise distances too large -- very hacky!!
        dmax = D.max() * 10
        D = D + dmax * D_invalid
    else:
        X_valid = torch.ones_like(X_f[:, :, 0])
        Y_valid = torch.ones_like(Y_f[:, :, 0])

    idx = D.argKmin(K=K, dim=2)  # K nearest neighbors
    dists = D.Kmin(K=K, dim=2)

    return dists, idx, X_valid.bool()


def pykeops_geometry_aware_knn(
    xyz_0, xyz_1, feat_0, feat_1, alpha, N_0=None, N_1=None, K=1
):
    max_norm = max(feat_0.norm(dim=-1).max(), feat_1.norm(dim=-1).max())
    assert abs(1 - max_norm) < 1e-5, f"assuming normalized features, max={max_norm}"

    lazy_X0 = LazyTensor(xyz_0[:, :, None, :].contiguous())
    lazy_X1 = LazyTensor(xyz_1[:, None, :, :].contiguous())
    lazy_F0 = LazyTensor(feat_0[:, :, None, :].contiguous())
    lazy_F1 = LazyTensor(feat_1[:, None, :, :].contiguous())

    dist_F = 1 - (lazy_F0 * lazy_F1).sum(dim=-1)
    dist_X = ((lazy_X0 - lazy_X1) ** 2).sum(dim=-1).sqrt()
    dist = dist_F + alpha * dist_X

    if N_0 is not None:
        raise NotImplementedError("Untested")
        # create valid tensors to handle heterogenous batch
        valid_0 = torch.zeros_like(xyz_0[:, :, 0])
        valid_1 = torch.zeros_like(xyz_1[:, :, 0])

        # set invalid to 0
        for i in range(xyz_0.shape[0]):
            valid_0[i, 0 : N_0[i]] = 1.0
            valid_1[i, 0 : N_1[i]] = 1.0

        # All valid pairs are assigned 0; invalid to 1
        lazy_V0 = LazyTensor(valid_0[:, :, None])
        lazy_V1 = LazyTensor(valid_1[:, None, :])
        invalid = 1 - (lazy_V0 * lazy_V1)

        # make invalid pairwise distances too large -- very hacky!!
        dmax = dist.max() * 10
        dist = dist + dmax * invalid
    else:
        valid_0 = torch.ones_like(xyz_0[:, :, 0])
        valid_1 = torch.ones_like(xyz_1[:, :, 0])

    idx = dist.argKmin(K=K, dim=2)  # K nearest neighbors
    dists = dist.Kmin(K=K, dim=2)

    return dists, idx, valid_0.bool()


def pykeops_geometry_aware_knn_idxs(xyz_0, xyz_1, feat_0, feat_1, alpha, K=1):
    max_norm = max(feat_0.norm(dim=-1).max(), feat_1.norm(dim=-1).max())
    assert abs(1 - max_norm) < 1e-5, f"assuming normalized features, max={max_norm}"

    lazy_X0 = LazyTensor(xyz_0[:, :, None, :].contiguous())
    lazy_X1 = LazyTensor(xyz_1[:, None, :, :].contiguous())
    lazy_F0 = LazyTensor(feat_0[:, :, None, :].contiguous())
    lazy_F1 = LazyTensor(feat_1[:, None, :, :].contiguous())

    dist_F = 1 - (lazy_F0 * lazy_F1).sum(dim=-1)
    dist_X = ((lazy_X0 - lazy_X1) ** 2).sum(dim=-1).sqrt()
    dist = dist_F + alpha * dist_X

    idx = dist.argKmin(K=K, dim=2)  # K nearest neighbors

    return idx
