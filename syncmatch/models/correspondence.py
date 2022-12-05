# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from typing import Optional

import torch
from pytorch3d.ops.knn import knn_points as pt3d_knn
from torch.nn.functional import cosine_similarity, relu

from ..utils.faiss_knn import faiss_knn
from ..utils.pykeops_knn import pykeops_geometry_aware_knn_idxs, pykeops_knn
from ..utils.util import knn_gather, list_knn_gather


def knn_points(
    X_f,
    Y_f,
    X_n=None,
    Y_n=None,
    K=1,
    metric="euclidean",
    backend="faiss",
    normed_features=False,
):
    """
    Finds the kNN according to either euclidean distance or cosine distance. This is
    tricky since PyTorch3D's fast kNN kernel does euclidean distance, however, we can
    take advantage of the relation between euclidean distance and cosine distance for
    points sampled on an n-dimension sphere.

    Using the quadratic expansion, we find that finding the kNN between two normalized
    is the same regardless of whether the metric is euclidean distance or cosine
    similiarity.

        -2 * xTy = (x - y)^2 - x^2 - y^2
        -2 * xtY = (x - y)^2 - 1 - 1
        - xTy = 0.5 * (x - y)^2 - 1

    Hence, the metric that would maximize cosine similarity is the same as that which
    would minimize the euclidean distance between the points, with the distances being
    a simple linear transformation.
    """
    assert metric in ["cosine", "euclidean"]
    if metric == "cosine" and (not normed_features):
        X_f = torch.nn.functional.normalize(X_f, dim=-1)
        Y_f = torch.nn.functional.normalize(Y_f, dim=-1)

    assert backend in ["pykeops", "pytorch3d"]  # faiss is off for now ..
    if backend == "faiss":
        dists_faiss, idx = faiss_knn(X_f, Y_f, K, X_n, Y_n)
        valid = (dists_faiss != -1).any(dim=2)
    elif backend == "pytorch3d":
        assert K > 1, "Valid currently requires 2 values, could be improved"
        _, idx, _ = pt3d_knn(X_f, Y_f, X_n, Y_n, K=K)
        valid = idx[:, :, 0] != idx[:, :, 1]
    elif backend == "pykeops":
        _, idx, valid = pykeops_knn(X_f, Y_f, X_n, Y_n, K=K)

    # Calculate dists since faiss does not support backprop
    Y_nn = knn_gather(Y_f, idx)  # batch x n_points x 2 x F

    if metric == "euclidean":
        dists = (Y_nn - X_f[:, :, None, :]).norm(p=2, dim=3)
    elif metric == "cosine":
        dists = 1 - cosine_similarity(Y_nn, X_f[:, :, None, :], dim=-1)

    dists = dists * valid.float()[:, :, None]

    return dists, idx, valid


def knn_points_geometry_aware(
    X_x, Y_x, X_f, Y_f, alpha, K=1, normed_features=False, return_components=False
):
    """
    Finds the kNN in a space that includes both feature and geometric distance. The
    metric in that space is:
        dist(X0,X1,F0,F1) = cos_dist(F0, F1) + alpha * ||X0 - X1||

    Due to the composite function, we can only use PyKeOps.
    """
    assert X_x.shape[-1] == 3
    assert Y_x.shape[-1] == 3
    if not normed_features:
        X_f = torch.nn.functional.normalize(X_f, dim=-1)
        Y_f = torch.nn.functional.normalize(Y_f, dim=-1)

    idx = pykeops_geometry_aware_knn_idxs(X_x, Y_x, X_f, Y_f, alpha=alpha, K=K)
    Y_x_nn, Y_f_nn = list_knn_gather([Y_x, Y_f], idx)

    dist_F = 1 - cosine_similarity(Y_f_nn, X_f[:, :, None, :], dim=-1)
    dist_X = (Y_x_nn - X_x[:, :, None, :]).norm(p=2, dim=-1)
    dists = dist_F + alpha * dist_X

    if return_components:
        return dists, idx, Y_x_nn, Y_f_nn, dist_X, dist_F
    else:
        return dists, idx, Y_x_nn, Y_f_nn


def get_geometry_weighted_correspondences(
    P1_X,
    P2_X,
    P1_F,
    P2_F,
    num_corres,
    alpha,
    normed_features=False,
    P1_W=None,
    P2_W=None,
    bidirectional=False,
):
    assert P1_X.dim() == 3
    assert P2_X.dim() == 3

    # Get kNN, choose ID as closest point, compute weights
    dists_1, idx_1, P2_X_, P2_F_, _, dists_1_F = knn_points_geometry_aware(
        P1_X,
        P2_X,
        P1_F,
        P2_F,
        alpha,
        K=2,
        normed_features=normed_features,
        return_components=True,
    )
    idx_1 = idx_1[:, :, 0:1]
    weights_1 = calculate_ratio_test(dists_1)

    if P1_W is not None:
        assert P2_W is not None
        W2_in1 = P2_W.gather(1, idx_1)
        weights_1 = weights_1 * P1_W * W2_in1

    if bidirectional:
        dists_2, idx_2, _, _, _, dists_2_F = knn_points_geometry_aware(
            P2_X,
            P1_X,
            P2_F,
            P1_F,
            alpha,
            K=2,
            normed_features=normed_features,
            return_components=True,
        )
        idx_2 = idx_2[:, :, 0:1]
        weights_2 = calculate_ratio_test(dists_2)

        if P2_W is not None:
            assert P1_W is not None
            W1_in2 = P1_W.gather(1, idx_2)
            weights_2 = weights_2 * P2_W * W1_in2

        # get top half_corr from each direction
        half_corr = num_corres // 2
        m12_idx1, m12_idx2, m12_dist = get_topk_matches(weights_1, idx_1, half_corr)
        m21_idx2, m21_idx1, m21_dist = get_topk_matches(weights_2, idx_2, half_corr)

        # concatenate into correspondences and weights
        all_idx1 = torch.cat((m12_idx1, m21_idx1), dim=1)
        all_idx2 = torch.cat((m12_idx2, m21_idx2), dim=1)
        all_dist = torch.cat((m12_dist, m21_dist), dim=1)
    else:
        all_idx1, all_idx2, all_dist = get_topk_matches(weights_1, idx_1, num_corres)

    return all_idx1.squeeze(dim=2), all_idx2.squeeze(dim=2), all_dist.squeeze(dim=2)


def get_correspondences_ratio_test(
    P1,
    P2,
    num_corres,
    W1=None,
    W2=None,
    metric="cosine",
    normed_features=False,
    bidirectional=False,
):
    """
    Input:
        P1          pytorch3d's Pointclouds     features for first pointcloud
        P2          pytorch3d's Pointclouds     features for first pointcloud
        num_corres  Int                         number of correspondences
        metric      {cosine, euclidean}         metric to be used for kNN

    Returns:
        LongTensor (N x 2 * num_corres)         Indices for first pointcloud
        LongTensor (N x 2 * num_corres)         Indices for second pointcloud
        FloatTensor (N x 2 * num_corres)        Weights for each correspondace
        FloatTensor (N x 2 * num_corres)        Cosine distance between features
    """
    backend = "pykeops"
    if type(P1) == torch.Tensor:
        P1_N = None
        P2_N = None

        # reshape to pointcloud format for kNN
        if len(P1.shape) == 4:
            batch, feat_dim, H, W = P1.shape
            P1_F = P1.view(batch, feat_dim, H * W).permute(0, 2, 1).contiguous()
            P2_F = P2.view(batch, feat_dim, H * W).permute(0, 2, 1).contiguous()
        else:
            P1_F = P1
            P2_F = P2
    else:
        P1_F = P1.features_padded()
        P2_F = P2.features_padded()
        P1_N = P1.num_points_per_cloud()
        P2_N = P2.num_points_per_cloud()

    # Calculate kNN for k=2; both outputs are (N, P, K)
    # idx_1 returns the indices of the nearest neighbor in P2
    # output is cosine distance (0, 2)
    K = 2

    dists_1, idx_1, val_1 = knn_points(
        P1_F, P2_F, P1_N, P2_N, K, metric, backend, normed_features
    )
    idx_1 = idx_1[:, :, 0:1]
    weights_1 = calculate_ratio_test(dists_1)
    weights_1 = weights_1 * val_1.unsqueeze(-1)

    if W1 is not None:
        assert W2 is not None
        W2_in1 = W2.gather(1, idx_1)
        weights_1 = weights_1 * W1 * W2_in1

    # Take the nearest neighbor for the indices for k={1, 2}
    if bidirectional:
        dists_2, idx_2, val_2 = knn_points(
            P2_F, P1_F, P2_N, P1_N, K, metric, backend, normed_features
        )
        idx_2 = idx_2[:, :, 0:1]
        weights_2 = calculate_ratio_test(dists_2)
        weights_2 = weights_2 * val_2.unsqueeze(-1)

        if W1 is not None:
            assert W2 is not None
            W1_in2 = W1.gather(1, idx_2)
            weights_2 = weights_2 * W2 * W1_in2

        # Get topK matches in both directions
        num_corres = num_corres // 2
        if P1_N is None:
            n_corres_1 = num_corres
            n_corres_2 = num_corres
        else:
            n_corres_1 = min(num_corres, P1_N.min())
            n_corres_2 = min(num_corres, P2_N.min())
            if n_corres_1 < num_corres or n_corres_2 < num_corres:
                print(f"Min corresponds is {n_corres_1} and {n_corres_2}")

        m12_idx1, m12_idx2, m12_dist = get_topk_matches(weights_1, idx_1, n_corres_1)
        m21_idx2, m21_idx1, m21_dist = get_topk_matches(weights_2, idx_2, n_corres_2)

        # concatenate into correspondences and weights
        all_idx1 = torch.cat((m12_idx1, m21_idx1), dim=1)
        all_idx2 = torch.cat((m12_idx2, m21_idx2), dim=1)
        all_dist = torch.cat((m12_dist, m21_dist), dim=1)
    else:
        n_corres_1 = num_corres if P1_N is None else min(num_corres, P1_N.min())
        all_idx1, all_idx2, all_dist = get_topk_matches(weights_1, idx_1, n_corres_1)

    return all_idx1.squeeze(dim=2), all_idx2.squeeze(dim=2), all_dist.squeeze(dim=2)


@torch.jit.script
def calculate_ratio_test(
    dists: torch.Tensor,
    num_pos: int = 1,
    neg_id: int = 1,
    use_softmax: bool = False,
    exponential: bool = False,
    temperature: float = 1.0,
    sink_value: Optional[float] = None,
):
    """
    Calculate weights for matches based on the ratio between kNN distances.

    Input:
        (N, P, 2) Cosine Distance between point and nearest 2 neighbors
    Output:
        (N, P, 1) Weight based on ratio; higher is more unique match
    """
    # Ratio -- close to 0 is completely unique; 1 is same feature
    # Weight -- Convert so that higher is more unique
    # clamping because some dists will be 0 (when not in the pointcloud
    dists = dists.clamp(min=1e-9)
    pos_sample = dists[:, :, 0:num_pos]
    neg_sample = dists[:, :, neg_id].unsqueeze(dim=2)

    if use_softmax:
        assert num_pos > 1
        assert sink_value is not None

        # ratio (batch x num_points x num matches) [1 -> 1e4]
        ratio = neg_sample / pos_sample.clamp(min=1e-4)

        # add sink value
        pad = sink_value * torch.ones_like(ratio[:, :, 0:1])
        ratio_padded = torch.cat((ratio, pad), dim=2)

        # apply softmax and discard sin value
        weight = ratio_padded.softmax(dim=2)[:, :, :-1]
    else:
        ratio = pos_sample / neg_sample.clamp(min=1e-9)

        if exponential:
            weight = (-1 * temperature * ratio).exp()
        else:
            weight = 1 - ratio

    return weight


# @torch.jit.script
def get_topk_matches(dists, idx, num_corres: int):
    if dists.size(2) == 1:
        num_corres = min(num_corres, dists.shape[1])
        dist, idx_source = torch.topk(dists, k=num_corres, dim=1)
        idx_target = idx.gather(1, idx_source)
        return idx_source, idx_target, dist
    else:
        b, n, k = dists.shape
        dists = dists.view(b, n * k, 1)
        idx = idx.view(b, n * k, 1)
        num_corres = min(num_corres, dists.shape[1])
        dist, idx_source = torch.topk(dists, k=num_corres, dim=1)
        idx_target = idx.gather(1, idx_source)
        idx_source = idx_source // k
        return idx_source, idx_target, dist


# ------- Sinkhorn --------------
def log_sinkhorn_iterations(Z, log_mu, log_nu, iters: int):
    """Perform Sinkhorn Normalization in Log-space for stability"""
    u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
    for _ in range(iters):
        u = log_mu - torch.logsumexp(Z + v.unsqueeze(1), dim=2)
        v = log_nu - torch.logsumexp(Z + u.unsqueeze(2), dim=1)
    return Z + u.unsqueeze(2) + v.unsqueeze(1)


def log_optimal_transport(scores, alpha, iters: int):
    """Perform Differentiable Optimal Transport in Log-space for stability"""
    b, m, n = scores.shape
    one = scores.new_tensor(1)
    ms, ns = (m * one).to(scores), (n * one).to(scores)

    bins0 = alpha.expand(b, m, 1)
    bins1 = alpha.expand(b, 1, n)
    alpha = alpha.expand(b, 1, 1)

    couplings = torch.cat(
        [torch.cat([scores, bins0], -1), torch.cat([bins1, alpha], -1)], 1
    )

    norm = -(ms + ns).log()
    log_mu = torch.cat([norm.expand(m), ns.log()[None] + norm])
    log_nu = torch.cat([norm.expand(n), ms.log()[None] + norm])
    log_mu, log_nu = log_mu[None].expand(b, -1), log_nu[None].expand(b, -1)

    Z = log_sinkhorn_iterations(couplings, log_mu, log_nu, iters)
    Z = Z - norm  # multiply probabilities by M+N
    return Z
