# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import torch
from pytorch3d.ops import corresponding_points_alignment
from torch.nn.functional import normalize

from ..utils.ransac import o3d_3d_correspondence_registration
from ..utils.transformations import make_Rt
from ..utils.util import list_knn_gather, nn_gather


def kabsch_algorithm(corr_P, corr_Q, corr_W):
    """Runs the weighted kabsh algorithm ...

    Args:
        corr_P (FloatTensor): pointcloud P (batch, N, 3)
        corr_Q (FloatTensor): pointcloud Q (batch, N, 3)
        corr_W (FloatTensor): correspondence weights (batch, N)

    Returns:
        FloatTensor: (batch, 3, 4) estimated registration
    """
    corr_P = corr_P.double()
    corr_Q = corr_Q.double()
    corr_W = corr_W.double().clamp(min=1e-12)

    corr_W = normalize(corr_W, p=2, dim=-1)
    Rt_out = corresponding_points_alignment(corr_P, corr_Q, corr_W)
    return Rt_out


def align_o3d(corres, P, Q):
    """Align pointclouds using Open3D's RANSAC based aligner

    Args:
        corres (tuple): the corresponding indices between two pointclouds
        P (FloatTensor): pointcloud P (batch, N, 3)
        Q (FloatTensor): pointcloud Q (batch, N, 3)

    Returns:
        FloatTensor: (batch, 3, 4) estimated registration
    """
    # get useful variables
    corr_P_idx, corr_Q_idx = corres[:2]

    # get match features and coord
    corr_P = nn_gather(P, corr_P_idx).double()
    corr_Q = nn_gather(Q, corr_Q_idx).double()

    Rts = []
    for i in range(corr_P.shape[0]):
        P_i = corr_P[i]
        Q_i = corr_Q[i]

        Rt_i = o3d_3d_correspondence_registration(P_i, Q_i)
        Rt_i = torch.tensor(Rt_i.transformation).to(P_i)
        Rts.append(Rt_i)

    Rts = torch.stack(Rts, dim=0).float()
    return Rts


def align_cpa(corres, P, Q):
    """Estimate registration between 2 pointclouds based on a set of correspondences

    Args:
        corres (tuple): correspondence ids and weights
        P (FloatTensor): pointcloud P (batch, N, 3)
        Q (FloatTensor): pointcloud Q (batch, N, 3)

    Returns:
        FloatTensor: (batch, 3, 4) estimated registration
    """
    # get useful variables
    corr_P_idx, corr_Q_idx, weights = corres[:3]

    # get match features and coord
    corr_P = nn_gather(P, corr_P_idx).double()
    corr_Q = nn_gather(Q, corr_Q_idx).double()
    weights = weights.double()

    Rt_PtoQ = kabsch_algorithm(corr_P, corr_Q, weights)
    Rt_PtoQ = make_Rt(Rt_PtoQ.R, Rt_PtoQ.T).float()

    return Rt_PtoQ


def align_cpa_ransac(
    corr_P,
    corr_Q,
    weights,
    schedule=[(3, 128)],
    threshold=0.1,
    return_new_weights=False,
):
    """Estimate pairwise alignment from a list of correspondences

    Args:
        corr_P (FloatTensor): Correspondnces P
        corr_Q (_type_): _description_
        weights (_type_): _description_
        schedule (list, optional): _description_. Defaults to [(3, 128)].
        threshold (float, optional): _description_. Defaults to 0.1.
        return_new_weights (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    # get useful variables
    assert 1 <= len(schedule) <= 2

    corr_P = corr_P.double()
    corr_Q = corr_Q.double()
    weights = weights.double()

    with torch.no_grad():
        bs = corr_P.size(0)
        n_hot, n_samples = schedule[0]
        idxs = torch.multinomial(
            weights[:, None].expand(-1, n_samples, -1).flatten(0, 1), n_hot
        ).unflatten(0, (bs, n_samples))
        P, Q, W = list_knn_gather([corr_P, corr_Q, weights[:, :, None]], idxs)
        T = kabsch_algorithm(P.flatten(0, 1), Q.flatten(0, 1), W.view(-1, n_hot))
        R, t = T.R.unflatten(0, (bs, n_samples)), T.T.view(bs, n_samples, 1, 3)
        delta = (corr_P[:, None] @ R + t - corr_Q[:, None]).norm(2, dim=-1)
        inliers = torch.exp(-delta / threshold)

        if len(schedule) == 2:  # grow set of inliers?
            n_hot, n_samples = schedule[1]
            iq = inliers.sum(2)

            # pick inlierdistances corresponding to the best Rt (soft)
            idxs = torch.multinomial(iq, n_samples, replacement=True)
            inliers = inliers[torch.arange(bs)[:, None].expand(-1, n_samples), idxs]

            # resample inliers according to fit
            idxs = torch.multinomial(inliers.flatten(0, 1), n_hot).unflatten(
                0, (bs, n_samples)
            )
            P, Q, W = list_knn_gather([corr_P, corr_Q, weights[:, :, None]], idxs)
            T = kabsch_algorithm(P.flatten(0, 1), Q.flatten(0, 1), W.view(-1, n_hot))
            R, t = T.R.unflatten(0, (bs, n_samples)), T.T.view(bs, n_samples, 1, 3)
            delta = (corr_P[:, None] @ R + t - corr_Q[:, None]).norm(2, dim=-1)
            inliers = torch.exp(-delta / threshold)

        n_inliers = inliers.sum(2)
        best = n_inliers.argmax(dim=1)
        inliers = inliers[torch.arange(bs), best]

    inliers = normalize(inliers, dim=-1).clamp(min=1e-7) * inliers.shape[-1]
    new_weights = weights * inliers
    Rt_PtoQ = kabsch_algorithm(corr_P, corr_Q, new_weights)
    Rt_PtoQ = make_Rt(Rt_PtoQ.R, Rt_PtoQ.T)

    if return_new_weights:
        return Rt_PtoQ.float(), new_weights.float()
    else:
        return Rt_PtoQ.float()
