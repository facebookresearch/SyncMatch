# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import faiss
import faiss.contrib.torch_utils
import torch
from torch.nn import functional as nn_F

res = faiss.StandardGpuResources()  # use a single GPU


def faiss_knn_gpu(query, target, k):
    num_elements, feat_dim = query.shape
    index_flat = faiss.IndexFlatL2(feat_dim)
    gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, index_flat)

    gpu_index_flat.add(target)
    return gpu_index_flat.search(query, k)


def faiss_knn(feature_A, feature_B, k, num_points_A=None, num_points_B=None):
    b, nA, ch = feature_A.shape
    dist = []
    indx = []
    for i in range(b):
        torch.cuda.synchronize()
        if num_points_A is not None:
            assert num_points_B is not None, "let's keep it consistent, probably a bug"
            feat_A_i = feature_A[i, 0 : num_points_A[i]]
            feat_B_i = feature_B[i, 0 : num_points_B[i]]
        else:
            feat_A_i = feature_A[i]
            feat_B_i = feature_B[i]

        # knn logic part
        dist_i, indx_i = faiss_knn_gpu(feat_A_i, feat_B_i, k)

        dist.append(dist_i)
        indx.append(indx_i)

    # pad if heterogeneous batching
    if num_points_A is not None:
        max_pts = num_points_A.max()
        dist = [
            nn_F.pad(dist[i], (0, 0, 0, max_pts - num_points_A[i]), value=-1)
            for i in range(b)
        ]
        indx = [
            nn_F.pad(indx[i], (0, 0, 0, max_pts - num_points_A[i]), value=0)
            for i in range(b)
        ]

    dist = torch.stack(dist, dim=0)
    indx = torch.stack(indx, dim=0)

    return dist, indx
