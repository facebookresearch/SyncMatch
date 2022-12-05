# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import torch


class DenseBlockRidgeRegression:
    def __init__(self, n_var, lambd):
        self.n_var = n_var
        self.x = []
        self.y = []
        self.lambd = lambd

    def add_row(self, ix, y):
        self.n_dim = ix[0][1].size(-1)
        n = self.n_var
        if y.ndim < ix[0][1].ndim:
            y = y[..., None]
        X = [None for _ in range(n)]
        for i, x in ix:
            X[i] = x
        z = torch.zeros_like(x)
        for i in range(n):
            if X[i] is None:
                X[i] = z
        X = torch.cat(X, -1)
        self.x.append(X)
        self.y.append(y)

    def predict(self):
        x = torch.cat(self.x, -2)
        y = torch.cat(self.y, -2)
        beta = (
            torch.pinverse(
                x.transpose(-2, -1) @ x
                + self.lambd * torch.eye(x.size(-1), device=x.device)
            )
            @ x.transpose(-2, -1)
            @ y
        )
        return beta.view(*beta.shape[:-2], self.n_var, self.n_dim, -1)


class SparseBlockRidgeRegression:
    """
    Batched Ridge Regression
    Let '.T' denote '.transpose(-2,-1)
    Solve ridge regression using torch.pinverse(X.T@X+lambd*torch.eye(.)))@X.T@Y
    # X is a matrix with size [batch, N, n_var*n_dim]. Y has size [batch, N, 1]
    X is created as a series of rows (with the little n adding up to big N).
    Each row is made up of a series of blocks, some of which will be zero, i.e.
    # [0, ..., 0, torch.Tensor([batch, n, n_dim]), 0, ..., 0, torch.Tensor([batch, n, n_dim]), 0, ..., .0]
    # This is encoded as a list [(index, block), (index2, block2)]
    For each row of X, the corresponding slice of Y has size [batch, n, 1]

    We also allow Y to have shape [batch, N, k]; interpreted as k subtensors of shape [batch, N, 1]

    [Code assumes all columns have at least one entry.]
    """

    def __init__(self, n_var, lambd):
        self.n_var = n_var
        self.lambd = lambd

    def add_row(self, ix, y):
        if not hasattr(self, "xtx"):
            # First row being added
            t = ix[0][1]
            device = t.device
            n_dim = t.size(-1)
            n_var = self.n_var
            eye = self.lambd * torch.eye(n_dim, device=device)
            zero = torch.zeros(*t.shape[:-2], n_dim, n_dim, device=device)
            # store entries i,j at i*n_var+j
            self.xtx = [zero if i % (n_var + 1) else eye for i in range(n_var * n_var)]
            self.xty = [0 for _ in range(n_var)]
        if y.ndim < ix[0][1].ndim:
            y = y[..., None]
        for i, xi in ix:
            self.xty[i] = self.xty[i] + xi.transpose(-2, -1) @ y
            for j, xj in ix:
                # xtx[i,j]=xtx[j,i].transpose(-2,-1) so avoid redundant compute
                if i <= j:
                    k = i * self.n_var + j
                    self.xtx[k] = self.xtx[k] + xi.transpose(-2, -1) @ xj

    def predict(self):
        n = self.n_var
        m = self.xtx[0].size(-1)
        xty = torch.cat(self.xty, -2)
        xtx = self.xtx
        for i in range(1, n):
            # i>j, use symmetry
            for j in range(i):
                k1 = j * n + i
                k2 = i * n + j
                xtx[k2] = xtx[k1].transpose(-2, -1)
        xtx = torch.stack(xtx, -3)
        xtx = xtx.view(*xtx.shape[:-3], n, n, m, m)
        xtx = xtx.transpose(-3, -2)
        xtx = xtx.reshape(*xtx.shape[:-4], n * m, n * m)

        beta = torch.pinverse(xtx) @ xty
        beta = beta.view(*beta.shape[:-2], n, m, -1)
        return beta


def finetune_depth_with_factors_of_variation(
    xy1, P, depth, fov, correspondences, lambda_depth, dense=False
):
    """
    Calculate coefficients to optimize depth using ridge regression

    xy1:                [batch, views, HW, 3]
    P                   [batch, views, 3 or 4, 4]    world-to-cam
    depth predictions:  [batch, views, HW, 1]
    Depth FOV:          [batch, views, HW, N]
    correspondences:    dictionary {(i,j): [idx1,, idx2, weights]} #triplet of [batch, n_matches] tensors
    lambda_depth:       positive number
    dense:              Use DenseRidgeRegression or SparseRidgeRegression (?)

    P=[R|T]  or [ R |T]
                [000|1]
    R_i X_world + T_i = X_i     if {X_world, X_i} have shape 3xN
    X_world R_i^T + T_i = X_i   if {X_world, X_i} have shape Nx3
    (X_i - T_i) R_i = X_world   if {X_world, X_i} have shape Nx3
    """
    batch, views, HW, N = fov.shape
    R = P[:, :, :3, :3]  # [batch, views, 3, 3]
    T = P[:, :, :3, 3]  # [batch, views, 3]
    if depth.dim() == 3:
        depth = depth[:, :, :, None]  # [batch, views, n_matches, 1]

    xy1R = xy1 @ R  # rotate screen coorinates into world coordinates

    if dense:
        rr = DenseBlockRidgeRegression(views, lambda_depth)
    else:
        rr = SparseBlockRidgeRegression(views, lambda_depth)

    for p in correspondences:
        c = correspondences[p]  # [BM, BM, BM]  idx0, idx1, weightp

        IX = []
        Y = 0
        for i, k in enumerate(p):
            # gather rows based on the correspondences
            depth_cc, fov_cc, xy1R_cc = list_knn_gather_(
                [depth[:, k], fov[:, k], xy1R[:, k]], c[i]
            )  # BM, BMN, BM3
            """equivalent to:
            import pytorch3d.ops
            cc=c[:,:,i:i+1]
            depth_cc=pytorch3d.ops.knn_gather(depth[:,k],cc).squeeze(2)
            fov_cc=pytorch3d.ops.knn_gather(fov[:,k],cc).squeeze(2)
            xy1R_cc=pytorch3d.ops.knn_gather(xy1R[:,k],cc).squeeze(2)
            """
            x_cc = (
                fov_cc[:, :, None] * xy1R_cc[:, :, :, None] * c[2][:, :, None, None]
            )  # BM1N*BM31*BM11->BM3N
            x_cc = x_cc.view(batch, -1, N)  # B(M3)N
            IX.append([k, x_cc if i == 0 else -x_cc])

            y = (xy1R_cc * depth_cc) - T[:, k, None] @ R[:, k]
            y = y * c[2][:, :, None]  # BM3
            y = y.view(batch, -1)  # B(M3)
            Y = (Y - y) if i == 0 else (Y + y)
        rr.add_row(IX, Y)

    return rr.predict()


def list_knn_gather_(xs, idxs):
    # x[0] shape NM or NMU
    # idxs NK
    N, K = idxs.shape
    M = xs[0].size(1)
    idxs = idxs.add(torch.arange(N, device=xs[0].device)[:, None] * M).flatten(0, 1)
    return [x.flatten(0, 1)[idxs].view(N, K, *x.shape[2:]) for x in xs]
