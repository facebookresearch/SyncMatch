# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import functools

import easydict
import torch
import yaml


class Residual(torch.nn.Module):
    def __init__(self, m, r, dropout=0, dropout_mode="**--"):
        super().__init__()
        self.m = m
        self.r = r
        self.dropout = dropout
        self.dropout_mode = dropout_mode

    def forward(self, x):
        r = self.r(x)
        m = self.m(x)
        if self.training and self.dropout > 0:
            noise_shape = [
                s if c == "*" else 1 for s, c in zip(m.shape, self.dropout_mode)
            ]
            return (
                r
                + m
                * torch.rand(*noise_shape, device=x.device)
                .ge_(self.dropout)
                .div(1 - self.dropout)
                .detach()
            )

        else:
            return r + m


class Conv_BN(torch.nn.Sequential):
    def __init__(
        self,
        a,
        b,
        ks=1,
        stride=1,
        pad=0,
        dilation=1,
        groups=1,
        bn_weight_init=1,
        transpose=False,
        activation=None,
        dimension=2,
    ):
        super().__init__()
        if transpose:
            self.add_module(
                "c",
                getattr(torch.nn, f"ConvTranspose{dimension}d")(
                    a, b, ks, stride, pad, dilation=dilation, groups=groups, bias=False
                ),
            )
        else:
            self.add_module(
                "c",
                getattr(torch.nn, f"Conv{dimension}d")(
                    a, b, ks, stride, pad, dilation, groups, bias=False
                ),
            )
        if bn_weight_init == "na":
            bn = getattr(torch.nn, f"BatchNorm{dimension}d")(b, affine=False)
        else:
            bn = getattr(torch.nn, f"BatchNorm{dimension}d")(b)
            torch.nn.init.constant_(bn.weight, bn_weight_init)
            torch.nn.init.constant_(bn.bias, 0)
        self.add_module("bn", bn)
        if activation is not None:
            self.add_module("activation", activation())


Conv2d_BN = functools.partial(Conv_BN, dimension=2)


class UpSample(torch.nn.Module):
    def __init__(self, H, W):
        super().__init__()
        self.H = H
        self.W = W
        self.register_buffer("ones", torch.ones(1, 1, 2 + H % 2, 2 + W % 2))
        self.register_buffer(
            "div",
            torch.nn.functional.conv_transpose2d(
                torch.ones(1, 1, H // 2, W // 2), self.ones, stride=2
            ),
        )

    def forward(self, x):
        return (
            torch.nn.functional.conv_transpose2d(
                x.view(-1, 1, self.H // 2, self.W // 2), self.ones, stride=2
            ).view(x.size(0), x.size(1), self.H, self.W)
            / self.div
        )


def UNet(
    reps_up, reps_down, nPlanes, dropout=0, dropout_mode="**--", H=60, W=80, **kwargs
):
    def subblock(a, b):
        m = torch.nn.Sequential(
            Conv2d_BN(a, b, 3, 1, 1, activation=torch.nn.GELU), Conv2d_BN(b, b, 3, 1, 1)
        )
        if a == b:
            r = torch.nn.Identity()
        else:
            r = Conv2d_BN(a, b)
        m = torch.nn.Sequential(
            Residual(m, r, dropout=dropout, dropout_mode=dropout_mode), torch.nn.ReLU()
        )
        return m

    def block(a, b, r):
        m = []
        for _ in range(r):
            m.append(subblock(a, b))
            a = b
        return torch.nn.Sequential(*m)

    if len(nPlanes) == 1:
        a = nPlanes[0]
        return block(a, a, reps_up + reps_down)
    a, b = nPlanes[:2]

    downsample = Conv2d_BN(
        a,
        b,
        (4 - H % 2, 4 - W % 2),
        2,
        (1 - H % 2, 1 - W % 2),
        activation=torch.nn.ReLU,
    )
    upsample = UpSample(H, W)

    return UBuilderCat(
        block(a, a, reps_up),
        downsample,
        UNet(reps_up, reps_down, nPlanes[1:], dropout, H=H // 2, W=W // 2),
        upsample,
        block(a + b, a, reps_down),
    )


class UBuilderCat(torch.nn.Module):
    def __init__(self, left, up, top, down, right):
        super().__init__()
        self.l = left  # noqa: E741
        self.u = up
        self.t = top
        self.d = down
        self.r = right

    def forward(self, x):
        x = self.l(x)
        y = self.u(x)
        y = self.t(y)
        y = self.d(y)
        y = torch.cat([y, x], 1)
        return self.r(y)


class DepthPredictionNetwork(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.fov_split = cfg.fov_split
        self.n_fov_test = cfg.n_fov_test
        self.cfg = cfg
        net = []
        assert cfg.downsampling_factor in [4]
        activation = getattr(torch.nn, cfg.activation)
        if cfg.downsampling_factor == 4:
            net.append(
                torch.nn.Sequential(
                    Conv2d_BN(3, cfg.m // 2, 4, 2, 1, activation=activation),
                    Conv2d_BN(cfg.m // 2, cfg.m, 4, 2, 1, activation=activation),
                )
            )
        H = (cfg.crop[1] - cfg.crop[0]) // cfg.downsampling_factor
        W = (cfg.crop[3] - cfg.crop[2]) // cfg.downsampling_factor
        if cfg.wg == "linear":
            WG = [(i + 1) * cfg.m for i in range(cfg.layers)]
        elif cfg.wg == "constant":
            WG = [cfg.m for i in range(cfg.layers)]
        elif cfg.wg == "half":
            WG = [(i + 2) * cfg.m // 2 for i in range(cfg.layers)]
        net.append(
            globals()[cfg.name](
                cfg.reps_up,
                cfg.reps_down,
                WG,
                dropout=cfg.dropout,
                dropout_mode=cfg.dropout_mode,
                H=H,
                W=W,
            )
        )
        # net.append(Conv2d_BN(cfg.m, 1 + cfg.n_fov))
        net.append(torch.nn.Conv2d(cfg.m, 1 + cfg.n_fov, 1, 1, 0))
        with torch.no_grad():
            net[-1].weight[:, 0].mul_(0.01)

        x = cfg.downsampling_factor
        if "flubber" in cfg:
            x *= cfg.flubber
        if cfg.upsample == "n":
            net.append(torch.nn.Upsample(scale_factor=x, mode="nearest"))
        elif cfg.upsample == "b":
            net.append(
                torch.nn.Upsample(scale_factor=x, mode="bilinear", align_corners=True)
            )
        else:
            net.append(torch.nn.Identity())
        self.net = torch.nn.Sequential(*net)
        self.a = cfg.min_depth
        self.b = 1 / (cfg.max_depth - cfg.min_depth)
        if "flubber" in cfg:
            self.flubber = cfg.flubber
        else:
            self.flubber = 0

    def forward(self, x, full_size_output=True):
        if self.flubber:
            x = torch.nn.functional.avg_pool2d(x, self.flubber, self.flubber)
        x = self.net[:-1](x)
        mu, sigma = x[:, :1], x[:, 1:]
        rg = (sigma.var(1) - 1).relu().mean()
        mu = self.a + 1 / (self.b + torch.nn.functional.softplus(mu))
        if self.cfg.scale_fov_with_depth:
            sigma = sigma * mu
        r = self.fov_split
        if (not self.training) and 0 < self.n_fov_test < sigma.size(1):
            sigma = compress_fov(sigma, self.n_fov_test)

        if r != [1, 1, 1]:
            with torch.no_grad():
                m, M = mu.min(), mu.max()
                _, _, H, W = mu.shape
                q = (mu - m) * ((r[2] - 1) / (M - m))
                q = (
                    torch.linspace(0, r[0] - 1, H)[None, :].to(sigma.device)
                    - torch.linspace(-1, r[0] - 2, r[0])[:, None].to(sigma.device)[
                        None
                    ],
                    torch.linspace(0, r[1] - 1, W)[None, :].to(sigma.device)
                    - torch.linspace(-1, r[1] - 2, r[1])[:, None].to(sigma.device)[
                        None
                    ],
                    q
                    - torch.linspace(-1, r[2] - 2, r[2])[None, :, None, None].to(
                        sigma.device
                    ),
                )
                q = [torch.min(qq, 2 - qq).clamp(min=0) for qq in q]
                q = [
                    q[0][:, :, None, None, :, None],
                    q[1][:, None, :, None, None, :],
                    q[2][:, None, None, :, :, :],
                ]
                q = (q[0] * q[1] * q[2]).view(-1, r[0] * r[1] * r[2], H, W)
            sigma = (sigma[:, None, :] * q[:, :, None]).view(sigma.size(0), -1, H, W)
        if full_size_output:
            mu = self.net[-1](mu)
            sigma = self.net[-1](sigma)

        return mu, sigma, rg


def cfg_from_string(x):
    return easydict.EasyDict(yaml.safe_load(x))


def compress_fov(fov, n_feats=6):
    s = torch.svd(fov.flatten(2).transpose(1, 2))
    U = (
        s.U[:, :, :n_feats]
        .transpose(1, 2)
        .reshape(fov.size(0), n_feats, *fov.shape[2:])
    )
    S = s.S[:, :n_feats, None, None]
    return U * S
