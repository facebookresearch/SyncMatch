# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import copy

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D


def plot_point_hist(points, index):
    pts = points[index, :, :].detach().cpu().numpy()
    plt.subplot(1, 3, 1)
    plt.hist(pts[:, 0])
    plt.subplot(1, 3, 2)
    plt.hist(pts[:, 1])
    plt.subplot(1, 3, 3)
    plt.hist(pts[:, 2])
    plt.show()


# Some easy savers
def save_rgb(path, array):
    fig = Figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.axis("off")
    ax.imshow(array)
    fig.savefig(path, pad_inches=0.0, bbox_inches="tight")


def save_mask(path, array, colorbar=False, cmap="binary", vlim=[0.0, 1.0]):
    fig = Figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.axis("off")
    if len(array.shape) == 3:
        assert array.shape[0] == 1, "not a single-channel image"
        array = array[0]
    im_fig = ax.imshow(array, cmap=cmap, vmin=vlim[0], vmax=vlim[1])
    if colorbar:
        fig.colorbar(im_fig, ax=ax)
    fig.savefig(path, pad_inches=0.0, bbox_inches="tight")


def get_pointcloud_hist(pts):
    fig = Figure()
    assert len(pts.shape) == 2 and pts.shape[1] == 3
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]

    ax = fig.add_subplot(1, 3, 1)
    ax.hist(x)
    ax = fig.add_subplot(1, 3, 2)
    ax.hist(y)
    ax = fig.add_subplot(1, 3, 3)
    ax.hist(z)
    return fig


def get_pointcloud_fig(pts, lim=1.0):
    fig = Figure()
    ax = Axes3D(fig)
    assert len(pts.shape) == 2
    if pts.shape[1] == 3:
        x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
        s = np.ones_like(x)
    elif pts.shape[1] == 4:
        x, y, z, s = pts[:, 0], pts[:, 1], pts[:, 2], pts[:, 3]
    elif pts.shape[0] == 3:
        x, y, z = pts[0], pts[1], pts[2]
        s = np.ones_like(x)
    elif pts.shape[0] == 4:
        x, y, z, s = pts[0], pts[1], pts[2], pts[3]
    else:
        raise ValueError("Unknown point cloud dimensionality")

    # move from LH to RH
    ax.scatter3D(x, z, -y, s=s)

    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.set_zlabel("y")

    if lim is not None:
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_zlim(-lim, lim)

    ax.view_init(190, 40)
    return fig


def save_pointcloud(path, pts, lim=1.0):
    fig = get_pointcloud_fig(pts, lim)
    fig.savefig(path, pad_inches=0.0, bbox_inches="tight")


def save_pointcloud_hist(path, pts):
    fig = get_pointcloud_hist(pts)
    fig.savefig(path, pad_inches=0.0, bbox_inches="tight")


def save_vox(path, vox):
    fig = Figure()
    ax = Axes3D(fig)
    ax.voxels(vox)
    ax.view_init(190, 30)
    fig.savefig(path, pad_inches=0.0, bbox_inches="tight")


def plot_correspondances(
    img_0,
    img_1,
    pts_0,
    pts_1,
    error=None,
    weight=None,
    show_both=False,
    ax=None,
    fig=None,
):
    """
    Generates a matplotlib figure for correspondances between two images.

    Input:
        img_0       RGB Image of size (H, W, 3)
        img_1       RGB Image of size (H, W, 3)
        id_0        Match ideas in img_0 (K, 1)
        id_1        Match ideas in img_1 (K, 1)
        weights     (optional) Weight associated with each matched pair
    """
    img_h, img_w, _ = img_0.shape
    pts_0 = copy.deepcopy(pts_0)
    pts_1 = copy.deepcopy(pts_1)

    # create canvas and fill it with RGB
    spacing = img_w + 30
    canvas = np.ones((img_h, img_w + spacing, 3))
    canvas[:, :img_w, :] = img_0
    canvas[:, spacing:, :] = img_1

    # Add spacing to x value for second image
    pts_1[:, 0] += spacing

    # generate figure
    if ax is None:
        fig = Figure()
        ax = fig.add_subplot(111)

    # handle no points
    if len(pts_0) == 0:
        ax.imshow(canvas)
        return fig

    # handle colors
    if error is not None:
        assert weight is None
        n_l = 3
        s_l = 5.0
        error_new = np.floor(error * 100 / s_l).clip(max=n_l) / n_l
        # error_new = np.zeros_like(error)
        # error_new[error <= 0.01] = 0.00
        # error_new[error <= 0.05] = 0.33
        # error_new[error <= 0.10] = 0.66
        # error_new[error > 0.10] = 1.00

        color = plt.cm.RdYlGn_r(error_new)
    elif weight is not None:
        color = plt.cm.RdYlGn(weight)
    else:
        color = None

    # make custon colormap
    # from matplotlib.colors import ListedColormap
    # cmap = plt.cm.Reds
    # my_cmap = cmap(np.arange(cmap.N))
    # my_cmap[:, -1] = np.linspace(0, 1, cmap.N)
    # my_cmap = ListedColormap(my_cmap)

    def plot_corr(_ax, _seg, _color):
        _ax.imshow(canvas)

        # create collection
        if color is not None:
            lc = LineCollection(_seg, linewidths=(0.4,), colors=_color)
        else:
            lc = LineCollection(_seg, linewidths=(0.2,), color="red")

        _ax.add_collection(lc)
        # if error is not None:
        #     fig.colorbar(lc, ax=_ax)

    num_matches = pts_0.shape[0]
    seg = [(pts_0[i], pts_1[i]) for i in range(num_matches)]

    if show_both and ax is None:
        half_matches = num_matches // 2
        ax = fig.add_subplot(121)
        plot_corr(ax, seg[:half_matches], color[:half_matches])

        ax = fig.add_subplot(121)
        plot_corr(ax, seg[half_matches:], color[half_matches:])
    else:
        plot_corr(ax, seg, color)

    return fig
