# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import copy
import glob
import os
import shutil

import imageio
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
from PIL import Image

vis_dim = 2048
# vis_dim = 1024

try:
    import open3d as o3d
except:
    print("Unable to use Open3D as it's not installed")
    OPEN3D_AVAILABLE = False


def get_initial(pcd_0, pcd_1, offset=None, return_offset=False):
    """
    Given two point clouds of the scene, align them next to each other with spacing
    equivalent to their mean width. Additionally, flip them so that they are viewable by
    the Open3D visualizer's view port.

    Input and output: 2 Open3D point clouds.
    """
    pcd_0 = copy.deepcopy(pcd_0)
    pcd_1 = copy.deepcopy(pcd_1)

    if offset is None:
        bounds_0 = pcd_0.get_max_bound()
        bounds_1 = pcd_1.get_max_bound()

        bound = max(bounds_0[:2].max(), bounds_1[:2].max())
        x_offset = bound * 1.25
    else:
        x_offset = offset

    pcd_0.translate([-1 * x_offset, 0, 0])
    pcd_1.translate([x_offset, 0, 0])

    # flip for better
    flip_transform = [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
    pcd_0.transform(flip_transform)
    pcd_1.transform(flip_transform)

    if return_offset:
        return pcd_0, pcd_1, x_offset
    else:
        return pcd_0, pcd_1


def get_final(pcd_0, pcd_1, vp_0to1):
    """
    Given two point clouds and the pose the aligns the first to the second, transform
    the first point cloud so that it's aligned with the second point cloud in the second
    point clouds frame of reference.
    Additionally, flip them so that they are viewable by
    the Open3D visualizer's view port.

    Input: 2 Open3D point clouds and the relative pose between them.
    Output: Aligned point clouds
    """
    pcd_0 = copy.deepcopy(pcd_0)
    pcd_1 = copy.deepcopy(pcd_1)

    pcd_0.transform(vp_0to1)

    # flip for better
    flip_transform = [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
    pcd_0.transform(flip_transform)
    pcd_1.transform(flip_transform)

    return pcd_0, pcd_1


def separate_pointclouds(pcd_0, pcd_1, path, recolor=False):
    pcd_0, pcd_1 = get_initial(pcd_0, pcd_1)

    if recolor:
        pcd_0_color = np.asarray(pcd_0.points)
        pcd_1_color = np.asarray(pcd_1.points)

        pcd_0_color[:] = np.array([1, 0.706, 0])
        pcd_1_color[:] = np.array([0, 0.651, 0.929])

        pcd_0.colors = o3d.utility.Vector3dVector(pcd_0_color)
        pcd_1.colors = o3d.utility.Vector3dVector(pcd_1_color)

    vis = o3d.visualization.Visualizer()
    vis.create_window(width=vis_dim, height=vis_dim)

    vis.add_geometry(pcd_0)
    vis.add_geometry(pcd_1)

    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(path)
    vis.destroy_window()


def aligned_pointclouds(pcd_0, pcd_1, vp_0to1, path, recolor=False):
    pcd_0, pcd_1 = get_final(pcd_0, pcd_1, vp_0to1)

    if recolor:
        pcd_0.paint_uniform_color([1, 0.706, 0])
        pcd_1.paint_uniform_color([0, 0.651, 0.929])
        pcd_0.estimate_normals()
        pcd_1.estimate_normals()

    vis = o3d.visualization.Visualizer()
    vis.create_window(width=vis_dim, height=vis_dim)

    vis.add_geometry(pcd_0)
    vis.add_geometry(pcd_1)

    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(path)
    vis.destroy_window()


def mesh_sphere(pcd, sphere_radius):
    # Create a mesh sphere
    spheres = o3d.geometry.TriangleMesh()
    s = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius)
    s.compute_vertex_normals()

    for i, p in enumerate(pcd.points):
        si = copy.deepcopy(s)
        trans = np.identity(4)
        trans[:3, 3] = p
        si.transform(trans)
        si.paint_uniform_color(pcd.colors[i])
        spheres += si
    return spheres


def correspondance_animation(pcd_0, pcd_1, corres, path, num_views=60, colors=None):
    pcd_0, pcd_1 = copy.deepcopy(pcd_0), copy.deepcopy(pcd_1)
    pcd_0, pcd_1, offset = get_initial(pcd_0, pcd_1, return_offset=True)
    cor_0, cor_1 = get_initial(corres[0], corres[1], offset=offset)

    points_0 = np.asarray(cor_0.points)
    points_1 = np.asarray(cor_1.points)

    lines = [[i, i + len(points_0)] for i in range(len(points_0))]
    points = np.concatenate((points_0, points_1), axis=0)

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(list(points))
    line_set.lines = o3d.utility.Vector2iVector(lines)

    if colors is None:
        colors = [(0, 0, 0) for line in lines]
    line_set.colors = o3d.utility.Vector3dVector(colors)

    vis = o3d.visualization.Visualizer()
    vis.create_window(width=vis_dim, height=vis_dim)

    vis.add_geometry(pcd_0)
    vis.add_geometry(pcd_1)
    vis.add_geometry(line_set)

    render_option = vis.get_render_option()
    render_option.line_width = 3

    gif_name = os.path.basename(path).split(".")[0]
    temp_dir = f"temp_for_{gif_name}"
    os.makedirs(temp_dir)

    for i in range(num_views + 1):
        j = num_views - i
        points_1_cur = (i * points_1 + j * points_0) / num_views
        points_cur = np.concatenate((points_0, points_1_cur), axis=0)
        line_set.points = o3d.utility.Vector3dVector(points_cur)

        vis.update_geometry(line_set)
        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(f"{temp_dir}/{i:04d}.jpg")

    vis.destroy_window()

    images_to_video(temp_dir, path)
    shutil.rmtree(temp_dir)


def registration_animation(pcd_0, pcd_1, vp_0to1, path, num_views=60):
    pcd_0, pcd_1 = copy.deepcopy(pcd_0), copy.deepcopy(pcd_1)
    pcd_0_init, pcd_1_init = get_initial(pcd_0, pcd_1)
    pcd_0_final, pcd_1_final = get_final(pcd_0, pcd_1, vp_0to1)

    pcd_0_init_x = np.asarray(pcd_0_init.points)
    pcd_1_init_x = np.asarray(pcd_1_init.points)
    pcd_0_final_x = np.asarray(pcd_0_final.points)
    pcd_1_final_x = np.asarray(pcd_1_final.points)

    vis = o3d.visualization.Visualizer()
    vis.create_window(width=vis_dim, height=vis_dim)

    pcd_0.points = pcd_0_init.points
    pcd_1.points = pcd_1_init.points

    vis.add_geometry(pcd_0)
    vis.add_geometry(pcd_1)

    gif_name = os.path.basename(path).split(".")[0]
    temp_dir = f"temp_for_{gif_name}"
    os.makedirs(temp_dir)

    for i in range(num_views + 1):
        j = num_views - i
        cur_pc_0 = (j * pcd_0_init_x + i * pcd_0_final_x) / num_views
        cur_pc_1 = (j * pcd_1_init_x + i * pcd_1_final_x) / num_views

        pcd_0.points = o3d.utility.Vector3dVector(cur_pc_0)
        pcd_1.points = o3d.utility.Vector3dVector(cur_pc_1)

        vis.update_geometry(pcd_0)
        vis.update_geometry(pcd_1)

        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(f"{temp_dir}/{i:04d}.jpg")

    vis.destroy_window()

    images_to_video(temp_dir, path)
    shutil.rmtree(temp_dir)


def images_to_gif(image_dir, gif_path):
    """
    Given a directory of ordered jpeg images, generate a gif.
    """
    # filepaths
    fp_in = f"{image_dir}/*.jpg"

    # https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#gif
    img, *imgs = (Image.open(f) for f in sorted(glob.glob(fp_in)))
    img.save(
        fp=gif_path,
        format="GIF",
        append_images=imgs,
        save_all=True,
        duration=200,
        loop=0,
    )


def images_to_video(image_dir, video_path):
    """
    Given a directory of ordered jpeg images, generate a video (mp4)
    """

    writer = imageio.get_writer(video_path, fps=20)
    im_list = os.listdir(image_dir)
    im_list.sort()

    l, t = vis_dim // 8, vis_dim // 4
    w, h = 3 * vis_dim // 4, vis_dim // 2

    for im in im_list:
        im_path = os.path.join(image_dir, im)

        img = imageio.imread(im_path)
        img = img[t : t + h, l : l + w]
        writer.append_data(img)
    writer.close()


def correspondance_2d(img_0, img_1, pts_0, pts_1, path, colors=None):
    """
    Generates a matplotlib figure for correspondances between two images.
    Input:
        img_0       RGB Image of size (H, W, 3)
        img_1       RGB Image of size (H, W, 3)
        id_0        Match kps in img_0 (K, 2)
        id_1        Match kps in img_1 (K, 2)
        path        path to save image
        colors      (optional) Weight associated with each matched pair
    """
    img_h, img_w, _ = img_0.shape

    # create canvas and fill it with RGB
    gap = int(img_w * 0.2)
    spacing = img_w + gap
    canvas = np.zeros((img_h, img_w + spacing, 3))
    canvas[:, :img_w, :] = img_0
    canvas[:, spacing:, :] = img_1

    # Add spacing to x value for second image
    pts_1[:, 0] += spacing

    # calculate number of matches in each pair (half total)
    num_matches = pts_0.shape[0]
    fig = Figure()

    # First pair
    ax = fig.add_subplot(111)
    ax.imshow(canvas)
    ax.axis("off")

    # create collection
    num_matches = pts_0.shape[0]
    seg = [(pts_0[i], pts_1[i]) for i in range(num_matches)]
    lc = LineCollection(seg, linewidths=(0.5,), colors=colors)
    ax.add_collection(lc)
    fig.savefig(path, bbox_inches="tight", pad_inches=0, dpi=300)

    return None


def single_pointcloud(pcd, path):
    pcd = copy.deepcopy(pcd)

    # flip for better
    flip_transform = [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
    pcd.transform(flip_transform)

    vis = o3d.visualization.Visualizer()
    vis.create_window(width=vis_dim, height=vis_dim)

    vis.add_geometry(pcd)

    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(path)
    vis.destroy_window()


def plot_pointcloud(path, xyz, rgb=None, fill_out="none", vox_dim=0.025):
    """
    input:
        path: path to save pointcloud
        xyz:  FloatTensor (Nx3)
        rgb:  FloatTensor (Nx3)
        fill_out: option to mesh ("none", "sphere", "normals")
    """
    assert fill_out in ["none", "sphere", "normals"], "Unsupoorted option"

    # create pointcloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz.detach().cpu().numpy())
    if rgb is not None:
        pcd.colors = o3d.utility.Vector3dVector(rgb.detach().cpu().numpy())

    # fillout
    if fill_out == "sphere":
        pcd = mesh_sphere(pcd, vox_dim)
    elif fill_out == "normals":
        pcd.estimate_normals()

    single_pointcloud(pcd, path)
