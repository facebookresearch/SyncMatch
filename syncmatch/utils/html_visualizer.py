# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import os

import matplotlib
import numpy as np
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from .io import makedir
from .point_vis import plot_pointcloud, registration_animation
from .pyhtml import Element, Table, TableRow, TableWriter, imgElement, vidElement
from .visual import plot_correspondances

try:
    import open3d as o3d
except:
    print("Unable to use Open3D as it's not installed")
    OPEN3D_AVAILABLE = False


class HTML_Visualizer:
    def __init__(self, html_root, exp_name, columns):
        # define main directories; now just main one and images
        self.html_dir = os.path.join(html_root, exp_name)
        self.img_dir = os.path.join(self.html_dir, "imgs")
        self.exp_name = exp_name

        makedir(self.html_dir, replace_existing=True)
        makedir(self.img_dir)
        print("=" * 80)
        print(f" ++ HTML Visualizer saving to: {self.html_dir} ++ ")
        print("=" * 80)

        # create primary HTML table
        self.columns = columns

        # create html table
        header_row = TableRow(isHeader=True)
        header_row.addElement(Element("Instance ID"))
        header_row.addElement(Element("Training Step"))
        for header in self.columns:
            header_row.addElement(Element(header))

        self.html_table = Table(path=self.html_dir)
        self.html_table.addRow(header_row)

        # define a path dictionary that will be added to
        self.table_dictionary = {}

    def update_table(self, instance_id, step, refresh_table=False):
        instance_data = self.table_dictionary[instance_id][step]
        curr_row = TableRow()
        curr_row.addElement(Element(str(instance_id)))
        curr_row.addElement(Element(str(step)))

        for col in self.columns:
            if col not in instance_data:
                c_value, c_type = "N/A", "text"
            else:
                c_value, c_type = instance_data[col]

            if c_type == "text":
                e = Element(c_value)
            elif c_type == "other":
                e = Element(str(c_value))
            elif c_type == "image":
                e = imgElement(c_value, 256)
            elif c_type == "alt_image":
                e = imgElement(c_value[0], 256, c_value[1])
            elif c_type == "video":
                e = vidElement(c_value)
            else:
                raise ValueError(f"Unknown column type {c_type}")

            curr_row.addElement(e)

        self.html_table.addRow(curr_row)

        if refresh_table:
            self.write_table()

    def write_table(self):
        tw = TableWriter(self.html_table, head=self.exp_name, rowsPerPage=20)
        tw.write()

    def update_dictionary(self, instance, column, step, value):
        if instance not in self.table_dictionary:
            self.table_dictionary[instance] = {}

        if step not in self.table_dictionary[instance]:
            self.table_dictionary[instance][step] = {}

        self.table_dictionary[instance][step][column] = value

    def add_text(self, instance, column, step, value):
        self.update_dictionary(instance, column, step, (value, "text"))

    def add_other(self, instance, column, step, value):
        self.update_dictionary(instance, column, step, (value, "other"))

    def add_rgb(self, instance, column, step, rgb):
        # define path
        rel_path = f"imgs/{instance}_{column}_{step}.png"
        path = os.path.join(self.html_dir, rel_path)
        value = (rel_path, "image")
        self.update_dictionary(instance, column, step, value)

        # plot figure
        fig = Figure()
        ax = fig.add_subplot(111)
        rgb = rgb.astype(float)
        ax.imshow(rgb.astype(float))
        ax.axis("off")
        fig.savefig(path, bbox_inches="tight", pad_inches=0, dpi=300)

    def add_depth(self, instance, column, step, dep, vmax=None):
        # define path
        rel_path = f"imgs/{instance}_{column}_{step}.png"
        path = os.path.join(self.html_dir, rel_path)
        value = (rel_path, "image")
        self.update_dictionary(instance, column, step, value)

        # plot figure
        fig = Figure()
        ax = fig.add_subplot(111)
        dep = dep.astype(float)

        # set visualization min/max
        cmap = "inferno"
        vmin = 0
        if vmax is None:
            vmax = np.max(dep)

        # plot
        norm = matplotlib.colors.PowerNorm(gamma=0.5, vmin=vmin, vmax=vmax)
        im = ax.imshow(dep, cmap=cmap, norm=norm)
        fig.colorbar(im, ax=ax)
        ax.axis("off")
        fig.savefig(path, bbox_inches="tight", pad_inches=0, dpi=300)

    def add_alt_depth(self, instance, column, step, dep_f, dep_b, vmax=None):
        # define path
        rel_path_f = f"imgs/{instance}_{column}_{step}_fore.png"
        rel_path_b = f"imgs/{instance}_{column}_{step}_back.png"
        path_f = os.path.join(self.html_dir, rel_path_f)
        path_b = os.path.join(self.html_dir, rel_path_b)
        value = ((rel_path_f, rel_path_b), "alt_image")
        self.update_dictionary(instance, column, step, value)

        # plot figure
        dep_f = dep_f.astype(float)
        dep_b = dep_b.astype(float)

        # set visualization min/max
        cmap = "inferno"
        vmin = 0
        if vmax is None:
            vmax_f = np.max(dep_f)  # np.nanpercentile(dep_f, 95)
            vmax_b = np.max(dep_b)  # np.nanpercentile(dep_b, 95)
            vmax = max(vmax_f, vmax_b)

        # plot
        norm = matplotlib.colors.PowerNorm(gamma=0.5, vmin=vmin, vmax=vmax)
        for dep, path in [(dep_f, path_f), (dep_b, path_b)]:
            fig = Figure()
            ax = fig.add_subplot(111)
            im = ax.imshow(dep, cmap=cmap, norm=norm)
            fig.colorbar(im, ax=ax)
            ax.axis("off")
            fig.savefig(path, bbox_inches="tight", pad_inches=0, dpi=300)

    def add_alt_rgb(self, instance, column, step, rgb_f, rgb_b):
        # define path
        rel_path_f = f"imgs/{instance}_{column}_{step}_fore.png"
        rel_path_b = f"imgs/{instance}_{column}_{step}_back.png"
        path_f = os.path.join(self.html_dir, rel_path_f)
        path_b = os.path.join(self.html_dir, rel_path_b)
        value = ((rel_path_f, rel_path_b), "alt_image")
        self.update_dictionary(instance, column, step, value)

        # plot figure
        for rgb, path in [(rgb_f, path_f), (rgb_b, path_b)]:
            fig = Figure()
            ax = fig.add_subplot(111)
            ax.imshow(rgb.astype(float))
            ax.axis("off")
            fig.savefig(path, bbox_inches="tight", pad_inches=0, dpi=300)

    def add_correpsondence(
        self, instance, column, step, rgb_0, rgb_1, corr, use_weight=False
    ):
        # define path
        rel_path = f"imgs/{instance}_{column}_{step}.png"
        path = os.path.join(self.html_dir, rel_path)
        value = (rel_path, "image")
        self.update_dictionary(instance, column, step, value)

        cpx_0, cpx_1, c_cnf, c_err = corr
        cpx_0 = cpx_0.detach().cpu().numpy()
        cpx_1 = cpx_1.detach().cpu().numpy()
        c_err = c_err.detach().cpu().numpy()
        c_cnf = c_cnf.squeeze().detach().cpu().numpy()
        fig = Figure()
        ax = fig.add_subplot(111)

        if use_weight:
            plot_correspondances(rgb_0, rgb_1, cpx_0, cpx_1, ax=ax, weight=c_cnf)
        else:
            plot_correspondances(rgb_0, rgb_1, cpx_0, cpx_1, ax=ax, error=c_err)

        ax.axis("off")
        fig.savefig(path, bbox_inches="tight", pad_inches=0, dpi=200)

    def add_multiview_correpsondence(
        self, instance, column, step, rgbs, pw_corr, views, use_weights=False
    ):
        if views == 2:
            rgb_i = rgbs[0]
            rgb_j = rgbs[1]
            corr = pw_corr[(0, 1)]
            self.add_correpsondence(
                instance, column, step, rgb_i, rgb_j, corr, use_weights
            )
            return

        # define path
        rel_path = f"imgs/{instance}_{column}_{step}.png"
        path = os.path.join(self.html_dir, rel_path)
        value = (rel_path, "image")
        self.update_dictionary(instance, column, step, value)

        size = 2
        dpi = 100
        fig = Figure(figsize=(5 * size * views, 2 * size * views))

        for i in range(views):
            for j in range(views):
                ax_ij = fig.add_subplot(views, views, i * views + j + 1)

                if i == j:
                    ax_ij.imshow(rgbs[i])
                elif (i, j) in pw_corr:
                    cpx_i, cpx_j, c_cnf, c_err = pw_corr[(i, j)]
                    cpx_i = cpx_i.detach().cpu().numpy()
                    cpx_j = cpx_j.detach().cpu().numpy()
                    c_err = c_err.detach().cpu().numpy()
                    c_cnf = c_cnf.detach().cpu().numpy()
                    rgb_i = rgbs[i]
                    rgb_j = rgbs[j]

                    if use_weights:
                        plot_correspondances(
                            rgb_i, rgb_j, cpx_i, cpx_j, weight=c_cnf, ax=ax_ij, fig=fig
                        )
                    else:
                        plot_correspondances(
                            rgb_i, rgb_j, cpx_i, cpx_j, error=c_err, ax=ax_ij, fig=fig
                        )

                ax_ij.axis("off")

        fig.savefig(path, bbox_inches="tight", pad_inches=0, dpi=dpi)

    def add_animation(self, instance, column, step, xyz_0, xyz_1, Rt_01):
        # define path
        rel_path = f"imgs/{instance}_{column}_{step}.png"
        path = os.path.join(self.html_dir, rel_path)
        value = (rel_path, "video")
        self.update_dictionary(instance, column, step, value)

        # create two point clouds
        pcd_0 = o3d.geometry.PointCloud()
        pcd_1 = o3d.geometry.PointCloud()
        pcd_0.points = o3d.utility.Vector3dVector(xyz_0.numpy())
        pcd_1.points = o3d.utility.Vector3dVector(xyz_1.numpy())

        P_01 = np.eye(4)
        P_01[:3, :] = Rt_01.numpy()

        registration_animation(pcd_0, pcd_1, P_01, path)

    def add_pointcloud(
        self, instance, column, step, xyz, rgb=None, fill_out="none", vox_dim=0.025
    ):
        # define path
        rel_path = f"imgs/{instance}_{column}_{step}.png"
        path = os.path.join(self.html_dir, rel_path)
        value = (rel_path, "image")
        self.update_dictionary(instance, column, step, value)
        plot_pointcloud(path, xyz, rgb, fill_out, vox_dim)
