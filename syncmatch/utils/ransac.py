# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import numpy as np

try:
    import open3d as o3d

    OPEN3D_AVAILABLE = True
except:
    print("Unable to import Open3D")
    OPEN3D_AVAILABLE = False


def o3d_3d_correspondence_registration(xyz_0, xyz_1):
    """
    Input:
        xyz_{0, 1}      FloatTensor(Nx3)
    """
    o3d_reg = o3d.pipelines.registration
    source = o3d.geometry.PointCloud()
    target = o3d.geometry.PointCloud()

    source.points = o3d.utility.Vector3dVector(xyz_0.cpu().numpy())
    target.points = o3d.utility.Vector3dVector(xyz_1.cpu().numpy())

    corres_range = np.arange(xyz_0.shape[0]).astype(np.int32)
    o3d_corres = np.stack((corres_range, corres_range), axis=1)
    o3d_corres = o3d.utility.Vector2iVector(o3d_corres)
    distance_threshold = 0.1
    out = o3d_reg.registration_ransac_based_on_correspondence(
        source,
        target,
        o3d_corres,
        distance_threshold,
        o3d_reg.TransformationEstimationPointToPoint(False),
        ransac_n=3,
        criteria=o3d_reg.RANSACConvergenceCriteria(10000, 500),
    )
    return out
