# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import os

import cv2
import numpy as np
import torch
from pytorch3d.ops import (
    corresponding_cameras_alignment,
    corresponding_points_alignment,
)
from pytorch3d.ops.knn import knn_gather, knn_points
from pytorch3d.renderer.cameras import PerspectiveCameras
from pytorch3d.structures import Pointclouds
from pytorch3d.transforms import so3_relative_angle, so3_rotation_angle
from torch.nn.functional import cosine_similarity

from .faiss_knn import faiss_knn
from .ransac import OPEN3D_AVAILABLE, o3d_3d_correspondence_registration
from .transformations import make_Rt, transform_points_Rt
from .util import depth_to_pointclouds, xyz_to_camera


def evaluate_multiview_Rt(pr, gt, align=True, scale=True, K=None, dep=None):
    assert pr.shape == gt.shape, f"{pr.shape} != {gt.shape}"
    batch, views, dim_3, dim_4 = pr.shape
    assert dim_3 == 4 and dim_4 == 4, f"{pr.shape} != B x V x 4 x 4"

    # Align two sets of cameras together -- pytorch implementation is not batched
    R_err = torch.zeros(batch, views).to(gt)
    t_err = torch.zeros(batch, views).to(gt)
    R_mag = torch.zeros(batch, views).to(gt)
    t_mag = torch.zeros(batch, views).to(gt)
    t_ang = torch.zeros(batch, views).to(gt)

    if dep is not None:
        assert K is not None
        p_err = -1 * torch.zeros(batch, views).to(gt)
        pcs = [depth_to_pointclouds(_dep, K).points_list() for _dep in dep]

    for i in range(batch):
        if align and views > 2:
            # account for optimization being done assuming right hand multiplication
            cam_pr = PerspectiveCameras(
                R=pr[i, :, :3, :3].transpose(-2, -1), T=pr[i, :, :3, 3]
            )
            cam_gt = PerspectiveCameras(
                R=gt[i, :, :3, :3].transpose(-2, -1), T=gt[i, :, :3, 3]
            )

            cam_pr_aligned = corresponding_cameras_alignment(
                cam_pr, cam_gt, estimate_scale=scale, mode="extrinsics"
            )

            # undo change
            pr_R = cam_pr_aligned.R.transpose(-2, -1).to(gt)
            pr_t = cam_pr_aligned.T.to(gt)
        else:
            pr_R = pr[i, :, :3, :3].to(gt)
            pr_t = pr[i, :, :3, 3].to(gt)

        gt_R = gt[i, :, :3, :3]
        gt_t = gt[i, :, :3, 3]

        try:
            # compute rotation error
            R_dif_i = torch.bmm(pr_R.double(), gt_R.double().permute(0, 2, 1))
            R_dif_i = R_dif_i.clamp(min=-1.0, max=1.0)  # to avoid numerical errors
            R_err_i = so3_rotation_angle(R_dif_i, cos_angle=True, eps=1e-4)
            R_mag_i = so3_rotation_angle(gt_R, cos_angle=True, eps=1e-4)
            R_err[i] = R_err_i.clamp(min=-1.0, max=1.0).acos() * 180.0 / np.pi
            R_mag[i] = R_mag_i.clamp(min=-1.0, max=1.0).acos() * 180.0 / np.pi
        except:
            print("Something went wrong with Rotation error computation")
            print("pr_R", pr_R)
            print("gt_R", gt_R)
            R_err[i] = torch.ones(views, device=gt_R.device) * 100
            R_mag[i] = torch.ones(views, device=gt_R.device) * 100

        # calculate absolute xyz error
        t_err[i] = (pr_t - gt_t).norm(p=2, dim=1) * 100.0
        t_mag[i] = gt_t.norm(p=2, dim=1) * 100.0

        # add epsilon to handle 0s)
        eps = 1e-5  # 1mm
        gt_t = gt_t + eps * (gt_t.norm(dim=-1, keepdim=True) < eps).float()
        pr_t = pr_t + eps * (pr_t.norm(dim=-1, keepdim=True) < eps).float()

        t_ang_i = cosine_similarity(gt_t, pr_t)
        t_ang_i = t_ang_i.clamp(max=1.0).acos()
        t_ang_i = t_ang_i * 180 / np.pi
        t_ang_i = torch.stack((t_ang_i, 180 - t_ang_i), dim=0).min(dim=0).values
        t_ang[i] = t_ang_i

        if dep is not None:
            for v_i in range(views):
                pc_vi_pr = transform_points_Rt(pcs[v_i][i], pr[i, v_i], inverse=True)
                pc_vi_gt = transform_points_Rt(pcs[v_i][i], gt[i, v_i], inverse=True)
                p_err[i, v_i] = (pc_vi_pr - pc_vi_gt).norm(p=2, dim=-1).mean()

    output = {
        "vp-mag_R": R_mag,
        "vp-mag_t": t_mag,
        "vp-error_R": R_err,
        "vp-error_t": t_err,
        "vp-angle_t": t_ang,
    }

    if dep is not None:
        output["point-error"] = p_err

    return output


def evaluate_3d_correspondances(
    xyz_0_all, xyz_1_all, K_all, Rt_all, img_dim, o3d_reg=False
):
    """
    Inputs:
        xyz_0     FloatTensor       batch_size x 3 x num_matches
        xyz_1     FloatTensor       batch_size x 3 x num_matches
        K         FloatTensor       batch_size x 3 x 3
        Rt        FloatTensor       batch_size x 3 x 4
        img_dim   tuple(int, int)   (image height, image width)

    Output:
        corr_error      FloatTensor     batch_size x num_matches
    """
    batch_size = len(xyz_0_all)
    output_all = []
    both_dirs = False

    for i in range(batch_size):
        xyz_0 = xyz_0_all[i]
        xyz_1 = xyz_1_all[i]
        K = K_all[i]
        Rt = Rt_all[i]

        if len(xyz_0) > 1:

            xyz_1in0 = transform_points_Rt(xyz_1, Rt, inverse=True)

            if both_dirs:
                xyz_0in1 = transform_points_Rt(xyz_0, Rt, inverse=False)
                xyz_all_0 = torch.cat((xyz_0, xyz_0in1), dim=-2)
                xyz_all_1 = torch.cat((xyz_1in0, xyz_1), dim=-2)
            else:
                xyz_all_0 = xyz_0
                xyz_all_1 = xyz_1in0

            uv_all_0 = xyz_to_camera(xyz_all_0, K)
            uv_all_1 = xyz_to_camera(xyz_all_1, K)

            H, W = img_dim

            error_2d = (uv_all_0 - uv_all_1).norm(dim=-1, p=2)
            error_3d = (xyz_all_0 - xyz_all_1).norm(dim=-1, p=2)

            # align via ransac and compute errors
            output = {
                "corr3d-meanError": error_3d.mean(dim=-1),
                "corr2d-meanError": error_2d.mean(dim=-1),
            }

            if o3d_reg and OPEN3D_AVAILABLE:
                out = o3d_3d_correspondence_registration(xyz_0.detach(), xyz_1.detach())
                Rt_pr = torch.tensor(out.transformation).to(xyz_0)

                R_err = so3_relative_angle(
                    Rt_pr[None, :3, :3], Rt[None, :3, :3], cos_angle=True
                )
                R_err = R_err.clamp(min=-1.0, max=1.0).acos() * 180.0 / np.pi
                t_err = (Rt_pr[:3, 3] - Rt[:3, 3]).norm(p=2) * 100.0
                output["corr3d-ransacError_R"] = R_err.squeeze()
                output["corr3d-ransacError_t"] = t_err.squeeze()

            for pix_thresh in [1, 2, 5, 10, 20]:
                inlier_thresh = error_2d <= pix_thresh
                # inlier_percent = (inlier_thresh * valid).float().mean(dim=-1)
                inlier_percent = inlier_thresh.float().mean(dim=-1)
                output[f"corr2d-within{pix_thresh}px"] = inlier_percent

            for cm_thresh in [1, 2, 5, 10, 20]:
                inlier_thresh = error_3d <= (cm_thresh / 100.0)
                inlier_percent = inlier_thresh.float().mean(dim=-1)
                output[f"corr3d-within{cm_thresh}cm"] = inlier_percent

            output["corr3d-num"] = torch.ones(1).to(Rt_all)[0] * xyz_all_0.shape[0]
        else:
            ZERO = torch.zeros(1).to(Rt_all)[0]
            output = {
                "corr3d-meanError": ZERO,
                "corr2d-meanError": ZERO,
                "corr3d-num": ZERO,
            }

            if o3d_reg and OPEN3D_AVAILABLE:
                output["corr3d-ransacError_R"] = ZERO
                output["corr3d-ransacError_t"] = ZERO

            for _thresh in [1, 2, 5, 10, 20]:
                output[f"corr2d-within{_thresh}px"] = ZERO
                output[f"corr3d-within{_thresh}cm"] = ZERO

        output_all.append(output)

    keys = list(output_all[0].keys())
    new_output = {}
    for key in keys:
        vals = [out[key] for out in output_all]
        new_output[key] = torch.stack(vals, dim=0)

    return new_output


def evaluate_feature_match(
    pc_0, pc_1, Rt_gt, dist_thresh, inlier_thresh, num_sample=5000
):
    num_instances = len(pc_0)

    # make 1 pc less, and the other mode -- very ugly
    pc_0_N = pc_0.num_points_per_cloud()
    pc_1_N = pc_1.num_points_per_cloud()

    # rotate pc_0 and maintain heterogenous batching
    pc_0_X = pc_0.points_padded()
    pc_0_X = transform_points_Rt(pc_0_X, Rt_gt, inverse=True)
    pc_0_X = [pc_0_X[i][0 : pc_0_N[i]] for i in range(num_instances)]

    # rest are just normal lists
    pc_1_X = pc_1.points_list()
    pc_0_F = pc_0.features_list()
    pc_1_F = pc_1.features_list()

    pc_less = []
    pc_more = []

    for i in range(num_instances):
        if pc_0_N[i] < pc_1_N[i]:
            pc_less.append((pc_0_X[i], pc_0_F[i]))
            pc_more.append((pc_1_X[i], pc_1_F[i]))
        else:
            pc_more.append((pc_0_X[i], pc_0_F[i]))
            pc_less.append((pc_1_X[i], pc_1_F[i]))

    pc_samp = []
    for i in range(num_instances):
        _pc_x, _pc_f = pc_less[i]
        if len(_pc_x) > num_sample:
            sprob = torch.ones(len(_pc_x))
            s_ids = sprob.multinomial(num_sample, replacement=False)
            _pc_x = _pc_x[s_ids]
            _pc_f = _pc_f[s_ids]
        pc_samp.append((_pc_x, _pc_f))

    pc_less = Pointclouds([pc[0] for pc in pc_samp], features=[pc[1] for pc in pc_samp])
    pc_more = Pointclouds([pc[0] for pc in pc_more], features=[pc[1] for pc in pc_more])

    # now I can do the normal computations!
    _, idx_nn, _ = knn_points(
        pc_less.features_padded(),
        pc_more.features_padded(),
        pc_less.num_points_per_cloud(),
        pc_more.num_points_per_cloud(),
        K=1,
    )

    pc_less_N = pc_less.num_points_per_cloud()
    pc_less_x = pc_less.points_padded()
    pc_more_x = knn_gather(pc_more.points_padded(), idx_nn).squeeze(dim=2)
    dist_diff = (pc_less_x - pc_more_x).norm(p=2, dim=2)

    for i in range(num_instances):
        dist_diff[i, pc_less_N[i] :] = 100 * dist_thresh

    num_matches = (dist_diff < dist_thresh).float().sum(dim=1)
    fmr = num_matches / pc_less_N.float()
    fmr_inlier = (fmr > inlier_thresh).float()
    return fmr, fmr_inlier


def evaluate_multiview_depth(pred, gt):
    """
    Common Pixel-wise Depth Error Metrics - https://arxiv.org/pdf/1805.01328.pdf
    """
    output = {}
    thresh = 1.25
    output["dep_thresh1"] = pixel_treshold(pred, gt, thresh ** 1, dep_dim=2)
    output["dep_thresh2"] = pixel_treshold(pred, gt, thresh ** 2, dep_dim=2)
    output["dep_thresh3"] = pixel_treshold(pred, gt, thresh ** 3, dep_dim=2)
    output["dep_absRMSE"] = abs_rms(pred, gt)
    output["dep_logRMSE"] = log_rms(pred, gt)
    output["dep_relDiff"] = absolute_rel_diff(pred, gt)

    # calcualte mean errors | keep separate in case I return for visual
    for key in output:
        output[key] = output[key].mean(dim=(2, 3, 4))

    return output


def pixel_treshold(pred, gt, threshold, dep_dim=1):
    # clip so that nothing is 0
    ratio = torch.cat((pred / gt, gt / pred.clamp(1e-9)), dep_dim)
    ratio = ratio.max(dim=dep_dim, keepdim=True)[0] < threshold
    return ratio.float()


def absolute_rel_diff(pred, gt):
    assert (gt > 0).all(), "GT Depth cannot be 0 w/ no mask"
    diff = (pred - gt).abs() / gt
    return diff


def abs_rms(pred, gt):
    diff = (pred - gt).pow(2)
    diff = diff.sqrt()
    return diff


def log_rms(pred, gt):
    return abs_rms(pred.clamp(1e-9).log(), gt.log())


def estimate_essential_matrix_pose(
    kpts0, kpts1, K, thresh=1.0, conf=0.99999, max_iter=None
):
    """
    based on estimate_pose function from @magicleap/SuperGluePretrainedNetwork

    Estimates the relative pose between two images based on 2D keypoints. This is
    done by estimating the essential matrix using OpenCV

    Inputs:
        kpts0       Numpy Float Array (N, 2)    2D keypoints in camera 0
        kpts1       Numpy Float Array (M, 2)    2D keypoints in camera 1
        K           Numpy Float Array (3, 3)    Camera instrinsics matrix
        thresh      float                       ??
        conf        float                       ??
        max_iter    int                         max number of iterations

    return  3-element tuple (R, t, ??)
        R           Numpy Float Array (3, 3)    relative rotation
        t           Numpy Float Array (3, )     translation (unit norm, no scale)
        ??

    """

    assert len(kpts0) >= 5, "Cannot solve with less than 5 points"

    # convert to numpy
    kpts0 = kpts0.cpu().numpy()
    kpts1 = kpts1.cpu().numpy()
    K = K.cpu().numpy()

    f_mean = np.mean([K[0, 0], K[1, 1]])
    norm_thresh = thresh / f_mean

    kpts0 = (kpts0 - K[[0, 1], [2, 2]][None]) / K[[0, 1], [0, 1]][None]
    kpts1 = (kpts1 - K[[0, 1], [2, 2]][None]) / K[[0, 1], [0, 1]][None]

    if max_iter is not None:
        E, mask = cv2.findEssentialMat(
            kpts0,
            kpts1,
            np.eye(3),
            threshold=norm_thresh,
            prob=conf,
            method=cv2.RANSAC,
            maxIters=max_iter,
        )
    else:
        E, mask = cv2.findEssentialMat(
            kpts0, kpts1, np.eye(3), threshold=norm_thresh, prob=conf, method=cv2.RANSAC
        )

    assert E is not None

    best_num_inliers = 0
    ret = None
    for _E in np.split(E, len(E) / 3):
        n, R, t, _ = cv2.recoverPose(_E, kpts0, kpts1, np.eye(3), 1e9, mask=mask)
        if n > best_num_inliers:
            best_num_inliers = n
            R = torch.tensor(R)
            t = torch.tensor(t[:, 0])
            mask = torch.tensor(mask.ravel() > 0)
            ret = (R, t, mask)

    return ret


def compute_epipolar_error(kpts0, kpts1, T_0to1, K):
    """
    taken from @magicleap/SuperGluePretrainedNetwork

    Estimates the relative pose between two images based on 2D keypoints. This is
    done by estimating the essential matrix using OpenCV

    Inputs:
        kpts0       Numpy Float Array (N, 2)    2D keypoints in camera 0
        kpts1       Numpy Float Array (M, 2)    2D keypoints in camera 1
        K           Numpy Float Array (3, 3)    Camera instrinsics matrix
        T_0to1      Numpy Float Array (4, 4)    Camera motion from 0 to 1
        conf        float                       ??

    return  epipolar error

    """
    kpts0 = kpts0.cpu().numpy()
    kpts1 = kpts1.cpu().numpy()
    T_0to1 = T_0to1.cpu().numpy()
    K = K.cpu().numpy()

    def to_homogeneous(points):
        return np.concatenate([points, np.ones_like(points[:, :1])], axis=-1)

    kpts0 = (kpts0 - K[[0, 1], [2, 2]][None]) / K[[0, 1], [0, 1]][None]
    kpts1 = (kpts1 - K[[0, 1], [2, 2]][None]) / K[[0, 1], [0, 1]][None]
    kpts0 = to_homogeneous(kpts0)
    kpts1 = to_homogeneous(kpts1)

    t0, t1, t2 = T_0to1[:3, 3]
    t_skew = np.array([[0, -t2, t1], [t2, 0, -t0], [-t1, t0, 0]])
    E = t_skew @ T_0to1[:3, :3]

    Ep0 = kpts0 @ E.T  # N x 3
    p1Ep0 = np.sum(kpts1 * Ep0, -1)  # N
    Etp1 = kpts1 @ E  # N x 3
    d = p1Ep0 ** 2 * (
        1.0 / (Ep0[:, 0] ** 2 + Ep0[:, 1] ** 2)
        + 1.0 / (Etp1[:, 0] ** 2 + Etp1[:, 1] ** 2)
    )
    return d


def evaluate_2d_correspondences(
    corr2d, K, gt_Rt, thresh=1.0, conf=0.99999, max_iter=None
):
    """
    Computes 2D correpsondences quality based on essential matrix computation.

    Input:
        corr2D      tuple(corr_i, corr_j, corr_w)       2D Correspondences
            corr_i      FloatTensor(N, K, 2) or List of FloatTensor(L, 2)
            corr_j      FloatTensor(N, K, 2) or List of FloatTensor(L, 2)
            corr_w      FloatTensor(N, K) or List of FloatTensor(L)

        K           FloatTensor(N, 3, 3)                Camera Intrinsics
        gt_Rt       FloatTensor(N, 4, 4)                6-DoF i -> j

    Return dict:
        E_R_error       FloatTensor(N, )        Rotation error (deg)
        E_t_error       FloatTensor(N, )        Translation angular error (deg)
        E_valid         FloatTensor(N, )        0 if essential matrix failed
    """

    batch_size = len(corr2d[0])

    valid = torch.zeros(batch_size)
    pr_Rt = torch.stack([torch.eye(4) for i in range(batch_size)])
    precision = torch.zeros(batch_size)
    match_score = torch.zeros(batch_size)

    for i in range(batch_size):
        Ki = K[i]
        kpts0 = corr2d[0][i]
        kpts1 = corr2d[1][i]

        # kconf = corr2d[2][i].cpu().numpy()
        # c_thresh = 0.4 # kconf.mean()
        # kpts0 = kpts0[kconf >= c_thresh]
        # kpts1 = kpts1[kconf >= c_thresh]
        # print(
        #     f"Mean: {kconf.mean():.3f} --",
        #     f"median: {np.median(kconf):.3f} --",
        #     f"num pass: {len(kpts0)}",
        # )

        if len(kpts0) >= 8:
            ret = estimate_essential_matrix_pose(
                kpts0, kpts1, Ki, thresh=thresh, max_iter=max_iter, conf=conf
            )
            if ret is not None:
                valid[i] = 1.0
                pr_Rt[i] = make_Rt(ret[0].T, ret[1])

            epi_errs = compute_epipolar_error(kpts0, kpts1, gt_Rt[i], Ki)
            correct = epi_errs < 5e-4
            num_correct = np.sum(correct)
            precision[i] = np.mean(correct) if len(correct) > 0 else 0
            match_score[i] = num_correct / len(kpts0) if len(kpts0) > 0 else 0

    # add epsilon to handle 0s
    pr_Rt = pr_Rt.to(gt_Rt)
    eps = 1e-5  # 1mm
    pr_t = pr_Rt[:, :3, 3]
    gt_t = gt_Rt[:, :3, 3]
    pr_R = pr_Rt[:, :3, :3]
    gt_R = gt_Rt[:, :3, :3]
    gt_t = gt_t + eps * (gt_t.norm(dim=-1, keepdim=True) < eps).float()
    pr_t = pr_t + eps * (pr_t.norm(dim=-1, keepdim=True) < eps).float()

    t_ang = cosine_similarity(gt_t, pr_t)
    t_ang = t_ang.clamp(min=-1.0, max=1.0).acos() * 180 / np.pi
    t_ang = torch.stack((t_ang, 180 - t_ang), dim=0).min(dim=0).values

    R_err = so3_relative_angle(pr_R, gt_R, cos_angle=True)
    R_err = R_err.clamp(min=-1.0, max=1.0).acos() * 180.0 / np.pi

    return R_err, t_ang, valid, precision, match_score


def evaluate_multiview_reconstruction(
    pr_Rt, gt_Rt, pr_dep, gt_dep, K, align_cameras=False, align_pointclouds=True
):
    """
    Measure how well aligned the two reconstructions are

    input:
        pr_Rt   FloatTensor     batch x num_views x 4 x 4
        gt_Rt   FloatTensor     batch x num_views x 4 x 4
        pr_dep  FloatTensor     batch x num_views x 1 x H x W
        gt_dep  FloatTensor     batch x num_views x 1 x H x W
        K       FloatTensor     batch x 3 x 3

    returns mean point error in meters (FloatTensor (batch, ))

    """

    assert pr_dep.shape == gt_dep.shape
    batch, num_views, _, _ = gt_Rt.shape
    p_err = torch.zeros(batch).to(gt_Rt)

    # make sure only valid points are included: valid pr/gt depth
    valid = (pr_dep > 0) & (gt_dep > 0)
    pr_dep = pr_dep * valid
    gt_dep = gt_dep * valid

    K = K[:, None, :, :].repeat(1, num_views, 1, 1)
    pr_pcs = [depth_to_pointclouds(pr_dep[i], K[i]).points_list() for i in range(batch)]
    gt_pcs = [depth_to_pointclouds(gt_dep[i], K[i]).points_list() for i in range(batch)]

    for i in range(batch):
        pr_Rt_i = pr_Rt[i]
        gt_Rt_i = gt_Rt[i]
        pr_pc_i = pr_pcs[i]
        gt_pc_i = gt_pcs[i]

        if align_cameras:
            assert not align_pointclouds
            # account for optimization being done assuming right hand multiplication
            cam_pr = PerspectiveCameras(
                R=pr_Rt_i[:, :3, :3].transpose(-2, -1), T=pr_Rt_i[:, :3, 3]
            )
            cam_gt = PerspectiveCameras(
                R=gt_Rt_i[:, :3, :3].transpose(-2, -1), T=gt_Rt_i[:, :3, 3]
            )

            scale = False
            cam_pr_aligned = corresponding_cameras_alignment(
                cam_pr, cam_gt, estimate_scale=scale, mode="extrinsics"
            )

            # undo change
            pr_Rt_i = make_Rt(cam_pr_aligned.R.to(gt_Rt), cam_pr_aligned.T.to(gt_Rt))

            # rotate pr and gt
            pr_pc_R = [
                transform_points_Rt(pr_pc_i[v_i], pr_Rt_i[v_i], inverse=True)
                for v_i in range(num_views)
            ]
            gt_pc_R = [
                transform_points_Rt(gt_pc_i[v_i], gt_Rt_i[v_i], inverse=True)
                for v_i in range(num_views)
            ]

            pr_pc_R = torch.cat(pr_pc_R, dim=0)
            gt_pc_R = torch.cat(gt_pc_R, dim=0)
        elif align_pointclouds:
            assert not align_cameras
            # rotate pr and gt
            pr_pc_R = [
                transform_points_Rt(pr_pc_i[v_i], pr_Rt_i[v_i], inverse=True)
                for v_i in range(num_views)
            ]
            gt_pc_R = [
                transform_points_Rt(gt_pc_i[v_i], gt_Rt_i[v_i], inverse=True)
                for v_i in range(num_views)
            ]

            pr_pc_R = torch.cat(pr_pc_R, dim=0)
            gt_pc_R = torch.cat(gt_pc_R, dim=0)

            Rt_pr2gt = corresponding_points_alignment(
                pr_pc_R[None, :, :],
                gt_pc_R[None, :, :],
                torch.ones_like(pr_pc_R[None, :, 0]),
            )
            Rt_pr2gt = make_Rt(Rt_pr2gt.R, Rt_pr2gt.T)[0]
            pr_pc_R = transform_points_Rt(pr_pc_R, Rt_pr2gt)
        else:
            raise ValueError("Either set align_pointclouds or align_cameras to True")

        p_err[i] = (pr_pc_R - gt_pc_R).norm(dim=1, p=2).mean()

    return p_err


def get_multiview_overlap(Rt, dep, K):
    """
    Measure how well aligned the two reconstructions are

    input:
        pr_Rt   FloatTensor     batch x num_views x 4 x 4
        gt_Rt   FloatTensor     batch x num_views x 4 x 4
        pr_dep  FloatTensor     batch x num_views x 1 x H x W
        gt_dep  FloatTensor     batch x num_views x 1 x H x W
        K       FloatTensor     batch x 3 x 3

    returns mean point error in meters (FloatTensor (batch, ))

    """

    batch, num_views, _, _ = Rt.shape

    K = K[:, None, :, :].repeat(1, num_views, 1, 1)
    pcs = [depth_to_pointclouds(dep[i], K[i]).points_list() for i in range(batch)]
    overlap = {}

    for i in range(batch):
        Rt_i = Rt[i]
        pc_i = pcs[i]
        pc_i = [
            transform_points_Rt(pc_i[v_i], Rt_i[v_i], inverse=True)
            for v_i in range(num_views)
        ]

        for v_i in range(num_views):
            pc_i_vi = pc_i[v_i][None, :]
            for v_j in range(v_i + 1, num_views):
                pc_i_vj = pc_i[v_j][None, :]

                dists_faiss, idx = faiss_knn(pc_i_vi, pc_i_vj, 1)
                dists = dists_faiss.sqrt()
                overlap_val = (dists < 0.05).float().mean()

                if (v_i, v_j) in overlap:
                    overlap[(v_i, v_j)].append(overlap_val)
                else:
                    overlap[(v_i, v_j)] = [overlap_val]

    # aggregate overlap accoss_views
    for v_ij in overlap:
        overlap[v_ij] = torch.stack(overlap[v_ij], dim=0)

    return overlap
