# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import pathlib

import numpy as np
import pytorch_lightning as zeus
import torch
from pytorch3d.transforms.rotation_conversions import matrix_to_quaternion

from ..models import build_model
from ..utils.html_visualizer import HTML_Visualizer
from ..utils.metrics import (
    evaluate_3d_correspondances,
    evaluate_multiview_Rt,
    get_multiview_overlap,
)
from ..utils.transformations import SE3_inverse, get_relative_Rt, transform_points_Rt
from ..utils.util import detach_dictionary, modify_keys


def pose_recall(errors, thresholds):
    recall = []
    for t in thresholds:
        recall.append(100.0 * (errors <= t).astype(float).mean())
    return recall


def error_auc(errors, thresholds):
    """
    Args:
        errors (list): [N,]
        thresholds (list)
    """
    errors = [0] + sorted(list(errors))
    recall = list(np.linspace(0, 1, len(errors)))

    aucs = []
    for thr in thresholds:
        last_index = np.searchsorted(errors, thr)
        y = recall[:last_index] + [recall[last_index - 1]]
        x = errors[:last_index] + [thr]
        aucs.append(np.trapz(y, x) / thr)

    return aucs


class MultiviewRegistrationTest(zeus.LightningModule):
    def __init__(self, cfg):
        super().__init__()

        # define hyperparameters
        self.cfg = cfg
        self.save_hyperparameters()

        # define model
        self.model = build_model(cfg.model)
        self.visualize_test = False

    def on_test_start(self):
        if self.visualize_test:
            # define visualizer for training
            columns = ["split", "sequence"]
            columns += [f"frame_{i}" for i in range(self.cfg.dataset.num_views)]
            columns += [f"rgb_{i}" for i in range(self.cfg.dataset.num_views)]
            # columns += [f"dep_{i}" for i in range(self.cfg.dataset.num_views)]
            columns += ["R_err", "t_err", "R_mag", "t_mag"]
            columns += ["correspondence-0", "correspondence-1"]

            if "experiment" in self.cfg:
                num_steps = self.model.cfg.refinement.num_steps
                exp_name = f"{self.cfg.experiment.full_name}_{num_steps}"
            elif self.model.cfg.name == "GenericAligner":
                feature = self.model.cfg.baseline.feature
                num_steps = self.model.cfg.refinement.num_steps
                exp_name = f"GenericAligner_{feature}_{num_steps}"
            else:
                num_steps = self.model.cfg.refinement.num_steps
                exp_name = f"{self.model.cfg.name}_{num_steps}"

            self.visualizer = HTML_Visualizer(
                self.cfg.paths.html_visual_dir, exp_name, columns
            )

    def on_test_batch_end(self, outputs, batch, batch_idx, dataloader_idx):
        if self.visualize_test:
            self.visualize_batch(outputs, batch, batch_idx, dataloader_idx, "test")

    def process_batch(self, batch):
        rgb = [batch[f"rgb_{i}"] for i in range(self.cfg.dataset.num_views)]
        dep = [batch[f"depth_{i}"] for i in range(self.cfg.dataset.num_views)]
        if "Rt_0" in batch:
            Rts = [batch[f"Rt_{i}"] for i in range(self.cfg.dataset.num_views)]
        else:
            Rts = None
        K = batch["K"]
        K_full = batch["K_full"]
        return rgb, dep, Rts, K, K_full

    def test_step(self, batch, batch_idx):
        p_batch = self.process_batch(batch)
        gt_rgb, gt_dep, gt_Rt, K = p_batch[:4]

        with torch.inference_mode():
            output = self.model(rgbs=gt_rgb, deps=gt_dep, K=K, gt_Rts=gt_Rt)

        compute_overlap = False
        if compute_overlap:
            output["view_overlap"] = get_multiview_overlap(
                torch.stack(gt_Rt, dim=1), torch.stack(gt_dep, dim=1), K
            )

        loss, losses = self.get_losses(batch, output)
        metrics = self.calculate_metrics(p_batch, output)
        metrics.update(losses)

        frames = torch.stack([batch[f"frame_{i}"] for i in range(len(gt_rgb))], dim=1)
        output["gt_Rt"] = gt_Rt
        if "tum_Rt_0" in batch:
            output["frame_id"] = frames
            output["tum_Rt"] = torch.stack(
                [batch[f"tum_Rt_{i}"] for i in range(len(gt_rgb))], dim=1
            )

        output = detach_dictionary(output)
        metrics = detach_dictionary(metrics)

        if "tum_Rt_0" in batch:
            output["timestamp"] = [batch[f"timestamp_{i}"] for i in range(len(gt_rgb))]

        return {"loss": loss, "output": output, "metrics": metrics}

    def get_losses(self, batch, output):
        metrics, losses = {}, {}

        loss = output["loss"].mean()

        # ==== Loss evaluation ====
        for key in output:
            if "loss" in key and "_" in key:
                key_split = key.split("_")
                loss_type = key_split[0]
                # aggregate loss
                loss_val = output[key]
                if type(loss_val) == dict:
                    loss_val = [loss_val[_k] for _k in loss_val]

                if type(loss_val) == list:
                    loss_val = sum(loss_val) / len(loss_val)

                if loss_type in losses:
                    losses[loss_type] += loss_val
                else:
                    losses[loss_type] = loss_val

        for loss_key in losses:
            metrics[f"loss_{loss_key}"] = losses[loss_key].detach()

        return loss, metrics

    def calculate_metrics(self, batch, output):
        metrics = {}

        # ==== Compute Metrics ====
        rgbs, gt_deps, gt_Rts, K, K_full = batch
        if gt_Rts is not None:
            gt_Rts = torch.stack(gt_Rts, dim=1)

        for step_i in range(self.model.cfg.refinement.num_steps):
            # Evaluate rotations
            if (gt_Rts is not None) and (f"Rts_{step_i}" in output):
                pr_Rts = output[f"Rts_{step_i}"]
                Rt_metrics = evaluate_multiview_Rt(pr_Rts, gt_Rts, K=K, dep=gt_deps)
            else:
                Rt_metrics = {}

            # evaluate correspondences
            pw_corr = output[f"pw_corr_{step_i}"]
            img_dim = rgbs[0].shape[2:]
            corr_metrics = {}
            for i, j in pw_corr:
                Rt_ij = get_relative_Rt(gt_Rts[:, i], gt_Rts[:, j])
                xyz_0, xyz_1, c_conf = pw_corr[(i, j)]
                ij_metrics = evaluate_3d_correspondances(
                    xyz_0, xyz_1, K_full, Rt_ij, img_dim
                )
                corr_metrics.update(modify_keys(ij_metrics, suffix=f"/({i},{j})"))

            step_metrics = {**Rt_metrics, **corr_metrics}

            metrics.update(modify_keys(step_metrics, prefix=f"step-{step_i}_"))

        # add last step metrics without step prefix
        metrics.update(step_metrics)

        return metrics

    def tum_evaluation(self, test_step_outputs):
        pw_Rts = [to["output"]["pw_Rts_1"] for to in test_step_outputs]
        frames = [to["output"]["frame_id"] for to in test_step_outputs]
        timestamps = [to["output"]["timestamp"] for to in test_step_outputs]

        all_timestamps = {}
        num_views = 0

        for b_i in range(len(pw_Rts)):
            frames_b = frames[b_i]

            for ins_i in range(frames_b.shape[0]):
                for f_i, f_j in pw_Rts[ins_i]:
                    ins_fi = frames_b[ins_i][f_i].item()
                    ins_fj = frames_b[ins_i][f_j].item()
                    all_timestamps[ins_fi] = timestamps[b_i][f_i][ins_i]
                    all_timestamps[ins_fj] = timestamps[b_i][f_j][ins_i]
                    num_views = max(num_views, ins_fj)

        gt_Rt = torch.stack(test_step_outputs[0]["output"]["gt_Rt"], dim=1)
        pr_Rt = test_step_outputs[0]["output"]["Rts_1"]

        Rt_metrics = evaluate_multiview_Rt(pr_Rt, gt_Rt)
        print(f"Post-Sync: VP - Rotation:    {Rt_metrics['vp-error_R'].mean():.2f}")
        print(f"Post-Sync: VP - Translation: {Rt_metrics['vp-error_t'].mean():.2f}")

        # convert Rts to TUM's time stamp - pose
        num_views = pr_Rt.shape[1]
        sequence = self.cfg.dataset.sequence
        lines = [
            "# estimated trajectory \n",
            f"# sequence: {sequence} \n",
            "# timestamp tx ty tz qx qy qz qw \n",
        ]

        for i in range(num_views):
            Rt_i = SE3_inverse(pr_Rt[0, i])
            R = Rt_i[:3, :3]
            t = Rt_i[:3, 3].numpy().tolist()
            q = matrix_to_quaternion(R).numpy().tolist()
            pose = [*t, q[3], *q[0:3]]
            line_i = all_timestamps[i] + " " + " ".join([str(_p) for _p in pose])

            lines.append(line_i + "\n")

        root_path = pathlib.Path(__file__).parent.parent.parent
        if self.cfg.dataset.name == "ETH":
            split = self.cfg.dataset.split
            save_path = root_path / "eth_outputs" / split / f"{sequence}.txt"
        else:
            save_path = root_path / "tum_outputs" / f"{sequence}.txt"

        save_path = str(save_path.resolve())
        print(f"Save output to {save_path}")
        with open(save_path, "w") as f:
            f.writelines(lines)

    def test_epoch_end(self, test_step_outputs):
        if "tum_Rt" in test_step_outputs[0]["output"]:
            self.tum_evaluation(test_step_outputs)
            return

        test_out = test_step_outputs
        summary = {}
        result_cols = [
            "corr3d-num/(0,1)",
            "corr3d-within1cm/(0,1)",
            "corr3d-within5cm/(0,1)",
            "corr3d-within10cm/(0,1)",
            "corr3d-meanError/(0,1)",
            "corr2d-within1px/(0,1)",
            "corr2d-within2px/(0,1)",
            "corr2d-within5px/(0,1)",
            "corr2d-meanError/(0,1)",
            "vp-error_R",
            "vp-error_t",
            "vp-mag_R",
            "vp-mag_t",
        ]

        num_corr = torch.cat([t_o["output"]["num_corr"] for t_o in test_out])

        print(
            f"Num corr: {num_corr.mean():.4f} |",
            f">5corr {(num_corr >= 5).float().mean():.4f}",
        )

        for key in result_cols:
            if key in test_out[0]["metrics"]:
                vals = [t_o["metrics"][key] for t_o in test_out]
                vals = torch.cat(vals)
                valid_vals = vals[num_corr >= 5].mean()
                print(f"{key} -- mean: {vals.mean():.4f} - valid mean {valid_vals:.4f}")
                summary[key] = valid_vals

        print("=" * 50)

        if test_out[0]["metrics"]["vp-error_R"].shape[1] == 2:
            R_err = torch.cat([t_o["metrics"]["vp-error_R"][:, 1] for t_o in test_out])
            t_err = torch.cat([t_o["metrics"]["vp-error_t"][:, 1] for t_o in test_out])
            err3d = torch.cat(
                [t_o["metrics"]["corr3d-meanError/(0,1)"] for t_o in test_out]
            )
            err2d = torch.cat(
                [t_o["metrics"]["corr2d-meanError/(0,1)"] for t_o in test_out]
            )

            R_err = R_err.cpu().numpy()
            t_err = t_err.cpu().numpy()
            err3d = err3d.cpu().numpy()
            err2d = err2d.cpu().numpy()

            # filter out invalid
            r_recall = pose_recall(R_err, [1, 5, 10])
            t_recall = pose_recall(t_err, [1, 5, 10])
            recall3d = pose_recall(err3d, [0.05, 0.1, 0.25])
            recall2d = pose_recall(err2d, [5, 10, 20])

            r_auc = error_auc(R_err, [5])
            t_auc = error_auc(t_err, [10])
            auc3d = error_auc(err3d, [0.1])
            auc2d = error_auc(err2d, [10])

            print("-" * 80)
            print(
                "Pose AUC (5deg, 10cm)    |  ",
                f"{100 * r_auc[0]:.1f}, {100 * t_auc[0]:.1f}",
            )
            print(
                "Corr AUC (10cm, 10px)    |  ",
                f"{100 * auc3d[0]:.1f}, {100 * auc2d[0]:.1f},",
            )
            print("-" * 80)

            # correspondences
            print(
                "Correspondences (3D, 2D) |  ",
                # f"{summary['corr3d-num/(0,1)']:.1f}  ",
                f"{100.0 * summary['corr3d-within1cm/(0,1)']:.1f} ",
                f"{100.0 * summary['corr3d-within5cm/(0,1)']:.1f} ",
                f"{100.0 * summary['corr3d-within10cm/(0,1)']:.1f} ",
                f"{100.0 * summary['corr2d-within1px/(0,1)']:.1f} ",
                f"{100.0 * summary['corr2d-within2px/(0,1)']:.1f} ",
                f"{100.0 * summary['corr2d-within5px/(0,1)']:.1f}",
            )
            print(
                "Corr means (3D, 2D)      |  ",
                f"{recall3d[0]:.1f}  {recall3d[1]:.1f}  {recall3d[2]:.1f} ",
                f"{recall2d[0]:.1f}  {recall2d[1]:.1f}  {recall2d[2]:.1f}",
            )

            print(
                "Pose estimation (R, t)   |  ",
                f"{r_recall[0]:.1f}  {r_recall[1]:.1f}  {r_recall[2]:.1f} ",
                f"{t_recall[0]:.1f}  {t_recall[1]:.1f}  {t_recall[2]:.1f}",
            )
            print(
                "Results for Sheet        |  ",
                f"{100.0 * summary['corr3d-within1cm/(0,1)']:.1f},",
                f"{100.0 * summary['corr3d-within5cm/(0,1)']:.1f},",
                f"{100.0 * summary['corr3d-within10cm/(0,1)']:.1f},",
                f"{100.0 * summary['corr2d-within1px/(0,1)']:.1f},",
                f"{100.0 * summary['corr2d-within2px/(0,1)']:.1f},",
                f"{100.0 * summary['corr2d-within5px/(0,1)']:.1f},",
                f"{r_recall[0]:.1f}, {r_recall[1]:.1f}, {r_recall[2]:.1f},",
                f"{t_recall[0]:.1f}, {t_recall[1]:.1f}, {t_recall[2]:.1f}",
            )
        else:
            R_err = torch.cat(
                [t_o["metrics"]["vp-error_R"].mean(dim=1) for t_o in test_out]
            )
            t_err = torch.cat(
                [t_o["metrics"]["vp-error_t"].mean(dim=1) for t_o in test_out]
            )

            R_err = R_err.cpu().numpy()
            t_err = t_err.cpu().numpy()

            r_auc = error_auc(R_err, [5])
            t_auc = error_auc(t_err, [10])

            print("-" * 80)
            print(
                "Pose AUC (5deg, 10cm)    |  ",
                f"{100 * r_auc[0]:.1f}, {100 * t_auc[0]:.1f}",
            )
            print("-" * 80)

    def visualize_batch(self, outputs, batch, batch_idx, dataloader_idx, split):
        """Visualize elements on the end of a batch every vis_step steps.
        Args:
            outputs (dictionary): batch_outputs
            batch (dictionary): batch of instances
            batch_idx (id): id within epoch
            dataloader_idx (id): ???
        """
        step = self.global_step

        uid = batch["uid"].detach().cpu().numpy()
        batch_size = len(uid)
        rgbs, gt_Rts, deps, gt_deps = [], [], [], []
        for i in range(self.cfg.dataset.num_views):
            rgb_i = (batch[f"rgb_{i}"] * 0.5 + 0.5).clip(min=0, max=1)
            rgbs.append(rgb_i.permute(0, 2, 3, 1).cpu().numpy())
            gt_deps.append(batch[f"depth_{i}"][:, 0].cpu().numpy())
            if "depth" in outputs["output"]:
                deps.append(outputs["output"]["depth"][:, i, 0].cpu().numpy())
            else:
                deps.append(None)
            if "Rt_0" in batch:
                gt_Rts.append(batch[f"Rt_{i}"])
            else:
                gt_Rts = None

        if gt_Rts is not None:
            err_R = outputs["metrics"]["vp-error_R"].numpy()
            err_t = outputs["metrics"]["vp-error_t"].numpy()
            mag_R = outputs["metrics"]["vp-mag_R"].numpy()
            mag_t = outputs["metrics"]["vp-mag_t"].numpy()

        for i in range(batch_size):
            for v_i in range(self.cfg.dataset.num_views):
                frame_i = batch[f"frame_{v_i}"][i].item()
                self.visualizer.add_other(uid[i], f"frame_{v_i}", step, frame_i)
                self.visualizer.add_rgb(uid[i], f"rgb_{v_i}", step, rgbs[v_i][i])
                # if deps[v_i] is None:
                #     self.visualizer.add_depth(
                #         uid[i], f"dep_{v_i}", step, gt_deps[v_i][i]
                #     )

                # else:
                #     self.visualizer.add_alt_depth(
                #         uid[i], f"dep_{v_i}", step, deps[v_i][i], gt_deps[v_i][i]
                #     )

            seq_id = batch["sequence_id"][i]
            self.visualizer.add_other(uid[i], "split", step, split)
            self.visualizer.add_other(uid[i], "sequence", step, seq_id)

            # add metrics if gt_Rt is available
            if gt_Rts is not None:
                _range = range(len(err_R[i]))
                err_R_i = "\n".join([f"{err_R[i][j]:.2f}" for j in _range])
                err_t_i = "\n".join([f"{err_t[i][j]:.2f}" for j in _range])
                mag_R_i = "\n".join([f"{mag_R[i][j]:.2f}" for j in _range])
                mag_t_i = "\n".join([f"{mag_t[i][j]:.2f}" for j in _range])

                self.visualizer.add_other(uid[i], "R_err", step, err_R_i)
                self.visualizer.add_other(uid[i], "t_err", step, err_t_i)
                self.visualizer.add_other(uid[i], "R_mag", step, mag_R_i)
                self.visualizer.add_other(uid[i], "t_mag", step, mag_t_i)

            instance_rgbs = [rgb_v[i] for rgb_v in rgbs]
            num_steps = self.model.cfg.refinement.num_steps
            for it_step in range(num_steps):
                # compute correspondence error
                pw_corr = outputs["output"][f"pw_corr_{it_step}"]
                pw_corr_vis = {}
                for v_i, v_j in pw_corr:
                    c_xyz_i, c_xyz_j, c_weight = pw_corr[(v_i, v_j)]
                    c_xyz_i = c_xyz_i[i]
                    c_xyz_j = c_xyz_j[i]
                    c_weight = c_weight[i]

                    if gt_Rts is not None:
                        gt_Rt_ij = get_relative_Rt(gt_Rts[v_i][i], gt_Rts[v_j][i])
                        gt_Rt_ij = gt_Rt_ij.to(c_xyz_i)
                        c_xyz_i_r = transform_points_Rt(c_xyz_i, gt_Rt_ij)
                        c_error = (c_xyz_i_r - c_xyz_j).norm(p=2, dim=-1)
                    else:
                        # if there's no GT Rts; visualized with green for all (0 error)
                        c_error = torch.zeros_like(c_xyz_i[..., 0])

                    # convert to camera xy
                    K = batch["K"][i].to(c_xyz_i)
                    c_xyz_i = c_xyz_i @ K.transpose(-2, -1)
                    c_xyz_j = c_xyz_j @ K.transpose(-2, -1)
                    c_xy_i = c_xyz_i[..., :2] / c_xyz_i[..., 2:3]
                    c_xy_j = c_xyz_j[..., :2] / c_xyz_j[..., 2:3]

                    pw_corr_vis[(v_i, v_j)] = (c_xy_i, c_xy_j, c_weight, c_error)

                self.visualizer.add_multiview_correpsondence(
                    uid[i],
                    f"correspondence-{it_step}",
                    step,
                    instance_rgbs,
                    pw_corr_vis,
                    views=self.cfg.dataset.num_views,
                )

            self.visualizer.update_table(uid[i], step)

        self.visualizer.write_table()
