# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import os
import pickle

import numpy
import torch
from pytorch3d.transforms.rotation_conversions import quaternion_to_matrix
from torchvision import transforms as transforms
from tqdm import tqdm

from ..utils.transformations import SE3_inverse, get_relative_Rt, make_Rt
from ..utils.util import fill_depth
from .abstract import AbstractDataset


class RGBD_Video_Dataset(AbstractDataset):
    def __init__(self, cfg, root, split):
        super().__init__(cfg.name, split, root)

        self.cfg = cfg
        self.split = split
        self.num_views = cfg.num_views
        self.square_crop = True
        self.square_crop = True

        assert "ETH" in cfg.name
        # aspect ratio for ETH is ~1.61
        self.img_shape = (self.cfg.img_dim, int(1.61 * self.cfg.img_dim))
        assert self.num_views > 0

        self.rgb_transform = transforms.Compose(
            [
                transforms.Resize(self.img_shape),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

        # get sequences
        self.sequences = os.listdir(root)
        # remove sequences with no light as they are impossible to match visually
        # sequences with "dark" in the name
        self.sequences = [seq for seq in self.sequences if "light_changes" not in seq]
        self.sequences = [seq for seq in self.sequences if "dark" not in seq]
        self.data_dict, self.K_dict, self.instances = self.get_instances(self.sequences)

        # Print out dataset stats
        print(f"Stats for {self.name} - {split}")
        print(f"Numer of sequences {len(self.sequences)}")
        print(f"Numer of instances {len(self.instances)}")
        print("Configs:")
        print(cfg)
        print("================================")

    def __len__(self):
        return len(self.instances)

    def dep_transform(self, dep):
        dep = torch.Tensor(dep[None, None, :, :]).float()
        # interpolation=0 is nearest
        dep = transforms.functional.resize(dep, self.img_shape, interpolation=0)[0, 0]

        if self.cfg.fill_depth:
            dep = torch.tensor(fill_depth(dep.numpy())).float()

        return dep[None, :, :]

    def __getitem__(self, index):
        sequence, f_ids = self.instances[index]

        output = {"uid": index, "class_id": self.name, "sequence_id": sequence}

        # -- Transform K to handle image resize and crop
        K = self.K_dict[sequence].clone().float()

        # get first image info
        view_0 = self.data_dict[sequence][f_ids[0]]
        rgb_path_0 = os.path.join(sequence, view_0["rgb_path"])
        rgb = self.get_rgb(rgb_path_0)

        # Resize K
        output["K_full"] = torch.tensor(K).float()
        K[0, :] *= self.img_shape[1] / rgb.width
        K[1, :] *= self.img_shape[0] / rgb.height

        if self.square_crop:
            side_crop = (self.img_shape[1] - self.img_shape[0]) // 2
            K[0, 2] -= side_crop

        K = torch.tensor(K).float()
        output["K"] = K

        # get rgb and dep
        for i, id_i in enumerate(f_ids):
            # get instance data
            view_i = self.data_dict[sequence][id_i]

            # append sequence to each path
            rgb_path = os.path.join(sequence, view_i["rgb_path"])
            dep_path = os.path.join(sequence, view_i["depth_path"])

            # log info
            output[f"frame_{i}"] = id_i
            output[f"timestamp_{i}"] = str(view_i["timestamp_rgb"])

            # get rgb
            rgb = self.get_rgb(rgb_path)
            rgb = self.rgb_transform(rgb)

            # get depth (divide by 5000)
            # https://vision.in.tum.de/data/datasets/rgbd-dataset/file_formats
            dep = self.get_img(dep_path)
            dep = dep / 5000.0
            dep = self.dep_transform(dep)

            if self.square_crop:
                rgb = rgb[:, :, side_crop:-side_crop]
                dep = dep[:, :, side_crop:-side_crop]

            # add to outout
            output[f"rgb_{i}"] = rgb
            output[f"depth_{i}"] = dep

            # convert pose
            pose_qt = view_i["pose"]
            qt_t = torch.tensor(pose_qt[0:3])
            qt_q = torch.tensor([pose_qt[6], *pose_qt[3:6]])
            qt_R = quaternion_to_matrix(qt_q)

            # camera to world
            P_wtoci = SE3_inverse(make_Rt(qt_R.T, qt_t)).float()
            output[f"tum_Rt_{i}"] = P_wtoci.clone()

            if i == 0:
                P_wtoc0 = P_wtoci

            # Let's center the world at X0
            P_wtoci = get_relative_Rt(P_wtoc0, P_wtoci)

            output[f"Rt_{i}"] = P_wtoci
            output[f"P_{i}"] = P_wtoci

        return output

    def get_instances(self, sequences):
        """Get the instances belonging to a set of sequences

        Args:
            sequences (list): A list of sequence names that match the directory
            structure of the dataset

        Returns:
            tuple:
                data_dict: dictionary of paths for each sequence
                K_dict: dictionary of intrinisic matrices for each sequence
                instance: (list) each instance is a sequence and list of frames
        """
        data_dict = {}
        K_dict = {}
        instances = []

        for sequence in sequences:
            sequence_path = os.path.join(self.root, sequence)

            # get intrinsics
            # https://vision.in.tum.de/data/datasets/rgbd-dataset/intrinsic_calibration
            calib_file = os.path.join(sequence_path, "calibration.txt")
            if os.path.exists(calib_file):
                fx, fy, cx, cy = numpy.loadtxt(calib_file)
            else:
                fx, fy, cx, cy = 525.0, 525.0, 319.5, 239.5

            K_dict[sequence] = torch.FloatTensor(
                [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1]]
            )

            # get instances
            with open(os.path.join(sequence_path, "sync_dict.pkl"), "rb") as f:
                data_dict[sequence] = pickle.load(f)

            num_frames = len(data_dict[sequence])
            for i in range(0, num_frames):
                for view_spacing in [1, 2, 3]:
                    frames_i = [i + v * view_spacing for v in range(self.num_views)]
                    if any([v >= num_frames for v in frames_i]):
                        continue

                    inst = (sequence, frames_i)
                    instances.append(inst)

        return data_dict, K_dict, instances
