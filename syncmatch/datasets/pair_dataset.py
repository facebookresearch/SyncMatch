# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os

import numpy as np
import torch
from torchvision import transforms as transforms

from ..utils.transformations import SE3_inverse, get_relative_Rt, make_Rt
from .abstract import AbstractDataset


class PairedDataset(AbstractDataset):
    def __init__(self, root_path):
        super().__init__("ScanNet Test Pairs", "test", root_path)

        self.num_views = 2
        self.rgb_transform = transforms.Compose(
            [
                transforms.Resize((480, 640)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

        # parse files for data
        self.instances = self.get_instances(root_path)

        # Print out dataset stats
        print(f"ScanNet Pairs dataset: Size: {len(self.instances)}.")

    def get_instances(self, root_path):
        K_dict = dict(np.load(f"{root_path}/intrinsics.npz"))
        data = np.load(f"{root_path}/test.npz")["name"]
        instances = []

        for i in range(len(data)):
            room_id, seq_id, ins_0, ins_1 = data[i]
            scene_id = f"scene{room_id:04d}_{seq_id:02d}"
            K_i = torch.tensor(K_dict[scene_id]).float()

            instances.append((scene_id, ins_0, ins_1, K_i))

        return instances

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, index):
        s_id, ins_0, ins_1, K = self.instances[index]
        output = {
            "uid": index,
            "class_id": "ScanNet_test",
            "sequence_id": s_id,
            "frame_0": int(ins_0),
            "frame_1": int(ins_1),
            "K_full": K,
            "K": K,
        }

        # get rgb
        rgb_path_0 = os.path.join(self.root, s_id, f"color/{ins_0}.jpg")
        rgb_path_1 = os.path.join(self.root, s_id, f"color/{ins_1}.jpg")
        output["rgb_0"] = self.rgb_transform(self.get_rgb(rgb_path_0))
        output["rgb_1"] = self.rgb_transform(self.get_rgb(rgb_path_1))

        # get poses
        pose_path_0 = os.path.join(self.root, s_id, f"pose/{ins_0}.txt")
        pose_path_1 = os.path.join(self.root, s_id, f"pose/{ins_1}.txt")
        P_0 = torch.tensor(np.loadtxt(pose_path_0, delimiter=" "))
        P_1 = torch.tensor(np.loadtxt(pose_path_1, delimiter=" "))
        P_0 = SE3_inverse(make_Rt(P_0[:3, :3].T, P_0[:3, 3]))
        P_1 = SE3_inverse(make_Rt(P_1[:3, :3].T, P_1[:3, 3]))
        P_01 = get_relative_Rt(P_0, P_1).float()
        P_00 = torch.eye(4).float()
        output["Rt_0"], output["P_0"] = P_00, P_00
        output["Rt_1"], output["P_1"] = P_01, P_01

        # get depths
        dep_path_0 = os.path.join(self.root, s_id, f"depth/{ins_0}.png")
        dep_path_1 = os.path.join(self.root, s_id, f"depth/{ins_1}.png")
        dep_0 = torch.tensor(self.get_img(dep_path_0)) / 1000
        dep_1 = torch.tensor(self.get_img(dep_path_1)) / 1000
        output["depth_0"] = dep_0[None, :, :].float()
        output["depth_1"] = dep_1[None, :, :].float()

        return output
