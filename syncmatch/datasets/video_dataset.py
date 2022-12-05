# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import torch
from torchvision import transforms as transforms
from tqdm import tqdm

from ..utils.transformations import SE3_inverse, get_relative_Rt, make_Rt
from ..utils.util import fill_depth
from .abstract import AbstractDataset


class VideoDataset(AbstractDataset):
    def __init__(self, cfg, root_path, data_dict, split, pairs=None):
        name = cfg.name
        super().__init__(name, split, root_path)
        self.cfg = cfg
        self.split = split
        self.num_views = cfg.num_views

        self.data_dict = data_dict

        self.square_crop = True
        self.img_shape = (self.cfg.img_dim, int(4 * self.cfg.img_dim / 3))

        self.rgb_transform = transforms.Compose(
            [
                transforms.Resize(self.img_shape),
                *(
                    [transforms.ColorJitter(*cfg.color_jitter)]
                    if (split == "train" and "color_jitter" in cfg)
                    else []
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

        # The option to do strided frame pairs is to under sample Validation and Test
        # sets since there's a huge number of frames to start with. Since RGBD is much
        # smaller, we use the non-strided version. An example of strided vs non strided
        # for a view spacing of 10 and frame pairs
        # strided:      (0, 10), (10, 20), (20, 30), etc
        # non-strided:  (0, 10), ( 1, 11), ( 2, 12), etc
        strided = split in ["valid", "test"]
        if pairs is not None:
            assert self.num_views == 2
            self.instances = self.pairs_to_instances(self.data_dict, pairs)
        else:
            self.instances = self.dict_to_instances(self.data_dict, strided)

        # Print out dataset stats
        print(
            f"Dataset: {self.name} - {split}. Size: {len(self.instances)}.",
            f"Num Sequences: {self.num_sequences}.",
        )
        print("Configs:")
        print(cfg)

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
        cls_id, s_id, f_ids = self.instances[index]
        s_instance = self.data_dict[cls_id][s_id]["instances"]
        output = {"uid": index, "class_id": cls_id, "sequence_id": s_id}

        # -- Transform K to handle image resize and crop
        rgb = self.get_rgb(s_instance[f_ids[0]]["rgb_path"])

        # Resize K
        K = s_instance[f_ids[0]]["intrinsic"][:3, :3].copy()
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
            output[f"path_{i}"] = s_instance[id_i]["rgb_path"]
            output[f"frame_{i}"] = id_i

            # get rgb
            rgb = self.get_rgb(s_instance[id_i]["rgb_path"])
            rgb = self.rgb_transform(rgb)

            # Resize depth and scale to meters according to ScanNet Docs
            # http://kaldir.vc.in.tum.de/scannet_benchmark/documentation
            dep_path = s_instance[id_i]["dep_path"]
            dep_ext = dep_path.split(".")[1]

            if dep_ext == "png":
                dep = self.get_img(dep_path)
                dep = self.dep_transform(dep)
                dep = dep / 1000.0
            elif dep_ext == "npy":
                dep = self.get_npy(dep_path)
                dep = self.dep_transform(dep)

            if self.square_crop:
                rgb = rgb[:, :, side_crop:-side_crop]
                dep = dep[:, :, side_crop:-side_crop]

            output[f"rgb_{i}"] = rgb
            output[f"depth_{i}"] = dep

            E = torch.tensor(s_instance[id_i]["extrinsic"]).float()
            # ScanNet is left multiplication, camera to world
            P_wtoci = SE3_inverse(make_Rt(E[:3, :3].T, E[:3, 3]))

            if i == 0:
                P_wtoc0 = P_wtoci

            # Let's center the world at X0
            P_wtoci = get_relative_Rt(P_wtoc0, P_wtoci)

            output[f"Rt_{i}"] = P_wtoci
            output[f"P_{i}"] = P_wtoci

        return output

    def pairs_to_instances(self, data_dict, pairs):
        """
        converts the data dictionary into a list of instances
        Input: data_dict -- sturcture  <classes>/<models>/<instances>

        Output: all dataset instances
        """
        instances = []
        scenes = []

        # populate dictionary
        for pair in tqdm(pairs):
            cls_id, s_id, f_i, f_j = pair[:4]
            scenes.append(s_id)

            # check that we're getting the right frames
            s_id = f"scene{s_id}"
            rgb_i = data_dict[cls_id][s_id]["instances"][f_i]["rgb_path"]
            rgb_j = data_dict[cls_id][s_id]["instances"][f_j]["rgb_path"]

            assert f_i == int(rgb_i.split("/")[-1].split(".")[0])
            assert f_j == int(rgb_j.split("/")[-1].split(".")[0])

            instances.append([cls_id, s_id, (f_i, f_j)])

        print(f"Num pairs: {len(instances)}")
        self.num_sequences = len(set(scenes))
        return instances

    def dict_to_instances(self, data_dict, strided):
        """
        converts the data dictionary into a list of instances
        Input: data_dict -- sturcture  <classes>/<models>/<instances>

        Output: all dataset instances
        """
        instances = []

        # populate dictionary
        num_classes = 0
        num_sequences = 0
        for cls_id in data_dict:
            num_classes += 1
            for i, s_id in enumerate(data_dict[cls_id]):
                num_sequences += 1
                frames = list(data_dict[cls_id][s_id]["instances"].keys())
                frames.sort()
                vs = self.cfg.view_spacing

                while len(frames) < (self.num_views + 1) * vs:
                    vs = vs * 2 // 3
                    print(vs, self.cfg.view_spacing, "!", self.split)
                assert vs > 0

                if strided:
                    frames = frames[::vs]
                    stride = 1
                else:
                    stride = vs
                num_frames = len(frames)

                for i in range(num_frames - self.num_views * stride):
                    # Some frames might be skipped due to missing projection matrix.
                    # This will affect overlap matrix which is defined over valid
                    # frames only. Hence, we need to index it with the frame index,
                    # not by the frame number.
                    f_ids = []
                    i_ids = []
                    for v in range(self.num_views):
                        f_ids.append(frames[i + v * stride])
                        i_ids.append(i + v * stride)

                    instances.append([cls_id, s_id, tuple(f_ids)])

        self.num_classes = num_classes
        self.num_sequences = num_sequences
        return instances
