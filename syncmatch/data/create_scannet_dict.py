# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import argparse
import os
import pickle
from multiprocessing import Pool

import numpy as np
import pandas as pd
from tqdm import tqdm


def get_split_scenes(file_name):
    with open(file_name) as f:
        scenes = f.readlines()
        scenes = [s.strip() for s in scenes]
    return scenes


def process_scene(input):

    data_root, split, s_id = input

    # get relative paths
    rgb_rel = os.path.join(split, "scans", s_id, "color")
    dep_rel = os.path.join(split, "scans", s_id, "depth")
    ext_rel = os.path.join(split, "scans", s_id, "pose")

    # -- Get frames through ext_files --
    rgb_dir = os.path.join(data_root, rgb_rel)
    rgb_files = os.listdir(rgb_dir)

    # -- get intrisnics --
    int_path = os.path.join(
        data_root, split, "scans", s_id, "intrinsic/intrinsic_color.txt"
    )
    int_mat = pd.read_csv(int_path, header=None, delimiter=" ").values
    inst_dict = {}
    skipped = 0
    for f_id in rgb_files:
        # get frame id from {frame_id}.jpg
        f_id = int(f_id.split(".")[0])

        # get values
        ext_path = os.path.join(data_root, ext_rel, f"{f_id}.txt")
        ext_mat = pd.read_csv(ext_path, header=None, delimiter=" ").values

        if np.isinf(ext_mat).any():
            skipped += 1
            continue

        # populate dictionary
        inst_dict[f_id] = {
            "rgb_path": os.path.join(rgb_rel, f"{f_id}.jpg"),
            "dep_path": os.path.join(dep_rel, f"{f_id}.png"),
            "extrinsic": ext_mat,
            "intrinsic": int_mat,
        }

    if skipped > 0:
        print(f"Skipped {skipped}/{len(rgb_files)} frames for scene {s_id}")

    return {"instances": inst_dict}


def create_scannet_dicts(data_root, scene_dir, split):
    # to handle earlier downloads
    data_dict = {}

    scans_root = os.path.join(data_root, scene_dir, "scans")
    scenes = os.listdir(scans_root)

    # filter scenes
    print(f"filter scenes for split {split}")
    scene_input = []
    for s_id in tqdm(scenes):
        if "scene" not in s_id:
            print(f"Skipping {s_id} as it doesn't contain scene")
            continue

        s_id = s_id.strip()
        scene_input.append((data_root, scene_dir, s_id))

    # multiprocessing
    print(f"run multiprocessing for split {split}")
    num_workers = len(os.sched_getaffinity(0))
    with Pool(num_workers) as p:
        r = list(tqdm(p.imap(process_scene, scene_input), total=len(scene_input)))

    for i in range(len(scene_input)):
        s_id = scene_input[i][2]
        scene_dict = r[i]
        data_dict[s_id] = scene_dict

    # save dictionary as pickle in output path
    split_dict = {"ScanNet": data_dict}
    dict_path = f"scannet_{split}.pkl"

    print(f"Saving dict for {split} with {len(data_dict)}")

    with open(dict_path, "wb") as f:
        pickle.dump(split_dict, f, protocol=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_root", type=str)
    parser.add_argument("subdir", type=str)
    parser.add_argument("split", type=str)
    args = parser.parse_args()
    split = args.split

    create_scannet_dicts(args.data_root, args.subdir, args.split)
