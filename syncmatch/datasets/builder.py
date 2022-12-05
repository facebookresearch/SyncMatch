# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import os

import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..utils.io import load_pickle
from .pair_dataset import PairedDataset
from .rgbd_video_dataset import RGBD_Video_Dataset
from .video_dataset import VideoDataset

# Define some important paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))


def build_dataset(cfg, split, overfit=None):
    """
    Builds a dataset from the provided dataset configs.
    Configs can be seen is configs/config.py
    """
    if cfg.name == "ScanNet":
        dict_path = os.path.join(PROJECT_ROOT, f"data/scannet_{split}.pkl")
        data_dict = load_pickle(dict_path)
        dataset = VideoDataset(cfg, cfg.root, data_dict, split)

        # Reduce ScanNet validation size to allow for more frequent validation
        if split == "valid":
            dataset.instances = dataset.instances[::10]
        elif split == "test":
            dataset.instances = dataset.instances[::5]
    if cfg.name == "ScanNet_Small":
        dict_path = os.path.join(PROJECT_ROOT, f"data/scannet_{split}.pkl")
        data_dict = load_pickle(dict_path)
        dataset = VideoDataset(cfg, cfg.root, data_dict, split)

        # Reduce dataset size
        smaller_size = len(dataset.instances) // 11
        dataset.instances = dataset.instances[0:smaller_size]

        # Reduce ScanNet validation size to allow for more frequent validation
        if split == "valid":
            dataset.instances = dataset.instances[::10]
    elif cfg.name == "ScanNet_TestPairs":
        assert split == "test", "Split only defined for test data"
        dataset = PairedDataset(cfg.root)
    elif cfg.name == "ETH_Video":
        eth_root = os.path.join(cfg.root, cfg.split)
        dataset = RGBD_Video_Dataset(cfg, eth_root, split)

        # generate fake validation
        if split == "train":
            dataset.instances = [
                v for i, v in enumerate(dataset.instances) if i % 20 != 0
            ]
        elif split == "valid":
            dataset.instances = [
                v for i, v in enumerate(dataset.instances) if i % 20 == 0
            ]

    # Overfit only loads a single batch for easy debugging/sanity checks
    if overfit is not None:
        assert type(overfit) is int
        dataset.instances = dataset.instances[: cfg.batch_size] * overfit

    return dataset


def build_loader(cfg, split, overfit=None):
    """
    Builds the dataset loader (including getting the dataset).
    """
    dataset = build_dataset(cfg, split, overfit)
    shuffle = (split == "train") and (not overfit)
    batch_size = cfg.batch_size

    num_workers = min(len(os.sched_getaffinity(0)), 20)

    loader = DataLoader(
        dataset=dataset,
        batch_size=int(batch_size),
        shuffle=shuffle,
        pin_memory=False,
        num_workers=num_workers,
    )

    return loader
