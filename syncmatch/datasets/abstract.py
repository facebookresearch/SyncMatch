# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import os

import numpy as np
import torch
from PIL import Image, ImageFile
from pytorch3d.io import IO

ImageFile.LOAD_TRUNCATED_IMAGES = True


class AbstractDataset(torch.utils.data.Dataset):
    def __init__(self, name, split, data_root):
        # dataset parameters
        self.name = name
        self.root = data_root
        self.split = split
        self.p3dIO = IO()

    def __len__(self):
        pass

    def __getitem__(self, index):
        pass

    def get_rgba(self, path, bbox=None):
        if self.root is not None:
            path = os.path.join(self.root, path)
        with open(path, "rb") as f:
            with Image.open(f) as img:
                img = img.convert("RGBA")
                if bbox is not None:
                    img = img.crop(box=bbox)
                return img

    def get_rgb_alpha(self, path, bbox=None):
        if self.root is not None:
            path = os.path.join(self.root, path)
        with open(path, "rb") as f:
            with Image.open(f) as img:
                _, _, _, a = img.split()
                img = img.convert("RGB")
                if bbox is not None:
                    img = img.crop(box=bbox)
                    a = a.crop(box=bbox)
                a = np.array(a).astype(dtype=np.float)
                return img, a

    def get_alpha(self, path, bbox=None):
        if self.root is not None:
            path = os.path.join(self.root, path)
        with open(path, "rb") as f:
            with Image.open(f) as img:
                r, g, b, a = img.split()
                if bbox is not None:
                    a = a.crop(box=bbox)
                a = np.array(a).astype(dtype=np.float)
        return a

    def get_img(self, path, bbox=None):
        if self.root is not None:
            path = os.path.join(self.root, path)
        with open(path, "rb") as f:
            with Image.open(f) as img:
                if bbox is not None:
                    img = img.crop(box=bbox)
                return np.array(img)

    def get_npy(self, path):
        if self.root is not None:
            path = os.path.join(self.root, path)
        return np.load(path)

    def get_rgb(self, path, bbox=None):
        if self.root is not None:
            path = os.path.join(self.root, path)
        with open(path, "rb") as f:
            with Image.open(f) as img:
                img = img.convert("RGB")
                if bbox is not None:
                    img = img.crop(box=bbox)
                return img

    def get_pointcloud(self, path):
        if self.root is not None:
            path = os.path.join(self.root, path)
        return self.p3dIO.load_pointcloud(path=path)
