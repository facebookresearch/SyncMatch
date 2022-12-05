# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from torch import nn as nn
from torchvision.models.resnet import BasicBlock, conv1x1


def get_visual_backbone(cfg):
    """Creates a visual backbone .. currently useless because there's only one
    backbone option, but you could add more

    Args:
        cfg (DictConfig): Config defining the visual model

    Raises:
        ValueError: Raises errors if anything but ResNet18 is defined

    Returns:
        visual backbone
    """
    if cfg.backbone == "ResNet18":
        # set stride to downsample from first layer
        strides = np.ones(5, dtype=int)
        dsample_layers = int(np.log2(cfg.downsampling_factor))
        strides[0:dsample_layers] = 2
        vb = ResNet([2, 2, 2, 2], cfg.dim, strides, zero_mean=cfg.zero_mean)
    else:
        raise ValueError(f"Backbone {cfg.backbone} not defined")

    return vb


class ResNet(nn.Module):
    def __init__(self, layers, feat_dim, strides=[1, 1, 1, 1, 1], zero_mean=False):
        """Creates a ResNet class based on stucture hyperparameters

        Args:
            layers (list): defines the layer sizes for each of the layers
            feat_dim (int): feature dimensions
            strides (list, optional): list of stride sizes
            zero_mean (bool, optional): whether to zero mean the outputs
        """
        super().__init__()

        block = BasicBlock
        norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.zero_mean = zero_mean

        self.inplanes = 64
        self.dilation = 1

        self.groups = 1
        self.base_width = 64

        self.conv1 = nn.Conv2d(
            3, self.inplanes, kernel_size=7, stride=strides[0], padding=3, bias=False
        )
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=strides[1])
        self.layer2 = self._make_layer(
            block, max(64, feat_dim // 2), layers[1], stride=strides[2]
        )
        self.layer3 = self._make_layer(
            block, max(64, feat_dim * 3 // 4), layers[2], stride=strides[3]
        )
        self.layer4 = self._make_layer(block, feat_dim, layers[3], stride=strides[4])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.zero_mean:
            x = x - x.mean(dim=1, keepdim=True)

        return x
