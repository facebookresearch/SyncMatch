# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from .generic_aligner import GenericAligner
from .pairwise_superglue import PairwiseSuperGlue
from .syncmatch import SyncMatch

try:
    from .pairwise_loftr import PairwiseLoFTR
except:
    print("Unable to import LoFTR")


def build_model(cfg):
    if cfg.name == "SyncMatch":
        model = SyncMatch(cfg)
    elif cfg.name == "SuperGlue":
        model = PairwiseSuperGlue(cfg)
    elif cfg.name == "LoFTR":
        model = PairwiseLoFTR(cfg)
    elif cfg.name == "LoFTR_Coarse":
        model = PairwiseLoFTR(cfg, fine=False)
    elif cfg.name == "GenericAligner":
        model = GenericAligner(cfg)
    else:
        raise ValueError(f"Model {cfg.name} is not recognized.")

    return model
