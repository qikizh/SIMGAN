# -*- encoding: utf-8 -*-
from __future__ import division
from __future__ import print_function

import os.path as osp
import numpy as np
from easydict import EasyDict as edict

__C = edict()
cfg = __C

# Dataset name: flower, bird, coco
__C.DATASET_NAME = 'bird'
__C.CONFIG_NAME = ''
__C.DATA_DIR = ''
__C.SAVE_DIR = ''
__C.WORKERS = 4
__C.RNN_TYPE = 'LSTM'   # 'GRU'
__C.CUDA = True

__C.TREE = edict()
__C.TREE.BRANCH_NUM = 3
__C.TREE.BASE_SIZE = 64

# Training options
__C.TRAIN = edict()
__C.TRAIN.USE_ATTR = True
__C.TRAIN.USE_UNPAIR = True
__C.TRAIN.USE_CLASS = True
__C.TRAIN.CLASS_NUM = 200

__C.TRAIN.NET_E = ''
__C.TRAIN.NET_G = ''
__C.TRAIN.BATCH_SIZE = 64
__C.TRAIN.MAX_EPOCH = 600
__C.TRAIN.WARMUP_EPOCHS = 200
__C.TRAIN.GSAVE_INTERVAL = 10
__C.TRAIN.DSAVE_INTERVAL = 10

# learning rate
__C.TRAIN.DISCRIMINATOR_LR = 2e-4
__C.TRAIN.GENERATOR_LR = 2e-4
__C.TRAIN.ENCODER_LR = 2e-4
__C.TRAIN.RNN_GRAD_CLIP = 0.25

# hyper-parameters
__C.TRAIN.SMOOTH = edict()
__C.TRAIN.SMOOTH.GAMMA1 = 5.0
__C.TRAIN.SMOOTH.GAMMA3 = 10.0
__C.TRAIN.SMOOTH.GAMMA2 = 5.0
__C.TRAIN.SMOOTH.LAMBDA = 1.0

# Modal options
__C.GAN = edict()
__C.GAN.GF_DIM = 64
__C.GAN.DF_DIM = 64
__C.GAN.Z_DIM = 100
__C.GAN.CONDITION_DIM = 100

# text embedding setting
__C.TEXT = edict()
__C.TEXT.MAX_ATTR_NUM = 3
__C.TEXT.MAX_ATTR_LEN = 5
__C.TEXT.CAPTIONS_PER_IMAGE = 10
__C.TEXT.EMBEDDING_DIM = 256
__C.TEXT.WORDS_NUM = 20
__C.TEXT.DAMSM_NAME = ''
__C.TEXT.SIM_DAMSM_NAME = ''

def _merge_a_into_b(a, b):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    if type(a) is not edict:
        return

    for k, v in a.items():
        # a must specify keys that are in b
        if k not in b:
            raise KeyError('{} is not a valid config key'.format(k))

        # the types must match, too
        old_type = type(b[k])
        if old_type is not type(v):
            if isinstance(b[k], np.ndarray):
                v = np.array(v, dtype=b[k].dtype)
            else:
                raise ValueError(('Type mismatch ({} vs. {}) '
                                  'for config key: {}').format(type(b[k]),
                                                               type(v), k))

        # recursively merge dicts
        if type(v) is edict:
            try:
                _merge_a_into_b(a[k], b[k])
            except:
                print('Error under config key: {}'.format(k))
                raise
        else:
            b[k] = v


def cfg_from_file(filename):
    """Load a config file and merge it into the default options."""
    import yaml
    with open(filename, 'r') as f:
        yaml_cfg = edict(yaml.load(f))

    _merge_a_into_b(yaml_cfg, __C)
