#!/usr/bin/python
# -*- coding: utf-8 -*-

import torch
import numpy as np
import random


'''
1.训练loss的选择
2.net的选择
3.是否进行数据增强
4。lr选择
5.
'''
# EPSILON = 1e-5
transform = None
DATA_PATH = "/home/zhang/Downloads/4040967758_mathcoder3/B题-全部数据/"
# DATA_PATH = "/home/a117/B题-全部数据/"
PATH_MODEL_TEST = '/home/zhang/下载/B题示例数据/lovasz_hinge 0.75.h5'
PATH_MODEL_SAVE = 'model.h5'

SHUFFLE = TRUE



TRAIN = True
TRAIN_CONTINUE = False

BATCH_SIZE = 4
MIN_BOUND = 0.0
MAX_BOUND = 2000.0


REPRODUCT = True
SEED = 2019


if REPRODUCT:
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(SEED)
    # torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True
