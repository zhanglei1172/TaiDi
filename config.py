#!/usr/bin/python
# -*- coding: utf-8 -*-



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
PATH_MODEL_TEST = '/home/zhang/下载/model(5).h5'
PATH_CHECKPOINT = './'
PATH_MODEL_BEST = '_model.pth'

SHUFFLE = True



TRAIN = True
initial_checkpoint = None

BATCH_SIZE = 2
MIN_BOUND = 0.0
MAX_BOUND = 2000.0


REPRODUCT = True




