#!/usr/bin/env python
# coding=utf-8

import copy
import time
from collections import defaultdict
import tqdm
# import math
from model.resUnet import ResNetUNet
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
# from torchsummary import summary
# import numpy as np
# from tensorboardX import SummaryWriter
from model.unet_ import *
from preprocessing import *
from loss import *
from eval import *

transform = None



class Data(Dataset):

    def __init__(self, dcm_series, labels, transform=None):
        self.transorm = transform
        # self.df = df
        self.dcm_series = dcm_series
        self.labels = labels

    def __len__(self):
        return len(self.dcm_series)

    def __getitem__(self, item):

        # X = torch.FloatTensor(np.expand_dims(np.clip((get_pixel_hu(
        #     dicom.read_file(self.dcm_series[item])) - MIN_BOUND) / (MAX_BOUND - MIN_BOUND), 0.,
        #                                             1.) , 0))
        X = torch.FloatTensor(np.expand_dims(np.clip((itk_read(
            (self.dcm_series[item])) - MIN_BOUND) / (MAX_BOUND - MIN_BOUND), 0.,
                                                     1.), 0))

        y = torch.FloatTensor(np.expand_dims(np.array(read_mask(self.labels[item]))/255, 0))
        # Generate data
        # X, y = self.__data_generation(list_IDs_temp)
        if self.transorm is not None:

            seed = np.random.rand()
            random.seed(seed)
            X = self.transorm(X)
            random.seed(seed)
            y = self.transorm(y)
            # random.seed(seed)
            # X = self.transorm(X)
            # random.seed(seed)
            # y = self.transorm(y)

        return X, y

