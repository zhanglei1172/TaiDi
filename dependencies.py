import pydicom
from itertools import filterfalse as ifilterfalse
from torch.autograd import Variable
from torchvision import transforms
import tqdm
from collections import defaultdict
import time
import copy
from torch.utils.data import Dataset, DataLoader
from torch.optim import lr_scheduler
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import numpy as np
import random
import cv2
import os
import glob
import SimpleITK as sitk
import pandas as pd
from datasets.preprocessing import *
from traning.loss import *
from datasets.data import *
from config import *
from traning.metrics import *
EPS = 1e-12
PI  = np.pi
INF = np.inf
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
#
#
