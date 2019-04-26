from librarys import *
from config import *

# if isMemoryEnough:
#
#     class Data(Dataset):
#
#         def __init__(self, dcm_series, labels, transform=None):
#             self.transorm = transform
#             # self.df = df
#             self.dcm_series = dcm_series
#             self.labels = labels
#             self.length = len(self.dcm_series)
#             self.X = np.zeros([self.length, 1, 512, 512])
#             self.Y = np.zeros([self.length, 1, 512, 512])
#
#             for i in range(self.length):
#                 self.X[i, 0, ...] = np.clip((itk_read(
#                     (self.dcm_series.iloc[i])) - MIN_BOUND) / (MAX_BOUND - MIN_BOUND), 0.,
#                         1.)
#                 self.Y[i, 0, ...] = np.array(read_mask(self.labels.ix[i]))
#             self.X = torch.FloatTensor(self.X)
#             self.Y = torch.FloatTensor(self.Y)/255
#
#         def __len__(self):
#             return self.length
#
#         def __getitem__(self, item):
#             X = self.X[item, 0, ...]
#             y = self.Y[item, 0, ...]
#
#
#             if self.transorm is not None:
#
#                 seed = np.random.rand()
#                 random.seed(seed)
#                 X = self.transorm(X)
#                 random.seed(seed)
#                 y = self.transorm(y)
#
#             return X, y

# else:
class Data(Dataset):

    def __init__(self, dcm_series, labels, transform=None):
        self.transorm = transform
        # self.df = df
        self.dcm_series = dcm_series
        self.labels = labels

    def __len__(self):
        return len(self.dcm_series)

    def __getitem__(self, item):
        X = torch.FloatTensor(np.expand_dims(np.clip((itk_read(
            (self.dcm_series[item])) - MIN_BOUND) / (MAX_BOUND - MIN_BOUND), 0.,
                                                     1.), 0))

        y = torch.FloatTensor(np.expand_dims(
            np.array(read_mask(self.labels[item])) / 255, 0))

        if self.transorm is not None:
            seed = np.random.rand()
            random.seed(seed)
            X = self.transorm(X)
            random.seed(seed)
            y = self.transorm(y)

        return X, y