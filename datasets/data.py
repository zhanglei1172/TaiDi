from dependencies import *


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
            np.array(read_mask(self.labels[item]))/255, 0))

        if self.transorm is not None:

            seed = np.random.rand()
            random.seed(seed)
            X = self.transorm(X)
            random.seed(seed)
            y = self.transorm(y)

        return X, y
