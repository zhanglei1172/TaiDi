#!/usr/bin/env python
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from preprocessing import *
import numpy as np
import seaborn as sns

if __name__ == '__main__':

    df = process_data("数据集1/*/*/*.dcm")
    idxs = []
    array = []
    # count = 0
    for idx, sample in enumerate(df.iterrows()):
        dcm = itk_read(sample[1]['path_dcm'])
        # img = cv2.imread(sample[1]['path_mask'], cv2.IMREAD_GRAYSCALE)
        # if img.any():
            # count += 1
            # idxs.append(idx)
            # dcm = itk_read(sample[1]['path_dcm'])
        # array.append(dcm)
        array.append(dcm)
    tmp = np.concatenate(array, None)
    plt.hist(tmp)
    plt.show()
    sns.distplot(np.concatenate(array, None))
    # print(count)



    # df = process_data("数据集1/*/*/*.dcm")
    # # idxs = []
    # array = []
    # for idx, sample in enumerate(df.iterrows()):
    #     img = cv2.imread(sample[1]['path_mask'], cv2.IMREAD_GRAYSCALE)
    #     array.append(img)
    # temp = np.concatenate(array, None)
    # plt.hist(temp)
    # plt.show()
    # temp //= 255
    # print((temp.size-temp.sum())/temp.sum())
    #
    # df = process_data("数据集1/*/*/*.dcm")
    # # idxs = []
    # array = []
    # for idx, sample in enumerate(df.iterrows()):
    #     img = cv2.imread(sample[1]['path_mask'], cv2.IMREAD_GRAYSCALE)
    #     if img.any():
    #         # idxs.append(idx)
    #         dcm = itk_read(sample[1]['path_dcm'])
    #         plt.imshow(np.logical_and(dcm>-50, dcm<100))
    #         plt.imshow(dcm, plt.cm.bone)
    #         plt.show()
    #         array.append(dcm[np.nonzero(dcm * img)] )
    # plt.hist(np.concatenate(array, None))
    # plt.show()
    pass