#!/usr/bin/env python
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from preprocessing import *
import numpy as np
import seaborn as sns
from skimage import data, io, filters
import matplotlib
myfont = matplotlib.font_manager.FontProperties(fname="/usr/share/fonts/Consolas+YaHei+hybrid.ttf", size=12)
if __name__ == '__main__':
    #
    df = process_data("数据集1/*/*/*.dcm")
    # df = df[df['phase'] == 'arterial phase']
    idxs = []
    array = []
    # count = 0
    for idx, sample in enumerate(df.iterrows()):
        # dcm = itk_read(sample[1]['path_dcm'])
        # img = cv2.imread(sample[1]['path_mask'], cv2.IMREAD_GRAYSCALE)
        # if img.any():
            # count += 1
            # idxs.append(idx)
        dcm = itk_read_(sample[1]['path_dcm']).astype('int16')
        array.append(dcm)
        # array.append(dcm)
        #     plt.imshow(filters.sobel(dcm/2500))
        #     plt.show()
        #     array.append(dcm)
    tmp = np.stack(array, 0)    # tmp = np.concatenate(array, None)
    # sns.distplot(tmp.tolist())
    # sns.heatmap(tmp.mean(0))
    plt.hist2d(tmp.sum(0))
    plt.xlabel('HU', fontproperties=myfont)
    # plt.title('标记区域hu值分布', fontproperties=myfont)
    plt.savefig('HU无坐标.svg', tig=10, format='svg')

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
    # idxs = []
    # array = []
    # for idx, sample in enumerate(df.iterrows()):
    #     img = cv2.imread(sample[1]['path_mask'], cv2.IMREAD_GRAYSCALE)
    #     if img.any():
    #         # idxs.append(idx)
    #         dcm = itk_read(sample[1]['path_dcm'])
    #         # plt.imshow(np.logical_and(dcm>-50, dcm<100))
    #         # plt.imshow(dcm, plt.cm.bone)
    #         plt.show()
    #         array.append(dcm[np.nonzero(dcm * img)] )
    # plt.hist(np.concatenate(array, None))
    # plt.show()
    # pass