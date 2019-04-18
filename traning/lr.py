#!/usr/bin/env python
# coding=utf-8

from dependencies import *


def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def get_learning_rate(optimizer):
    lr = []
    for param_group in optimizer.param_groups:
        lr += [param_group['lr']]

    assert (len(lr) == 1)  # we support only one param_group
    lr = lr[0]

    return lr

if __name__ == '__main__':
    lr = []
    # epoch = 1
    scheduler = lambda x: (0.01 / 2) * (np.cos(PI * (np.mod(x - 1, 10*797) / (10*797))) + 1)
    for i in range(50*797):
        # print(scheduler(i+1))
        lr.append(scheduler(i+1))
    import matplotlib.pyplot as plt
    plt.plot(lr)
    plt.show()