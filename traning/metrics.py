#!/usr/bin/env python
# coding=utf-8
from dependencies import *


def print_metrics(metrics, epoch_samples, phase):
    outputs = []
    for k in metrics.keys():
        outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))

    print("{}: {}".format(phase, ", ".join(outputs)))
#### test-demo


def plot(outputs, labels):
    import matplotlib.pyplot as plt
    fig = plt.figure()
    a1 = fig.add_subplot(121)
    a2 = fig.add_subplot(122)
    a1.imshow((labels).cpu().squeeze())
    a2.imshow((outputs>0.5).cpu().squeeze())
    plt.show()
####
def cal_dice(logits, labels):
    '''

    :param logits:
    :param labels:
    :return: batchsize个image的dice之和/batchsize -> 单个image平均的dice
    '''
    dice = 0
    for i in range(labels.size(0)):
        logit = logits[i].detach()
        label = labels[i].detach()
        prob = F.sigmoid(logit)
        x1 = (prob > 0.5).cpu().squeeze().numpy()
        x2 = (label).cpu().squeeze().numpy()
        intersection = (x1*x2).sum()
        if  x1.any() or  x2.any():
            dice += ((2. * intersection) / (x1.sum()+x2.sum()))
        else:
            dice += 1
    return dice/labels.size(0)

def show(inputs, logits, labels):
    import matplotlib.pyplot as plt
    for i in range(labels.size(0)):
        logit = logits[i].detach()
        label = labels[i].detach().cpu().squeeze().numpy()
        input_ = inputs[i].detach().cpu().squeeze().numpy()
        prob = F.sigmoid(logit).cpu().squeeze().numpy()

        fig = plt.figure(i)
        fig.add_subplot(141)
        plt.imshow(input_)
        plt.title('原图')

        fig.add_subplot(142)
        plt.imshow(prob)
        plt.title('prob')

        fig.add_subplot(143)
        plt.imshow(prob > 0.5)
        plt.title('prob>0.5')

        fig.add_subplot(144)
        plt.imshow(label)
        plt.title('label')
        plt.show()
        # plt.savefig('%s.png' % i)

def observe_grad(net):
    for i, m in enumerate(net.modules()):
        if isinstance(m, nn.Conv2d):
            print(m.weight.grad.norm(2))