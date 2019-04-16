#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from itertools import filterfalse as ifilterfalse
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import matplotlib.pyplot as plt


class ConvBlock(nn.Module):

    def __init__(self, in_chan, out_chan):

        super(ConvBlock, self).__init__()
        self.down = nn.Sequential(
            nn.Conv2d(in_chan, out_chan, 3, padding=1),
            # nn.BatchNorm2d(out_chan),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_chan, out_chan, 3, padding=1),
            # nn.BatchNorm2d(out_chan),
            nn.ReLU(inplace=True),

        )
        # self.conv1_3x3 = nn.Conv2d(in_chan, out_chan, 3, padding=1)
        # self.conv2_3x3 = nn.Conv2d(out_chan, out_chan, 3, padding=1)
        #
        # if bn:
        #     self.bn1 = nn.BatchNorm2d(out_chan)
        #     self.bn2 = nn.BatchNorm2d(out_chan)
        #
        # self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        # returns the block output and the shortcut to use in the uppooling blocks
        x = self.down(x)
        return F.max_pool2d(x), x


class MiddleBlock(nn.Module):

    def __init__(self, in_chan, out_chan):
        super(MiddleBlock, self).__init__()

        self.down = nn.Sequential(
            nn.Conv2d(in_chan, out_chan, 3, padding=1),
            # nn.BatchNorm2d(out_chan),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_chan, out_chan, 3, padding=1),
            # nn.BatchNorm2d(out_chan),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.down(x)
        return x


class DeconvBlock(nn.Module):

    def __init__(self, in_chan, out_chan):
        super(DeconvBlock, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = nn.Sequential(
            nn.Conv2d(in_chan, out_chan, 3, padding=1),
            # nn.BatchNorm2d(out_chan),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_chan, out_chan, 3, padding=1),
            # nn.BatchNorm2d(out_chan),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, shortcut):
        x = self.up(x)
        x = torch.cat([x, shortcut], dim=1)
        x = self.conv(x)
        return x


class Net(nn.Module):
    def __init__(self, start_neurons=32):
        super(Net, self).__init__()
        self.down1 = ConvBlock(1, start_neurons)
        self.down2 = ConvBlock(start_neurons, start_neurons*2)
        self.down3 = ConvBlock(start_neurons*2, start_neurons*4)
        self.down4 = ConvBlock(start_neurons*4, start_neurons*8)

        self.middle = MiddleBlock(start_neurons*8, start_neurons*16)

        self.up4 = DeconvBlock(start_neurons*16, start_neurons*8)
        self.up3 = DeconvBlock(start_neurons * 8, start_neurons * 4)
        self.up2 = DeconvBlock(start_neurons * 4, start_neurons * 2)
        self.up1 = DeconvBlock(start_neurons * 2, start_neurons)

        self.out = nn.Conv2d(start_neurons, 1, 1, padding=1)

    def forward(self, x):
        x, shortcut1 = self.down1(x)
        # 512 -> 256
        x, shortcut2 = self.down2(x)
        # 256 -> 128
        x, shortcut3 = self.down3(x)
        # 128 -> 64
        x, shortcut4 = self.down4(x)
        # 64 -> 32
        x = self.middle(x)

        x = self.up4(x, shortcut4)
        # 32 -> 64
        x = self.up3(x, shortcut3)
        x = self.up2(x, shortcut2)
        x = self.up1(x, shortcut1)

        x = self.out(x)
        self.reset_params()
        # return logits
        return x

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0.)

    def reset_params(self):
        for m in self.modules():
            self.weight_init(m)


# def build_model(start_neurons, bn=False, dropout=None):
#     input_layer = Input((128, 128, 1))
#     # 128 -> 64
#     conv1, shortcut1 = conv_block(start_neurons, input_layer, bn, dropout)
#     # 64 -> 32
#     conv2, shortcut2 = conv_block(start_neurons * 2, conv1, bn, dropout)
#     # 32 -> 16
#     conv3, shortcut3 = conv_block(start_neurons * 4, conv2, bn, dropout)
#     # 16 -> 8
#     conv4, shortcut4 = conv_block(start_neurons * 8, conv3, bn, dropout)
#     # Middle
#     convm = middle_block(start_neurons * 16, conv4, bn, dropout)
#     # 8 -> 16
#     deconv4 = deconv_block(start_neurons * 8, convm, shortcut4, bn, dropout)
#     # 16 -> 32
#     deconv3 = deconv_block(start_neurons * 4, deconv4, shortcut3, bn, dropout)
#     # 32 -> 64
#     deconv2 = deconv_block(start_neurons * 2, deconv3, shortcut2, bn, dropout)
#     # 64 -> 128
#     deconv1 = deconv_block(start_neurons, deconv2, shortcut1, bn, dropout)
#     # uconv1 = Dropout(0.5)(uconv1)
#     output_layer = nn.Conv2d(1, (1, 1), padding="same", activation="sigmoid")(deconv1)
#     model = Model(input_layer, output_layer)
#
#     return model
class ConvBn2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(ConvBn2d, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class sSE(nn.Module):
    def __init__(self, out_channels):
        super(sSE, self).__init__()
        self.conv = ConvBn2d(in_channels=out_channels,
                             out_channels=1,
                             kernel_size=1,
                             padding=0)
    def forward(self,x):
        x=self.conv(x)
        #print('spatial',x.size())
        x=F.sigmoid(x)
        return x

class cSE(nn.Module):
    def __init__(self, out_channels):
        super(cSE, self).__init__()
        self.conv1 = ConvBn2d(in_channels=out_channels,
                              out_channels=int(out_channels/2),
                              kernel_size=1,
                              padding=0)
        self.conv2 = ConvBn2d(in_channels=int(out_channels/2),
                              out_channels=out_channels,
                              kernel_size=1,
                              padding=0)
    def forward(self,x):
        x=nn.AvgPool2d(x.size()[2:])(x)
        #print('channel',x.size())
        x=self.conv1(x)
        x=F.relu(x)
        x=self.conv2(x)
        x=F.sigmoid(x)
        return x

class Decoder(nn.Module):
    def __init__(self, in_channels, channels, out_channels):
        super(Decoder, self).__init__()
        self.conv1 = ConvBn2d(in_channels, channels,
                              kernel_size=3, padding=1)
        self.conv2 = ConvBn2d(channels, out_channels,
                              kernel_size=3, padding=1)
        self.spatial_gate = sSE(out_channels)
        self.channel_gate = cSE(out_channels)

    def forward(self, x, e=None):
        x = F.upsample(x, scale_factor=2, mode='bilinear', align_corners=True)
        #print('x',x.size())
        #print('e',e.size())
        if e is not None:
            x = torch.cat([x,e],1)

        x = F.relu(self.conv1(x),inplace=True)
        x = F.relu(self.conv2(x),inplace=True)
        #print('x_new',x.size())
        g1 = self.spatial_gate(x)
        #print('g1',g1.size())
        g2 = self.channel_gate(x)
        #print('g2',g2.size())
        x = g1*x + g2*x
        return x



from torchvision import models


class UNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.resnet = models.resnet34(pretrained=True)

        self.conv1 = nn.Sequential(
            self.resnet.conv1,
            self.resnet.bn1,
            self.resnet.relu,
        )

        self.encoder2 = self.resnet.layer1  # 64
        self.encoder3 = self.resnet.layer2  # 128
        self.encoder4 = self.resnet.layer3  # 256
        self.encoder5 = self.resnet.layer4  # 512

        self.center = nn.Sequential(
            ConvBn2d(512, 512),
            nn.ReLU(inplace=True),
            ConvBn2d(512, 256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.decoder5 = Decoder(256 + 512, 512, 64)
        self.decoder4 = Decoder(64 + 256, 256, 64)
        self.decoder3 = Decoder(64 + 128, 128, 64)
        self.decoder2 = Decoder(64 + 64, 64, 64)
        self.decoder1 = Decoder(64, 32, 64)

        self.logit = nn.Sequential(
            nn.Conv2d(384, 64, kernel_size=3, padding=1),
            nn.ELU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1, padding=0),
        )


    def forward(self, x):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        x = torch.cat([
            (x - mean[2]) / std[2],
            (x - mean[1]) / std[1],
            (x - mean[0]) / std[0],
        ], 1)

        e1 = self.conv1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        e5 = self.encoder5(e4)

        f = self.center(e5)
        d5 = self.decoder5(f, e5)
        d4 = self.decoder4(d5, e4)
        d3 = self.decoder3(d4, e3)
        d2 = self.decoder2(d3, e2)
        d1 = self.decoder1(d2)

        f = torch.cat((
            F.upsample(e1, scale_factor=2,
                       mode='bilinear', align_corners=False),
            d1,
            F.upsample(d2, scale_factor=2,
                       mode='bilinear', align_corners=False),
            F.upsample(d3, scale_factor=4,
                       mode='bilinear', align_corners=False),
            F.upsample(d4, scale_factor=8,
                       mode='bilinear', align_corners=False),
            F.upsample(d5, scale_factor=16,
                       mode='bilinear', align_corners=False),
        ), 1)

        f = F.dropout2d(f, p=0.50)
        logit = self.logit(f)

        return logit



class FocalLoss2d(nn.Module):
    def __init__(self, gamma=2, size_average=True):
        super(FocalLoss2d, self).__init__()
        self.gamma = gamma
        self.size_average = size_average

    def forward(self, logit, target, metrics, class_weight=None, type='sigmoid'):
        target = target.view(-1, 1).long()
        if type == 'sigmoid':
            if class_weight is None:
                class_weight = [1] * 2  # [0.5, 0.5]
            prob = F.sigmoid(logit)
            prob = prob.view(-1, 1)
            prob = torch.cat((1 - prob, prob), 1)
            select = torch.FloatTensor(len(prob), 2).zero_().cuda()
            select.scatter_(1, target, 1.)
        elif type == 'softmax':
            B, C, H, W = logit.size()
            if class_weight is None:
                class_weight = [1] * C  # [1/C]*C
            logit = logit.permute(0, 2, 3, 1).contiguous().view(-1, C)
            prob = F.softmax(logit, 1)
            select = torch.FloatTensor(len(prob), C).zero_().cuda()
            select.scatter_(1, target, 1.)
        class_weight = torch.FloatTensor(class_weight).cuda().view(-1, 1)
        class_weight = torch.gather(class_weight, 0, target)
        prob = (prob * select).sum(1).view(-1, 1)
        prob = torch.clamp(prob, 1e-8, 1 - 1e-8)
        batch_loss = - class_weight * (torch.pow((1 - prob), self.gamma)) * prob.log()
        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss
        # metrics['bce'] += bce.detach().item() * target.size(0)
        # metrics['dice'] += dice.detach().item() * target.size(0)
        metrics['loss'] += loss.detach().item() * target.size(0)
        return loss

def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1: # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard

def isnan(x):
    return x != x


def mean(l, ignore_nan=False, empty=0):
    """
    nanmean compatible with generators.
    """
    l = iter(l)
    if ignore_nan:
        l = ifilterfalse(isnan, l)
    try:
        n = 1
        acc = next(l)
    except StopIteration:
        if empty == 'raise':
            raise ValueError('Empty mean')
        return empty
    for n, v in enumerate(l, 2):
        acc += v
    if n == 1:
        return acc
    return acc / n


def lovasz_hinge(logits, labels, metrics, per_image=True, ignore=None):
    """
    Binary Lovasz hinge loss
      logits: [B, H, W] Variable, logits at each pixel (between -\infty and +\infty)
      labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
      per_image: compute the loss per image instead of per batch
      ignore: void class id
    """
    if per_image:
        loss = mean(lovasz_hinge_flat(*flatten_binary_scores(log.unsqueeze(0), lab.unsqueeze(0), ignore))
                          for log, lab in zip(logits, labels))
    else:
        loss = lovasz_hinge_flat(*flatten_binary_scores(logits, labels, ignore))
    metrics['loss'] += loss.detach().item() * labels.size(0)
    return loss


def lovasz_hinge_flat(logits, labels):
    """
    Binary Lovasz hinge loss
      logits: [P] Variable, logits at each prediction (between -\infty and +\infty)
      labels: [P] Tensor, binary ground truth labels (0 or 1)
      ignore: label to ignore
    """
    if len(labels) == 0:
        # only void pixels, the gradients should be 0
        return logits.sum() * 0.
    signs = 2. * labels.float() - 1.
    errors = (1. - logits * Variable(signs))
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    perm = perm.data
    gt_sorted = labels[perm]
    grad = lovasz_grad(gt_sorted)
    loss = torch.dot(F.relu(errors_sorted), Variable(grad))
    return loss


def flatten_binary_scores(scores, labels, ignore=None):
    """
    Flattens predictions in the batch (binary case)
    Remove labels equal to 'ignore'
    """
    scores = scores.view(-1)
    labels = labels.view(-1)
    if ignore is None:
        return scores, labels
    valid = (labels != ignore)
    vscores = scores[valid]
    vlabels = labels[valid]
    return vscores, vlabels





# metrc

def dice_loss(pred, target, smooth=1.):
    # pred = pred.contiguous()
    # target = target.contiguous()

    intersection = (pred * target).sum(dim=2).sum(dim=2)

    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))

    return loss.mean()


def calc_loss(pred, target, metrics, bce_weight=.5):
    class_weight = torch.FloatTensor([1, 1]) #### TODO 1:10?
    bce = F.binary_cross_entropy(pred, target, weight=class_weight[target.long()].to(device))

    # pred = F.sigmoid(pred)
    dice = dice_loss(pred, target)

    loss = bce * bce_weight + dice * (1 - bce_weight)

    metrics['bce'] += bce.detach().item() * target.size(0)
    metrics['dice'] += dice.detach().item() * target.size(0)
    metrics['loss'] += loss.detach().item() * target.size(0)

    return loss

def print_metrics(metrics, epoch_samples, phase):
    outputs = []
    for k in metrics.keys():
        outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))
    print("{}: {}".format(phase, ", ".join(outputs)))
#### test-demo
def plot(outputs, labels):
    # import matplotlib.pyplot as plt
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
        # plt.show()
        plt.savefig('%s.png' % i)



