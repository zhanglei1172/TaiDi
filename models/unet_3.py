#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch.autograd import Variable
# from itertools import filterfalse as ifilterfalse
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# import matplotlib.pyplot as plt
# import numpy as np

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
