import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models

models.resnet18()


class Block(nn.Module):

    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ch_in, ch_in // 2, 1),  # 降维
            nn.BatchNorm2d(ch_in // 2),
            nn.ReLU(inplace=True),

            nn.Conv2d(ch_in // 2, ch_in // 2, 3, stride=1, padding=1),  # feature 提取
            nn.BatchNorm2d(ch_in // 2),
            nn.ReLU(inplace=True),

            nn.Conv2d(ch_in // 2, ch_out, 1),  # 升维
            nn.BatchNorm2d(ch_out)
        )
        self.extra = nn.Sequential()
        if ch_in != ch_out:
            self.extra = nn.Conv2d(ch_in, ch_out, 3, stride=1, padding=1)

    def forward(self, x):
        x = F.relu(self.block(x) + self.extra(x), inplace=True)
        return x


class Down(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            Block(ch_in, ch_out)
        )

    def forward(self, x):
        x = self.mpconv(x)

        return x


class Up(nn.Module):

    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = Block(ch_in, ch_out)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class Outconv(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(Outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x


class ResNetUNet(nn.Module):

    def __init__(self, ch_in):
        super().__init__()

        # Use ResNet18 as the encoder with the pretrained weights
        # self.base_model = models.resnet18(pretrained=True)
        # self.base_layers = list(self.base_model.children())
        self.inconv = nn.Sequential(
            nn.Conv2d(ch_in, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        self.down4 = Down(256, 256)

        self.up4 = Up(256 + 256, 128)
        self.up3 = Up(128 + 128, 64)
        self.up2 = Up(64 + 64, 32)
        self.up1 = Up(32 + 32, 32)
        self.outc = Outconv(32, 1)
        self.reset_params()

    def forward(self, x):
        x1 = self.inconv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up4(x5, x4)
        x = self.up3(x, x3)
        x = self.up2(x, x2)
        x = self.up1(x, x1)
        x = self.outc(x)
        return F.sigmoid(x)

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_normal(m.weight)
            nn.init.constant(m.bias, 0)

    def reset_params(self):
        for i, m in enumerate(self.modules()):
            self.weight_init(m)

#
# class SaveFeatures:
#
#     features = None
#
#     def __init__(self, m): self.hook = m.register_forward_hook(self.hook_fn)
#
#     def hook_fn(self, module, input, output): self.features = output
#
#     def remove(self): self.hook.remove()
#
#
# class UnetBlock(nn.Module):
#     def __init__(self, up_in, down_in, n_out, dp=False, ps=0.25):
#         super().__init__()
#         up_out = down_out = n_out // 2
#         self.tr_conv = nn.ConvTranspose2d(up_in, up_out, 2, 2, bias=False)
#         self.conv = nn.Conv2d(down_in, down_out, 1, bias=False)
#         self.bn = nn.BatchNorm2d(n_out)
#         self.dp = dp
#         if dp: self.dropout = nn.Dropout(ps, inplace=True)
#
#     def forward(self, up_x, down_x):
#         x1 = self.tr_conv(up_x)
#         x2 = self.conv(down_x)
#         x = torch.cat([x1, x2], dim=1)
#         x = self.bn(F.relu(x))
#         return self.dropout(x) if self.dp else x
#
#
# class Unet34(nn.Module):
#
#     def __init__(self, rn, drop_i=False, ps_i=None, drop_up=False, ps=None):
#
#         super().__init__()
#         self.rn = rn
#         self.sfs = [SaveFeatures(rn[i]) for i in [2, 4, 5, 6]]
#         self.drop_i = drop_i
#         if drop_i:
#             self.dropout = nn.Dropout(ps_i, inplace=True)
#         if ps_i is None: ps_i = 0.1
#         if ps is not None: assert len(ps) == 4
#         if ps is None: ps = [0.1] * 4
#         self.up1 = UnetBlock(512, 256, 256, drop_up, ps[0])
#         self.up2 = UnetBlock(256, 128, 256, drop_up, ps[1])
#         self.up3 = UnetBlock(256, 64, 256, drop_up, ps[2])
#         self.up4 = UnetBlock(256, 64, 256, drop_up, ps[3])
#         self.up5 = nn.ConvTranspose2d(256, 1, 2, 2)
#
#     def forward(self, x):
#         x = F.relu(self.rn(x))
#         x = self.dropout(x) if self.drop_i else x
#         x = self.up1(x, self.sfs[3].features)
#         x = self.up2(x, self.sfs[2].features)
#         x = self.up3(x, self.sfs[1].features)
#         x = self.up4(x, self.sfs[0].features)
#         x = self.up5(x)
#         return x[:, 0]
#
#     def close(self):
#         for o in self.sfs: o.remove()
