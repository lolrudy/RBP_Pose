# !/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    model.py
# @Project:     GuideNet
# @Author:      jie
# @Time:        2021/3/14 7:50 PM

import torch
import torch.nn as nn
from scipy.stats import truncnorm
import math
from torch.autograd import Function
import GuideConv  # install from the authors code
import torch.nn.functional as F

__all__ = [
    'GN',
    'GNS',
]


def Conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def Conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


class Basic2d(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer=None, kernel_size=3, padding=1):
        super().__init__()
        if norm_layer:
            conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                             stride=1, padding=padding, bias=False)
        else:
            conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                             stride=1, padding=padding, bias=True)
        self.conv = nn.Sequential(conv, )
        if norm_layer:
            self.conv.add_module('bn', norm_layer(out_channels))
        self.conv.add_module('relu', nn.ReLU(inplace=True))

    def forward(self, x):
        out = self.conv(x)
        return out


class Basic2dTrans(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer=None):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.conv = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3,
                                       stride=2, padding=1, output_padding=1, bias=False)
        self.bn = norm_layer(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        return out


class Guide(nn.Module):

    def __init__(self, input_planes, weight_planes, norm_layer=None, weight_ks=3):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.conv11 = Basic2d(input_planes + weight_planes, input_planes, None)
        self.br = nn.Sequential(
            norm_layer(num_features=input_planes),
            nn.ReLU(inplace=True),
        )
        self.conv3 = Basic2d(input_planes, input_planes, norm_layer)

    def forward(self, input, weight):
        weight = torch.cat([input, weight], 1)
        out = self.conv11(weight)
        out = self.br(out)
        out = self.conv3(out)
        return out


class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=None, act=True):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.conv1 = Conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = Conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
        self.act = act

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        if self.act:
            out = self.relu(out)
        return out


class GuideNet(nn.Module):
    """
    Not activate at the ref
    Init change to trunctated norm
    """
    # basic2d: conv+norm
    def __init__(self, block=BasicBlock, bc=16, img_layers=[2, 2, 2, 2, 2],
                 depth_layers=[2, 2, 2, 2, 2], norm_layer=nn.BatchNorm2d, guide=Guide, weight_ks=3, output_channel=32):
        super(GuideNet, self).__init__()
        self._norm_layer = norm_layer
        # here padding is determined by the input image. In our settings, the input image is
        # 3x256x256
        self.conv_img = Basic2d(3, bc * 2, norm_layer=norm_layer, kernel_size=5, padding=2)
        in_channels = bc * 2
        self.inplanes = in_channels
        self.layer1_img = self._make_layer(block, in_channels * 2, img_layers[0], stride=2)
        # 2 3x3 conv
        self.guide1 = guide(in_channels * 2, in_channels * 2, norm_layer, weight_ks)
        self.inplanes = in_channels * 2 * block.expansion
        self.layer2_img = self._make_layer(block, in_channels * 4, img_layers[1], stride=2)

        self.guide2 = guide(in_channels * 4, in_channels * 4, norm_layer, weight_ks)
        self.inplanes = in_channels * 4 * block.expansion
        self.layer3_img = self._make_layer(block, in_channels * 8, img_layers[2], stride=2)

        self.guide3 = guide(in_channels * 8, in_channels * 8, norm_layer, weight_ks)
        self.inplanes = in_channels * 8 * block.expansion
        self.layer4_img = self._make_layer(block, in_channels * 8, img_layers[3], stride=2)

        self.guide4 = guide(in_channels * 8, in_channels * 8, norm_layer, weight_ks)
        self.inplanes = in_channels * 8 * block.expansion
        self.layer5_img = self._make_layer(block, in_channels * 8, img_layers[4], stride=2)

        self.layer1d_img = Basic2dTrans(in_channels * 2, in_channels, norm_layer)
        self.layer2d_img = Basic2dTrans(in_channels * 4, in_channels * 2, norm_layer)
        self.layer3d_img = Basic2dTrans(in_channels * 8, in_channels * 4, norm_layer)
        self.layer4d_img = Basic2dTrans(in_channels * 8, in_channels * 8, norm_layer)
        self.layer5d_img = Basic2dTrans(in_channels * 8, in_channels * 8, norm_layer)

        # Construct output layer, for object mask detection
        self.mask_feature = block(bc * 2, bc * 2, norm_layer=norm_layer, act=False)
        # self.obj_mask = nn.Conv2d(bc * 2, 1, kernel_size=3, stride=1, padding=1)

        self.conv_lidar = Basic2d(1, bc * 2, norm_layer=None, kernel_size=5, padding=2)

        self.inplanes = in_channels
        self.layer1_lidar = self._make_layer(block, in_channels * 2, depth_layers[0], stride=2)
        self.inplanes = in_channels * 2 * block.expansion
        self.layer2_lidar = self._make_layer(block, in_channels * 4, depth_layers[1], stride=2)
        self.inplanes = in_channels * 4 * block.expansion
        self.layer3_lidar = self._make_layer(block, in_channels * 8, depth_layers[2], stride=2)
        self.inplanes = in_channels * 8 * block.expansion
        self.layer4_lidar = self._make_layer(block, in_channels * 8, depth_layers[3], stride=2)
        self.inplanes = in_channels * 8 * block.expansion
        self.layer5_lidar = self._make_layer(block, in_channels * 8, depth_layers[4], stride=2)

        self.layer1d = Basic2dTrans(in_channels * 2, in_channels, norm_layer)
        self.layer2d = Basic2dTrans(in_channels * 4, in_channels * 2, norm_layer)
        self.layer3d = Basic2dTrans(in_channels * 8, in_channels * 4, norm_layer)
        self.layer4d = Basic2dTrans(in_channels * 8, in_channels * 8, norm_layer)
        self.layer5d = Basic2dTrans(in_channels * 8, in_channels * 8, norm_layer)

        self.conv_mask = nn.Conv2d(bc * 2, 2, kernel_size=3, stride=1, padding=1)   # generate mask, 2 class, bg and fg
        self.log_softmax = nn.LogSoftmax()

        self.ref_mask = block(bc * 2, bc * 2, norm_layer=norm_layer, act=False)

        self.conv_seman = nn.Conv2d(bc * 2, output_channel, kernel_size=3, stride=1, padding=1)
        self.ref_seman = block(bc * 2, bc * 2, norm_layer=norm_layer, act=False)

        # self._initialize_weights()

    def forward(self, img, lidar):
        c0_img = self.conv_img(img)
        c1_img = self.layer1_img(c0_img)
        c2_img = self.layer2_img(c1_img)
        c3_img = self.layer3_img(c2_img)
        c4_img = self.layer4_img(c3_img)
        c5_img = self.layer5_img(c4_img)
        # final RGB_feature_getted

        dc5_img = self.layer5d_img(c5_img)
        c4_mix = dc5_img + c4_img
        dc4_img = self.layer4d_img(c4_mix)
        c3_mix = dc4_img + c3_img
        dc3_img = self.layer3d_img(c3_mix)
        c2_mix = dc3_img + c2_img
        dc2_img = self.layer2d_img(c2_mix)
        c1_mix = dc2_img + c1_img
        # dc1_img = self.layer1d_img(c1_mix)
        # c0_mix = c0_img + dc1_img
        # obj_mask_feature = self.mask_feature(c0_mix)
        # obj_mask = self.obj_mask(obj_mask_feature)

        c0_lidar = self.conv_lidar(lidar)
        c1_lidar = self.layer1_lidar(c0_lidar)
        c1_lidar_dyn = self.guide1(c1_lidar, c1_mix)
        c2_lidar = self.layer2_lidar(c1_lidar_dyn)
        c2_lidar_dyn = self.guide2(c2_lidar, c2_mix)
        c3_lidar = self.layer3_lidar(c2_lidar_dyn)
        c3_lidar_dyn = self.guide3(c3_lidar, c3_mix)
        c4_lidar = self.layer4_lidar(c3_lidar_dyn)
        c4_lidar_dyn = self.guide4(c4_lidar, c4_mix)
        c5_lidar = self.layer5_lidar(c4_lidar_dyn)
        # final Depth_feature_getted


        c5 = c5_img + c5_lidar
        dc5 = self.layer5d(c5)
        c4 = dc5 + c4_lidar_dyn
        dc4 = self.layer4d(c4)
        c3 = dc4 + c3_lidar_dyn
        dc3 = self.layer3d(c3)
        c2 = dc3 + c2_lidar_dyn
        dc2 = self.layer2d(c2)
        c1 = dc2 + c1_lidar_dyn
        dc1 = self.layer1d(c1)
        c0 = dc1 + c0_lidar
        mask_feature = self.ref_mask(c0)
        mask = self.conv_mask(mask_feature)              # remember to do softmax
        # mask = self.log_softmax(mask)
        seman_feature = self.ref_seman(c0)
        seman_feature = self.conv_seman(seman_feature)   # bs x 6 x h x w
        return seman_feature, mask

    def _make_layer(self, block, planes, blocks, stride=1):
        norm_layer = self._norm_layer
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                Conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        def truncated_normal_(num, mean=0., std=1.):
            lower = -2 * std
            upper = 2 * std
            X = truncnorm((lower - mean) / std, (upper - mean) / std, loc=mean, scale=std)
            samples = X.rvs(num)
            output = torch.from_numpy(samples)
            return output

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                '''
                print(m.weight.nelement())
                if m.weight.nelement() > 10000000:
                    sdf = 1
                '''
                data = truncated_normal_(m.weight.nelement(), mean=0, std=math.sqrt(1.3 * 2. / n))
                data = data.type_as(m.weight.data)
                m.weight.data = data.view_as(m.weight.data)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


def GN():
    # return GuideNet(norm_layer=encoding.nn.SyncBatchNorm, guide=Guide)
    return GuideNet(norm_layer=nn.BatchNorm2d, guide=Guide)

def GNS(output_channel=32):
    # return GuideNet(norm_layer=encoding.nn.SyncBatchNorm, guide=Guide, weight_ks=1)
    return GuideNet(norm_layer=nn.BatchNorm2d, guide=Guide, weight_ks=1, output_channel=output_channel)

    # syncbatchnorm is for multi-gpu training. Simutanouesly normalize data on all devices