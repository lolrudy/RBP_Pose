from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging


import torch
import torch.nn as nn
import torch.nn.functional as F
from depth_attention_module import DAM

BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class MakeBL(nn.Module):
    def __init__(self, BasicBlock, num_branches, num_blocks, num_channels, multi_scale_output=True):
        super(MakeBL, self).__init__()
        self.BasicBlock = BasicBlock
        self.num_branches = num_branches
        self.num_blocks = num_blocks
        self.num_channels = num_channels
        self.multi_scale_output = multi_scale_output
        self.branches = self._make_branches(BasicBlock, num_branches, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(True)

    def _make_one_branch(self, BasicBlock, branch_index, num_blocks, num_channels, stride=1):
        layers = []
        layers.append(
            BasicBlock(
                self.num_channels[branch_index],
                num_channels[branch_index],
                stride
            )
        )
        for i in range(1, num_blocks[branch_index]):
            layers.append(
                BasicBlock(
                    self.num_channels[branch_index],
                    num_channels[branch_index]
                )
            )

        return nn.Sequential(*layers)

    def _make_branches(self, BasicBlock, num_branches, num_blocks, num_channels):
        branches = []

        for i in range(num_branches):
            branches.append(
                self._make_one_branch(BasicBlock, i, num_blocks, num_channels)
            )

        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        num_inchannels = self.num_channels
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(
                        nn.Sequential(
                            nn.Conv2d(
                                num_inchannels[j],
                                num_inchannels[i],
                                1, 1, 0, bias=False
                            ),
                            nn.BatchNorm2d(num_inchannels[i]),
                            nn.Upsample(scale_factor=2 ** (j - i), mode='nearest')
                        )
                    )
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.append(
                                nn.Sequential(
                                    nn.Conv2d(
                                        num_inchannels[j],
                                        num_outchannels_conv3x3,
                                        3, 2, 1, bias=False
                                    ),
                                    nn.BatchNorm2d(num_outchannels_conv3x3)
                                )
                            )
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.append(
                                nn.Sequential(
                                    nn.Conv2d(
                                        num_inchannels[j],
                                        num_outchannels_conv3x3,
                                        3, 2, 1, bias=False
                                    ),
                                    nn.BatchNorm2d(num_outchannels_conv3x3),
                                    nn.ReLU(True)
                                )
                            )
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def forward(self, x):
        if self.num_branches == 4:
            for i in range(self.num_branches-1):
                x[i] = self.branches[i](x[i])

            x_fuse = []

            for i in range(len(self.fuse_layers)):
                # n = len(self.fuse_layers)
                y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
                for j in range(1, self.num_branches):
                    if i == j:
                        y = y + x[j]
                    else:
                        y = torch.cat((y, self.fuse_layers[i][j](x[j])), 1)
                x_fuse.append(self.relu(y))
            return x_fuse

        else:
            if self.num_branches == 1:
                return [self.branches[0](x[0])]

            for i in range(self.num_branches-1):
                x[i] = self.branches[i](x[i])
            if self.num_branches == 3:

                x_fuse = []
                x_cat_list_stage3 = []

                for i in range(len(self.fuse_layers)):
                    y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
                    for j in range(1, self.num_branches):
                        if i == j:
                            y = y + x[j]
                        else:
                            if i == 0:
                                if(i == 0 and j == 1):
                                    x_cat_list_stage3.append(x[0])
                                x_cat_list_stage3.append(self.fuse_layers[i][j](x[j]))
                            y = y + self.fuse_layers[i][j](x[j])

                    x_fuse.append(self.relu(y))
                    x_cat = torch.cat((x_cat_list_stage3[0], x_cat_list_stage3[1]), 1)
                    x_stage3_cat = torch.cat((x_cat, x_cat_list_stage3[2]), 1)
                return x_fuse, x_stage3_cat
            else:
                x_fuse = []
                x_cat_list_stage2 = []
                x_stage2_mask = []
                for i in range(len(self.fuse_layers)):

                    y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
                    for j in range(1, self.num_branches):
                        if i == j:
                            y = y + x[j]
                        else:
                            x_cat_list_stage2.append(y)
                            x_cat_list_stage2.append(self.fuse_layers[i][j](x[j]))
                            y = y + self.fuse_layers[i][j](x[j])
                    x_stage2_mask = torch.cat((x_cat_list_stage2[0], x_cat_list_stage2[1]), 1)
                    x_fuse.append(self.relu(y))

                return x_fuse, x_stage2_mask

class LadderNet(nn.Module):

    def __init__(self, cfg):
        self.inplanes = 64
        super(LadderNet, self).__init__()
        #stage2
        self.stage2_cfg = cfg['stage2_cfg']
        self.stage2 = self._make_stage(self.stage2_cfg)
        #stage3
        self.stage3_cfg = cfg['stage3_cfg']
        self.stage3 = self._make_stage(self.stage3_cfg)
        # stage4
        self.stage4_cfg = cfg['stage4_cfg']
        self.stage4 = self._make_stage(self.stage4_cfg,  multi_scale_output=False)

        # stage2 mask supervision
        self.final_stage2_layer1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1), nn.BatchNorm2d(64, momentum=BN_MOMENTUM), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=1), nn.BatchNorm2d(64, momentum=BN_MOMENTUM), nn.ReLU(inplace=True),
        )
        self.final_stage2_layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1), nn.BatchNorm2d(64, momentum=BN_MOMENTUM), nn.ReLU(inplace=True),
            nn.Conv2d(64, 2, kernel_size=1)
        )

        # stage3 mask supervision
        self.final_stage3_layer1 = nn.Sequential(
            nn.Conv2d(96, 64, kernel_size=1), nn.BatchNorm2d(64, momentum=BN_MOMENTUM), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=1), nn.BatchNorm2d(64, momentum=BN_MOMENTUM), nn.ReLU(inplace=True),
        )
        self.final_stage3_layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1), nn.BatchNorm2d(64, momentum=BN_MOMENTUM), nn.ReLU(inplace=True),
            nn.Conv2d(64, 2, kernel_size=1)
        )

        # stage4 mask supervision
        self.final_layer1 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=1), nn.BatchNorm2d(128, momentum=BN_MOMENTUM), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=1), nn.BatchNorm2d(128, momentum=BN_MOMENTUM), nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=1), nn.BatchNorm2d(64, momentum=BN_MOMENTUM), nn.ReLU(inplace=True),
        )
        self.final_layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1), nn.BatchNorm2d(64, momentum=BN_MOMENTUM), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=1), nn.BatchNorm2d(64, momentum=BN_MOMENTUM), nn.ReLU(inplace=True),
            nn.Conv2d(64, 2, kernel_size=1)
        )

        # transition
        self.trans2_0 = nn.Sequential(
            conv3x3(128, 32, 1), nn.BatchNorm2d(32, momentum=BN_MOMENTUM), nn.ReLU(inplace=True)
        )
        self.trans3_1 = nn.Sequential(
            conv3x3(256, 64, 1), nn.BatchNorm2d(64, momentum=BN_MOMENTUM), nn.ReLU(inplace=True)
        )
        self.trans4_2 = nn.Sequential(
            conv3x3(512, 128, 1), nn.BatchNorm2d(128, momentum=BN_MOMENTUM), nn.ReLU(inplace=True)
        )
        self.trans5_3 = nn.Sequential(
            conv3x3(512, 256, 1), nn.BatchNorm2d(256, momentum=BN_MOMENTUM),nn.ReLU(inplace=True)
        )

        # depth_stage2
        self.depth_attention_stage2 = DAM(
            inplanes=32,
            planes=32,
        )
        # depth_stage3
        self.depth_attention_stage3 = DAM(
            inplanes=32,
            planes=32,
        )
        # depth_stage4
        self.depth_attention_stage4 = DAM(
            inplanes=128,
            planes=128,
        )

    def _make_stage(self, layer_config, multi_scale_output=True):
        num_branches = layer_config['NUM_BRANCHES']
        num_blocks = layer_config['NUM_BLOCKS']
        num_channels = layer_config['NUM_CHANNELS']
        modules = []
        modules.append(
            MakeBL(
                BasicBlock,
                num_branches,
                num_blocks,
                num_channels,
                multi_scale_output
            )
        )
        return nn.Sequential(*modules)

    def forward(self, h2, h3, h4, h5, d2, d3, d4):

        f0 = self.trans2_0(h2)
        f1 = self.trans3_1(h3)
        f2 = self.trans4_2(h4)
        f3 = self.trans5_3(h5)

        x_list = []
        for i in range(2):
            if i == 0:
                x_list.append(f0)
            else:
                x_list.append(f1)

        y_list, y_stage2_mask = self.stage2(x_list)
        dfmap2 = self.depth_attention_stage2(y_list[0], d2)

        y_list[0] = dfmap2
        x_list = []
        for i in range(3):
            if i == 2:
                x_list.append(f2)
            else:
                x_list.append(y_list[i])

        y_list, y_stage3_mask = self.stage3(x_list)
        dfmap3 = self.depth_attention_stage3(y_list[0], d3)
        y_list[0] = dfmap3
        x_list = []
        for i in range(4):
            if i == 3:
                x_list.append(f3)
            else:
                x_list.append(y_list[i])
        y_list = self.stage4(x_list)

        # stage2 supervision
        predict_stage2_mask_1 = F.upsample(y_stage2_mask, scale_factor=2, mode='bilinear', align_corners=True)
        predict_stage2_mask_2 = self.final_stage2_layer1(predict_stage2_mask_1)
        predict_stage2_mask_3 = F.upsample(predict_stage2_mask_2, scale_factor=2, mode='bilinear', align_corners=True)
        predict_stage2_mask = self.final_stage2_layer2(predict_stage2_mask_3)
        # stage3 supervision
        predict_stage3_mask_1 = F.upsample(y_stage3_mask, scale_factor=2, mode='bilinear', align_corners=True)
        predict_stage3_mask_2 = self.final_stage3_layer1(predict_stage3_mask_1)
        predict_stage3_mask_3 = F.upsample(predict_stage3_mask_2, scale_factor=2, mode='bilinear', align_corners=True)
        predict_stage3_mask = self.final_stage3_layer2(predict_stage3_mask_3)
        # stage4 supervision
        dfmap4 = self.depth_attention_stage4(y_list[0], d4)
        predict_mask = F.upsample(dfmap4, scale_factor=2, mode='bilinear', align_corners=True)
        predict_mask = self.final_layer1(predict_mask)
        predict_mask = F.upsample(predict_mask, scale_factor=2, mode='bilinear', align_corners=True)
        predict_mask = self.final_layer2(predict_mask)
        return F.sigmoid(predict_stage2_mask), F.sigmoid(predict_stage3_mask), F.sigmoid(predict_mask)
        # return F.sigmoid(predict_mask)

    def init_weights(self):
        logger.info('=> init weights from normal distribution')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)


