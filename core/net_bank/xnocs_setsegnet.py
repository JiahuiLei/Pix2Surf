# Borrowed and extensively modified  from https://github.com/meetshah1995/pytorch-semseg
# refer to https://github.com/drsrinathsridhar/xnocs

import torch.nn.functional as F
import os, sys, torch
import torch.nn as nn
import torchvision.models as models
from core.net_bank.modules import segnetDown2, segnetDown3, segnetUp2, segnetUp3


# Permutation equivariant set version of SegNet (with or without skip connections)
class SetSegNet(nn.Module):
    def __init__(self, n_classes=21, in_channels=3, is_unpooling=True, pretrained=True, withSkipConnections=False,
                 enablePermEq=True):
        super().__init__()

        self.in_channels = in_channels
        self.is_unpooling = is_unpooling
        self.withSkipConnections = withSkipConnections
        self.enablePermEq = enablePermEq

        self.down1 = segnetDown2(self.in_channels, 64, withFeatureMap=self.withSkipConnections)
        self.down2 = segnetDown2(64, 128, withFeatureMap=self.withSkipConnections)
        self.down3 = segnetDown3(128, 256, withFeatureMap=self.withSkipConnections)
        self.down4 = segnetDown3(256, 512, withFeatureMap=self.withSkipConnections)
        self.down5 = segnetDown3(512, 512, withFeatureMap=self.withSkipConnections)

        self.up5 = segnetUp3(512, 512, withSkipConnections=self.withSkipConnections)
        self.up4 = segnetUp3(512, 256, withSkipConnections=self.withSkipConnections)
        self.up3 = segnetUp3(256, 128, withSkipConnections=self.withSkipConnections)
        self.up2 = segnetUp2(128, 64, withSkipConnections=self.withSkipConnections)
        self.up1 = segnetUp2(64, n_classes, withSkipConnections=self.withSkipConnections)

        if pretrained:
            vgg16 = models.vgg16(pretrained=True)
            Arch = 'SetSegNet'
            if self.withSkipConnections:
                Arch = 'SetSegNetSkip'
            print(
                '[ INFO ]: Using pre-trained weights from VGG16 with {}. Permutation equivariant layers are {}.'.format(
                    Arch, 'ENABLED' if self.enablePermEq else 'DISABLED'))
            self.init_vgg16_params(vgg16)

    def avgSubtract(self, FeatureMap):
        # print('-'*50)
        # print('FeatureMap:', FeatureMap.size())
        B, S, C, W, H = FeatureMap.size()
        FeatureMap_p = FeatureMap.view(B, S, -1)
        # print('View:', FeatureMap_p.size())
        FeatureMap_p = FeatureMap_p.permute(0, 2, 1)  # Set size should be the last dimension
        # print('Permuted:', FeatureMap_p.size())
        # print('S:', S)
        # MP = FeatureMap_p[:, :, 0].unsqueeze(2)  # TEMP TESTING TODO
        MP = F.avg_pool1d(FeatureMap_p, S)
        # MP = self.MaxPool1D(FeatureMap_p)
        # print('MaxPooled:', MP.size())
        MP = MP.permute(0, 2, 1)  # Undo previous permute
        # print('Permuted:', MP.size())
        MP = MP.view(B, 1, C, W, H)
        # print('Final:', MP.size())

        MS = []
        for i in range(S):
            MS.append(FeatureMap[:, i, :, :, :].unsqueeze(1) - MP)

        MS = torch.cat(MS, dim=1)
        # print('MaxSubtracted:', MS.size())
        # print('-' * 50)

        return MS

    def maxSubtract(self, FeatureMap):
        # TODO: There is a CUDA bug in max_pool1d, so using avgSubtract
        return self.avgSubtract(FeatureMap)

        # print('-'*50)
        # print('FeatureMap:', FeatureMap.size())
        B, S, C, W, H = FeatureMap.size()
        FeatureMap_p = FeatureMap.view(B, S, -1)
        # print('View:', FeatureMap_p.size())
        FeatureMap_p = FeatureMap_p.permute(0, 2, 1)  # Set size should be the last dimension
        # print('Permuted:', FeatureMap_p.size())
        # print('S:', S)
        # MP = FeatureMap_p[:, :, 0].unsqueeze(2)  # TEMP TESTING TODO
        MP = F.max_pool1d(FeatureMap_p, S)
        # MP = self.MaxPool1D(FeatureMap_p)
        # print('MaxPooled:', MP.size())
        MP = MP.permute(0, 2, 1)  # Undo previous permute
        # print('Permuted:', MP.size())
        MP = MP.view(B, 1, C, W, H)
        # print('Final:', MP.size())

        MS = []
        for i in range(S):
            MS.append(FeatureMap[:, i, :, :, :].unsqueeze(1) - MP)

        MS = torch.cat(MS, dim=1)
        # print('MaxSubtracted:', MS.size())
        # print('-' * 50)

        return MS

    def forward(self, inputs):
        B, S, C, W, H = inputs.size()
        # print('B, S, C, W, H:', B, S, C, W, H)
        if self.withSkipConnections:
            downs1 = []
            all_indices_1 = []
            all_unpool_shape1 = []
            all_FM1 = []
            for s in range(S):
                down1, indices_1, unpool_shape1, FM1 = self.down1(torch.squeeze(inputs[:, s, :, :, :], dim=1))
                downs1.append(down1.unsqueeze(dim=1))
                all_indices_1.append(indices_1)
                all_unpool_shape1.append(unpool_shape1)
                all_FM1.append(FM1)
            downs1 = torch.cat(downs1, dim=1)
            if self.enablePermEq:
                downs1 = self.maxSubtract(downs1)

            downs2 = []
            all_indices_2 = []
            all_unpool_shape2 = []
            all_FM2 = []
            for s in range(S):
                down2, indices_2, unpool_shape2, FM2 = self.down2(torch.squeeze(downs1[:, s, :, :, :], dim=1))
                downs2.append(down2.unsqueeze(dim=1))
                all_indices_2.append(indices_2)
                all_unpool_shape2.append(unpool_shape2)
                all_FM2.append(FM2)
            downs2 = torch.cat(downs2, dim=1)
            if self.enablePermEq:
                downs2 = self.maxSubtract(downs2)

            downs3 = []
            all_indices_3 = []
            all_unpool_shape3 = []
            all_FM3 = []
            for s in range(S):
                down3, indices_3, unpool_shape3, FM3 = self.down3(torch.squeeze(downs2[:, s, :, :, :], dim=1))
                downs3.append(down3.unsqueeze(dim=1))
                all_indices_3.append(indices_3)
                all_unpool_shape3.append(unpool_shape3)
                all_FM3.append(FM3)
            downs3 = torch.cat(downs3, dim=1)
            if self.enablePermEq:
                downs3 = self.maxSubtract(downs3)

            downs4 = []
            all_indices_4 = []
            all_unpool_shape4 = []
            all_FM4 = []
            for s in range(S):
                down4, indices_4, unpool_shape4, FM4 = self.down4(torch.squeeze(downs3[:, s, :, :, :], dim=1))
                downs4.append(down4.unsqueeze(dim=1))
                all_indices_4.append(indices_4)
                all_unpool_shape4.append(unpool_shape4)
                all_FM4.append(FM4)
            downs4 = torch.cat(downs4, dim=1)
            if self.enablePermEq:
                downs4 = self.maxSubtract(downs4)

            downs5 = []
            all_indices_5 = []
            all_unpool_shape5 = []
            all_FM5 = []
            for s in range(S):
                down5, indices_5, unpool_shape5, FM5 = self.down5(torch.squeeze(downs4[:, s, :, :, :], dim=1))
                downs5.append(down5.unsqueeze(dim=1))
                all_indices_5.append(indices_5)
                all_unpool_shape5.append(unpool_shape5)
                all_FM5.append(FM5)
            downs5 = torch.cat(downs5, dim=1)
            if self.enablePermEq:
                downs5 = self.maxSubtract(downs5)

            ups5 = []
            ups4 = []
            ups3 = []
            ups2 = []
            ups1 = []
            for s in range(S):
                up5 = self.up5(torch.squeeze(downs5[:, s, :, :, :], dim=1), all_indices_5[s], all_unpool_shape5[s],
                               SkipFeatureMap=all_FM5[s])
                ups5.append(up5.unsqueeze(dim=1))
            ups5 = torch.cat(ups5, dim=1)
            if self.enablePermEq:
                ups5 = self.maxSubtract(ups5)

            for s in range(S):
                up4 = self.up4(torch.squeeze(ups5[:, s, :, :, :], dim=1), all_indices_4[s], all_unpool_shape4[s],
                               SkipFeatureMap=all_FM4[s])
                ups4.append(up4.unsqueeze(dim=1))
            ups4 = torch.cat(ups4, dim=1)
            if self.enablePermEq:
                ups4 = self.maxSubtract(ups4)

            for s in range(S):
                up3 = self.up3(torch.squeeze(ups4[:, s, :, :, :], dim=1), all_indices_3[s], all_unpool_shape3[s],
                               SkipFeatureMap=all_FM3[s])
                ups3.append(up3.unsqueeze(dim=1))
            ups3 = torch.cat(ups3, dim=1)
            if self.enablePermEq:
                ups3 = self.maxSubtract(ups3)

            for s in range(S):
                up2 = self.up2(torch.squeeze(ups3[:, s, :, :, :], dim=1), all_indices_2[s], all_unpool_shape2[s],
                               SkipFeatureMap=all_FM2[s])
                ups2.append(up2.unsqueeze(dim=1))
            ups2 = torch.cat(ups2, dim=1)
            if self.enablePermEq:
                ups2 = self.maxSubtract(ups2)

            for s in range(S):
                up1 = self.up1(torch.squeeze(ups2[:, s, :, :, :], dim=1), all_indices_1[s], all_unpool_shape1[s],
                               SkipFeatureMap=all_FM1[s])
                ups1.append(up1.unsqueeze(dim=1))

        SetUps1 = torch.cat(ups1, dim=1)
        # print(SetUps1.size())

        return SetUps1

    def init_vgg16_params(self, vgg16):
        blocks = [self.down1, self.down2, self.down3, self.down4, self.down5]

        features = list(vgg16.features.children())

        vgg_layers = []
        for _layer in features:
            if isinstance(_layer, nn.Conv2d):
                vgg_layers.append(_layer)

        merged_layers = []
        for idx, conv_block in enumerate(blocks):
            if idx < 2:
                units = [conv_block.conv1.cbr_unit, conv_block.conv2.cbr_unit]
            else:
                units = [
                    conv_block.conv1.cbr_unit,
                    conv_block.conv2.cbr_unit,
                    conv_block.conv3.cbr_unit,
                ]
            for _unit in units:
                for _layer in _unit:
                    if isinstance(_layer, nn.Conv2d):
                        merged_layers.append(_layer)

        assert len(vgg_layers) == len(merged_layers)

        for l1, l2 in zip(vgg_layers, merged_layers):
            if isinstance(l1, nn.Conv2d) and isinstance(l2, nn.Conv2d):
                assert l1.weight.size() == l2.weight.size()
                assert l1.bias.size() == l2.bias.size()
                l2.weight.data = l1.weight.data
                l2.bias.data = l1.bias.data


if __name__ == '__main__':
    with torch.no_grad():
        model = SetSegNet(n_classes=21, in_channels=3, withSkipConnections=True, enablePermEq=True).cuda()
        x = torch.rand(1, 5, 3, 240, 320).cuda()
        y = model(x)
        print(y.shape)
