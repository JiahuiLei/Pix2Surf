# Borrowed from https://github.com/meetshah1995/pytorch-semseg
# refer to https://github.com/drsrinathsridhar/xnocs
# minor modified

import torch.nn as nn
import torchvision.models as models
from core.net_bank.modules import segnetDown2, segnetDown3, segnetUp2, segnetUp3
import torch

class SegNet(nn.Module):
    def __init__(self, out_channels=8, in_channels=3,
                 pretrained=True, withSkipConnections=True, new_version=False, additional=None):
        """
        :param out_channels:
        :param in_channels:
        :param pretrained:
        :param withSkipConnections:
        :param new_version:
        :param additional: all additional output layer are new version
        """
        super().__init__()

        self.in_channels = in_channels
        self.withSkipConnections = withSkipConnections

        self.down1 = segnetDown2(self.in_channels, 64, withFeatureMap=self.withSkipConnections)
        self.down2 = segnetDown2(64, 128, withFeatureMap=self.withSkipConnections)
        self.down3 = segnetDown3(128, 256, withFeatureMap=self.withSkipConnections)
        self.down4 = segnetDown3(256, 512, withFeatureMap=self.withSkipConnections)
        self.down5 = segnetDown3(512, 512, withFeatureMap=self.withSkipConnections)

        self.up5 = segnetUp3(512, 512, withSkipConnections=self.withSkipConnections)
        self.up4 = segnetUp3(512, 256, withSkipConnections=self.withSkipConnections)
        self.up3 = segnetUp3(256, 128, withSkipConnections=self.withSkipConnections)
        self.up2 = segnetUp2(128, 64, withSkipConnections=self.withSkipConnections)
        self.up1 = segnetUp2(64, out_channels, last_layer=True if new_version else False,
                             withSkipConnections=self.withSkipConnections)
        if additional is not None:
            self.additional_last_layer = segnetUp2(64, additional, last_layer=True,
                                                   withSkipConnections=self.withSkipConnections)
            self.additional = True
        else:
            self.additional = False

        if pretrained:
            vgg16 = models.vgg16(pretrained=True)
            Arch = 'SegNet'
            if self.withSkipConnections:
                Arch = 'SegNetSkip'
            print('[ INFO ]: Using pre-trained weights from VGG16 with {}.'.format(Arch))
            self.init_vgg16_params(vgg16)

    def forward(self, inputs, return_code=False):
        down1, indices_1, unpool_shape1, FM1 = self.down1(inputs)
        down2, indices_2, unpool_shape2, FM2 = self.down2(down1)
        down3, indices_3, unpool_shape3, FM3 = self.down3(down2)
        down4, indices_4, unpool_shape4, FM4 = self.down4(down3)
        down5, indices_5, unpool_shape5, FM5 = self.down5(down4)

        up5 = self.up5(down5, indices_5, unpool_shape5, SkipFeatureMap=FM5)
        up4 = self.up4(up5, indices_4, unpool_shape4, SkipFeatureMap=FM4)
        up3 = self.up3(up4, indices_3, unpool_shape3, SkipFeatureMap=FM3)
        up2 = self.up2(up3, indices_2, unpool_shape2, SkipFeatureMap=FM2)
        up1 = self.up1(up2, indices_1, unpool_shape1, SkipFeatureMap=FM1)

        # # DEBUG: print sizes
        # print('down1:', down1.size())
        # print('down2:', down2.size())
        # print('down3:', down3.size())
        # print('down4:', down4.size())
        # print('down5:', down5.size())
        #
        # print('up5:', up5.size())
        # print('up4:', up4.size())
        # print('up3:', up3.size())
        # print('up2:', up2.size())
        # print('up1:', up1.size())
        if self.additional:
            add = self.additional_last_layer(up2, indices_1, unpool_shape1, SkipFeatureMap=FM1)
            up1 = torch.cat((up1, add), dim=1)
        if return_code:
            return up1, down5
        else:
            return up1

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


class SegNetGroup(nn.Module):
    def __init__(self, out_channels=8, in_channels=3,
                 pretrained=True, withSkipConnections=True, new_version=False, additional=None):
        """
        :param out_channels:
        :param in_channels:
        :param pretrained:
        :param withSkipConnections:
        :param new_version:
        :param additional: all additional output layer are new version
        """
        super().__init__()

        self.in_channels = in_channels
        self.withSkipConnections = withSkipConnections

        self.down1 = segnetDown2(self.in_channels, 64, withFeatureMap=self.withSkipConnections)
        self.down2 = segnetDown2(64, 128, withFeatureMap=self.withSkipConnections)
        self.down3 = segnetDown3(128, 256, withFeatureMap=self.withSkipConnections)
        self.down4 = segnetDown3(256, 512, withFeatureMap=self.withSkipConnections)
        self.down5 = segnetDown3(512, 512, withFeatureMap=self.withSkipConnections)

        self.up5 = segnetUp3(512, 512, withSkipConnections=self.withSkipConnections)
        self.up4 = segnetUp3(512, 256, withSkipConnections=self.withSkipConnections)
        self.up3 = segnetUp3(256, 128, withSkipConnections=self.withSkipConnections)
        self.up2 = segnetUp2(128, 64, withSkipConnections=self.withSkipConnections)
        self.up1 = segnetUp2(64, out_channels, last_layer=True if new_version else False,
                             withSkipConnections=self.withSkipConnections)
        if additional is not None:
            self.additional_last_layer = segnetUp2(64, additional, last_layer=True,
                                                   withSkipConnections=self.withSkipConnections)
            self.additional = True
        else:
            self.additional = False

        if pretrained:
            vgg16 = models.vgg16(pretrained=True)
            Arch = 'SegNet'
            if self.withSkipConnections:
                Arch = 'SegNetSkip'
            print('[ INFO ]: Using pre-trained weights from VGG16 with {}.'.format(Arch))
            self.init_vgg16_params(vgg16)

    def forward(self, inputs, return_code=False):
        """
        :param inputs: is a list of rgb
        :param return_code: before the max pooling
        :return:
        """
        l1, l2, l3, l4, l5 = [], [], [], [], []
        l5_feature = []
        for rgb in inputs:
            down1, indices_1, unpool_shape1, FM1 = self.down1(rgb)
            down2, indices_2, unpool_shape2, FM2 = self.down2(down1)
            down3, indices_3, unpool_shape3, FM3 = self.down3(down2)
            down4, indices_4, unpool_shape4, FM4 = self.down4(down3)
            down5, indices_5, unpool_shape5, FM5 = self.down5(down4)
            l1.append([indices_1, unpool_shape1, FM1])
            l2.append([indices_2, unpool_shape2, FM2])
            l3.append([indices_3, unpool_shape3, FM3])
            l4.append([indices_4, unpool_shape4, FM4])
            l5.append([indices_5, unpool_shape5, FM5])
            l5_feature.append(down5.unsqueeze(0))
        max_pooled_feature = torch.max(torch.cat(l5_feature, 0), dim=0).values
        f_dim = max_pooled_feature.shape[1]

        pred_list = []
        for i in range(len(inputs)):
            down5 = torch.cat((max_pooled_feature[:, :f_dim // 2, :, :],
                               l5_feature[i].squeeze(0)[:, f_dim // 2:, :, :]), dim=1)
            up5 = self.up5(down5, *l5[i])
            up4 = self.up4(up5, *l4[i])
            up3 = self.up3(up4, *l3[i])
            up2 = self.up2(up3, *l2[i])
            up1 = self.up1(up2, *l1[i])

            if self.additional:
                add = self.additional_last_layer(up2, *l1[i])
                up1 = torch.cat((up1, add), dim=1)
            pred_list.append(up1)
        feature_list = [item.squeeze(0) for item in l5_feature]

        if return_code:
            return pred_list, feature_list
        else:
            return pred_list

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


class SegNetEncoder(nn.Module):
    def __init__(self, in_channels=3, withSkipConnections=True):
        """
        :param in_channels:
        :param pretrained:
        :param withSkipConnections:
        """
        super().__init__()

        self.in_channels = in_channels
        self.withSkipConnections = withSkipConnections

        self.down1 = segnetDown2(self.in_channels, 64, withFeatureMap=self.withSkipConnections)
        self.down2 = segnetDown2(64, 128, withFeatureMap=self.withSkipConnections)
        self.down3 = segnetDown3(128, 256, withFeatureMap=self.withSkipConnections)
        self.down4 = segnetDown3(256, 512, withFeatureMap=self.withSkipConnections)
        self.down5 = segnetDown3(512, 512, withFeatureMap=self.withSkipConnections)

    def forward(self, inputs, return_code=False):
        down1, indices_1, unpool_shape1, FM1 = self.down1(inputs)
        down2, indices_2, unpool_shape2, FM2 = self.down2(down1)
        down3, indices_3, unpool_shape3, FM3 = self.down3(down2)
        down4, indices_4, unpool_shape4, FM4 = self.down4(down3)
        down5, indices_5, unpool_shape5, FM5 = self.down5(down4)
        return down5


if __name__ == '__main__':
    import torch

    net = SegNetGroup(withSkipConnections=True, out_channels=8).cuda()
    x = torch.rand(2, 3, 320, 240).cuda()
    y, f = net([x, x], return_code=True)
    print(f.shape)
    print(y.shape)
