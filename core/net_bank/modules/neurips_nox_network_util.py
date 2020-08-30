# Borrowed from https://github.com/meetshah1995/pytorch-semseg
# refer to https://github.com/drsrinathsridhar/xnocs
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from torch.autograd import Variable


class conv2DBatchNorm(nn.Module):
    def __init__(
            self,
            in_channels,
            n_filters,
            k_size,
            stride,
            padding,
            bias=True,
            dilation=1,
            is_batchnorm=True,
    ):
        super(conv2DBatchNorm, self).__init__()

        conv_mod = nn.Conv2d(
            int(in_channels),
            int(n_filters),
            kernel_size=k_size,
            padding=padding,
            stride=stride,
            bias=bias,
            dilation=dilation,
        )

        if is_batchnorm:
            self.cb_unit = nn.Sequential(conv_mod, nn.BatchNorm2d(int(n_filters)))
        else:
            self.cb_unit = nn.Sequential(conv_mod)

    def forward(self, inputs):
        outputs = self.cb_unit(inputs)
        return outputs


class conv2DGroupNorm(nn.Module):
    def __init__(
            self, in_channels, n_filters, k_size, stride, padding, bias=True, dilation=1, n_groups=16
    ):
        super(conv2DGroupNorm, self).__init__()

        conv_mod = nn.Conv2d(
            int(in_channels),
            int(n_filters),
            kernel_size=k_size,
            padding=padding,
            stride=stride,
            bias=bias,
            dilation=dilation,
        )

        self.cg_unit = nn.Sequential(conv_mod, nn.GroupNorm(n_groups, int(n_filters)))

    def forward(self, inputs):
        outputs = self.cg_unit(inputs)
        return outputs


class deconv2DBatchNorm(nn.Module):
    def __init__(self, in_channels, n_filters, k_size, stride, padding, bias=True):
        super(deconv2DBatchNorm, self).__init__()

        self.dcb_unit = nn.Sequential(
            nn.ConvTranspose2d(
                int(in_channels),
                int(n_filters),
                kernel_size=k_size,
                padding=padding,
                stride=stride,
                bias=bias,
            ),
            nn.BatchNorm2d(int(n_filters)),
        )

    def forward(self, inputs):
        outputs = self.dcb_unit(inputs)
        return outputs


class conv2DBatchNormRelu(nn.Module):
    def __init__(
            self,
            in_channels,
            n_filters,
            k_size,
            stride,
            padding,
            bias=True,
            dilation=1,
            is_batchnorm=True,
    ):
        super(conv2DBatchNormRelu, self).__init__()

        conv_mod = nn.Conv2d(
            int(in_channels),
            int(n_filters),
            kernel_size=k_size,
            padding=padding,
            stride=stride,
            bias=bias,
            dilation=dilation,
        )

        if is_batchnorm:
            self.cbr_unit = nn.Sequential(
                conv_mod, nn.BatchNorm2d(int(n_filters)), nn.ReLU(inplace=True)
            )
        else:
            self.cbr_unit = nn.Sequential(conv_mod, nn.ReLU(inplace=True))

    def forward(self, inputs):
        outputs = self.cbr_unit(inputs)
        return outputs


class conv2DGroupNormRelu(nn.Module):
    def __init__(
            self, in_channels, n_filters, k_size, stride, padding, bias=True, dilation=1, n_groups=16
    ):
        super(conv2DGroupNormRelu, self).__init__()

        conv_mod = nn.Conv2d(
            int(in_channels),
            int(n_filters),
            kernel_size=k_size,
            padding=padding,
            stride=stride,
            bias=bias,
            dilation=dilation,
        )

        self.cgr_unit = nn.Sequential(
            conv_mod, nn.GroupNorm(n_groups, int(n_filters)), nn.ReLU(inplace=True)
        )

    def forward(self, inputs):
        outputs = self.cgr_unit(inputs)
        return outputs


class deconv2DBatchNormRelu(nn.Module):
    def __init__(self, in_channels, n_filters, k_size, stride, padding, bias=True):
        super(deconv2DBatchNormRelu, self).__init__()

        self.dcbr_unit = nn.Sequential(
            nn.ConvTranspose2d(
                int(in_channels),
                int(n_filters),
                kernel_size=k_size,
                padding=padding,
                stride=stride,
                bias=bias,
            ),
            nn.BatchNorm2d(int(n_filters)),
            nn.ReLU(inplace=True),
        )

    def forward(self, inputs):
        outputs = self.dcbr_unit(inputs)
        return outputs


class segnetDown2(nn.Module):
    def __init__(self, in_size, out_size, withFeatureMap=False):
        super(segnetDown2, self).__init__()
        self.conv1 = conv2DBatchNormRelu(in_size, out_size, 3, 1, 1)
        self.conv2 = conv2DBatchNormRelu(out_size, out_size, 3, 1, 1)
        self.maxpool_with_argmax = nn.MaxPool2d(2, 2, return_indices=True)
        self.withFeatureMap = withFeatureMap

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        FeatureMap = outputs
        unpooled_shape = outputs.size()
        outputs, indices = self.maxpool_with_argmax(outputs)
        if self.withFeatureMap:
            return outputs, indices, unpooled_shape, FeatureMap
        return outputs, indices, unpooled_shape, None


class segnetDown3(nn.Module):
    def __init__(self, in_size, out_size, withFeatureMap=False):
        super(segnetDown3, self).__init__()
        self.conv1 = conv2DBatchNormRelu(in_size, out_size, 3, 1, 1)
        self.conv2 = conv2DBatchNormRelu(out_size, out_size, 3, 1, 1)
        self.conv3 = conv2DBatchNormRelu(out_size, out_size, 3, 1, 1)
        self.maxpool_with_argmax = nn.MaxPool2d(2, 2, return_indices=True)
        self.withFeatureMap = withFeatureMap

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        outputs = self.conv3(outputs)
        FeatureMap = outputs
        unpooled_shape = outputs.size()
        outputs, indices = self.maxpool_with_argmax(outputs)
        if self.withFeatureMap:
            return outputs, indices, unpooled_shape, FeatureMap
        return outputs, indices, unpooled_shape, None


class segnetUp2(nn.Module):
    def __init__(self, in_size, out_size, last_layer=False, withSkipConnections=False):
        super().__init__()
        self.withSkipConnections = withSkipConnections
        self.unpool = nn.MaxUnpool2d(2, 2)
        if self.withSkipConnections:
            self.conv1 = conv2DBatchNormRelu(2 * in_size, 2 * in_size, 3, 1, 1)
            if last_layer:
                self.conv2 = nn.Conv2d(in_channels=2 * in_size, out_channels=out_size, kernel_size=3, padding=1,
                                       stride=1)
            else:
                self.conv2 = conv2DBatchNormRelu(2 * in_size, out_size, 3, 1, 1)
        else:
            self.conv1 = conv2DBatchNormRelu(in_size, in_size, 3, 1, 1)
            if last_layer:
                self.conv2 = nn.Conv2d(in_channels=in_size, out_channels=out_size, kernel_size=3, padding=1, stride=1)
            else:
                self.conv2 = conv2DBatchNormRelu(in_size, out_size, 3, 1, 1)

    def forward(self, inputs, indices, output_shape, SkipFeatureMap=None):
        if self.withSkipConnections and SkipFeatureMap is None:
            raise RuntimeError('Created SegNet with skip connections. But no feature map is passed.')

        outputs = self.unpool(input=inputs, indices=indices, output_size=output_shape)
        if self.withSkipConnections:
            outputs = torch.cat((SkipFeatureMap, outputs), 1)
        outputs = self.conv1(outputs)
        outputs = self.conv2(outputs)

        return outputs


class segnetUp3(nn.Module):
    def __init__(self, in_size, out_size, withSkipConnections=False):
        super().__init__()
        self.withSkipConnections = withSkipConnections
        self.unpool = nn.MaxUnpool2d(2, 2)
        if self.withSkipConnections:
            self.conv1 = conv2DBatchNormRelu(2 * in_size, 2 * in_size, 3, 1, 1)
            self.conv2 = conv2DBatchNormRelu(2 * in_size, 2 * in_size, 3, 1, 1)
            self.conv3 = conv2DBatchNormRelu(2 * in_size, out_size, 3, 1, 1)
        else:
            self.conv1 = conv2DBatchNormRelu(in_size, in_size, 3, 1, 1)
            self.conv2 = conv2DBatchNormRelu(in_size, in_size, 3, 1, 1)
            self.conv3 = conv2DBatchNormRelu(in_size, out_size, 3, 1, 1)

    def forward(self, inputs, indices, output_shape, SkipFeatureMap=None):
        if self.withSkipConnections and SkipFeatureMap is None:
            raise RuntimeError('Created SegNet with skip connections. But no feature map is passed.')

        outputs = self.unpool(input=inputs, indices=indices, output_size=output_shape)
        if self.withSkipConnections:
            outputs = torch.cat((SkipFeatureMap, outputs), 1)
        outputs = self.conv1(outputs)
        outputs = self.conv2(outputs)
        outputs = self.conv3(outputs)
        return outputs


# ------- UNet

class UNet_ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()

        self.Conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.BN = nn.BatchNorm2d(out_channels)
        self.ReLU = nn.ReLU()

    def forward(self, x):
        x = self.Conv(x)
        x = self.BN(x)
        x = self.ReLU(x)

        return x


class UNet_DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.Block1 = UNet_ConvBlock(in_channels, out_channels, kernel_size=(3, 3), stride=1,
                                     padding=0)  # Fixed kernel sizes
        self.Block2 = UNet_ConvBlock(out_channels, out_channels, kernel_size=(3, 3), stride=1, padding=0)
        self.Pool = nn.MaxPool2d(kernel_size=(2, 2), stride=2)

    def forward(self, x):
        x = self.Block1(x)
        x = self.Block2(x)
        FeatureMap = x
        x = self.Pool(x)

        return x, FeatureMap


class UNet_UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, up_size):
        super().__init__()

        # Doing what's in the original paper: Upsample the feature map and then a 2x2 conv (and another upsample to match feature sizes)
        self.UpSample = nn.Upsample(size=up_size, mode='bilinear')
        self.Conv2 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(2, 2), stride=1,
                               padding=0)
        self.UpSample2 = nn.Upsample(size=up_size, mode='bilinear')
        self.Block1 = UNet_ConvBlock(in_channels, out_channels, kernel_size=(3, 3), stride=1, padding=0)
        self.Block2 = UNet_ConvBlock(out_channels, out_channels, kernel_size=(3, 3), stride=1, padding=0)

    def CopyCropConcat(self, Upsampled, CopiedFeatureMap):
        PadHalfSize = (CopiedFeatureMap.size()[2] - Upsampled.size()[2]) // 2  # Floor division //
        # print('PadHalfSize:', PadHalfSize)
        # Crop copied feature map
        # Remove PadHalfSize from both sides for both dimensions (starting from the last: width, then height)
        CopiedFeatureMap = F.pad(CopiedFeatureMap, (-PadHalfSize, -PadHalfSize, -PadHalfSize, -PadHalfSize))
        # print('CopiedFeatureMap:', CopiedFeatureMap.size())
        # print('Upsampled:', Upsampled.size())
        # Concat the features
        Concated = torch.cat((CopiedFeatureMap, Upsampled), 1)  # Is this correct?
        # print('Concated:', Concated.size())
        return Concated

    def forward(self, x, CopiedFeatureMap):
        # print('-----------------------')
        # print('Input:', x.size())
        # Doing what's in the original paper: Upsample the feature map and then a 2x2 conv
        x = self.UpSample(x)
        x = self.Conv2(x)
        x = self.UpSample2(x)
        # Copy and crop here
        x = self.CopyCropConcat(x, CopiedFeatureMap)
        # print('After copycropconcat:', x.size())
        x = self.Block1(x)
        x = self.Block2(x)
        # print('Output:', x.size())
        # print('-----------------------')

        return x
