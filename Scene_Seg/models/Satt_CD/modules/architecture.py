# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.utils import init_weights
import numpy as np
from functools import partial
from models.utils import unetConv2,unetUp,unetConv2_res,unetConv2_res_IN,ResidualBlock
from models.MS_Attention.attention import ChannelAttention, SpatialAttention, SCAttention, ChannelAttention1, \
    PAM_Module, CAM_Module,SpatialRefine
from models.MS_Attention.attention import SCSEBlock
from . import block as B
from models.Satt_CD.modules.senet import se_resnext50_32x4d
import logging
logger = logging.getLogger('base')
import math
import numbers
from dropblock import DropBlock2D, LinearScheduler
#from . import pac



class REBNCONV(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, dirate=1):
        super(REBNCONV, self).__init__()

        self.conv_s1 = nn.Conv2d(in_ch, out_ch, 3, padding=1 * dirate, dilation=1 * dirate)
        self.bn_s1 = nn.BatchNorm2d(out_ch)
        self.relu_s1 = nn.ReLU(inplace=True)

    def forward(self, x):
        hx = x
        xout = self.relu_s1(self.bn_s1(self.conv_s1(hx)))

        return xout


nonlinearity = partial(F.relu, inplace=True)


class LocalAttenModule(nn.Module):
    def __init__(self, inplane):
        super(LocalAttenModule, self).__init__()
        self.dconv1 = nn.Sequential(
            nn.Conv2d(inplane, inplane,kernel_size=3, groups=inplane, stride=2),
            nn.BatchNorm2d(inplane),
            nn.ReLU(inplace=False)
        )
        self.dconv2 = nn.Sequential(
            nn.Conv2d(inplane, inplane, kernel_size=3, groups=inplane, stride=2),
            nn.BatchNorm2d(inplane),
            nn.ReLU(inplace=False)
        )
        self.dconv3 = nn.Sequential(
            nn.Conv2d(inplane, inplane, kernel_size=3, groups=inplane, stride=2),
            nn.BatchNorm2d(inplane),
            nn.ReLU(inplace=False)
        )
        self.sigmoid_spatial = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()#[1,512,97,97]
        res1 = x
        res2 = x
        x = self.dconv1(x)#[1,512,48,48]
        x = self.dconv2(x)#[1,512,23,23]
        x = self.dconv3(x)#[1,512,11,11]
        x = F.upsample(x, size=(h, w), mode="bilinear", align_corners=True)#[1,512,97,97]
        x_mask = self.sigmoid_spatial(x)#[1,512,97,97]

        res1 = res1 * x_mask#[1,512,97,97]

        return res2 + res1

class GALD(nn.Module):
    def __init__(self, in_channel=512):
        super(GALD, self).__init__()
        self.GA=ASPP(in_channel=in_channel)
        self.LD=LocalAttenModule(in_channel)


    def forward(self, x):
        size = x.size()[2:]
        #x = self.down(x)  # [1,512,48,48]
        x = self.GA(x)  # [1,512,48,48]
        # local attention
        x = F.upsample(x, size=size, mode="bilinear", align_corners=True)  # [1,512,97,97]
        res = x
        x = self.LD(x)  # [1,512,97,97]
        return x + res


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels,
                               bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x
class Ladder_ASPP(nn.Module):
    def __init__(self, channel):
        super(Ladder_ASPP, self).__init__()
        self.dilate1 = SeparableConv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = SeparableConv2d(channel * 2, channel, kernel_size=3, dilation=2, padding=2)
        self.dilate3 = SeparableConv2d(channel * 3, channel, kernel_size=3, dilation=5, padding=5)
        self.dilate4 = SeparableConv2d(channel * 4, channel, kernel_size=3, dilation=7, padding=7)
        self.bn = nn.BatchNorm2d(channel)
        self.drop = nn.Dropout2d(0.5)
        self.sg = nn.Sigmoid()

        self.finalchannel = channel

        self.conv1x1_1 = SeparableConv2d(channel * 5, channel * 3, kernel_size=1, dilation=1, padding=0)
        self.conv1x1_2 = SeparableConv2d(channel * 3, channel, kernel_size=1, dilation=1, padding=0)

        # Master branch
        self.conv_master = SeparableConv2d(channel, channel, kernel_size=1, bias=False)
        self.bn_master = nn.BatchNorm2d(channel)

        # Global pooling branch
        self.conv_gpb = SeparableConv2d(channel, channel, kernel_size=1, bias=False)
        self.bn_gpb = nn.BatchNorm2d(channel)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):  # [4,512,32,32]

        x_gpb = self.avg_pool(x)  # [4,512,1,1]
        x_gpb = self.conv_gpb(x_gpb)  # [4,512,1,1]
        x_gpb = self.bn_gpb(x_gpb)
        x_gpb = self.sg(x_gpb)  # [4,512,1,1]

        x_se = x_gpb * x  # [4,512,32,32]

        # first block rate1
        d1 = self.dilate1(x)  # [4,512,32,32]
        d1 = self.bn(d1)

        # second block rate3
        d2 = torch.cat([d1, x], 1)  # [4,512*2,32,32]
        d2 = self.dilate2(d2)  # [4,512,32,32]
        d2 = self.bn(d2)

        # third block rate5
        d3 = torch.cat([d1, d2, x], 1)  # [4,512*3,32,32]
        d3 = self.dilate3(d3)  # [4,512,32,32]
        d3 = self.bn(d3)

        # last block rate7
        d4 = torch.cat([d1, d2, d3, x], 1)  # [4,512*4,32,32]
        d4 = self.dilate4(d4)  # [4,512,32,32]
        d4 = self.bn(d4)

        out = torch.cat([d1, d2, d3, d4, x_se], 1)  # [4,512*5,32,32]
        out = self.drop(out)
        out = self.conv1x1_1(out)  # [4,512*3,32,32]
        out = self.conv1x1_2(out)  # [4,512,32,32]

        return out




'''
https://blog.csdn.net/qq_36530992/article/details/102628455
'''
class ASPP(nn.Module):
    def __init__(self, in_channel=512):
        super(ASPP, self).__init__()
        # global average pooling : init nn.AdaptiveAvgPool2d ;also forward torch.mean(,,keep_dim=True)
        depth = int(in_channel)
        self.mean = nn.AdaptiveAvgPool2d((1, 1))
        self.conv = nn.Conv2d(in_channel, depth, 1, 1)
        # k=1 s=1 no pad
        self.atrous_block1 = nn.Conv2d(in_channel, depth, 1, 1)# default dilation=1
        self.atrous_block2 = nn.Conv2d(in_channel, depth, 3, 1, padding=3, dilation=3)
        self.atrous_block4 = nn.Conv2d(in_channel, depth, 3, 1, padding=7, dilation=7)
        self.atrous_block8 = nn.Conv2d(in_channel, depth, 3, 1, padding=15, dilation=15)

        self.conv_1x1_output = nn.Conv2d(depth * 5, depth, 1, 1)
        self.ReLU = nn.ReLU(inplace=True)
        self.att = False
        if self.att:
            self.pam = PAM_Module(depth)

    def forward(self, x):
        '''
        using sigle conv , thus the ROF is 1, 7, 15,31 for d=1,3,7,15
        :param x:
        :return:
        '''
        size = x.shape[2:]

        image_features = self.mean(x)
        image_features = self.conv(image_features)
        image_features = F.upsample(image_features, size=size, mode='bilinear')

        atrous_block1 = self.ReLU(self.atrous_block1(x))

        atrous_block2 = self.ReLU(self.atrous_block2(x))

        atrous_block4 = self.ReLU(self.atrous_block4(x))

        atrous_block8 = self.ReLU(self.atrous_block8(x))
        # if self.att:
        #     image_features = self.pam(image_features) * image_features
        #     astrous_block1 = self.pam(atrous_block1) * atrous_block1
        #     astrous_block2 = self.pam(atrous_block2) * atrous_block2
        #     astrous_block4 = self.pam(atrous_block4) * atrous_block4
        #     astrous_block8 = self.pam(atrous_block8) * atrous_block8

        net = self.conv_1x1_output(torch.cat([image_features, atrous_block1, atrous_block2,
                                              atrous_block4, atrous_block8], dim=1))
        return net


class Dblock(nn.Module):
    def __init__(self, channel,refine=False,conv_num=4):
        super(Dblock, self).__init__()
        # self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        # self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=2, padding=2)
        # self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=4, padding=4)
        # self.dilate4 = nn.Conv2d(channel, channel, kernel_size=3, dilation=8, padding=8)

        self.dilate1 = REBNCONV(channel, channel, dirate=1)#RF 3
        self.dilate2 = REBNCONV(channel, channel, dirate=2)# RF 4+3=7
        self.dilate3 = REBNCONV(channel, channel, dirate=4)# RF 8+7=15
        self.dilate4 = REBNCONV(channel, channel, dirate=8)#RF 16+15=31

        #self.conv_final=nn.Conv2d(channel, channel, kernel_size=3, padding=1)

        self.refine=refine
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        # dilate1_out = nonlinearity(self.dilate1(x))
        # dilate2_out = nonlinearity(self.dilate2(dilate1_out))
        # dilate3_out = nonlinearity(self.dilate3(dilate2_out))
        # dilate4_out = nonlinearity(self.dilate4(dilate3_out))
        '''
        using 4-layer conv, thus the ROF is 1, 7, 15,31
        :param x:
        :return:
        '''
        dilate1_out = self.dilate1(x)
        dilate2_out = self.dilate2(dilate1_out)

        dilate3_out = self.dilate3(dilate2_out)
        dilate4_out = self.dilate4(dilate3_out)

        out = x+dilate1_out + dilate2_out + dilate3_out + dilate4_out  # + dilate5_out
        # if self.refine:
        #     return self.conv_final(out)+x


        return out

#=================to reduce grid effect in astroud conv, using avg filter or gaussian filter to procesing the image before sending to astous conv====
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class AvgFilter(nn.Module):#[2,3,128,128]==>[2,3,128,128]
    def __init__(self,kernel_size=3):
        super(AvgFilter, self).__init__()



        kernel_value = 1.0 / (kernel_size * kernel_size)
        kernel=torch.empty(1, 1, kernel_size, kernel_size).fill_(kernel_value)

        pad_size=kernel_size//2
        self.pad=nn.ReflectionPad2d(pad_size)
        #kernel_avg= torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0).to(device)
        kernel_avg = torch.FloatTensor(kernel).to(device)

        self.weight_avg = nn.Parameter(data = kernel_avg, requires_grad = False)


    def forward(self, x):

        # x0 = x[:, 0]
        # x1 = x[:, 1]
        # x2 = x[:, 2]
        # x0_lap = F.conv2d(x0.unsqueeze(1), self.weight_lap, padding = 1)
        # x1_lap = F.conv2d(x1.unsqueeze(1), self.weight_lap, padding = 1)
        # x2_lap = F.conv2d(x2.unsqueeze(1), self.weight_lap, padding = 1)
        # x0_lap = torch.abs(torch.tanh(x0_lap))
        # x1_lap = torch.abs(torch.tanh(x1_lap))
        # x2_lap = torch.abs(torch.tanh(x2_lap))
        # x_grad = torch.cat([x0_lap, x1_lap, x2_lap], dim=1)
        x=self.pad(x)
        x_avg=F.conv2d(x,self.weight_avg,groups=x.shape[1])
        return x_avg

class GaussianFilter(nn.Module):
    def __init__(self, kernel_size=5, stride=1, padding=4):
        super(GaussianFilter, self).__init__()
        # initialize guassian kernel
        mean = (kernel_size - 1) / 2.0
        variance = (kernel_size / 6.0) ** 2.0
        # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
        x_coord = torch.arange(kernel_size)
        x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
        y_grid = x_grid.t()
        xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

        # Calculate the 2-dimensional gaussian kernel
        gaussian_kernel = torch.exp(-torch.sum((xy_grid - mean) ** 2., dim=-1) / (2 * variance))

        # Make sure sum of values in gaussian kernel equals 1.
        gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

        # Reshape to 2d depthwise convolutional weight
        gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
        gaussian_kernel = gaussian_kernel.repeat(3, 1, 1, 1)

        # create gaussian filter as convolutional layer
        self.gaussian_filter = nn.Conv2d(3, 3, kernel_size, stride=stride, padding=padding, groups=3, bias=False)
        self.gaussian_filter.weight.data = gaussian_kernel
        self.gaussian_filter.weight.requires_grad = False

    def forward(self, x):
        return self.gaussian_filter(x)


class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
        The actual weights are registered as a buffer

self.register_buffer('weight', kernel)
Since they are not parameters (torch.nn.Parameter) they are not trained and will remain static when you use the layer.
    """
    def __init__(self, channels, kernel_size, sigma, dim=2):
        super(GaussianSmoothing, self).__init__()
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                      torch.exp(-((mgrid - mean) / std) ** 2 / 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)#
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
            )
        self.pad_size=kernel_size[0]//2
    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        input=F.pad(input, (self.pad_size, self.pad_size, self.pad_size, self.pad_size), mode='reflect')

        return self.conv(input, weight=self.weight, groups=self.groups)








class FilterLow(nn.Module):
    def __init__(self, recursions=1, kernel_size=5, stride=1, padding=True, include_pad=True, gaussian=False):
        super(FilterLow, self).__init__()
        if padding:
            pad = int((kernel_size - 1) / 2)
        else:
            pad = 0
        if gaussian:
            self.filter = GaussianFilter(kernel_size=kernel_size, stride=stride, padding=pad)
        else:
            self.filter = nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=pad, count_include_pad=include_pad)
        self.recursions = recursions

    def forward(self, img):
        for i in range(self.recursions):
            img = self.filter(img)
        return img



class Dblock2(nn.Module):#with dynamic AS blocks
    def __init__(self, channel, refine=False, conv_num=4):
        super(Dblock2, self).__init__()

        self.dilate1 = REBNCONV(channel, channel, dirate=1)  # RF 3
        self.dilate2 = REBNCONV(channel, channel, dirate=2)  # RF 4+3=7
        self.dilate3 = REBNCONV(channel, channel, dirate=3)  # RF 8+7=15
        #self.dilate3 = REBNCONV(channel, channel, dirate=4)  # RF 8+7=15
        self.dilate4 = REBNCONV(channel, channel, dirate=8)  # RF 16+15=31

        # self.conv_final=nn.Conv2d(channel, channel, kernel_size=3, padding=1)
        self.conv_num=conv_num
        self.refine = refine
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        '''
        using 4-layer conv, thus the ROF is 1, 7, 15,31
        :param x:
        :return:
        '''
        dilate1_out = self.dilate1(x)
        dilate2_out = self.dilate2(dilate1_out)
        if self.conv_num==2:
            out = x + dilate1_out + dilate2_out
        elif self.conv_num==3:
            dilate3_out = self.dilate3(dilate2_out)
            out = x + dilate1_out + dilate2_out + dilate3_out
        else:
            dilate3_out = self.dilate3(dilate2_out)
            dilate4_out = self.dilate4(dilate3_out)
            out = x + dilate1_out + dilate2_out + dilate3_out + dilate4_out



        return out


class Dblock_Smooth(nn.Module):
    def __init__(self, channel, smooth_mode='gaussian',conv_num=4):
        super(Dblock_Smooth, self).__init__()

        # self.dilate1 = REBNCONV(channel, channel, dirate=1)  # RF 3
        # self.dilate2 = REBNCONV(channel, channel, dirate=3)  # RF 6+3=9
        # self.dilate3 = REBNCONV(channel, channel, dirate=5)  # RF 10+9=19
        # self.dilate4 = REBNCONV(channel, channel, dirate=7)  # RF 14+19=33

        self.dilate1 = REBNCONV(channel, channel, dirate=1)  # RF 3
        self.dilate2 = REBNCONV(channel, channel, dirate=2)  # RF 4+3=7
        self.dilate3 = REBNCONV(channel, channel, dirate=4)  # RF 8+7=15
        self.dilate4 = REBNCONV(channel, channel, dirate=8)  # RF 16+15=31

        # self.conv_final=nn.Conv2d(channel, channel, kernel_size=3, padding=1)
        self.conv_num=conv_num
        if smooth_mode=='gaussian':
            self.smooth_conv2=GaussianSmoothing(channel,3,1)#channels, kernel_size, sigma, dim=2):
            self.smooth_conv3=GaussianSmoothing(channel,5,1)
            self.smooth_conv4 = GaussianSmoothing(channel, 7, 1)
        else:
            self.smooth_conv2 = AvgFilter(3)  # kernel_size=3:
            self.smooth_conv3 = AvgFilter(5)
            self.smooth_conv4 = AvgFilter(7)


        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        '''
        using 4-layer conv, thus the ROF is 1, 7, 15,31
        :param x:
        :return:
        '''
        dilate1_out = self.dilate1(x)
        dilate1_out=self.smooth_conv2(dilate1_out)
        dilate2_out = self.dilate2(dilate1_out)
        if self.conv_num==2:
            out = x + dilate1_out + dilate2_out
        elif self.conv_num==3:
            dilate2_out = self.smooth_conv3(dilate2_out)
            dilate3_out = self.dilate3(dilate2_out)
            out = x + dilate1_out + dilate2_out + dilate3_out
        else:
            dilate2_out = self.smooth_conv3(dilate2_out)
            dilate3_out = self.dilate3(dilate2_out)
            dilate3_out = self.smooth_conv4(dilate3_out)
            dilate4_out = self.dilate4(dilate3_out)
            out = x + dilate1_out + dilate2_out + dilate3_out + dilate4_out



        return out

class ConvBNReLU(nn.Module):
    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1, dilation=1, *args, **kwargs):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_chan,
                              out_chan,
                              kernel_size=ks,
                              stride=stride,
                              padding=padding,
                              dilation=dilation,
                              bias=True)
        self.bn = nn.BatchNorm2d(out_chan)
        self.relu = nn.ReLU(inplace=False)
        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)


class HADCLayer(nn.Module):
    def __init__(self, in_chan, out_chan, ks=3, stride=1, dilation=1, mode='parallel', *args, **kwargs):
        super(HADCLayer, self).__init__()
        self.mode = mode
        self.ks = ks
        if ks > 3:
            padding = int(dilation * ((ks - 1) // 2))
            if mode == 'cascade':
                self.hadc_layer = nn.Sequential(ConvBNReLU(in_chan, out_chan,
                                                           ks=[3, ks], dilation=[1, dilation],
                                                           padding=[1, padding]),
                                                ConvBNReLU(out_chan, out_chan,
                                                           ks=[ks, 3], dilation=[dilation, 1],
                                                           padding=[padding, 1]))
            elif mode == 'parallel':
                self.hadc_layer1 = ConvBNReLU(in_chan, out_chan,
                                              ks=[3, ks], dilation=[1, dilation],
                                              padding=[1, padding])
                self.hadc_layer2 = ConvBNReLU(in_chan, out_chan,
                                              ks=[ks, 3], dilation=[dilation, 1],
                                              padding=[padding, 1])
            else:
                raise Exception('No %s mode, please choose from cascade and parallel' % mode)

        elif ks == 3:
            self.hadc_layer = ConvBNReLU(in_chan, out_chan, ks=ks, dilation=dilation, padding=dilation)

        else:
            self.hadc_layer = ConvBNReLU(in_chan, out_chan, ks=ks, dilation=1, padding=0)

        self.init_weight()

    def forward(self, x):
        if self.mode == 'cascade' or self.ks <= 3:
            return self.hadc_layer(x)
        elif self.mode == 'parallel' and self.ks > 3:
            x1 = self.hadc_layer1(x)  # [1,256,48,48]
            x2 = self.hadc_layer2(x)  # [1,256,48,48]
            return x1 + x2  # [1,256,48,48]

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)
class LKPBlock(nn.Module):
    def __init__(self, in_chan, out_chan, ks, dilation=[1, 2, 3], mode='parallel', *args, **kwargs):
        super(LKPBlock, self).__init__()
        if ks >= 3:
            self.lkpblock = nn.Sequential(HADCLayer(in_chan, out_chan,
                                                    ks=ks, dilation=dilation[0], mode=mode),
                                          HADCLayer(out_chan, out_chan,
                                                    ks=ks, dilation=dilation[1], mode=mode),
                                          HADCLayer(out_chan, out_chan,
                                                    ks=ks, dilation=dilation[2], mode=mode))
        else:
            self.lkpblock = HADCLayer(in_chan, out_chan, ks=ks)

        self.init_weight()

    def forward(self, x):
        return self.lkpblock(x)

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)
class LKPP(nn.Module):
    def __init__(self, in_chan=2048, out_chan=256, ks_list=[7, 5, 3, 1], mode='parallel', with_gp=True, *args,
                 **kwargs):
        super(LKPP, self).__init__()
        self.with_gp = with_gp
        self.conv1 = LKPBlock(in_chan, out_chan, ks=ks_list[0], dilation=[1, 2, 3], mode=mode)
        self.conv2 = LKPBlock(in_chan, out_chan, ks=ks_list[1], dilation=[1, 2, 3], mode=mode)
        self.conv3 = LKPBlock(in_chan, out_chan, ks=ks_list[2], dilation=[1, 2, 3], mode=mode)
        self.conv4 = LKPBlock(in_chan, out_chan, ks=ks_list[3], mode=mode)
        if self.with_gp:
            self.avg = nn.AdaptiveAvgPool2d((1, 1))
            self.conv1x1 = ConvBNReLU(in_chan, out_chan, ks=1)
            self.conv_out = ConvBNReLU(out_chan * 5, out_chan*4, ks=1,padding=0)
        else:
            self.conv_out = ConvBNReLU(out_chan * 4, out_chan*4, ks=1,padding=0)

        self.init_weight()

    def forward(self, x):
        H, W = x.size()[2:]  # [48,48]
        feat1 = self.conv1(x)  # [1,256,48,48]
        feat2 = self.conv2(x)  # [1,256,48,48]
        feat3 = self.conv3(x)  # [1,256,48,48]
        feat4 = self.conv4(x)  # [1,256,48,48]  Expanding block
        if self.with_gp:
            avg = self.avg(x)
            feat5 = self.conv1x1(avg)
            feat5 = F.interpolate(feat5, (H, W), mode='bilinear', align_corners=True)  # [1,256,48,48]
            feat = torch.cat([feat1, feat2, feat3, feat4, feat5], 1)  ##[1,256*5,48,48]

        else:
            feat = torch.cat([feat1, feat2, feat3, feat4], 1)

        feat = self.conv_out(feat)
        return feat

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params = []
        non_wd_params = []
        for name, param in self.named_parameters():
            if 'bn' in name or 'bias' in name:
                non_wd_params.append(param)
            else:
                wd_params.append(param)
        return wd_params, non_wd_params











class FPAv2(nn.Module):#E:\TEST2020\DownLoadPrj\CD\kaggle_salt_bes_phalanx-master\kaggle_salt_bes_phalanx-master\phalanx\unet_model.py
    def __init__(self, input_dim, output_dim):
        super(FPAv2, self).__init__()
        self.glob = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                  nn.Conv2d(input_dim, output_dim, kernel_size=1, bias=False))

        self.down2_1 = nn.Sequential(nn.Conv2d(input_dim, input_dim, kernel_size=5, stride=2, padding=2, bias=False),
                                     nn.BatchNorm2d(input_dim),
                                     nn.ELU(True))
        self.down2_2 = nn.Sequential(nn.Conv2d(input_dim, output_dim, kernel_size=5, padding=2, bias=False),
                                     nn.BatchNorm2d(output_dim),
                                     nn.ELU(True))

        self.down3_1 = nn.Sequential(nn.Conv2d(input_dim, input_dim, kernel_size=3, stride=2, padding=1, bias=False),
                                     nn.BatchNorm2d(input_dim),
                                     nn.ELU(True))
        self.down3_2 = nn.Sequential(nn.Conv2d(input_dim, output_dim, kernel_size=3, padding=1, bias=False),
                                     nn.BatchNorm2d(output_dim),
                                     nn.ELU(True))

        self.conv1 = nn.Sequential(nn.Conv2d(input_dim, output_dim, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(output_dim),
                                   nn.ELU(True))

    def forward(self, x):
        # x shape: 512, 16, 16
        x_glob = self.glob(x)  # 256, 1, 1
        x_glob = F.upsample(x_glob, scale_factor=16, mode='bilinear', align_corners=True)  # 256, 16, 16

        d2 = self.down2_1(x)  # 512, 8, 8
        d3 = self.down3_1(d2)  # 512, 4, 4

        d2 = self.down2_2(d2)  # 256, 8, 8
        d3 = self.down3_2(d3)  # 256, 4, 4

        d3 = F.upsample(d3, scale_factor=2, mode='bilinear', align_corners=True)  # 256, 8, 8
        d2 = d2 + d3

        d2 = F.upsample(d2, scale_factor=2, mode='bilinear', align_corners=True)  # 256, 16, 16
        x = self.conv1(x)  # 256, 16, 16
        x = x * d2

        x = x + x_glob

        return x



class unetConv2_res0(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm, n=1, ks=3, stride=1, padding=1, use_res=True, use_att=False):
        super(unetConv2_res0, self).__init__()
        self.n = n
        self.ks = ks
        self.stride = stride
        self.padding = padding
        s = stride
        p = padding
        self.use_res = use_res
        self.use_att = use_att
        self.channel_Att = ChannelAttention(out_size)
        self.spatial_Att = SpatialAttention()
        self.SC_Att = SCAttention(out_size)
        if is_batchnorm:
            for i in range(1, n + 1):
                conv = nn.Sequential(nn.Conv2d(in_size, out_size, ks, s, p),
                                     nn.BatchNorm2d(out_size),
                                     nn.ReLU(inplace=True), )
                setattr(self, 'conv%d' % i, conv)
                in_size = out_size

        else:
            for i in range(1, n + 1):
                conv = nn.Sequential(nn.Conv2d(in_size, out_size, ks, s, p),
                                     nn.ReLU(inplace=True), )
                setattr(self, 'conv%d' % i, conv)
                in_size = out_size

        # initialise the blocks
        for m in self.children():
            init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        x = inputs
        for i in range(1, self.n + 1):
            conv = getattr(self, 'conv%d' % i)
            x = conv(x)  # [9,16,256,256]
            # if self.use_att:
            #     xc=self.channel_Att(x)*x#[9,16,1,1]*[9,16,256,256]==>[9,16,256,256] [9,16,256,256]*[9,16,1,1]==>[9,16,256,256]
            #     xs=self.spatial_Att(x)*x#[9,1,256,256]*[9,16,256,256]==>[9,16,256,256]  [9,16,256,256]*[9,1,256,256]==>[9,16,256,256]
            #     x=xc+xs
            if i == 1:
                x_identity = x
        if self.use_att:
            # x = self.channel_Att(x) * x  # [9,16,1,1]*[9,16,256,256]==>[9,16,256,256] [9,16,256,256]*[9,16,1,1]==>[9,16,256,256]
            # x = self.spatial_Att(x) * x  # [9,1,256,256]*[9,16,256,256]==>[9,16,256,256]  [9,16,256,256]*[9,1,256,256]==>[9,16,256,256]
            x = self.SC_Att(x) * x
        if self.use_res:
            x = x_identity + x
        return x


class unetConv2_stem(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm=True, n=2, ks=3, stride=1, padding=1):
        super(unetConv2_stem, self).__init__()
        self.n = n
        self.ks = ks
        self.stride = stride
        self.padding = padding
        s = stride
        p = padding
        if is_batchnorm:
            self.conv_short = nn.Sequential(nn.Conv2d(in_size, out_size, ks, s, p),
                                            nn.BatchNorm2d(out_size),
                                            )
            for i in range(1, n + 1):
                conv = nn.Sequential(  # nn.Conv2d(out_size, out_size, ks, s, p),
                    nn.BatchNorm2d(out_size),
                    nn.ReLU(inplace=False),
                    nn.Conv2d(out_size, out_size, ks, s, p),
                )
                setattr(self, 'conv%d' % i, conv)
                # in_size = out_size

        else:
            self.conv_short = nn.Sequential(nn.Conv2d(in_size, out_size, ks, s, p),

                                            )
            for i in range(1, n + 1):
                conv = nn.Sequential(  # nn.Conv2d(out_size, out_size, ks, s, p),
                    nn.ReLU(inplace=False),
                    nn.Conv2d(out_size, out_size, ks, s, p),
                )
                setattr(self, 'conv%d' % i, conv)
                # in_size = out_size
        self.conv0 = nn.Conv2d(in_size, out_size, ks, s, p)

        # initialise the blocks
        for m in self.children():
            init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        x = inputs
        x = self.conv0(x)
        for i in range(1, self.n + 1):
            conv = getattr(self, 'conv%d' % i)
            x = conv(x)
        x_short = self.conv_short(inputs)

        return x + x_short




# ===================================for unet refinement=========================
class RefUnet(nn.Module):
    def __init__(self, in_ch=1, inc_ch=16):
        super(RefUnet, self).__init__()
        # inc_ch=inc_ch
        self.conv0 = nn.Conv2d(in_ch, inc_ch, 3, padding=1)

        self.conv1 = nn.Conv2d(inc_ch, inc_ch, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(inc_ch)
        self.relu1 = nn.ReLU(inplace=True)

        self.pool1 = nn.MaxPool2d(2, 2, ceil_mode=True)

        self.conv2 = nn.Conv2d(inc_ch, inc_ch, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(inc_ch)
        self.relu2 = nn.ReLU(inplace=True)

        self.pool2 = nn.MaxPool2d(2, 2, ceil_mode=True)

        self.conv3 = nn.Conv2d(inc_ch, inc_ch, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(inc_ch)
        self.relu3 = nn.ReLU(inplace=True)

        self.pool3 = nn.MaxPool2d(2, 2, ceil_mode=True)

        self.conv4 = nn.Conv2d(inc_ch, inc_ch, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(inc_ch)
        self.relu4 = nn.ReLU(inplace=True)

        self.pool4 = nn.MaxPool2d(2, 2, ceil_mode=True)

        #####

        self.conv5 = nn.Conv2d(inc_ch, inc_ch, 3, padding=1)
        self.bn5 = nn.BatchNorm2d(inc_ch)
        self.relu5 = nn.ReLU(inplace=True)

        #####

        self.conv_d4 = nn.Conv2d(inc_ch * 2, inc_ch, 3, padding=1)
        self.bn_d4 = nn.BatchNorm2d(inc_ch)
        self.relu_d4 = nn.ReLU(inplace=True)

        self.conv_d3 = nn.Conv2d(inc_ch * 2, inc_ch, 3, padding=1)
        self.bn_d3 = nn.BatchNorm2d(inc_ch)
        self.relu_d3 = nn.ReLU(inplace=True)

        self.conv_d2 = nn.Conv2d(inc_ch * 2, inc_ch, 3, padding=1)
        self.bn_d2 = nn.BatchNorm2d(inc_ch)
        self.relu_d2 = nn.ReLU(inplace=True)

        self.conv_d1 = nn.Conv2d(inc_ch * 2, inc_ch, 3, padding=1)
        self.bn_d1 = nn.BatchNorm2d(inc_ch)
        self.relu_d1 = nn.ReLU(inplace=True)

        self.conv_d0 = nn.Conv2d(inc_ch, 1, 3, padding=1)

        self.upscore2 = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, x):
        hx = x
        hx = self.conv0(hx)

        hx1 = self.relu1(self.bn1(self.conv1(hx)))
        hx = self.pool1(hx1)

        hx2 = self.relu2(self.bn2(self.conv2(hx)))
        hx = self.pool2(hx2)

        hx3 = self.relu3(self.bn3(self.conv3(hx)))
        hx = self.pool3(hx3)

        hx4 = self.relu4(self.bn4(self.conv4(hx)))
        hx = self.pool4(hx4)

        hx5 = self.relu5(self.bn5(self.conv5(hx)))

        hx = self.upscore2(hx5)

        d4 = self.relu_d4(self.bn_d4(self.conv_d4(torch.cat((hx, hx4), 1))))
        hx = self.upscore2(d4)

        d3 = self.relu_d3(self.bn_d3(self.conv_d3(torch.cat((hx, hx3), 1))))
        hx = self.upscore2(d3)

        d2 = self.relu_d2(self.bn_d2(self.conv_d2(torch.cat((hx, hx2), 1))))
        hx = self.upscore2(d2)

        d1 = self.relu_d1(self.bn_d1(self.conv_d1(torch.cat((hx, hx1), 1))))

        residual = self.conv_d0(d1)

        return x + residual


# ==========================================================================================
class UNet_3Plus(nn.Module):

    #def __init__(self, in_channels=6, n_classes=1, feature_scale=4, is_deconv=True, is_batchnorm=True):
    def __init__(self,in_channels=3, n_classes=1, filters=[],feature_scale=4, is_deconv=True, is_batchnorm=True,use_res=True,
                 use_dense=False,use_deep_sup=True,att_type=None,dblock_type='AS',use_rfnet=False):
        super(UNet_3Plus, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale
        self.n_classes = n_classes
        # filters = [64, 128, 256, 512, 1024]
        filters = [16, 32, 64, 128, 256]
        self.use_dense = use_dense
        if self.use_dense:
            self.conv33 = unetConv2(filters[0] + filters[1], filters[2], self.is_batchnorm)
            self.conv44 = unetConv2(filters[0] + filters[1] + filters[2], filters[3], self.is_batchnorm)
            self.conv55 = unetConv2(filters[0] + filters[1] + filters[2] + filters[3], filters[4], self.is_batchnorm)
            logger.info("use dense connection")
        else:
            logger.info("without dense connection")

        ## -------------Encoder--------------
        self.conv1 = unetConv2(self.in_channels, filters[0], self.is_batchnorm)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = unetConv2(filters[0], filters[1], self.is_batchnorm)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = unetConv2(filters[1], filters[2], self.is_batchnorm)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4 = unetConv2(filters[2], filters[3], self.is_batchnorm)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)

        self.conv5 = unetConv2(filters[3], filters[4], self.is_batchnorm)


        ## -------------Decoder--------------
        self.CatChannels = filters[0]
        self.CatBlocks = 5
        self.UpChannels = self.CatChannels * self.CatBlocks

        '''stage 4d'''
        # h1->320*320, hd4->40*40, Pooling 8 times
        self.h1_PT_hd4 = nn.MaxPool2d(8, 8, ceil_mode=True)
        self.h1_PT_hd4_conv = nn.Conv2d(filters[0], self.CatChannels, 3, padding=1)
        self.h1_PT_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.h1_PT_hd4_relu = nn.ReLU(inplace=True)

        # h2->160*160, hd4->40*40, Pooling 4 times
        self.h2_PT_hd4 = nn.MaxPool2d(4, 4, ceil_mode=True)
        self.h2_PT_hd4_conv = nn.Conv2d(filters[1], self.CatChannels, 3, padding=1)
        self.h2_PT_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.h2_PT_hd4_relu = nn.ReLU(inplace=True)

        # h3->80*80, hd4->40*40, Pooling 2 times
        self.h3_PT_hd4 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.h3_PT_hd4_conv = nn.Conv2d(filters[2], self.CatChannels, 3, padding=1)
        self.h3_PT_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.h3_PT_hd4_relu = nn.ReLU(inplace=True)

        # h4->40*40, hd4->40*40, Concatenation
        self.h4_Cat_hd4_conv = nn.Conv2d(filters[3], self.CatChannels, 3, padding=1)
        self.h4_Cat_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.h4_Cat_hd4_relu = nn.ReLU(inplace=True)

        # hd5->20*20, hd4->40*40, Upsample 2 times
        self.hd5_UT_hd4 = nn.Upsample(scale_factor=2, mode='bilinear')  # 14*14
        self.hd5_UT_hd4_conv = nn.Conv2d(filters[4], self.CatChannels, 3, padding=1)
        self.hd5_UT_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd5_UT_hd4_relu = nn.ReLU(inplace=True)

        # fusion(h1_PT_hd4, h2_PT_hd4, h3_PT_hd4, h4_Cat_hd4, hd5_UT_hd4)
        self.conv4d_1 = nn.Conv2d(self.UpChannels, self.UpChannels, 3, padding=1)  # 16
        self.bn4d_1 = nn.BatchNorm2d(self.UpChannels)
        self.relu4d_1 = nn.ReLU(inplace=True)

        '''stage 3d'''
        # h1->320*320, hd3->80*80, Pooling 4 times
        self.h1_PT_hd3 = nn.MaxPool2d(4, 4, ceil_mode=True)
        self.h1_PT_hd3_conv = nn.Conv2d(filters[0], self.CatChannels, 3, padding=1)
        self.h1_PT_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.h1_PT_hd3_relu = nn.ReLU(inplace=True)

        # h2->160*160, hd3->80*80, Pooling 2 times
        self.h2_PT_hd3 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.h2_PT_hd3_conv = nn.Conv2d(filters[1], self.CatChannels, 3, padding=1)
        self.h2_PT_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.h2_PT_hd3_relu = nn.ReLU(inplace=True)

        # h3->80*80, hd3->80*80, Concatenation
        self.h3_Cat_hd3_conv = nn.Conv2d(filters[2], self.CatChannels, 3, padding=1)
        self.h3_Cat_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.h3_Cat_hd3_relu = nn.ReLU(inplace=True)

        # hd4->40*40, hd4->80*80, Upsample 2 times
        self.hd4_UT_hd3 = nn.Upsample(scale_factor=2, mode='bilinear')  # 14*14
        self.hd4_UT_hd3_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd4_UT_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd4_UT_hd3_relu = nn.ReLU(inplace=True)

        # hd5->20*20, hd4->80*80, Upsample 4 times
        self.hd5_UT_hd3 = nn.Upsample(scale_factor=4, mode='bilinear')  # 14*14
        self.hd5_UT_hd3_conv = nn.Conv2d(filters[4], self.CatChannels, 3, padding=1)
        self.hd5_UT_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd5_UT_hd3_relu = nn.ReLU(inplace=True)

        # fusion(h1_PT_hd3, h2_PT_hd3, h3_Cat_hd3, hd4_UT_hd3, hd5_UT_hd3)
        self.conv3d_1 = nn.Conv2d(self.UpChannels, self.UpChannels, 3, padding=1)  # 16
        self.bn3d_1 = nn.BatchNorm2d(self.UpChannels)
        self.relu3d_1 = nn.ReLU(inplace=True)

        '''stage 2d '''
        # h1->320*320, hd2->160*160, Pooling 2 times
        self.h1_PT_hd2 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.h1_PT_hd2_conv = nn.Conv2d(filters[0], self.CatChannels, 3, padding=1)
        self.h1_PT_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.h1_PT_hd2_relu = nn.ReLU(inplace=True)

        # h2->160*160, hd2->160*160, Concatenation
        self.h2_Cat_hd2_conv = nn.Conv2d(filters[1], self.CatChannels, 3, padding=1)
        self.h2_Cat_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.h2_Cat_hd2_relu = nn.ReLU(inplace=True)

        # hd3->80*80, hd2->160*160, Upsample 2 times
        self.hd3_UT_hd2 = nn.Upsample(scale_factor=2, mode='bilinear')  # 14*14
        self.hd3_UT_hd2_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd3_UT_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd3_UT_hd2_relu = nn.ReLU(inplace=True)

        # hd4->40*40, hd2->160*160, Upsample 4 times
        self.hd4_UT_hd2 = nn.Upsample(scale_factor=4, mode='bilinear')  # 14*14
        self.hd4_UT_hd2_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd4_UT_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd4_UT_hd2_relu = nn.ReLU(inplace=True)

        # hd5->20*20, hd2->160*160, Upsample 8 times
        self.hd5_UT_hd2 = nn.Upsample(scale_factor=8, mode='bilinear')  # 14*14
        self.hd5_UT_hd2_conv = nn.Conv2d(filters[4], self.CatChannels, 3, padding=1)
        self.hd5_UT_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd5_UT_hd2_relu = nn.ReLU(inplace=True)

        # fusion(h1_PT_hd2, h2_Cat_hd2, hd3_UT_hd2, hd4_UT_hd2, hd5_UT_hd2)
        self.conv2d_1 = nn.Conv2d(self.UpChannels, self.UpChannels, 3, padding=1)  # 16
        self.bn2d_1 = nn.BatchNorm2d(self.UpChannels)
        self.relu2d_1 = nn.ReLU(inplace=True)

        '''stage 1d'''
        # h1->320*320, hd1->320*320, Concatenation
        self.h1_Cat_hd1_conv = nn.Conv2d(filters[0], self.CatChannels, 3, padding=1)
        self.h1_Cat_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.h1_Cat_hd1_relu = nn.ReLU(inplace=True)

        # hd2->160*160, hd1->320*320, Upsample 2 times
        self.hd2_UT_hd1 = nn.Upsample(scale_factor=2, mode='bilinear')  # 14*14
        self.hd2_UT_hd1_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd2_UT_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd2_UT_hd1_relu = nn.ReLU(inplace=True)

        # hd3->80*80, hd1->320*320, Upsample 4 times
        self.hd3_UT_hd1 = nn.Upsample(scale_factor=4, mode='bilinear')  # 14*14
        self.hd3_UT_hd1_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd3_UT_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd3_UT_hd1_relu = nn.ReLU(inplace=True)

        # hd4->40*40, hd1->320*320, Upsample 8 times
        self.hd4_UT_hd1 = nn.Upsample(scale_factor=8, mode='bilinear')  # 14*14
        self.hd4_UT_hd1_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd4_UT_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd4_UT_hd1_relu = nn.ReLU(inplace=True)

        # hd5->20*20, hd1->320*320, Upsample 16 times
        self.hd5_UT_hd1 = nn.Upsample(scale_factor=16, mode='bilinear')  # 14*14
        self.hd5_UT_hd1_conv = nn.Conv2d(filters[4], self.CatChannels, 3, padding=1)
        self.hd5_UT_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd5_UT_hd1_relu = nn.ReLU(inplace=True)

        # fusion(h1_Cat_hd1, hd2_UT_hd1, hd3_UT_hd1, hd4_UT_hd1, hd5_UT_hd1)
        self.conv1d_1 = nn.Conv2d(self.UpChannels, self.UpChannels, 3, padding=1)  # 16
        self.bn1d_1 = nn.BatchNorm2d(self.UpChannels)
        self.relu1d_1 = nn.ReLU(inplace=True)

        # output
        self.outconv1 = nn.Conv2d(self.UpChannels, n_classes, 3, padding=1)

        # #=======initialise weights
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         init_weights(m, init_type='kaiming')
        #     elif isinstance(m, nn.BatchNorm2d):
        #         init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        ## -------------Encoder-------------
        h1 = self.conv1(inputs)  # h1->320*320*64
        h2 = self.maxpool1(h1)
        h2 = self.conv2(h2)  # h2->160*160*128
        if self.use_dense:

            h3 = self.conv33(torch.cat([F.max_pool2d(h1, 4), F.max_pool2d(h2, 2)], dim=1))
            h4 = self.conv44(torch.cat([F.max_pool2d(h1, 8), F.max_pool2d(h2, 4), F.max_pool2d(h3, 2)], dim=1))
            hd5 = self.conv55(
                torch.cat([F.max_pool2d(h1, 16), F.max_pool2d(h2, 8), F.max_pool2d(h3, 4), F.max_pool2d(h4, 2)], dim=1))
        else:
            h3 = self.maxpool2(h2)
            h3 = self.conv3(h3)  # h3->80*80*256

            h4 = self.maxpool3(h3)
            h4 = self.conv4(h4)  # h4->40*40*512

            h5 = self.maxpool4(h4)  # [20*20*512]
            hd5 = self.conv5(h5)  # h5->20*20*1024

        ## -------------Decoder-------------
        h1_PT_hd4 = self.h1_PT_hd4_relu(self.h1_PT_hd4_bn(self.h1_PT_hd4_conv(self.h1_PT_hd4(h1))))  # [2,64,40,40]
        h2_PT_hd4 = self.h2_PT_hd4_relu(self.h2_PT_hd4_bn(self.h2_PT_hd4_conv(self.h2_PT_hd4(h2))))  # [2,64,40,40]
        h3_PT_hd4 = self.h3_PT_hd4_relu(self.h3_PT_hd4_bn(self.h3_PT_hd4_conv(self.h3_PT_hd4(h3))))  # [2,64,40,40]
        h4_Cat_hd4 = self.h4_Cat_hd4_relu(self.h4_Cat_hd4_bn(self.h4_Cat_hd4_conv(h4)))  # [2,64,40,40]
        hd5_UT_hd4 = self.hd5_UT_hd4_relu(
            self.hd5_UT_hd4_bn(self.hd5_UT_hd4_conv(self.hd5_UT_hd4(hd5))))  # [2,64,40,40]
        hd4 = self.relu4d_1(self.bn4d_1(self.conv4d_1(
            torch.cat((h1_PT_hd4, h2_PT_hd4, h3_PT_hd4, h4_Cat_hd4, hd5_UT_hd4),
                      1))))  # [2,320,40,40] hd4->40*40*UpChannels

        h1_PT_hd3 = self.h1_PT_hd3_relu(self.h1_PT_hd3_bn(self.h1_PT_hd3_conv(self.h1_PT_hd3(h1))))
        h2_PT_hd3 = self.h2_PT_hd3_relu(self.h2_PT_hd3_bn(self.h2_PT_hd3_conv(self.h2_PT_hd3(h2))))
        h3_Cat_hd3 = self.h3_Cat_hd3_relu(self.h3_Cat_hd3_bn(self.h3_Cat_hd3_conv(h3)))
        hd4_UT_hd3 = self.hd4_UT_hd3_relu(self.hd4_UT_hd3_bn(self.hd4_UT_hd3_conv(self.hd4_UT_hd3(hd4))))
        hd5_UT_hd3 = self.hd5_UT_hd3_relu(self.hd5_UT_hd3_bn(self.hd5_UT_hd3_conv(self.hd5_UT_hd3(hd5))))
        hd3 = self.relu3d_1(self.bn3d_1(self.conv3d_1(
            torch.cat((h1_PT_hd3, h2_PT_hd3, h3_Cat_hd3, hd4_UT_hd3, hd5_UT_hd3), 1))))  # hd3->80*80*UpChannels

        h1_PT_hd2 = self.h1_PT_hd2_relu(self.h1_PT_hd2_bn(self.h1_PT_hd2_conv(self.h1_PT_hd2(h1))))
        h2_Cat_hd2 = self.h2_Cat_hd2_relu(self.h2_Cat_hd2_bn(self.h2_Cat_hd2_conv(h2)))
        hd3_UT_hd2 = self.hd3_UT_hd2_relu(self.hd3_UT_hd2_bn(self.hd3_UT_hd2_conv(self.hd3_UT_hd2(hd3))))
        hd4_UT_hd2 = self.hd4_UT_hd2_relu(self.hd4_UT_hd2_bn(self.hd4_UT_hd2_conv(self.hd4_UT_hd2(hd4))))
        hd5_UT_hd2 = self.hd5_UT_hd2_relu(self.hd5_UT_hd2_bn(self.hd5_UT_hd2_conv(self.hd5_UT_hd2(hd5))))
        hd2 = self.relu2d_1(self.bn2d_1(self.conv2d_1(
            torch.cat((h1_PT_hd2, h2_Cat_hd2, hd3_UT_hd2, hd4_UT_hd2, hd5_UT_hd2), 1))))  # hd2->160*160*UpChannels

        h1_Cat_hd1 = self.h1_Cat_hd1_relu(self.h1_Cat_hd1_bn(self.h1_Cat_hd1_conv(h1)))
        hd2_UT_hd1 = self.hd2_UT_hd1_relu(self.hd2_UT_hd1_bn(self.hd2_UT_hd1_conv(self.hd2_UT_hd1(hd2))))
        hd3_UT_hd1 = self.hd3_UT_hd1_relu(self.hd3_UT_hd1_bn(self.hd3_UT_hd1_conv(self.hd3_UT_hd1(hd3))))
        hd4_UT_hd1 = self.hd4_UT_hd1_relu(self.hd4_UT_hd1_bn(self.hd4_UT_hd1_conv(self.hd4_UT_hd1(hd4))))
        hd5_UT_hd1 = self.hd5_UT_hd1_relu(self.hd5_UT_hd1_bn(self.hd5_UT_hd1_conv(self.hd5_UT_hd1(hd5))))
        hd1 = self.relu1d_1(self.bn1d_1(self.conv1d_1(
            torch.cat((h1_Cat_hd1, hd2_UT_hd1, hd3_UT_hd1, hd4_UT_hd1, hd5_UT_hd1), 1))))  # hd1->320*320*UpChannels

        d1 = self.outconv1(hd1)  # d1->320*320*n_classes
        # return F.sigmoid(d1)
        if self.n_classes == 1:
            return F.sigmoid(d1)
        return d1


class UNet_3PlusRes(nn.Module):

    def __init__(self, in_channels=3, n_classes=1, filters=[],feature_scale=4, is_deconv=True, is_batchnorm=True,use_res=True,
                 use_dense=True,use_deep_sup=True,att_type=None,dblock_type='AS',use_rfnet=False):
        super(UNet_3PlusRes, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale
        self.n_classes = n_classes
        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / 4) for x in filters]
        # filters=[16,32,64,128,256]
        self.use_dense = use_dense
        self.use_res = use_res
        resb_num = 1
        self.use_deep_sup = use_deep_sup
        self.dblock_type = dblock_type
        self.att_type = att_type
        if self.dblock_type=='ASPP':
           self.Dblock = ASPP(in_channel=filters[4])
        elif self.dblock_type=='AS':
           self.Dblock=Dblock(filters[4])
        else:
            logger.info('Without dblock module.')
            self.Dblock=None

        if self.att_type=="CA":
            self.cam5 = ChannelAttention(filters[4])
            self.cam14 = ChannelAttention(self.UpChannels)
            self.pam = SpatialAttention()
            self.cam345 = ChannelAttention(filters[4] + 2 * self.UpChannels)
            self.conv345 = nn.Conv2d(filters[4] + 2 * self.UpChannels, self.UpChannels, 1)
            self.conv12 = nn.Conv2d(2 * self.UpChannels, self.UpChannels, 3, padding=1)
            self.outconv12345 = nn.Conv2d(2 * self.UpChannels, n_classes, 3, padding=1)


        elif self.att_type=="PA":
            self.UpChannels_half = int(self.UpChannels / 2)
            self.pam12_1 = nn.Sequential(
                nn.Conv2d(self.UpChannels, self.UpChannels_half, (1, 9), padding=(0, 4)),
                nn.BatchNorm2d(self.UpChannels_half),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.UpChannels_half, 1, (9, 1), padding=(4, 0)),
                nn.BatchNorm2d(1),
                nn.ReLU(inplace=True)
            )
            self.pam12_2 = nn.Sequential(
                nn.Conv2d(self.UpChannels, self.UpChannels_half, (9, 1), padding=(4, 0)),
                nn.BatchNorm2d(self.UpChannels_half),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.UpChannels_half, 1, (1, 9), padding=(0, 4)),
                nn.BatchNorm2d(1),
                nn.ReLU(inplace=True)
            )
        else:
            self.att_type=None
            logger.info('Without attention module.')


            # self.cam5=CAM_Module(filters[4])
            # self.pam5=PAM_Module(filters[4])
            # self.cam34 = CAM_Module(self.UpChannels)
            # self.pam34 = PAM_Module(self.UpChannels)
            # self.cam12=ChannelAttention(self.UpChannels)
            # self.pam12=SpatialAttention()
        # =======refine=================
        self.use_rfnet=use_rfnet
        if self.use_rfnet:
           #self.refnet = RefUnet()
           self.refnet=Dblock(1)
           logger.info('With refine module.')
        ## -------------Encoder--------------
        self.conv1 = unetConv2_res(self.in_channels, filters[0], n=resb_num)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = unetConv2_res(filters[0], filters[1], n=resb_num)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = unetConv2_res(filters[1], filters[2], n=resb_num)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4 = unetConv2_res(filters[2], filters[3], n=resb_num)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)

        self.conv5 = unetConv2_res(filters[3], filters[4], n=resb_num)
        # =======================using dense connection in the encoder======================
        # self.conv33 = unetConv2_res(filters[0] + filters[1], filters[2], n=resb_num)
        # self.conv44 = unetConv2_res(filters[0] + filters[1] + filters[2], filters[3], n=resb_num)
        # self.conv55 = unetConv2_res(filters[0] + filters[1] + filters[2] + filters[3], filters[4], n=resb_num)

        self.conv33 = nn.Conv2d(filters[0]+filters[1], filters[1],1)
        self.conv44 = nn.Conv2d(filters[1]+filters[2], filters[2], 1)
        self.conv55 = nn.Conv2d(filters[2]+filters[3],filters[3], 1)

        # ====================Dblock======================
        # self.Dblock=Dblock(filters[4])

        # self.Dblock34=ASPP(in_channel=filters[0]*5)
        ## -------------Decoder--------------
        self.CatChannels = filters[0]
        self.CatBlocks = 5
        self.UpChannels = self.CatChannels * self.CatBlocks

        '''stage 4d'''
        # h1->320*320, hd4->40*40, Pooling 8 times
        self.h1_PT_hd4 = nn.MaxPool2d(8, 8, ceil_mode=True)
        self.h1_PT_hd4_conv = nn.Conv2d(filters[0], self.CatChannels, 3, padding=1)
        self.h1_PT_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.h1_PT_hd4_relu = nn.ReLU(inplace=True)

        # h2->160*160, hd4->40*40, Pooling 4 times
        self.h2_PT_hd4 = nn.MaxPool2d(4, 4, ceil_mode=True)
        self.h2_PT_hd4_conv = nn.Conv2d(filters[1], self.CatChannels, 3, padding=1)
        self.h2_PT_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.h2_PT_hd4_relu = nn.ReLU(inplace=True)

        # h3->80*80, hd4->40*40, Pooling 2 times
        self.h3_PT_hd4 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.h3_PT_hd4_conv = nn.Conv2d(filters[2], self.CatChannels, 3, padding=1)
        self.h3_PT_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.h3_PT_hd4_relu = nn.ReLU(inplace=True)

        # h4->40*40, hd4->40*40, Concatenation
        self.h4_Cat_hd4_conv = nn.Conv2d(filters[3], self.CatChannels, 3, padding=1)
        self.h4_Cat_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.h4_Cat_hd4_relu = nn.ReLU(inplace=True)

        # hd5->20*20, hd4->40*40, Upsample 2 times
        self.hd5_UT_hd4 = nn.Upsample(scale_factor=2, mode='bilinear')  # 14*14
        self.hd5_UT_hd4_conv = nn.Conv2d(filters[4], self.CatChannels, 3, padding=1)
        self.hd5_UT_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd5_UT_hd4_relu = nn.ReLU(inplace=True)

        # fusion(h1_PT_hd4, h2_PT_hd4, h3_PT_hd4, h4_Cat_hd4, hd5_UT_hd4)
        self.conv4d_1 = nn.Conv2d(self.UpChannels, self.UpChannels, 3, padding=1)  # 16
        self.bn4d_1 = nn.BatchNorm2d(self.UpChannels)
        self.relu4d_1 = nn.ReLU(inplace=True)

        self.conv4d_res = unetConv2_res(self.UpChannels, self.UpChannels, n=resb_num)

        '''stage 3d'''
        # h1->320*320, hd3->80*80, Pooling 4 times
        self.h1_PT_hd3 = nn.MaxPool2d(4, 4, ceil_mode=True)
        self.h1_PT_hd3_conv = nn.Conv2d(filters[0], self.CatChannels, 3, padding=1)
        self.h1_PT_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.h1_PT_hd3_relu = nn.ReLU(inplace=True)

        # h2->160*160, hd3->80*80, Pooling 2 times
        self.h2_PT_hd3 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.h2_PT_hd3_conv = nn.Conv2d(filters[1], self.CatChannels, 3, padding=1)
        self.h2_PT_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.h2_PT_hd3_relu = nn.ReLU(inplace=True)

        # h3->80*80, hd3->80*80, Concatenation
        self.h3_Cat_hd3_conv = nn.Conv2d(filters[2], self.CatChannels, 3, padding=1)
        self.h3_Cat_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.h3_Cat_hd3_relu = nn.ReLU(inplace=True)

        # hd4->40*40, hd4->80*80, Upsample 2 times
        self.hd4_UT_hd3 = nn.Upsample(scale_factor=2, mode='bilinear')  # 14*14
        self.hd4_UT_hd3_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd4_UT_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd4_UT_hd3_relu = nn.ReLU(inplace=True)

        # hd5->20*20, hd4->80*80, Upsample 4 times
        self.hd5_UT_hd3 = nn.Upsample(scale_factor=4, mode='bilinear')  # 14*14
        self.hd5_UT_hd3_conv = nn.Conv2d(filters[4], self.CatChannels, 3, padding=1)
        self.hd5_UT_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd5_UT_hd3_relu = nn.ReLU(inplace=True)

        # fusion(h1_PT_hd3, h2_PT_hd3, h3_Cat_hd3, hd4_UT_hd3, hd5_UT_hd3)
        self.conv3d_1 = nn.Conv2d(self.UpChannels, self.UpChannels, 3, padding=1)  # 16
        self.bn3d_1 = nn.BatchNorm2d(self.UpChannels)
        self.relu3d_1 = nn.ReLU(inplace=True)

        self.conv3d_res = unetConv2_res(self.UpChannels, self.UpChannels, n=resb_num)
        # self.conv3d_res6 = unetConv2_res(self.UpChannels+filters[3], self.UpChannels, n=resb_num)
        '''stage 2d '''
        # h1->320*320, hd2->160*160, Pooling 2 times
        self.h1_PT_hd2 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.h1_PT_hd2_conv = nn.Conv2d(filters[0], self.CatChannels, 3, padding=1)
        self.h1_PT_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.h1_PT_hd2_relu = nn.ReLU(inplace=True)

        # h2->160*160, hd2->160*160, Concatenation
        self.h2_Cat_hd2_conv = nn.Conv2d(filters[1], self.CatChannels, 3, padding=1)
        self.h2_Cat_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.h2_Cat_hd2_relu = nn.ReLU(inplace=True)

        # hd3->80*80, hd2->160*160, Upsample 2 times
        self.hd3_UT_hd2 = nn.Upsample(scale_factor=2, mode='bilinear')  # 14*14
        self.hd3_UT_hd2_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd3_UT_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd3_UT_hd2_relu = nn.ReLU(inplace=True)

        # hd4->40*40, hd2->160*160, Upsample 4 times
        self.hd4_UT_hd2 = nn.Upsample(scale_factor=4, mode='bilinear')  # 14*14
        self.hd4_UT_hd2_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd4_UT_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd4_UT_hd2_relu = nn.ReLU(inplace=True)

        # hd5->20*20, hd2->160*160, Upsample 8 times
        self.hd5_UT_hd2 = nn.Upsample(scale_factor=8, mode='bilinear')  # 14*14
        self.hd5_UT_hd2_conv = nn.Conv2d(filters[4], self.CatChannels, 3, padding=1)
        self.hd5_UT_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd5_UT_hd2_relu = nn.ReLU(inplace=True)

        # fusion(h1_PT_hd2, h2_Cat_hd2, hd3_UT_hd2, hd4_UT_hd2, hd5_UT_hd2)
        self.conv2d_1 = nn.Conv2d(self.UpChannels, self.UpChannels, 3, padding=1)  # 16
        self.bn2d_1 = nn.BatchNorm2d(self.UpChannels)
        self.relu2d_1 = nn.ReLU(inplace=True)

        self.conv2d_res = unetConv2_res(self.UpChannels, self.UpChannels, n=resb_num)
        # self.conv2d_res7 = unetConv2_res(self.UpChannels+filters[2]+filters[3], self.UpChannels, n=resb_num)
        '''stage 1d'''
        # h1->320*320, hd1->320*320, Concatenation
        self.h1_Cat_hd1_conv = nn.Conv2d(filters[0], self.CatChannels, 3, padding=1)
        self.h1_Cat_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.h1_Cat_hd1_relu = nn.ReLU(inplace=True)

        # hd2->160*160, hd1->320*320, Upsample 2 times
        self.hd2_UT_hd1 = nn.Upsample(scale_factor=2, mode='bilinear')  # 14*14
        self.hd2_UT_hd1_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd2_UT_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd2_UT_hd1_relu = nn.ReLU(inplace=True)

        # hd3->80*80, hd1->320*320, Upsample 4 times
        self.hd3_UT_hd1 = nn.Upsample(scale_factor=4, mode='bilinear')  # 14*14
        self.hd3_UT_hd1_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd3_UT_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd3_UT_hd1_relu = nn.ReLU(inplace=True)

        # hd4->40*40, hd1->320*320, Upsample 8 times
        self.hd4_UT_hd1 = nn.Upsample(scale_factor=8, mode='bilinear')  # 14*14
        self.hd4_UT_hd1_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd4_UT_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd4_UT_hd1_relu = nn.ReLU(inplace=True)

        # hd5->20*20, hd1->320*320, Upsample 16 times
        self.hd5_UT_hd1 = nn.Upsample(scale_factor=16, mode='bilinear')  # 14*14
        self.hd5_UT_hd1_conv = nn.Conv2d(filters[4], self.CatChannels, 3, padding=1)
        self.hd5_UT_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd5_UT_hd1_relu = nn.ReLU(inplace=True)

        # fusion(h1_Cat_hd1, hd2_UT_hd1, hd3_UT_hd1, hd4_UT_hd1, hd5_UT_hd1)
        self.conv1d_1 = nn.Conv2d(self.UpChannels, self.UpChannels, 3, padding=1)  # 16
        self.bn1d_1 = nn.BatchNorm2d(self.UpChannels)
        self.relu1d_1 = nn.ReLU(inplace=True)

        self.conv1d_res = unetConv2_res(self.UpChannels, self.UpChannels, n=resb_num)
        # self.conv1d_res8=unetConv2_res(self.UpChannels+filters[1]+filters[2]+filters[3], self.UpChannels, n=resb_num)
        # output
        self.outconv1 = nn.Conv2d(self.UpChannels, n_classes, 3, padding=1)
        # =====================MSOF=============================
        self.fuseconv = nn.Conv2d(n_classes * 5, n_classes, 3, padding=1)

        # -------------Bilinear Upsampling--------------
        # self.upscore6 = nn.Upsample(scale_factor=32, mode='bilinear')  ###
        self.upscore5 = nn.Upsample(scale_factor=16, mode='bilinear')
        self.upscore4 = nn.Upsample(scale_factor=8, mode='bilinear')
        self.upscore3 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.upscore2 = nn.Upsample(scale_factor=2, mode='bilinear')

        # DeepSup
        self.outconv1 = nn.Conv2d(self.UpChannels, n_classes, 3, padding=1)
        self.outconv2 = nn.Conv2d(self.UpChannels, n_classes, 3, padding=1)
        self.outconv3 = nn.Conv2d(self.UpChannels, n_classes, 3, padding=1)
        self.outconv4 = nn.Conv2d(self.UpChannels, n_classes, 3, padding=1)
        self.outconv5 = nn.Conv2d(filters[4], n_classes, 3, padding=1)



        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        ## -------------Encoder-------------
        h1 = self.conv1(inputs)  # h1->[4,16,512,512]
        h2_0 = self.maxpool1(h1) #[4,16,256,256]
        h2 = self.conv2(h2_0) #[4,32,256,256] # h2->160*1
        if self.use_dense:

            #h3 = self.conv33(torch.cat([F.max_pool2d(h1, 4), F.max_pool2d(h2, 2)], dim=1))
            # h4 = self.conv44(torch.cat([F.max_pool2d(h1, 8), F.max_pool2d(h2, 4), F.max_pool2d(h3, 2)], dim=1))
            # hd5 = self.conv55(torch.cat([F.max_pool2d(h1, 16), F.max_pool2d(h2, 8), F.max_pool2d(h3, 4), F.max_pool2d(h4, 2)], dim=1))
            #using skip connections between two successive pooling
            h3_0=self.conv33(torch.cat([h2_0,h2],dim=1))
            h3_1=self.maxpool1(h3_0)
            h3=self.conv3(h3_1)

            h4_0 = self.conv44(torch.cat([h3_1, h3], dim=1))
            h4_1 = self.maxpool1(h4_0)
            h4= self.conv4(h4_1)

            h5_0 = self.conv55(torch.cat([h4_1, h4], dim=1))
            h5_1 = self.maxpool1(h5_0)
            hd5 = self.conv5(h5_1)

            if self.Dblock:
                hd5 = self.Dblock(hd5)


        else:
            h3 = self.maxpool2(h2)
            h3 = self.conv3(h3)  # h3->80*80*256

            h4 = self.maxpool3(h3)
            h4 = self.conv4(h4)  # h4->40*40*512

            h5 = self.maxpool4(h4)  # [20*20*512]
            hd5 = self.conv5(h5)  # h5->20*20*1024
        # ========================DBlock==================================

        ## -------------Decoder-------------
        h1_PT_hd4 = self.h1_PT_hd4_relu(self.h1_PT_hd4_bn(self.h1_PT_hd4_conv(self.h1_PT_hd4(h1))))  # [2,64,40,40]
        h2_PT_hd4 = self.h2_PT_hd4_relu(self.h2_PT_hd4_bn(self.h2_PT_hd4_conv(self.h2_PT_hd4(h2))))  # [2,64,40,40]
        h3_PT_hd4 = self.h3_PT_hd4_relu(self.h3_PT_hd4_bn(self.h3_PT_hd4_conv(self.h3_PT_hd4(h3))))  # [2,64,40,40]
        h4_Cat_hd4 = self.h4_Cat_hd4_relu(self.h4_Cat_hd4_bn(self.h4_Cat_hd4_conv(h4)))  # [2,64,40,40]
        hd5_UT_hd4 = self.hd5_UT_hd4_relu(
            self.hd5_UT_hd4_bn(self.hd5_UT_hd4_conv(self.hd5_UT_hd4(hd5))))  # [2,64,40,40]
        if self.use_res:
            hd4 = self.conv4d_res(torch.cat([h1_PT_hd4, h2_PT_hd4, h3_PT_hd4, h4_Cat_hd4, hd5_UT_hd4], 1))
        else:
            hd4 = self.relu4d_1(self.bn4d_1(self.conv4d_1(
                torch.cat((h1_PT_hd4, h2_PT_hd4, h3_PT_hd4, h4_Cat_hd4, hd5_UT_hd4),
                          1))))  # [2,320,40,40] hd4->40*40*UpChannels

        h1_PT_hd3 = self.h1_PT_hd3_relu(self.h1_PT_hd3_bn(self.h1_PT_hd3_conv(self.h1_PT_hd3(h1))))
        h2_PT_hd3 = self.h2_PT_hd3_relu(self.h2_PT_hd3_bn(self.h2_PT_hd3_conv(self.h2_PT_hd3(h2))))
        h3_Cat_hd3 = self.h3_Cat_hd3_relu(self.h3_Cat_hd3_bn(self.h3_Cat_hd3_conv(h3)))
        hd4_UT_hd3 = self.hd4_UT_hd3_relu(self.hd4_UT_hd3_bn(self.hd4_UT_hd3_conv(self.hd4_UT_hd3(hd4))))
        hd5_UT_hd3 = self.hd5_UT_hd3_relu(self.hd5_UT_hd3_bn(self.hd5_UT_hd3_conv(self.hd5_UT_hd3(hd5))))
        if self.use_res:
            hd3 = self.conv3d_res(torch.cat([h1_PT_hd3, h2_PT_hd3, h3_Cat_hd3, hd4_UT_hd3, hd5_UT_hd3], 1))
            # hd3 = self.conv3d_res6(torch.cat([h1_PT_hd3, h2_PT_hd3, h3_Cat_hd3,self.upscore2(h4), hd4_UT_hd3, hd5_UT_hd3], 1))
        else:

            hd3 = self.relu3d_1(self.bn3d_1(self.conv3d_1(
                torch.cat((h1_PT_hd3, h2_PT_hd3, h3_Cat_hd3, hd4_UT_hd3, hd5_UT_hd3), 1))))  # hd3->80*80*UpChannels

        h1_PT_hd2 = self.h1_PT_hd2_relu(self.h1_PT_hd2_bn(self.h1_PT_hd2_conv(self.h1_PT_hd2(h1))))
        h2_Cat_hd2 = self.h2_Cat_hd2_relu(self.h2_Cat_hd2_bn(self.h2_Cat_hd2_conv(h2)))
        hd3_UT_hd2 = self.hd3_UT_hd2_relu(self.hd3_UT_hd2_bn(self.hd3_UT_hd2_conv(self.hd3_UT_hd2(hd3))))
        hd4_UT_hd2 = self.hd4_UT_hd2_relu(self.hd4_UT_hd2_bn(self.hd4_UT_hd2_conv(self.hd4_UT_hd2(hd4))))
        hd5_UT_hd2 = self.hd5_UT_hd2_relu(self.hd5_UT_hd2_bn(self.hd5_UT_hd2_conv(self.hd5_UT_hd2(hd5))))
        if self.use_res:
            hd2 = self.conv2d_res(torch.cat([h1_PT_hd2, h2_Cat_hd2, hd3_UT_hd2, hd4_UT_hd2, hd5_UT_hd2], 1))
            # hd2 = self.conv2d_res7(torch.cat([h1_PT_hd2, h2_Cat_hd2,self.upscore2(h3),self.upscore3(h4), hd3_UT_hd2, hd4_UT_hd2, hd5_UT_hd2], 1))
        else:

            hd2 = self.relu2d_1(self.bn2d_1(self.conv2d_1(
                torch.cat((h1_PT_hd2, h2_Cat_hd2, hd3_UT_hd2, hd4_UT_hd2, hd5_UT_hd2), 1))))  # hd2->160*160*UpChannels

        h1_Cat_hd1 = self.h1_Cat_hd1_relu(self.h1_Cat_hd1_bn(self.h1_Cat_hd1_conv(h1)))
        hd2_UT_hd1 = self.hd2_UT_hd1_relu(self.hd2_UT_hd1_bn(self.hd2_UT_hd1_conv(self.hd2_UT_hd1(hd2))))
        hd3_UT_hd1 = self.hd3_UT_hd1_relu(self.hd3_UT_hd1_bn(self.hd3_UT_hd1_conv(self.hd3_UT_hd1(hd3))))
        hd4_UT_hd1 = self.hd4_UT_hd1_relu(self.hd4_UT_hd1_bn(self.hd4_UT_hd1_conv(self.hd4_UT_hd1(hd4))))
        hd5_UT_hd1 = self.hd5_UT_hd1_relu(self.hd5_UT_hd1_bn(self.hd5_UT_hd1_conv(self.hd5_UT_hd1(hd5))))
        if self.use_res:
            hd1 = self.conv1d_res(torch.cat([h1_Cat_hd1, hd2_UT_hd1, hd3_UT_hd1, hd4_UT_hd1, hd5_UT_hd1], 1))
            # hd1 = self.conv1d_res8(torch.cat([h1_Cat_hd1, self.upscore2(h2),self.upscore3(h3),self.upscore4(h4),hd2_UT_hd1, hd3_UT_hd1, hd4_UT_hd1, hd5_UT_hd1], 1))
        else:
            hd1 = self.relu1d_1(self.bn1d_1(self.conv1d_1(
                torch.cat((h1_Cat_hd1, hd2_UT_hd1, hd3_UT_hd1, hd4_UT_hd1, hd5_UT_hd1), 1))))  # hd1->320*320*UpChannels

        if self.use_deep_sup:
            if self.att_type:
                # ================method 1 0.7324========
                # cam5 = self.cam5(hd5)
                # pam5 = self.pam5(hd5)
                # d5 = self.outconv5(cam5+pam5)
                # d5 = self.upscore5(d5)  # 16->256
                #
                # cam4 = self.cam34(hd4)
                # pam4 = self.pam34(hd4)
                # d4 = self.outconv4(cam4+ pam4)
                # d4 = self.upscore4(d4)  # 32->256
                #
                # cam3 = self.cam34(hd3)
                # pam3 = self.pam34(hd3)
                # d3 = self.outconv3(cam3+ pam3)
                # d3 = self.upscore3(d3)  # 64->256
                #
                # d2 = self.cam12(hd2) * hd2
                # d2 = self.pam12(d2) * d2
                # d2 = self.outconv2(d2)
                # d2 = self.upscore2(d2)  # 128->256
                #
                # d1 = self.cam12(hd1) * hd1
                # d1 = self.pam12(d1) * d1
                # d1 = self.outconv1(d1)  # 256
                # fuse = torch.cat([d1, d2, d3, d4, d5], dim=1)
                # fuse = self.fuseconv(fuse)
                # =====method2 0.7346==================================
                # d5=self.cam5(hd5)*hd5
                # d5=self.pam(d5)*d5
                # d5 = self.outconv5(d5)
                # d5 = self.upscore5(d5)  # 16->256
                #
                # d4 = self.cam14(hd4) * hd4
                # d4= self.pam(d4) * d4
                # d4 = self.outconv4(d4)
                # d4 = self.upscore4(d4)  # 32->256
                #
                # d3 = self.cam14(hd3)* hd3
                # d3 = self.pam(d3) * d3
                # d3 = self.outconv3(d3)
                # d3 = self.upscore3(d3)  # 64->256
                #
                # d2 = self.cam14(hd2) * hd2
                # d2 = self.pam(d2) * d2
                # d2 = self.outconv2(d2)
                # d2 = self.upscore2(d2)  # 128->256
                #
                # d1 = self.cam14(hd1) * hd1
                # d1 = self.pam(d1) * d1
                # d1 = self.outconv1(d1)  # 256
                # fuse = torch.cat([d1, d2, d3, d4, d5], dim=1)
                # fuse = self.fuseconv(fuse)
                # ===============================method3 high-level CA+low-level SA 0.7257===============
                # if self.Dblock:
                #    hd4 = self.Dblock34(hd4)  # [8,80,32,32]
                #    hd3 = self.Dblock34(hd3)  # [8,80,64,64]
                hd345 = torch.cat(
                    [F.upsample_bilinear(hd5, scale_factor=4), F.upsample_bilinear(hd4, scale_factor=2), hd3], dim=1)
                d345 = self.cam345(hd345) * hd345
                d345 = F.upsample_bilinear(self.conv345(d345), scale_factor=4)

                hd12 = self.conv12(torch.cat([hd1, F.upsample_bilinear(hd2, scale_factor=2)], dim=1))
                att_1 = self.pam12_1(hd12)
                att_2 = self.pam12_2(hd12)
                att12 = F.sigmoid(att_1 + att_2)
                d12 = att12 * hd12

                d12345 = torch.cat([d345, d12], dim=1)
                dout = self.outconv12345(d12345)

                return F.sigmoid(dout)


            else:

                d5 = self.outconv5(hd5)
                d5 = self.upscore5(d5)  # 16->256

                d4 = self.outconv4(hd4)
                d4 = self.upscore4(d4)  # 32->256

                d3 = self.outconv3(hd3)
                d3 = self.upscore3(d3)  # 64->256

                d2 = self.outconv2(hd2)
                d2 = self.upscore2(d2)  # 128->256

                d1 = self.outconv1(hd1)  # 256
                fuse = torch.cat([d1, d2, d3, d4, d5], dim=1)
                fuse = self.fuseconv(fuse)

            # dout = self.refnet(fuse)
            # fuse = torch.cat((d5, d4, d3, d2, d1), dim=1)
            if self.use_rfnet:
                fuse=self.refnet(fuse)
            return F.sigmoid(fuse), F.sigmoid(d1), F.sigmoid(d2), F.sigmoid(d3), F.sigmoid(d4), F.sigmoid(d5)
        else:
            d1 = self.outconv1(hd1)  # d1->320*320*n_classes
            dout = self.refnet(d1)
            if self.n_classes == 1:
                return F.sigmoid(dout)
            return dout


'''
    UNet 3+ with deep supervision
'''


class UNet_3Plus_DeepSup(nn.Module):
    def __init__(self, in_channels=3, n_classes=1, feature_scale=4, is_deconv=True, is_batchnorm=True):
        super(UNet_3Plus_DeepSup, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / 4) for x in filters]
        ## -------------Encoder--------------
        self.conv1 = unetConv2(self.in_channels, filters[0], self.is_batchnorm)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = unetConv2(filters[0], filters[1], self.is_batchnorm)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = unetConv2(filters[1], filters[2], self.is_batchnorm)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4 = unetConv2(filters[2], filters[3], self.is_batchnorm)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)

        self.conv5 = unetConv2(filters[3], filters[4], self.is_batchnorm)

        ## -------------Decoder--------------
        self.CatChannels = filters[0]
        self.CatBlocks = 5
        self.UpChannels = self.CatChannels * self.CatBlocks

        '''stage 4d'''
        # h1->320*320, hd4->40*40, Pooling 8 times
        self.h1_PT_hd4 = nn.MaxPool2d(8, 8, ceil_mode=True)
        self.h1_PT_hd4_conv = nn.Conv2d(filters[0], self.CatChannels, 3, padding=1)
        self.h1_PT_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.h1_PT_hd4_relu = nn.ReLU(inplace=True)

        # h2->160*160, hd4->40*40, Pooling 4 times
        self.h2_PT_hd4 = nn.MaxPool2d(4, 4, ceil_mode=True)
        self.h2_PT_hd4_conv = nn.Conv2d(filters[1], self.CatChannels, 3, padding=1)
        self.h2_PT_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.h2_PT_hd4_relu = nn.ReLU(inplace=True)

        # h3->80*80, hd4->40*40, Pooling 2 times
        self.h3_PT_hd4 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.h3_PT_hd4_conv = nn.Conv2d(filters[2], self.CatChannels, 3, padding=1)
        self.h3_PT_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.h3_PT_hd4_relu = nn.ReLU(inplace=True)

        # h4->40*40, hd4->40*40, Concatenation
        self.h4_Cat_hd4_conv = nn.Conv2d(filters[3], self.CatChannels, 3, padding=1)
        self.h4_Cat_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.h4_Cat_hd4_relu = nn.ReLU(inplace=True)

        # hd5->20*20, hd4->40*40, Upsample 2 times
        self.hd5_UT_hd4 = nn.Upsample(scale_factor=2, mode='bilinear')  # 14*14
        self.hd5_UT_hd4_conv = nn.Conv2d(filters[4], self.CatChannels, 3, padding=1)
        self.hd5_UT_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd5_UT_hd4_relu = nn.ReLU(inplace=True)

        # fusion(h1_PT_hd4, h2_PT_hd4, h3_PT_hd4, h4_Cat_hd4, hd5_UT_hd4)
        self.conv4d_1 = nn.Conv2d(self.UpChannels, self.UpChannels, 3, padding=1)  # 16
        self.bn4d_1 = nn.BatchNorm2d(self.UpChannels)
        self.relu4d_1 = nn.ReLU(inplace=True)

        '''stage 3d'''
        # h1->320*320, hd3->80*80, Pooling 4 times
        self.h1_PT_hd3 = nn.MaxPool2d(4, 4, ceil_mode=True)
        self.h1_PT_hd3_conv = nn.Conv2d(filters[0], self.CatChannels, 3, padding=1)
        self.h1_PT_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.h1_PT_hd3_relu = nn.ReLU(inplace=True)

        # h2->160*160, hd3->80*80, Pooling 2 times
        self.h2_PT_hd3 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.h2_PT_hd3_conv = nn.Conv2d(filters[1], self.CatChannels, 3, padding=1)
        self.h2_PT_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.h2_PT_hd3_relu = nn.ReLU(inplace=True)

        # h3->80*80, hd3->80*80, Concatenation
        self.h3_Cat_hd3_conv = nn.Conv2d(filters[2], self.CatChannels, 3, padding=1)
        self.h3_Cat_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.h3_Cat_hd3_relu = nn.ReLU(inplace=True)

        # hd4->40*40, hd4->80*80, Upsample 2 times
        self.hd4_UT_hd3 = nn.Upsample(scale_factor=2, mode='bilinear')  # 14*14
        self.hd4_UT_hd3_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd4_UT_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd4_UT_hd3_relu = nn.ReLU(inplace=True)

        # hd5->20*20, hd4->80*80, Upsample 4 times
        self.hd5_UT_hd3 = nn.Upsample(scale_factor=4, mode='bilinear')  # 14*14
        self.hd5_UT_hd3_conv = nn.Conv2d(filters[4], self.CatChannels, 3, padding=1)
        self.hd5_UT_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd5_UT_hd3_relu = nn.ReLU(inplace=True)

        # fusion(h1_PT_hd3, h2_PT_hd3, h3_Cat_hd3, hd4_UT_hd3, hd5_UT_hd3)
        self.conv3d_1 = nn.Conv2d(self.UpChannels, self.UpChannels, 3, padding=1)  # 16
        self.bn3d_1 = nn.BatchNorm2d(self.UpChannels)
        self.relu3d_1 = nn.ReLU(inplace=True)

        '''stage 2d '''
        # h1->320*320, hd2->160*160, Pooling 2 times
        self.h1_PT_hd2 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.h1_PT_hd2_conv = nn.Conv2d(filters[0], self.CatChannels, 3, padding=1)
        self.h1_PT_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.h1_PT_hd2_relu = nn.ReLU(inplace=True)

        # h2->160*160, hd2->160*160, Concatenation
        self.h2_Cat_hd2_conv = nn.Conv2d(filters[1], self.CatChannels, 3, padding=1)
        self.h2_Cat_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.h2_Cat_hd2_relu = nn.ReLU(inplace=True)

        # hd3->80*80, hd2->160*160, Upsample 2 times
        self.hd3_UT_hd2 = nn.Upsample(scale_factor=2, mode='bilinear')  # 14*14
        self.hd3_UT_hd2_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd3_UT_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd3_UT_hd2_relu = nn.ReLU(inplace=True)

        # hd4->40*40, hd2->160*160, Upsample 4 times
        self.hd4_UT_hd2 = nn.Upsample(scale_factor=4, mode='bilinear')  # 14*14
        self.hd4_UT_hd2_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd4_UT_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd4_UT_hd2_relu = nn.ReLU(inplace=True)

        # hd5->20*20, hd2->160*160, Upsample 8 times
        self.hd5_UT_hd2 = nn.Upsample(scale_factor=8, mode='bilinear')  # 14*14
        self.hd5_UT_hd2_conv = nn.Conv2d(filters[4], self.CatChannels, 3, padding=1)
        self.hd5_UT_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd5_UT_hd2_relu = nn.ReLU(inplace=True)

        # fusion(h1_PT_hd2, h2_Cat_hd2, hd3_UT_hd2, hd4_UT_hd2, hd5_UT_hd2)
        self.conv2d_1 = nn.Conv2d(self.UpChannels, self.UpChannels, 3, padding=1)  # 16
        self.bn2d_1 = nn.BatchNorm2d(self.UpChannels)
        self.relu2d_1 = nn.ReLU(inplace=True)

        '''stage 1d'''
        # h1->320*320, hd1->320*320, Concatenation
        self.h1_Cat_hd1_conv = nn.Conv2d(filters[0], self.CatChannels, 3, padding=1)
        self.h1_Cat_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.h1_Cat_hd1_relu = nn.ReLU(inplace=True)

        # hd2->160*160, hd1->320*320, Upsample 2 times
        self.hd2_UT_hd1 = nn.Upsample(scale_factor=2, mode='bilinear')  # 14*14
        self.hd2_UT_hd1_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd2_UT_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd2_UT_hd1_relu = nn.ReLU(inplace=True)

        # hd3->80*80, hd1->320*320, Upsample 4 times
        self.hd3_UT_hd1 = nn.Upsample(scale_factor=4, mode='bilinear')  # 14*14
        self.hd3_UT_hd1_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd3_UT_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd3_UT_hd1_relu = nn.ReLU(inplace=True)

        # hd4->40*40, hd1->320*320, Upsample 8 times
        self.hd4_UT_hd1 = nn.Upsample(scale_factor=8, mode='bilinear')  # 14*14
        self.hd4_UT_hd1_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd4_UT_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd4_UT_hd1_relu = nn.ReLU(inplace=True)

        # hd5->20*20, hd1->320*320, Upsample 16 times
        self.hd5_UT_hd1 = nn.Upsample(scale_factor=16, mode='bilinear')  # 14*14
        self.hd5_UT_hd1_conv = nn.Conv2d(filters[4], self.CatChannels, 3, padding=1)
        self.hd5_UT_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd5_UT_hd1_relu = nn.ReLU(inplace=True)

        # fusion(h1_Cat_hd1, hd2_UT_hd1, hd3_UT_hd1, hd4_UT_hd1, hd5_UT_hd1)
        self.conv1d_1 = nn.Conv2d(self.UpChannels, self.UpChannels, 3, padding=1)  # 16
        self.bn1d_1 = nn.BatchNorm2d(self.UpChannels)
        self.relu1d_1 = nn.ReLU(inplace=True)

        # -------------Bilinear Upsampling--------------
        self.upscore6 = nn.Upsample(scale_factor=32, mode='bilinear')  ###
        self.upscore5 = nn.Upsample(scale_factor=16, mode='bilinear')
        self.upscore4 = nn.Upsample(scale_factor=8, mode='bilinear')
        self.upscore3 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.upscore2 = nn.Upsample(scale_factor=2, mode='bilinear')

        # DeepSup
        self.outconv1 = nn.Conv2d(self.UpChannels, n_classes, 3, padding=1)
        self.outconv2 = nn.Conv2d(self.UpChannels, n_classes, 3, padding=1)
        self.outconv3 = nn.Conv2d(self.UpChannels, n_classes, 3, padding=1)
        self.outconv4 = nn.Conv2d(self.UpChannels, n_classes, 3, padding=1)
        self.outconv5 = nn.Conv2d(filters[4], n_classes, 3, padding=1)
        # OCR
        last_inp_channels, ocr_mid_channels, ocr_key_channels = self.UpChannels * 4, 256, 256
        from models.OCRModule import SpatialGather_Module, SpatialOCR_Module
        # self.down5=
        self.conv3x3_ocr = nn.Sequential(
            nn.Conv2d(last_inp_channels, ocr_mid_channels,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(ocr_mid_channels),
            nn.ReLU(inplace=True),
        )
        self.ocr_gather_head = SpatialGather_Module(n_classes)

        self.ocr_distri_head = SpatialOCR_Module(in_channels=ocr_mid_channels,
                                                 key_channels=ocr_key_channels,
                                                 out_channels=ocr_mid_channels,
                                                 scale=1,
                                                 dropout=0.05,
                                                 )
        self.cls_head = nn.Conv2d(
            ocr_mid_channels, n_classes, kernel_size=1, stride=1, padding=0, bias=True)

        self.aux_head = nn.Sequential(
            nn.Conv2d(last_inp_channels, last_inp_channels,
                      kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(last_inp_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(last_inp_channels, n_classes,
                      kernel_size=1, stride=1, padding=0, bias=True)
        )
        self.use_sup = True
        self.seg_head = nn.Conv2d(
            5 * n_classes, n_classes, kernel_size=1, stride=1, padding=0, bias=True)
        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        ## -------------Encoder-------------
        h1 = self.conv1(inputs)  # h1->320*320*64

        h2 = self.maxpool1(h1)
        h2 = self.conv2(h2)  # h2->160*160*128

        h3 = self.maxpool2(h2)
        h3 = self.conv3(h3)  # h3->80*80*256

        h4 = self.maxpool3(h3)
        h4 = self.conv4(h4)  # h4->40*40*512

        h5 = self.maxpool4(h4)
        hd5 = self.conv5(h5)  # h5->20*20*1024

        ## -------------Decoder-------------
        h1_PT_hd4 = self.h1_PT_hd4_relu(self.h1_PT_hd4_bn(self.h1_PT_hd4_conv(self.h1_PT_hd4(h1))))
        h2_PT_hd4 = self.h2_PT_hd4_relu(self.h2_PT_hd4_bn(self.h2_PT_hd4_conv(self.h2_PT_hd4(h2))))
        h3_PT_hd4 = self.h3_PT_hd4_relu(self.h3_PT_hd4_bn(self.h3_PT_hd4_conv(self.h3_PT_hd4(h3))))
        h4_Cat_hd4 = self.h4_Cat_hd4_relu(self.h4_Cat_hd4_bn(self.h4_Cat_hd4_conv(h4)))
        hd5_UT_hd4 = self.hd5_UT_hd4_relu(self.hd5_UT_hd4_bn(self.hd5_UT_hd4_conv(self.hd5_UT_hd4(hd5))))
        hd4 = self.relu4d_1(self.bn4d_1(self.conv4d_1(
            torch.cat((h1_PT_hd4, h2_PT_hd4, h3_PT_hd4, h4_Cat_hd4, hd5_UT_hd4), 1))))  # hd4->40*40*UpChannels

        h1_PT_hd3 = self.h1_PT_hd3_relu(self.h1_PT_hd3_bn(self.h1_PT_hd3_conv(self.h1_PT_hd3(h1))))
        h2_PT_hd3 = self.h2_PT_hd3_relu(self.h2_PT_hd3_bn(self.h2_PT_hd3_conv(self.h2_PT_hd3(h2))))
        h3_Cat_hd3 = self.h3_Cat_hd3_relu(self.h3_Cat_hd3_bn(self.h3_Cat_hd3_conv(h3)))
        hd4_UT_hd3 = self.hd4_UT_hd3_relu(self.hd4_UT_hd3_bn(self.hd4_UT_hd3_conv(self.hd4_UT_hd3(hd4))))
        hd5_UT_hd3 = self.hd5_UT_hd3_relu(self.hd5_UT_hd3_bn(self.hd5_UT_hd3_conv(self.hd5_UT_hd3(hd5))))
        hd3 = self.relu3d_1(self.bn3d_1(self.conv3d_1(
            torch.cat((h1_PT_hd3, h2_PT_hd3, h3_Cat_hd3, hd4_UT_hd3, hd5_UT_hd3), 1))))  # hd3->80*80*UpChannels

        h1_PT_hd2 = self.h1_PT_hd2_relu(self.h1_PT_hd2_bn(self.h1_PT_hd2_conv(self.h1_PT_hd2(h1))))
        h2_Cat_hd2 = self.h2_Cat_hd2_relu(self.h2_Cat_hd2_bn(self.h2_Cat_hd2_conv(h2)))
        hd3_UT_hd2 = self.hd3_UT_hd2_relu(self.hd3_UT_hd2_bn(self.hd3_UT_hd2_conv(self.hd3_UT_hd2(hd3))))
        hd4_UT_hd2 = self.hd4_UT_hd2_relu(self.hd4_UT_hd2_bn(self.hd4_UT_hd2_conv(self.hd4_UT_hd2(hd4))))
        hd5_UT_hd2 = self.hd5_UT_hd2_relu(self.hd5_UT_hd2_bn(self.hd5_UT_hd2_conv(self.hd5_UT_hd2(hd5))))
        hd2 = self.relu2d_1(self.bn2d_1(self.conv2d_1(
            torch.cat((h1_PT_hd2, h2_Cat_hd2, hd3_UT_hd2, hd4_UT_hd2, hd5_UT_hd2), 1))))  # hd2->160*160*UpChannels

        h1_Cat_hd1 = self.h1_Cat_hd1_relu(self.h1_Cat_hd1_bn(self.h1_Cat_hd1_conv(h1)))
        hd2_UT_hd1 = self.hd2_UT_hd1_relu(self.hd2_UT_hd1_bn(self.hd2_UT_hd1_conv(self.hd2_UT_hd1(hd2))))
        hd3_UT_hd1 = self.hd3_UT_hd1_relu(self.hd3_UT_hd1_bn(self.hd3_UT_hd1_conv(self.hd3_UT_hd1(hd3))))
        hd4_UT_hd1 = self.hd4_UT_hd1_relu(self.hd4_UT_hd1_bn(self.hd4_UT_hd1_conv(self.hd4_UT_hd1(hd4))))
        hd5_UT_hd1 = self.hd5_UT_hd1_relu(self.hd5_UT_hd1_bn(self.hd5_UT_hd1_conv(self.hd5_UT_hd1(hd5))))
        hd1 = self.relu1d_1(self.bn1d_1(self.conv1d_1(
            torch.cat((h1_Cat_hd1, hd2_UT_hd1, hd3_UT_hd1, hd4_UT_hd1, hd5_UT_hd1), 1))))  # hd1->320*320*UpChannels

        # ===============deep supervisoin=========================
        if self.use_sup:
            d5 = self.outconv5(hd5)
            d5 = self.upscore5(d5)  # 16->256

            d4 = self.outconv4(hd4)
            d4 = self.upscore4(d4)  # 32->256

            d3 = self.outconv3(hd3)
            d3 = self.upscore3(d3)  # 64->256

            d2 = self.outconv2(hd2)
            d2 = self.upscore2(d2)  # 128->256

            d1 = self.outconv1(hd1)  # 256

            fuse = torch.cat((d5, d4, d3, d2, d1), dim=1)
            # return F.sigmoid(d1), F.sigmoid(d2), F.sigmoid(d3), F.sigmoid(d4), F.sigmoid(d5)
            return F.sigmoid(self.seg_head(fuse))

        else:  # use OCR
            # d5=self.upscore5(hd5)
            d4 = self.upscore4(hd4)
            d3 = self.upscore3(hd3)
            d2 = self.upscore2(hd2)
            d1 = hd1
            feats = torch.cat((d4, d3, d2, d1), dim=1)
            out_aux = self.aux_head(feats)
            feats = self.conv3x3_ocr(feats)
            # ref. Object-Contextual Representations for Semantic Segmentation
            context = self.ocr_gather_head(feats, out_aux)  # Eq.(4) [2,512,19,1]
            feats = self.ocr_distri_head(feats, context)
            out = self.cls_head(feats)
            out_aux_seg = []
            out_aux_seg.append(out_aux)
            out_aux_seg.append(out)
            return out_aux_seg
            '''
            out_aux_seg = []
  
          # ocr
          out_aux = self.aux_head(feats)#[2,19,64,64] generate object representation using conbined features
          # compute contrast feature
          feats = self.conv3x3_ocr(feats)#[2,512,64,64]
          #ref. Object-Contextual Representations for Semantic Segmentation
          context = self.ocr_gather_head(feats, out_aux)#Eq.(4) [2,512,19,1]
          feats = self.ocr_distri_head(feats, context)#Eq.(3) [2,512,64,64]
  
          out = self.cls_head(feats)
  
          out_aux_seg.append(out_aux)
          out_aux_seg.append(out)
            '''


'''
    UNet 3+ with deep supervision and class-guided module
'''


class UNet_3Plus_DeepSup_CGM(nn.Module):

    def __init__(self, in_channels=3, n_classes=1, feature_scale=4, is_deconv=True, is_batchnorm=True):
        super(UNet_3Plus_DeepSup_CGM, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale

        filters = [64, 128, 256, 512, 1024]

        ## -------------Encoder--------------
        self.conv1 = unetConv2(self.in_channels, filters[0], self.is_batchnorm)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = unetConv2(filters[0], filters[1], self.is_batchnorm)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = unetConv2(filters[1], filters[2], self.is_batchnorm)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4 = unetConv2(filters[2], filters[3], self.is_batchnorm)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)

        self.conv5 = unetConv2(filters[3], filters[4], self.is_batchnorm)

        ## -------------Decoder--------------
        self.CatChannels = filters[0]
        self.CatBlocks = 5
        self.UpChannels = self.CatChannels * self.CatBlocks

        '''stage 4d'''
        # h1->320*320, hd4->40*40, Pooling 8 times
        self.h1_PT_hd4 = nn.MaxPool2d(8, 8, ceil_mode=True)
        self.h1_PT_hd4_conv = nn.Conv2d(filters[0], self.CatChannels, 3, padding=1)
        self.h1_PT_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.h1_PT_hd4_relu = nn.ReLU(inplace=True)

        # h2->160*160, hd4->40*40, Pooling 4 times
        self.h2_PT_hd4 = nn.MaxPool2d(4, 4, ceil_mode=True)
        self.h2_PT_hd4_conv = nn.Conv2d(filters[1], self.CatChannels, 3, padding=1)
        self.h2_PT_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.h2_PT_hd4_relu = nn.ReLU(inplace=True)

        # h3->80*80, hd4->40*40, Pooling 2 times
        self.h3_PT_hd4 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.h3_PT_hd4_conv = nn.Conv2d(filters[2], self.CatChannels, 3, padding=1)
        self.h3_PT_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.h3_PT_hd4_relu = nn.ReLU(inplace=True)

        # h4->40*40, hd4->40*40, Concatenation
        self.h4_Cat_hd4_conv = nn.Conv2d(filters[3], self.CatChannels, 3, padding=1)
        self.h4_Cat_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.h4_Cat_hd4_relu = nn.ReLU(inplace=True)

        # hd5->20*20, hd4->40*40, Upsample 2 times
        self.hd5_UT_hd4 = nn.Upsample(scale_factor=2, mode='bilinear')  # 14*14
        self.hd5_UT_hd4_conv = nn.Conv2d(filters[4], self.CatChannels, 3, padding=1)
        self.hd5_UT_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd5_UT_hd4_relu = nn.ReLU(inplace=True)

        # fusion(h1_PT_hd4, h2_PT_hd4, h3_PT_hd4, h4_Cat_hd4, hd5_UT_hd4)
        self.conv4d_1 = nn.Conv2d(self.UpChannels, self.UpChannels, 3, padding=1)  # 16
        self.bn4d_1 = nn.BatchNorm2d(self.UpChannels)
        self.relu4d_1 = nn.ReLU(inplace=True)

        '''stage 3d'''
        # h1->320*320, hd3->80*80, Pooling 4 times
        self.h1_PT_hd3 = nn.MaxPool2d(4, 4, ceil_mode=True)
        self.h1_PT_hd3_conv = nn.Conv2d(filters[0], self.CatChannels, 3, padding=1)
        self.h1_PT_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.h1_PT_hd3_relu = nn.ReLU(inplace=True)

        # h2->160*160, hd3->80*80, Pooling 2 times
        self.h2_PT_hd3 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.h2_PT_hd3_conv = nn.Conv2d(filters[1], self.CatChannels, 3, padding=1)
        self.h2_PT_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.h2_PT_hd3_relu = nn.ReLU(inplace=True)

        # h3->80*80, hd3->80*80, Concatenation
        self.h3_Cat_hd3_conv = nn.Conv2d(filters[2], self.CatChannels, 3, padding=1)
        self.h3_Cat_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.h3_Cat_hd3_relu = nn.ReLU(inplace=True)

        # hd4->40*40, hd4->80*80, Upsample 2 times
        self.hd4_UT_hd3 = nn.Upsample(scale_factor=2, mode='bilinear')  # 14*14
        self.hd4_UT_hd3_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd4_UT_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd4_UT_hd3_relu = nn.ReLU(inplace=True)

        # hd5->20*20, hd4->80*80, Upsample 4 times
        self.hd5_UT_hd3 = nn.Upsample(scale_factor=4, mode='bilinear')  # 14*14
        self.hd5_UT_hd3_conv = nn.Conv2d(filters[4], self.CatChannels, 3, padding=1)
        self.hd5_UT_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd5_UT_hd3_relu = nn.ReLU(inplace=True)

        # fusion(h1_PT_hd3, h2_PT_hd3, h3_Cat_hd3, hd4_UT_hd3, hd5_UT_hd3)
        self.conv3d_1 = nn.Conv2d(self.UpChannels, self.UpChannels, 3, padding=1)  # 16
        self.bn3d_1 = nn.BatchNorm2d(self.UpChannels)
        self.relu3d_1 = nn.ReLU(inplace=True)

        '''stage 2d '''
        # h1->320*320, hd2->160*160, Pooling 2 times
        self.h1_PT_hd2 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.h1_PT_hd2_conv = nn.Conv2d(filters[0], self.CatChannels, 3, padding=1)
        self.h1_PT_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.h1_PT_hd2_relu = nn.ReLU(inplace=True)

        # h2->160*160, hd2->160*160, Concatenation
        self.h2_Cat_hd2_conv = nn.Conv2d(filters[1], self.CatChannels, 3, padding=1)
        self.h2_Cat_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.h2_Cat_hd2_relu = nn.ReLU(inplace=True)

        # hd3->80*80, hd2->160*160, Upsample 2 times
        self.hd3_UT_hd2 = nn.Upsample(scale_factor=2, mode='bilinear')  # 14*14
        self.hd3_UT_hd2_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd3_UT_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd3_UT_hd2_relu = nn.ReLU(inplace=True)

        # hd4->40*40, hd2->160*160, Upsample 4 times
        self.hd4_UT_hd2 = nn.Upsample(scale_factor=4, mode='bilinear')  # 14*14
        self.hd4_UT_hd2_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd4_UT_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd4_UT_hd2_relu = nn.ReLU(inplace=True)

        # hd5->20*20, hd2->160*160, Upsample 8 times
        self.hd5_UT_hd2 = nn.Upsample(scale_factor=8, mode='bilinear')  # 14*14
        self.hd5_UT_hd2_conv = nn.Conv2d(filters[4], self.CatChannels, 3, padding=1)
        self.hd5_UT_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd5_UT_hd2_relu = nn.ReLU(inplace=True)

        # fusion(h1_PT_hd2, h2_Cat_hd2, hd3_UT_hd2, hd4_UT_hd2, hd5_UT_hd2)
        self.conv2d_1 = nn.Conv2d(self.UpChannels, self.UpChannels, 3, padding=1)  # 16
        self.bn2d_1 = nn.BatchNorm2d(self.UpChannels)
        self.relu2d_1 = nn.ReLU(inplace=True)

        '''stage 1d'''
        # h1->320*320, hd1->320*320, Concatenation
        self.h1_Cat_hd1_conv = nn.Conv2d(filters[0], self.CatChannels, 3, padding=1)
        self.h1_Cat_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.h1_Cat_hd1_relu = nn.ReLU(inplace=True)

        # hd2->160*160, hd1->320*320, Upsample 2 times
        self.hd2_UT_hd1 = nn.Upsample(scale_factor=2, mode='bilinear')  # 14*14
        self.hd2_UT_hd1_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd2_UT_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd2_UT_hd1_relu = nn.ReLU(inplace=True)

        # hd3->80*80, hd1->320*320, Upsample 4 times
        self.hd3_UT_hd1 = nn.Upsample(scale_factor=4, mode='bilinear')  # 14*14
        self.hd3_UT_hd1_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd3_UT_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd3_UT_hd1_relu = nn.ReLU(inplace=True)

        # hd4->40*40, hd1->320*320, Upsample 8 times
        self.hd4_UT_hd1 = nn.Upsample(scale_factor=8, mode='bilinear')  # 14*14
        self.hd4_UT_hd1_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd4_UT_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd4_UT_hd1_relu = nn.ReLU(inplace=True)

        # hd5->20*20, hd1->320*320, Upsample 16 times
        self.hd5_UT_hd1 = nn.Upsample(scale_factor=16, mode='bilinear')  # 14*14
        self.hd5_UT_hd1_conv = nn.Conv2d(filters[4], self.CatChannels, 3, padding=1)
        self.hd5_UT_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd5_UT_hd1_relu = nn.ReLU(inplace=True)

        # fusion(h1_Cat_hd1, hd2_UT_hd1, hd3_UT_hd1, hd4_UT_hd1, hd5_UT_hd1)
        self.conv1d_1 = nn.Conv2d(self.UpChannels, self.UpChannels, 3, padding=1)  # 16
        self.bn1d_1 = nn.BatchNorm2d(self.UpChannels)
        self.relu1d_1 = nn.ReLU(inplace=True)

        # -------------Bilinear Upsampling--------------
        self.upscore6 = nn.Upsample(scale_factor=32, mode='bilinear')  ###
        self.upscore5 = nn.Upsample(scale_factor=16, mode='bilinear')
        self.upscore4 = nn.Upsample(scale_factor=8, mode='bilinear')
        self.upscore3 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.upscore2 = nn.Upsample(scale_factor=2, mode='bilinear')

        # DeepSup
        self.outconv1 = nn.Conv2d(self.UpChannels, n_classes, 3, padding=1)
        self.outconv2 = nn.Conv2d(self.UpChannels, n_classes, 3, padding=1)
        self.outconv3 = nn.Conv2d(self.UpChannels, n_classes, 3, padding=1)
        self.outconv4 = nn.Conv2d(self.UpChannels, n_classes, 3, padding=1)
        self.outconv5 = nn.Conv2d(filters[4], n_classes, 3, padding=1)

        self.cls = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Conv2d(filters[4], 2, 1),
            nn.AdaptiveMaxPool2d(1),
            nn.Sigmoid())

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')

    def dotProduct(self, seg, cls):
        B, N, H, W = seg.size()
        seg = seg.view(B, N, H * W)  # [2,1,320*320]
        final = torch.einsum("ijk,ij->ijk", [seg, cls])  # [2,1,320*320]*[2,1]=[2,1,320*320]
        final = final.view(B, N, H, W)
        return final

    def forward(self, inputs):
        ## -------------Encoder-------------
        h1 = self.conv1(inputs)  # h1->320*320*64

        h2 = self.maxpool1(h1)
        h2 = self.conv2(h2)  # h2->160*160*128

        h3 = self.maxpool2(h2)
        h3 = self.conv3(h3)  # h3->80*80*256

        h4 = self.maxpool3(h3)
        h4 = self.conv4(h4)  # h4->40*40*512

        h5 = self.maxpool4(h4)
        hd5 = self.conv5(h5)  # h5->20*20*1024

        # -------------Classification-------------
        cls_branch = self.cls(hd5).squeeze(3).squeeze(
            2)  # self.cls(hd5)==>[2,1024,20,20]==>[2,2,1,1]==>[1,2] (B,N,1,1)->(B,N)
        cls_branch_max = cls_branch.argmax(dim=1)  # [2]
        cls_branch_max = cls_branch_max[:, np.newaxis].float()  # [2,1]

        ## -------------Decoder-------------
        h1_PT_hd4 = self.h1_PT_hd4_relu(self.h1_PT_hd4_bn(self.h1_PT_hd4_conv(self.h1_PT_hd4(h1))))
        h2_PT_hd4 = self.h2_PT_hd4_relu(self.h2_PT_hd4_bn(self.h2_PT_hd4_conv(self.h2_PT_hd4(h2))))
        h3_PT_hd4 = self.h3_PT_hd4_relu(self.h3_PT_hd4_bn(self.h3_PT_hd4_conv(self.h3_PT_hd4(h3))))
        h4_Cat_hd4 = self.h4_Cat_hd4_relu(self.h4_Cat_hd4_bn(self.h4_Cat_hd4_conv(h4)))
        hd5_UT_hd4 = self.hd5_UT_hd4_relu(self.hd5_UT_hd4_bn(self.hd5_UT_hd4_conv(self.hd5_UT_hd4(hd5))))
        hd4 = self.relu4d_1(self.bn4d_1(self.conv4d_1(
            torch.cat((h1_PT_hd4, h2_PT_hd4, h3_PT_hd4, h4_Cat_hd4, hd5_UT_hd4), 1))))  # hd4->40*40*UpChannels

        h1_PT_hd3 = self.h1_PT_hd3_relu(self.h1_PT_hd3_bn(self.h1_PT_hd3_conv(self.h1_PT_hd3(h1))))
        h2_PT_hd3 = self.h2_PT_hd3_relu(self.h2_PT_hd3_bn(self.h2_PT_hd3_conv(self.h2_PT_hd3(h2))))
        h3_Cat_hd3 = self.h3_Cat_hd3_relu(self.h3_Cat_hd3_bn(self.h3_Cat_hd3_conv(h3)))
        hd4_UT_hd3 = self.hd4_UT_hd3_relu(self.hd4_UT_hd3_bn(self.hd4_UT_hd3_conv(self.hd4_UT_hd3(hd4))))
        hd5_UT_hd3 = self.hd5_UT_hd3_relu(self.hd5_UT_hd3_bn(self.hd5_UT_hd3_conv(self.hd5_UT_hd3(hd5))))
        hd3 = self.relu3d_1(self.bn3d_1(self.conv3d_1(
            torch.cat((h1_PT_hd3, h2_PT_hd3, h3_Cat_hd3, hd4_UT_hd3, hd5_UT_hd3), 1))))  # hd3->80*80*UpChannels

        h1_PT_hd2 = self.h1_PT_hd2_relu(self.h1_PT_hd2_bn(self.h1_PT_hd2_conv(self.h1_PT_hd2(h1))))
        h2_Cat_hd2 = self.h2_Cat_hd2_relu(self.h2_Cat_hd2_bn(self.h2_Cat_hd2_conv(h2)))
        hd3_UT_hd2 = self.hd3_UT_hd2_relu(self.hd3_UT_hd2_bn(self.hd3_UT_hd2_conv(self.hd3_UT_hd2(hd3))))
        hd4_UT_hd2 = self.hd4_UT_hd2_relu(self.hd4_UT_hd2_bn(self.hd4_UT_hd2_conv(self.hd4_UT_hd2(hd4))))
        hd5_UT_hd2 = self.hd5_UT_hd2_relu(self.hd5_UT_hd2_bn(self.hd5_UT_hd2_conv(self.hd5_UT_hd2(hd5))))
        hd2 = self.relu2d_1(self.bn2d_1(self.conv2d_1(
            torch.cat((h1_PT_hd2, h2_Cat_hd2, hd3_UT_hd2, hd4_UT_hd2, hd5_UT_hd2), 1))))  # hd2->160*160*UpChannels

        h1_Cat_hd1 = self.h1_Cat_hd1_relu(self.h1_Cat_hd1_bn(self.h1_Cat_hd1_conv(h1)))
        hd2_UT_hd1 = self.hd2_UT_hd1_relu(self.hd2_UT_hd1_bn(self.hd2_UT_hd1_conv(self.hd2_UT_hd1(hd2))))
        hd3_UT_hd1 = self.hd3_UT_hd1_relu(self.hd3_UT_hd1_bn(self.hd3_UT_hd1_conv(self.hd3_UT_hd1(hd3))))
        hd4_UT_hd1 = self.hd4_UT_hd1_relu(self.hd4_UT_hd1_bn(self.hd4_UT_hd1_conv(self.hd4_UT_hd1(hd4))))
        hd5_UT_hd1 = self.hd5_UT_hd1_relu(self.hd5_UT_hd1_bn(self.hd5_UT_hd1_conv(self.hd5_UT_hd1(hd5))))
        hd1 = self.relu1d_1(self.bn1d_1(self.conv1d_1(
            torch.cat((h1_Cat_hd1, hd2_UT_hd1, hd3_UT_hd1, hd4_UT_hd1, hd5_UT_hd1), 1))))  # hd1->320*320*UpChannels

        d5 = self.outconv5(hd5)
        d5 = self.upscore5(d5)  # [2,1,320,320]16->256

        d4 = self.outconv4(hd4)
        d4 = self.upscore4(d4)  # 32->256

        d3 = self.outconv3(hd3)
        d3 = self.upscore3(d3)  # 64->256

        d2 = self.outconv2(hd2)
        d2 = self.upscore2(d2)  # 128->256

        d1 = self.outconv1(hd1)  # [2,1,320,320]d1->320*320*n_classes

        d1 = self.dotProduct(d1, cls_branch_max)
        d2 = self.dotProduct(d2, cls_branch_max)
        d3 = self.dotProduct(d3, cls_branch_max)
        d4 = self.dotProduct(d4, cls_branch_max)
        d5 = self.dotProduct(d5, cls_branch_max)

        return F.sigmoid(d1), F.sigmoid(d2), F.sigmoid(d3), F.sigmoid(d4), F.sigmoid(d5)

class unet_2D(nn.Module):

    def __init__(self, feature_scale=4, n_classes=1, is_deconv=True, in_channels=3, is_batchnorm=True,use_att=False,use_res=True,
        dblock_type='AS',use_rfnet=False):
        super(unet_2D, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale
        self.n_class=n_classes
        self.use_att=False
        # self.use_PSP=False#

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]#[16,32,64,128,256]
        if use_res:
            conv_block=unetConv2_res

        else:
            conv_block=unetConv2

        # downsampling
        self.conv1 = conv_block(self.in_channels, filters[0], self.is_batchnorm)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = conv_block(filters[0], filters[1], self.is_batchnorm)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = conv_block(filters[1], filters[2], self.is_batchnorm)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4 = conv_block(filters[2], filters[3], self.is_batchnorm)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)

        self.center = conv_block(filters[3], filters[4], self.is_batchnorm)
        #self.dblock_typ=dblock_type
        if dblock_type=='ASPP':
           self.Dblock = ASPP(in_channel=filters[4])
           logger.info('With ASPP module.')
        elif dblock_type=='AS':
           self.Dblock=Dblock(filters[4])
           logger.info('With AS module.')
        else:
            logger.info('Without dblock module.')
            self.Dblock=None
        if use_rfnet:
            self.RefNet=Dblock(1)
            logger.info('With refine module.')
        else:
            logger.info('Without Refine module.')
            self.RefNet=None
        # upsampling
        self.up_concat4 = unetUp(filters[4], filters[3], self.is_deconv,use_res=use_res)
        self.up_concat3 = unetUp(filters[3], filters[2], self.is_deconv,use_res=use_res)
        self.up_concat2 = unetUp(filters[2], filters[1], self.is_deconv,use_res=use_res)
        self.up_concat1 = unetUp(filters[1], filters[0], self.is_deconv,use_res=use_res)


        # final conv (without any concat)
        self.final = nn.Conv2d(filters[0], n_classes, 1)

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')


    def forward(self, inputs):
        conv1 = self.conv1(inputs)#[1,6,256,256]==>[1,16,256,256]
        maxpool1 = self.maxpool1(conv1)#[1,16,256,256]==>[1,16,128,128]

        conv2 = self.conv2(maxpool1)#[1,16,128,128]==>[1,32,128,128]
        maxpool2 = self.maxpool2(conv2)#[1,32,128,128]==>[1,32,64,64]

        conv3 = self.conv3(maxpool2)#[1,32,64,64]==>[1,64,64,64]
        maxpool3 = self.maxpool3(conv3)#[1,64,64,64]==>[1,64,32,32]

        conv4 = self.conv4(maxpool3)#[1,64,32,32]==>[1,128,32,32]
        maxpool4 = self.maxpool4(conv4)#[1,128,32,32]==>[1,128,16,16]

        center = self.center(maxpool4)#[1,128,16,16]==>[1,256,16,16]
        if self.Dblock:
            center=self.Dblock(center)
        if self.use_att:

            up4 = self.up_concat4_att(conv4, center)  # [1,256,16,16]+[1,128,32,32]+conv==>[1,128,32,32]
            up3 = self.up_concat3_att(conv3, up4)  # [1,64,64,64]+[1,128,32,32]+conv==>[1,64,64,64]
            up2 = self.up_concat2_att(conv2, up3)  # [1,32,128,128]+[1,64,64,64]+conv==>[1,32,128,128]
            up1 = self.up_concat1_att(conv1, up2)
            #pass
        else:

           up4 = self.up_concat4(conv4, center)#[1,256,16,16]+[1,128,32,32]+conv==>[1,128,32,32]
           up3 = self.up_concat3(conv3, up4)#[1,64,64,64]+[1,128,32,32]+conv==>[1,64,64,64]
           up2 = self.up_concat2(conv2, up3)#[1,32,128,128]+[1,64,64,64]+conv==>[1,32,128,128]
           up1 = self.up_concat1(conv1, up2)#[1,16,256,256]+[1,32,128,128]+conv==>[1,16,256,256]

        final = self.final(up1)#[1,16,256,256]==>[1,1,256,256]
        if self.RefNet:
            final=self.RefNet(final)
        if self.n_class==1:
            return F.sigmoid(final)# for binary only
        else:
            return final

class unet_2D_Atk(nn.Module):

    def __init__(self, feature_scale=2, n_classes=1, is_deconv=True, in_channels=3, is_batchnorm=True,use_att=False,use_res=True,
        dblock_type="None",use_rfnet=False,lam_img=1/15):
        super(unet_2D_Atk, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale
        self.n_class=n_classes
        self.use_att=False
        self.lam_img=lam_img

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]#[16,32,64,128,256]
        if use_res:
            conv_block=unetConv2_res

        else:
            conv_block=unetConv2

        # downsampling
        self.conv1 = conv_block(self.in_channels, filters[0], False,act='lrelu')#no BN
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = conv_block(filters[0], filters[1], self.is_batchnorm,act='lrelu')
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = conv_block(filters[1], filters[2], self.is_batchnorm,act='lrelu')
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4 = conv_block(filters[2], filters[3], self.is_batchnorm,act='lrelu')
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)

        self.center = conv_block(filters[3], filters[4], self.is_batchnorm,act='lrelu')
        #self.dblock_typ=dblock_type
        if dblock_type=='ASPP':
           self.Dblock = ASPP(in_channel=filters[4])
           logger.info('With ASPP module.')
        elif dblock_type=='AS':
           self.Dblock=Dblock(filters[4])
           logger.info('With AS module.')
        else:
            logger.info('Without dblock module.')
            self.Dblock=None
        if use_rfnet:
            self.RefNet=Dblock(1)
            logger.info('With refine module.')
        else:
            logger.info('Without Refine module.')
            self.RefNet=None
        # upsampling
        self.up_concat4 = unetUp(filters[4], filters[3], self.is_deconv,use_res=use_res,act='lrelu')
        self.up_concat3 = unetUp(filters[3], filters[2], self.is_deconv,use_res=use_res,act='lrelu')
        self.up_concat2 = unetUp(filters[2], filters[1], self.is_deconv,use_res=use_res,act='lrelu')
        self.up_concat1 = unetUp(filters[1], filters[0], self.is_deconv,use_res=use_res,act='lrelu')


        # final conv (without any concat)
        self.final = nn.Conv2d(filters[0], n_classes, 1)
        self.tanh_act=nn.Tanh()



        # initialise weights
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         init_weights(m, init_type='kaiming')
        #     elif isinstance(m, nn.BatchNorm2d):
        #         init_weights(m, init_type='kaiming')


    def forward(self, inputs):
        conv1 = self.conv1(inputs)#[1,6,256,256]==>[1,16,256,256]
        maxpool1 = self.maxpool1(conv1)#[1,16,256,256]==>[1,16,128,128]

        conv2 = self.conv2(maxpool1)#[1,16,128,128]==>[1,32,128,128]
        maxpool2 = self.maxpool2(conv2)#[1,32,128,128]==>[1,32,64,64]

        conv3 = self.conv3(maxpool2)#[1,32,64,64]==>[1,64,64,64]
        maxpool3 = self.maxpool3(conv3)#[1,64,64,64]==>[1,64,32,32]

        conv4 = self.conv4(maxpool3)#[1,64,32,32]==>[1,128,32,32]
        maxpool4 = self.maxpool4(conv4)#[1,128,32,32]==>[1,128,16,16]

        center = self.center(maxpool4)#[1,128,16,16]==>[1,256,16,16]
        if self.Dblock:
            center=self.Dblock(center)
        if self.use_att:

            up4 = self.up_concat4_att(conv4, center)  # [1,256,16,16]+[1,128,32,32]+conv==>[1,128,32,32]
            up3 = self.up_concat3_att(conv3, up4)  # [1,64,64,64]+[1,128,32,32]+conv==>[1,64,64,64]
            up2 = self.up_concat2_att(conv2, up3)  # [1,32,128,128]+[1,64,64,64]+conv==>[1,32,128,128]
            up1 = self.up_concat1_att(conv1, up2)
            #pass
        else:

           up4 = self.up_concat4(conv4, center)#[1,256,16,16]+[1,128,32,32]+conv==>[1,128,32,32]
           up3 = self.up_concat3(conv3, up4)#[1,64,64,64]+[1,128,32,32]+conv==>[1,64,64,64]
           up2 = self.up_concat2(conv2, up3)#[1,32,128,128]+[1,64,64,64]+conv==>[1,32,128,128]
           up1 = self.up_concat1(conv1, up2)#[1,16,256,256]+[1,32,128,128]+conv==>[1,16,256,256]

        final = self.final(up1)#[1,16,256,256]==>[1,1,256,256]
        final=self.tanh_act(final)
        output=inputs*self.lam_img+final

        return final






class unet_2D_Att(nn.Module):

    def __init__(self, feature_scale=4, n_classes=1, is_deconv=True, in_channels=3, is_batchnorm=True,use_att=False,use_res=True,
        dblock_type='AS',use_rfnet=False):
        super(unet_2D_Att, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale
        self.n_class=n_classes
        self.use_att=use_att

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]#[16,32,64,128,256]
        if use_res:
            conv_block=unetConv2_res
            logger.info("using resblock conv...")
        else:
            conv_block=unetConv2
            logger.info("using basic conv...")

        if self.use_att:
            self.FPA=B.FPA(channels=filters[4])
            self.GAU4=B.GAU(filters[4],filters[3],use_res=True)
            self.GAU3 = B.GAU(filters[3],filters[2],use_res=True)
            self.GAU2 = B.GAU(filters[2],filters[1],use_res=True)
            self.GAU1 = B.GAU(filters[1],filters[0],use_res=True)
            logger.info("using FPA and GAU attetnion...")
        else:
            # upsampling
            self.up_concat4 = unetUp(filters[4], filters[3], self.is_deconv, use_res=use_res)
            self.up_concat3 = unetUp(filters[3], filters[2], self.is_deconv, use_res=use_res)
            self.up_concat2 = unetUp(filters[2], filters[1], self.is_deconv, use_res=use_res)
            self.up_concat1 = unetUp(filters[1], filters[0], self.is_deconv, use_res=use_res)
            logger.info("without FPA and GAU attetnion...")

        # downsampling
        self.conv1 = conv_block(self.in_channels, filters[0], self.is_batchnorm)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = conv_block(filters[0], filters[1], self.is_batchnorm)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = conv_block(filters[1], filters[2], self.is_batchnorm)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4 = conv_block(filters[2], filters[3], self.is_batchnorm)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)

        self.center = conv_block(filters[3], filters[4], self.is_batchnorm)
        #self.dblock_typ=dblock_type
        if dblock_type=='ASPP':
           self.Dblock = ASPP(in_channel=filters[4])
           logger.info('With ASPP module.')
        elif dblock_type=='AS':
           self.Dblock=Dblock(filters[4])
           logger.info('With AS module.')
        else:
            logger.info('Without dblock module.')
            self.Dblock=None
        if use_rfnet:
            self.RefNet=Dblock(1)
            logger.info('With refine module.')
        else:
            logger.info('Without Refine module.')
            self.RefNet=None



        # final conv (without any concat)
        self.final = nn.Conv2d(filters[0], n_classes, 1)

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')


    def forward(self, inputs):
        conv1 = self.conv1(inputs)#[1,6,256,256]==>[1,16,256,256]
        maxpool1 = self.maxpool1(conv1)#[1,16,256,256]==>[1,16,128,128]

        conv2 = self.conv2(maxpool1)#[1,16,128,128]==>[1,32,128,128]
        maxpool2 = self.maxpool2(conv2)#[1,32,128,128]==>[1,32,64,64]

        conv3 = self.conv3(maxpool2)#[1,32,64,64]==>[1,64,64,64]
        maxpool3 = self.maxpool3(conv3)#[1,64,64,64]==>[1,64,32,32]

        conv4 = self.conv4(maxpool3)#[1,64,32,32]==>[1,128,32,32]
        maxpool4 = self.maxpool4(conv4)#[1,128,32,32]==>[1,128,16,16]

        center = self.center(maxpool4)#[1,128,16,16]==>[1,256,16,16]
        if self.Dblock:
            center=self.Dblock(center)
        if self.use_att:
            center=self.FPA(center)
            up4=self.GAU4(center,conv4)
            up3 = self.GAU3(up4, conv3)
            up2 = self.GAU2(up3, conv2)
            up1 = self.GAU1(up2, conv1)

        else:

           up4 = self.up_concat4(conv4, center)#[1,256,16,16]+[1,128,32,32]+conv==>[1,128,32,32]
           up3 = self.up_concat3(conv3, up4)#[1,64,64,64]+[1,128,32,32]+conv==>[1,64,64,64]
           up2 = self.up_concat2(conv2, up3)#[1,32,128,128]+[1,64,64,64]+conv==>[1,32,128,128]
           up1 = self.up_concat1(conv1, up2)#[1,16,256,256]+[1,32,128,128]+conv==>[1,16,256,256]

        final = self.final(up1)#[1,16,256,256]==>[1,1,256,256]
        if self.RefNet:
            final=self.RefNet(final)
        if self.n_class==1:
            return F.sigmoid(final)# for binary only
        else:
            return final

class unet_2D_PreTrain_MS(nn.Module):
    '''
      unet2d+multi-scale input in the encoder: ref: U-Net Based Architecture for an Improved Multiresolution Segmentation in Medical Images
      unfortunatly, it does not work, the f1-score seem to drop during training
    '''

    def __init__(self, feature_scale=4, n_classes=1, is_deconv=True, in_channels=3, is_batchnorm=True,use_att=False,use_res=True,
        dblock_type='AS',use_rfnet=False,use_logits=False):
        super(unet_2D_PreTrain_MS, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale
        self.n_class=n_classes
        self.use_att=use_att

        # filters = [64, 128, 256, 512, 1024]
        # filters = [int(x / self.feature_scale) for x in filters]#[16,32,64,128,256]
        if use_res:
            conv_block=unetConv2_res
            logger.info("using resblock conv...")
        else:
            conv_block=unetConv2
            logger.info("using basic conv...")



        # downsampling
        from torchvision import models
        filters = [64, 128, 256, 512]
        resnet = models.resnet34(pretrained=True)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        #============for multi-scale input==================
        self.conv_ms0=conv_block(in_channels,filters[0])
        self.conv_ms1 = conv_block(filters[0], filters[0])
        self.conv_ms2 = conv_block(filters[0], filters[1])
        self.conv_ms3 = conv_block(filters[1], filters[2])
        self.conv_ms4 = conv_block(filters[2], filters[3])

        self.conv_c0=unetConv2(filters[0]*2,filters[0])
        self.conv_c1 = unetConv2(filters[0] * 2, filters[0])
        self.conv_c2 = unetConv2(filters[1] * 2, filters[1])
        self.conv_c3 = unetConv2(filters[2] * 2, filters[2])
        self.conv_c4 = unetConv2(filters[3] * 2, filters[3])
        #===================================================

        self.dblock = Dblock(512)

        if self.use_att:
            self.GAU4 = B.GAU(filters[3], filters[2], use_res=False)
            self.GAU3 = B.GAU(filters[2], filters[1], use_res=False)
            self.GAU2 = B.GAU(filters[1], filters[0], use_res=False)
            self.GAU1 = B.GAU(filters[0], filters[0], use_res=False)
            logger.info("using GAU attetnion...")
        else:
            self.decoder4 = B.DecoderBlock(filters[3], filters[2])
            self.decoder3 = B.DecoderBlock(filters[2], filters[1])
            self.decoder2 = B.DecoderBlock(filters[1], filters[0])
            self.decoder1 = B.DecoderBlock(filters[0], filters[0])
            logger.info("without GAU attetnion...")
        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)  # size*2
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, n_classes, 3, padding=1)




        # #self.dblock_typ=dblock_type
        # if dblock_type=='ASPP':
        #    self.Dblock = ASPP(in_channel=filters[4])
        #    logger.info('With ASPP module.')
        # elif dblock_type=='AS':
        #    self.Dblock=Dblock(filters[4])
        #    logger.info('With AS module.')
        # else:
        #     logger.info('Without dblock module.')
        #     self.Dblock=None

        if use_rfnet:
            self.RefNet=Dblock(1)
            logger.info('With refine module.')
        else:
            self.RefNet = None
            logger.info('Without Refine module.')


        self.use_logits=use_logits

    def forward(self, inputs):


        x = self.firstconv(inputs)  # [2,64,256,256]
        x = self.firstbn(x)
        x_conv = self.firstrelu(x)# [2,64,256,256]
        mx0=self.conv_ms0(F.max_pool2d(inputs,2))
        mx_c0=self.conv_c0(torch.cat([mx0,x_conv],dim=1))
        x1 = self.firstmaxpool(mx_c0)  # [2,64,128,128]

        e1_0 = self.encoder1(x1)  ##[2,64,128,128]
        mx1=self.conv_ms1(F.max_pool2d(mx0,2))
        e1=self.conv_c1(torch.cat([mx1,e1_0],dim=1))

        e2_0 = self.encoder2(e1)  # [2,128,64,64]
        mx2 = self.conv_ms2(F.max_pool2d(mx1, 2))
        e2 = self.conv_c2(torch.cat([mx2, e2_0], dim=1))

        e3_0 = self.encoder3(e2)  # [2,256,32,32]
        mx3 = self.conv_ms3(F.max_pool2d(mx2, 2))
        e3 = self.conv_c3(torch.cat([mx3, e3_0], dim=1))

        e4_0 = self.encoder4(e3)  # [2,512,16,16]
        mx4 = self.conv_ms4(F.max_pool2d(mx3, 2))
        e4 = self.conv_c4(torch.cat([mx4, e4_0], dim=1))


        # Center
        e4 = self.dblock(e4)  # [2,512,16,16]

        # Decoder
        if self.use_att:
            d4=self.GAU4(e4,e3)#16
            d3=self.GAU3(d4,e2)#32
            d2=self.GAU2(d3,e1)#64
            d1=self.GAU1(d2,mx_c0)#128
        else:
            d4 = self.decoder4(e4) + e3  # [2,256,16,16]
            d3 = self.decoder3(d4) + e2  # [2,128,32,32]
            d2 = self.decoder2(d3) + e1  # [2,64,64,64]
            d1 = self.decoder1(d2)  # [2,64,128,128]
        out = self.finaldeconv1(d1)  # [2,64,256,256]
        out = self.finalrelu1(out)
        out = self.finalconv2(out)  # [2,32,256,256]
        out = self.finalrelu2(out)
        final = self.finalconv3(out)  # [2,1,256,256]


        if self.RefNet:
            final=self.RefNet(final)
        if self.use_logits:
            return final
        else:
            if self.n_class == 1:
                return F.sigmoid(final)  # for binary only
            else:
                return final





class unet_2D_PreTrain(nn.Module):

    def __init__(self, feature_scale=4, n_classes=1, is_deconv=True, in_channels=3, is_batchnorm=True,use_att=False,use_res=True,
        dblock_type='AS',use_rfnet=False,use_logits=False,frozen_encoder=False,use_drop=False,drop_rate=0.2):
        super(unet_2D_PreTrain, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale
        self.n_class=n_classes
        self.use_att=use_att

        # filters = [64, 128, 256, 512, 1024]
        # filters = [int(x / self.feature_scale) for x in filters]#[16,32,64,128,256]
        if use_res:
            #conv_block=unetConv2_res
            logger.info("using resblock conv...")
        else:
            #conv_block=unetConv2
            logger.info("using basic conv...")


        # downsampling
        from torchvision import models
        filters = [64, 128, 256, 512]
        resnet = models.resnet34(pretrained=True)
        if frozen_encoder:
            for p in resnet.parameters():
                p.requires_grad=False
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.dblock = Dblock(512)

        if self.use_att:
            self.GAU4 = B.GAU(filters[3], filters[2], use_res=True)
            self.GAU3 = B.GAU(filters[2], filters[1], use_res=True)
            self.GAU2 = B.GAU(filters[1], filters[0], use_res=True)
            self.GAU1 = B.GAU(filters[0], filters[0], use_res=True)
            logger.info("using GAU attetnion...")
        else:
            self.decoder4 = B.DecoderBlock(filters[3], filters[2])
            self.decoder3 = B.DecoderBlock(filters[2], filters[1])
            self.decoder2 = B.DecoderBlock(filters[1], filters[0])
            self.decoder1 = B.DecoderBlock(filters[0], filters[0])
            logger.info("without GAU attetnion...")
        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)  # size*2
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, n_classes, 3, padding=1)





        if use_rfnet:
            self.RefNet=Dblock(1)
            logger.info('With refine module.')
        else:
            self.RefNet = None
            logger.info('Without Refine module.')


        self.use_logits=use_logits
        self.use_drop=use_drop
        if self.use_drop:
           self.drop_out=nn.Dropout(p=drop_rate)

    def forward(self, inputs):


        x = self.firstconv(inputs)  # [2,64,128,128]
        x = self.firstbn(x)
        x_conv = self.firstrelu(x)#[2,64,128,128]
        x1 = self.firstmaxpool(x_conv)  # [2,64,64,64]
        e1 = self.encoder1(x1)  ##[2,64,64,64]
        e2 = self.encoder2(e1)  # [2,128,32,32]
        e3 = self.encoder3(e2)  # [2,256,16,16]
        e4 = self.encoder4(e3)  # [2,512,8,8]

        # Center
        e4 = self.dblock(e4)  # [2,512,8,8]
        if self.use_drop:
            e4=self.drop_out(e4)

        # Decoder
        if self.use_att:
            d4=self.GAU4(e4,e3)#16
            d3=self.GAU3(d4,e2)#32
            d2=self.GAU2(d3,e1)#64
            d1=self.GAU1(d2,x_conv)#128
        else:
            d4 = self.decoder4(e4) + e3  # [2,256,16,16]
            d3 = self.decoder3(d4) + e2  # [2,128,32,32]
            d2 = self.decoder2(d3) + e1  # [2,64,64,64]
            d1 = self.decoder1(d2)  # [2,64,128,128]
        out = self.finaldeconv1(d1)  # [2,64,256,256]
        out = self.finalrelu1(out)
        out = self.finalconv2(out)  # [2,32,256,256]
        out = self.finalrelu2(out)
        final = self.finalconv3(out)  # [2,1,256,256]


        if self.RefNet:
            final=self.RefNet(final)
        if self.use_logits:
            return final
        else:
            if self.n_class == 1:
                return F.sigmoid(final)  # for binary only
            else:
                return final


class unet_2D_PreTrainInter(nn.Module):

    def __init__(self, feature_scale=4, n_classes=1, is_deconv=True, in_channels=3, is_batchnorm=True,use_att=False,use_res=True,
        dblock_type='AS',use_rfnet=False,use_logits=False,frozen_encoder=False,use_drop=False,drop_rate=0.2):
        super(unet_2D_PreTrainInter, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale
        self.n_class=n_classes
        self.use_att=use_att

        # filters = [64, 128, 256, 512, 1024]
        # filters = [int(x / self.feature_scale) for x in filters]#[16,32,64,128,256]
        if use_res:
            #conv_block=unetConv2_res
            logger.info("using resblock conv...")
        else:
            #conv_block=unetConv2
            logger.info("using basic conv...")


        # downsampling
        from torchvision import models
        filters = [64, 128, 256, 512]
        resnet = models.resnet34(pretrained=True)
        if frozen_encoder:
            for p in resnet.parameters():
                p.requires_grad=False
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.dblock = Dblock(512)

        if self.use_att:
            self.GAU4 = B.GAU(filters[3], filters[2], use_res=False)
            self.GAU3 = B.GAU(filters[2], filters[1], use_res=False)
            self.GAU2 = B.GAU(filters[1], filters[0], use_res=False)
            self.GAU1 = B.GAU(filters[0], filters[0], use_res=False)
            logger.info("using GAU attetnion...")
        else:
            self.decoder4 = B.DecoderBlock(filters[3], filters[2])
            self.decoder3 = B.DecoderBlock(filters[2], filters[1])
            self.decoder2 = B.DecoderBlock(filters[1], filters[0])
            self.decoder1 = B.DecoderBlock(filters[0], filters[0])
            logger.info("without GAU attetnion...")
        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)  # size*2
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        #self.finalconv3 = nn.Conv2d(32, n_classes, 3, padding=1)






    def forward(self, inputs):


        x = self.firstconv(inputs)  # [2,64,128,128]
        x = self.firstbn(x)
        x_conv = self.firstrelu(x)
        x1 = self.firstmaxpool(x_conv)  # [2,64,64,64]
        e1 = self.encoder1(x1)  ##[2,64,64,64]
        e2 = self.encoder2(e1)  # [2,128,32,32]
        e3 = self.encoder3(e2)  # [2,256,16,16]
        e4 = self.encoder4(e3)  # [2,512,8,8]

        # Center
        e4 = self.dblock(e4)  # [2,512,8,8]
        # if self.use_drop:
        #     e4=self.drop_out(e4)

        # Decoder
        if self.use_att:
            d4=self.GAU4(e4,e3)#16
            d3=self.GAU3(d4,e2)#32
            d2=self.GAU2(d3,e1)#64
            d1=self.GAU1(d2,x_conv)#128
        else:
            d4 = self.decoder4(e4) + e3  # [2,256,16,16]
            d3 = self.decoder3(d4) + e2  # [2,128,32,32]
            d2 = self.decoder2(d3) + e1  # [2,64,64,64]
            d1 = self.decoder1(d2)  # [2,64,128,128]
        out = self.finaldeconv1(d1)  # [2,64,256,256]
        out = self.finalrelu1(out)
        out = self.finalconv2(out)  # [2,32,256,256]
        out = self.finalrelu2(out)
        #final = self.finalconv3(out)  # [2,1,256,256]


        return  out









class unet_2D_Dense(nn.Module):
    '''
    cause too much GPU memory during training
    FCDenseNet(
        in_channels=3, down_blocks=(4,5,7,10,12),
        up_blocks=(12,10,7,5,4), bottleneck_layers=15,
        growth_rate=16, out_chans_first_conv=48, n_classes=n_classes,use_att=False)
    '''
    def __init__(self, feature_scale=4, n_classes=1, in_channels=3, use_att=False,use_logits=False,
        use_rfnet=False):
        super(unet_2D_Dense, self).__init__()

        self.in_channels = in_channels
        self.n_class=n_classes
        self.use_att=use_att
        from .Unet_Dense import FCDenseNet
        self.base_net=FCDenseNet(
        in_channels=in_channels, down_blocks=(4,5,7,10,12),
        up_blocks=(12,10,7,5,4), bottleneck_layers=15,
        growth_rate=16, out_chans_first_conv=48, n_classes=n_classes,use_att=use_att)

        if use_rfnet:
            self.RefNet=Dblock(1)
            logger.info('With refine module.')
        else:
            self.RefNet = None
            logger.info('Without Refine module.')
        self.use_logits=use_logits


    def forward(self, inputs):


        final=self.base_net(inputs)

        if self.RefNet:
            final=self.RefNet(final)
        if self.use_logits:
            return final
        else:
            if self.n_class == 1:
                return F.sigmoid(final)  # for binary only
            else:
                return final






class unet_2D_Encoder(nn.Module):
    def __init__(self, feature_scale=4, n_classes=1, is_deconv=True, in_channels=3, is_batchnorm=True, use_att=False,
                 use_res=False,
                 dblock_type='AS',use_se=False):
        super(unet_2D_Encoder, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale
        self.n_class = n_classes
        self.use_att = use_att


        if dblock_type=='AS':
            logger.info("using AS for center...")
            self.dblock=Dblock(512)
        elif dblock_type=='ASPP':
            logger.info("using ASPP for center...")
            self.dblock=ASPP(in_channel=512)
        else:
            logger.info("using FPA for center...")
            self.dblock =FPAv2(512,512)

        # downsampling
        from torchvision import models
        filters = [64, 128, 256, 512]
        resnet = models.resnet34(pretrained=True)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        if use_se:
            logger.info("using se for encoder...")
            self.encoder1 = nn.Sequential(resnet.layer1,
                                          SCSEBlock(filters[0]))
            self.encoder2 = nn.Sequential(resnet.layer2,
                                          SCSEBlock(filters[1]))
            self.encoder3 = nn.Sequential(resnet.layer3,
                                          SCSEBlock(filters[2]))
            self.encoder4 = nn.Sequential(resnet.layer4,
                                          SCSEBlock(filters[3]))
        else:
            self.encoder1 = resnet.layer1
            self.encoder2 = resnet.layer2
            self.encoder3 = resnet.layer3
            self.encoder4 = resnet.layer4

        #self.dblock = Dblock(512)

    def forward(self, inputs):

        x = self.firstconv(inputs)  # [2,64,128,128]
        x = self.firstbn(x)
        x_conv = self.firstrelu(x)
        x1 = self.firstmaxpool(x_conv)  # [2,64,64,64]
        e1 = self.encoder1(x1)  ##[2,64,64,64]
        e2 = self.encoder2(e1)  # [2,128,32,32]
        e3 = self.encoder3(e2)  # [2,256,16,16]
        e4 = self.encoder4(e3)  # [2,512,8,8]

        # # Center
        e4 = self.dblock(e4)  # [2,512,8,8]   in decoder
        return  x_conv,e1,e2,e3,e4

#================using max-pooling in the center part, so that more high-level features can be used===========================
class unet_2D_Encoder_New(nn.Module):
    def __init__(self, feature_scale=4, n_classes=1, is_deconv=True, in_channels=3, is_batchnorm=True, use_att=False,
                 use_res=False,
                 dblock_type='AS', use_se=False):
        super(unet_2D_Encoder_New, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale
        self.n_class = n_classes
        self.use_att = use_att

        if dblock_type == 'AS':
            logger.info("using AS for center...")
            self.dblock = nn.Sequential(Dblock(512),
                                        nn.MaxPool2d(2, 2))
        elif dblock_type == 'ASPP':
            logger.info("using ASPP for center...")
            self.dblock =nn.Sequential( ASPP(in_channel=512),
                                        nn.MaxPool2d(2, 2))
        else:
            logger.info("using FPA for center...")
            self.dblock = nn.Sequential(FPAv2(512, 512),
                                        nn.MaxPool2d(2, 2))

        # downsampling
        from torchvision import models
        filters = [64, 128, 256, 512]
        resnet = models.resnet34(pretrained=True)
        self.conv1=nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu
        )
        # self.firstconv = resnet.conv1
        # self.firstbn = resnet.bn1
        # self.firstrelu = resnet.relu
        # self.firstmaxpool = resnet.maxpool
        if use_se:
            logger.info("using se for encoder...")
            self.encoder1 = nn.Sequential(resnet.layer1,
                                          SCSEBlock(filters[0]))
            self.encoder2 = nn.Sequential(resnet.layer2,
                                          SCSEBlock(filters[1]))
            self.encoder3 = nn.Sequential(resnet.layer3,
                                          SCSEBlock(filters[2]))
            self.encoder4 = nn.Sequential(resnet.layer4,
                                          SCSEBlock(filters[3]))
        else:
            self.encoder1 = resnet.layer1
            self.encoder2 = resnet.layer2
            self.encoder3 = resnet.layer3
            self.encoder4 = resnet.layer4


    def forward(self, inputs):

        conv1=self.conv1(inputs)#[1,64,256,256]
        e1 = self.encoder1(conv1)  ##[1,64,256,256]
        e2 = self.encoder2(e1)  # [1,128,128,128]
        e3 = self.encoder3(e2)  # [1,256,64,64]
        e4 = self.encoder4(e3)  # [1,512,32,32]

        # # Center
        center = self.dblock(e4)  # [2,512,8,8]   in decoder
        return e1, e2, e3, e4,center


class unet_2D_Encoder_Res50(nn.Module):
    def __init__(self, feature_scale=4, n_classes=1, is_deconv=True, in_channels=3, is_batchnorm=True, use_att=False,
                 use_res=True,
                 dblock_type='AS',use_se=False):
        super(unet_2D_Encoder_Res50, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.n_class = n_classes
        self.use_att = use_att

        self.encoder = se_resnext50_32x4d()

        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Sequential(self.encoder.layer0.conv1,
                                   self.encoder.layer0.bn1,
                                   self.encoder.layer0.relu1)
        self.conv2 = self.encoder.layer1
        self.conv3 = self.encoder.layer2
        self.conv4 = self.encoder.layer3
        self.conv5 = self.encoder.layer4
        self.center = nn.Sequential(nn.Conv2d(512 * 4, 512, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(512),
                                    nn.ReLU(inplace=True),
                                    # nn.Conv2d(512, 256, kernel_size=3, padding=1),
                                    # nn.BatchNorm2d(256),
                                    # nn.ReLU(inplace=True),
                                    nn.MaxPool2d(kernel_size=2, stride=2),
                                    Dblock(512))

    def forward(self, inputs):

        conv1 = self.conv1(inputs)  # 1/2 [1,3,512,512]==>[1,64,256,256]
        conv2 = self.conv2(conv1)  # 1/2  [1,256,256,256]
        conv3 = self.conv3(conv2)  # 1/4  [1,512,128,128]
        conv4 = self.conv4(conv3)  # 1/8  [1,1024,64,64]
        conv5 = self.conv5(conv4)  # 1/16  [1,2048,32,32]
        center=self.center(conv5)#[1,256,16,16]




        # x = self.firstconv(inputs)  # [2,64,128,128]
        # x = self.firstbn(x)
        # x_conv = self.firstrelu(x)
        # x1 = self.firstmaxpool(x_conv)  # [2,64,64,64]
        # e1 = self.encoder1(x1)  ##[2,64,64,64]
        # e2 = self.encoder2(e1)  # [2,128,32,32]
        # e3 = self.encoder3(e2)  # [2,256,16,16]
        # e4 = self.encoder4(e3)  # [2,512,8,8]


        return  conv2,conv3,conv4,conv5,center

class unet_2D_Decoder(nn.Module):
    def __init__(self, feature_scale=4, n_classes=1, is_deconv=True, in_channels=3, is_batchnorm=True, use_att=False,
                 use_res=True,
                 dblock_type='AS',use_rfnet=False):
        super(unet_2D_Decoder, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale
        self.n_class = n_classes
        self.use_att = use_att
        if use_rfnet:
            self.RefNet = Dblock(1)
            logger.info('With refine module.')
        else:
            self.RefNet = None
            logger.info('Without Refine module.')

        if use_res:
            conv_block = unetConv2_res
            logger.info("using resblock conv...")
        else:
            conv_block = unetConv2
            logger.info("using basic conv...")
        filters = [64, 128, 256, 512]

        if self.use_att:
            self.GAU4 = B.GAU(filters[3], filters[2], use_res=False)
            self.GAU3 = B.GAU(filters[2], filters[1], use_res=False)
            self.GAU2 = B.GAU(filters[1], filters[0], use_res=False)
            self.GAU1 = B.GAU(filters[0], filters[0], use_res=False)
            logger.info("using GAU attetnion...")
        else:
            self.decoder4 = B.DecoderBlock(filters[3], filters[2])
            self.decoder3 = B.DecoderBlock(filters[2], filters[1])
            self.decoder2 = B.DecoderBlock(filters[1], filters[0])
            self.decoder1 = B.DecoderBlock(filters[0], filters[0])
            logger.info("without GAU attetnion...")
        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)  # size*2
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, n_classes, 3, padding=1)



    def forward(self, x_conv,e1,e2,e3,e4):

        # Center
        #e4 = self.dblock(e4)  # [2,512,8,8]   in decoder

        if self.use_att:
            d4 = self.GAU4(e4, e3)  # 16
            d3 = self.GAU3(d4, e2)  # 32
            d2 = self.GAU2(d3, e1)  # 64
            d1 = self.GAU1(d2, x_conv)  # 128
        else:
            d4 = self.decoder4(e4) + e3  # [2,256,16,16]
            d3 = self.decoder3(d4) + e2  # [2,128,32,32]
            d2 = self.decoder2(d3) + e1  # [2,64,64,64]
            d1 = self.decoder1(d2)  # [2,64,128,128]
        out = self.finaldeconv1(d1)  # [2,64,256,256]
        out = self.finalrelu1(out)
        out = self.finalconv2(out)  # [2,32,256,256]
        out = self.finalrelu2(out)
        final = self.finalconv3(out)  # [2,1,256,256]



        if self.RefNet:
            final = self.RefNet(final)
        else:
            if self.n_class == 1:
                return F.sigmoid(final)  # for binary only
            else:
                return final


class unet_2D_Encoder2(nn.Module):#downsample 4 times
    def __init__(self, feature_scale=4, n_classes=1, is_deconv=True, in_channels=3, is_batchnorm=True, use_att=False,
                 use_res=True,
                 dblock_type='AS'):
        super(unet_2D_Encoder2, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale
        self.n_class = n_classes
        self.use_att = use_att



        # downsampling
        from torchvision import models
        filters = [64, 128, 256]
        resnet = models.resnet34(pretrained=True)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3

        if dblock_type=='AS':
           #self.dblock = Dblock(256)
            self.dblock=Dblock2(256,conv_num=3)
        else:
            self.dblock=ASPP(256)

    def forward(self, inputs):

        x = self.firstconv(inputs)  # [2,64,64,64]   [4,3,128,128]==>[4,64,64,64]
        x = self.firstbn(x)
        x = self.firstrelu(x)# [2,64,64,64]
        e0 = self.firstmaxpool(x)  # [2,64,64,64]    [4,64,32,32]
        e1 = self.encoder1(e0)  ##[2,64,64,64]      [4,64,32,32]
        e2 = self.encoder2(e1)  # [2,128,32,32]     [4,128,16,16]
        e3 = self.encoder3(e2)  # [2,256,16,16]     [4,256,8,8]
        #e4 = self.encoder4(e3)  # [2,512,8,8]

        # # Center
        e3 = self.dblock(e3)  # [2,512,8,8]   in decoder
        #return  e0, e1, e2, e3
        return x,e1,e2,e3

class unet_2D_Decoder2(nn.Module):#upsample 3 times
    def __init__(self, feature_scale=4, n_classes=1, is_deconv=True, in_channels=3, is_batchnorm=True, use_att=False,
                 use_res=True,
                 dblock_type='AS', use_rfnet=False):
        super(unet_2D_Decoder2, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale
        self.n_class = n_classes
        self.use_att = use_att
        if use_rfnet:
            self.RefNet = Dblock(1)
            logger.info('With refine module.')
        else:
            self.RefNet = None
            logger.info('Without Refine module.')


        filters = [64, 128, 256]
        self.use_drop = True
        drop_rate = 0.1
        self.dropblock = LinearScheduler(
            DropBlock2D(drop_prob=drop_rate, block_size=7),
            start_value=0.,
            stop_value=drop_rate,
            nr_steps=10000
        )

        if self.use_att:

            # self.GAU3 = B.GAU(filters[2], filters[1], use_res=False)
            # self.GAU2 = B.GAU(filters[1], filters[0], use_res=False)
            # self.GAU1 = B.GAU(filters[0], filters[0], use_res=False)

            self.GAU3 = B.GAU_Res2(filters[2], filters[1])
            self.GAU2 = B.GAU_Res2(filters[1], filters[0])
            self.GAU1 = B.GAU_Res2(filters[0], filters[0])

            logger.info("using GAU attetnion...")
        else:
            #self.decoder4 = B.DecoderBlock(filters[3], filters[2])
            self.decoder3 = B.DecoderBlock(filters[2], filters[1])
            self.decoder2 = B.DecoderBlock(filters[1], filters[0])
            self.decoder1 = B.DecoderBlock(filters[0], filters[0])
            logger.info("without GAU attetnion...")
        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32,4,2,1)  # size*2
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, n_classes, 3, padding=1)

    def forward(self,e0, e1, e2, e3):

        if self.use_drop:
            self.dropblock.step()

        if self.use_att:
            #d4 = self.GAU4(e4, e3)  # 16
            d3 = self.GAU3(e3, e2)  # 32
            d2 = self.GAU2(d3, e1)  # 64
            d1 = self.GAU1(d2, e0)  # 128
        else:
            #d4 = self.decoder4(e4) + e3  # [2,256,16,16]
            d3 = self.decoder3(e3) + e2  # [2,128,32,32]
            d2 = self.decoder2(d3) + e1  # [2,64,64,64]
            d1 = self.decoder1(d2)  # [2,64,128,128]
        out = self.finaldeconv1(d1)  # [2,64,256,256]
        out = self.finalrelu1(out)
        out = self.finalconv2(out)  # [2,32,256,256]
        out = self.finalrelu2(out)
        out=self.dropblock(out)
        final = self.finalconv3(out)  # [2,1,256,256]

        if self.RefNet:
            final = self.RefNet(final)

        if self.n_class == 1:
            return F.sigmoid(final)  # for binary only
        else:
            return final


class unet_2D_Decoder2_Df(nn.Module):  # upsample 3 times
    def __init__(self, feature_scale=4, n_classes=1, is_deconv=True, in_channels=3, is_batchnorm=True, use_att=False,
                 use_res=True,
                 dblock_type='AS', use_rfnet=False):
        super(unet_2D_Decoder2_Df, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale
        self.n_class = n_classes
        self.use_att = use_att
        if use_rfnet:
            self.RefNet = Dblock(1)
            logger.info('With refine module.')
        else:
            self.RefNet = None
            logger.info('Without Refine module.')

        filters = [64, 128, 256]
        self.use_drop = False
        # drop_rate = 0.1
        # self.dropblock = LinearScheduler(
        #     DropBlock2D(drop_prob=drop_rate, block_size=7),
        #     start_value=0.,
        #     stop_value=drop_rate,
        #     nr_steps=1e4
        # )

        if self.use_att:

            # self.GAU3 = B.GAU(filters[2], filters[1], use_res=False)
            # self.GAU2 = B.GAU(filters[1], filters[0], use_res=False)
            # self.GAU1 = B.GAU(filters[0], filters[0], use_res=False)

            self.GAU3 = B.GAU_Res2(filters[2], filters[1])
            self.GAU2 = B.GAU_Res2(filters[1], filters[0])
            self.GAU1 = B.GAU_Res2(filters[0], filters[0])

            logger.info("using GAU attetnion...")
        else:
            # self.decoder4 = B.DecoderBlock(filters[3], filters[2])
            self.decoder3 = B.DecoderBlock(filters[2], filters[1])
            self.decoder2 = B.DecoderBlock(filters[1], filters[0])
            self.decoder1 = B.DecoderBlock(filters[0], filters[0])
            logger.info("without GAU attetnion...")
        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)  # size*2
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, n_classes, 3, padding=1)

    def forward(self, e0, e1, e2, e3):

        # if self.use_drop:
        #     self.dropblock.step()

        if self.use_att:
            # d4 = self.GAU4(e4, e3)  # 16
            d3 = self.GAU3(e3, e2)  # 32
            d2 = self.GAU2(d3, e1)  # 64
            d1 = self.GAU1(d2, e0)  # 128
        else:
            # d4 = self.decoder4(e4) + e3  # [2,256,16,16]
            d3 = self.decoder3(e3) + e2  # [2,128,32,32]
            d2 = self.decoder2(d3) + e1  # [2,64,64,64]
            d1 = self.decoder1(d2)  # [2,64,128,128]
        out = self.finaldeconv1(d1)  # [2,64,256,256]
        out = self.finalrelu1(out)
        out = self.finalconv2(out)  # [2,32,256,256]
        out = self.finalrelu2(out)
        #out= self.dropblock(out)
        final = self.finalconv3(out)  # [2,1,256,256]



        if self.n_class == 1:
            return F.sigmoid(final)  # for binary only
        else:
            return final



class unet_2D_Encoder2Ref(nn.Module):#downsample 4 times
    def __init__(self, feature_scale=4, n_classes=1, is_deconv=True, in_channels=3, is_batchnorm=True, use_att=False,
                 use_res=True,
                 dblock_type='AS'):
        super(unet_2D_Encoder2Ref, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale
        self.n_class = n_classes
        self.use_att = use_att



        # downsampling
        from torchvision import models
        filters = [64, 128, 256]
        resnet = models.resnet34(pretrained=True)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3

        if dblock_type=='AS':
           #self.dblock = Dblock(256)
            self.dblock=Dblock2(256,conv_num=3)
        else:
            self.dblock=ASPP(256)

    def forward(self, inputs):

        x = self.firstconv(inputs)  # [2,64,64,64]   [4,3,128,128]==>[4,64,64,64]
        x = self.firstbn(x)
        x = self.firstrelu(x)# [2,64,64,64]
        e0 = self.firstmaxpool(x)  # [2,64,64,64]    [4,64,32,32]
        e1 = self.encoder1(e0)  ##[2,64,64,64]      [4,64,32,32]
        e2 = self.encoder2(e1)  # [2,128,32,32]     [4,128,16,16]
        e3 = self.encoder3(e2)  # [2,256,16,16]     [4,256,8,8]
        #e4 = self.encoder4(e3)  # [2,512,8,8]

        # # Center
        e3 = self.dblock(e3)  # [2,512,8,8]   in decoder
        #return  e0, e1, e2, e3
        return inputs,x,e1,e2,e3


class unet_2D_Decoder2Ref(nn.Module):#upsample 3 times
    def __init__(self, feature_scale=4, n_classes=1, is_deconv=True, in_channels=3, is_batchnorm=True, use_att=False,
                 use_res=True,
                 dblock_type='AS', use_rfnet=True):
        super(unet_2D_Decoder2Ref, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale
        self.n_class = n_classes
        self.use_att = use_att
        self.use_rfnet=use_rfnet
        # if use_rfnet:
        #     self.RefNet = Dblock(1)
        #     logger.info('With refine module.')
        # else:
        #     self.RefNet = None
        #     logger.info('Without Refine module.')


        filters = [64, 128, 256]
        # self.use_drop = True
        # drop_rate = 0.1
        # self.dropblock = LinearScheduler(
        #     DropBlock2D(drop_prob=drop_rate, block_size=7),
        #     start_value=0.,
        #     stop_value=drop_rate,
        #     nr_steps=1e4
        # )

        if self.use_att:

            # self.GAU3 = B.GAU(filters[2], filters[1], use_res=False)
            # self.GAU2 = B.GAU(filters[1], filters[0], use_res=False)
            # self.GAU1 = B.GAU(filters[0], filters[0], use_res=False)

            self.GAU3 = B.GAU_Res2(filters[2], filters[1])
            self.GAU2 = B.GAU_Res2(filters[1], filters[0])
            self.GAU1 = B.GAU_Res2(filters[0], filters[0])

            logger.info("using GAU attetnion...")
        else:
            #self.decoder4 = B.DecoderBlock(filters[3], filters[2])
            self.decoder3 = B.DecoderBlock(filters[2], filters[1])
            self.decoder2 = B.DecoderBlock(filters[1], filters[0])
            self.decoder1 = B.DecoderBlock(filters[0], filters[0])
            logger.info("without GAU attetnion...")

        if use_rfnet:
            self.final_conv = nn.Sequential(
                nn.Conv2d(filters[0], 32, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, n_classes, 3, padding=1)
            )
            self.RefNet = SpatialRefine(67, 64)#pred_rv = torch.matmul(pred, aff).view(n, -1, h, w)  consume too much GPU
            #self.RefNet=pac.PacConv2d(filters[0],filters[0],kernel_size=3,padding=1)
            logger.info('With refine module.')
        else:
            logger.info('Without Refine module.')
            self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)  # size*2
            self.finalrelu1 = nonlinearity
            self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
            self.finalrelu2 = nonlinearity
            self.finalconv3 = nn.Conv2d(32, n_classes, 3, padding=1)


        # self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32,4,2,1)  # size*2
        # self.finalrelu1 = nonlinearity
        # self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        # self.finalrelu2 = nonlinearity
        # self.finalconv3 = nn.Conv2d(32, n_classes, 3, padding=1)

    def forward(self,input,e0, e1, e2, e3):

        # if self.use_drop:
        #     self.dropblock.step()

        if self.use_att:
            #d4 = self.GAU4(e4, e3)  # 16
            d3 = self.GAU3(e3, e2)  # 32
            d2 = self.GAU2(d3, e1)  # 64
            d1 = self.GAU1(d2, e0)  # 128
        else:
            #d4 = self.decoder4(e4) + e3  # [2,256,16,16]
            d3 = self.decoder3(e3) + e2  # [2,128,32,32]
            d2 = self.decoder2(d3) + e1  # [2,64,64,64]
            d1 = self.decoder1(d2)  # [2,64,128,128]
        # out = self.finaldeconv1(d1)  # [2,64,256,256]
        # out = self.finalrelu1(out)
        # out = self.finalconv2(out)  # [2,32,256,256]
        # out = self.finalrelu2(out)
        # out=self.dropblock(out)
        # final = self.finalconv3(out)  # [2,1,256,256]
        #
        # if self.RefNet:
        #     final = self.RefNet(final)
        if self.use_rfnet:
            out = self.final_conv(d1)#[8,1,64,64]
            _, _, H0, W0 = e0.size()
            _, _, H, W = input.size()
            fea0 = F.interpolate(input, (H0, W0), mode='bilinear', align_corners=True)
            fea = torch.cat([fea0, e0], dim=1)#[8,67,64,64]
            out = self.RefNet(out, fea)
            final = F.interpolate(out, (H, W), mode='bilinear', align_corners=True)

        else:
            out = self.finaldeconv1(d1)  # [2,64,256,256]
            out = self.finalrelu1(out)
            out = self.finalconv2(out)  # [2,32,256,256]
            out = self.finalrelu2(out)
            final = self.finalconv3(out)  # [2,1,256,256]




        if self.n_class == 1:
            return F.sigmoid(final)  # for binary only
        else:
            return final


class unet_2D_Decoder2New(nn.Module):#upsample 3 times
    def __init__(self, feature_scale=4, n_classes=1, is_deconv=True, in_channels=3, is_batchnorm=True, use_att=False,
                 use_res=True,
                 dblock_type='AS', use_rfnet=False):
        super(unet_2D_Decoder2New, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale
        self.n_class = n_classes
        self.use_att = use_att
        if use_rfnet:
            self.RefNet = Dblock(1)
            logger.info('With refine module.')
        else:
            self.RefNet = None
            logger.info('Without Refine module.')


        filters = [64, 128, 256]
        self.use_drop=True
        drop_rate=0.1
        self.dropblock = LinearScheduler(
            DropBlock2D(drop_prob=drop_rate, block_size=7),
            start_value=0.,
            stop_value=drop_rate,
            nr_steps=10000
        )

        if self.use_att:

            # self.GAU3 = B.GAU(filters[2], filters[1], use_res=False)
            # self.GAU2 = B.GAU(filters[1], filters[0], use_res=False)
            # self.GAU1 = B.GAU(filters[0], filters[0], use_res=False)

            self.GAU3 = B.GAU_Res2(filters[2], filters[1])
            self.GAU2 = B.GAU_Res2(filters[1], filters[0])
            self.GAU1 = B.GAU_Res2(filters[0], filters[0])

            logger.info("using GAU attetnion...")
        else:

            self.decoder3 = B.DecoderBlock(filters[2], filters[1])
            self.decoder2 = B.DecoderBlock(filters[1], filters[0])
            self.decoder1 = B.DecoderBlock(filters[0], filters[0])
            logger.info("without GAU attetnion...")
        # self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32,4,2,1)  # size*2
        # self.finalrelu1 = nonlinearity
        # self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        # self.finalrelu2 = nonlinearity
        # self.finalconv3 = nn.Conv2d(32, n_classes, 3, padding=1)

        self.class_conv1=nn.Sequential(
            nn.ConvTranspose2d(filters[0], 32, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            self.dropblock,
            nn.Conv2d(32, n_classes, 3, padding=1)
            #nn.Sigmoid()
        )
        self.class_conv2 = nn.Sequential(
            nn.ConvTranspose2d(filters[0], 32, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            self.dropblock,
            nn.Conv2d(32, n_classes, 3, padding=1)
            #nn.Sigmoid()
        )
    def forward(self,e0, e1, e2, e3):
        if self.use_drop:
            self.dropblock.step()

        if self.use_att:
            #d4 = self.GAU4(e4, e3)  # 16
            d3 = self.GAU3(e3, e2)  # 32
            d2 = self.GAU2(d3, e1)  # 64
            d1 = self.GAU1(d2, e0)  # 128
        else:
            #d4 = self.decoder4(e4) + e3  # [2,256,16,16]
            d3 = self.decoder3(e3) + e2  # [2,128,32,32]
            d2 = self.decoder2(d3) + e1  # [2,64,64,64]
            d1 = self.decoder1(d2)  # [2,64,128,128]

        pred1=self.class_conv1(d2)
        pred2=self.class_conv2(d1)
        # out = self.finaldeconv1(d1)  # [2,64,256,256]
        # out = self.finalrelu1(out)
        # out = self.finalconv2(out)  # [2,32,256,256]
        # out = self.finalrelu2(out)
        # final = self.finalconv3(out)  # [2,1,256,256]
        #
        # if self.RefNet:
        #     final = self.RefNet(final)
        # if self.n_class == 1:
        #     return F.sigmoid(final)  # for binary only
        # else:
        #     return final
        return pred1,pred2


#####################################################################################################################
#==================================CCT model=========================================================================
#####################################################################################################################


class unet_2D_PreTrainCCT(nn.Module):

    def __init__(self, train_opt, n_classes=1, in_channels=3):
        super(unet_2D_PreTrainCCT, self).__init__()

        self.encoder=unet_2D_Encoder2(in_channels=in_channels)
        self.decoder=unet_2D_Decoder2(n_classes=n_classes)
        # ===aug decoder
        decoder_in_ch = 256
        self.train_opt=train_opt
        vat_decoder = [B.VATDecoder(decoder_in_ch, xi=self.train_opt['xi'],
                                    eps=self.train_opt['eps']) for _ in range(self.train_opt['vat'])]
        drop_decoder = [B.DropOutDecoder(decoder_in_ch,
                                         drop_rate=self.train_opt['drop_rate'],
                                         spatial_dropout=self.train_opt['spatial'])
                        for _ in range(self.train_opt['drop'])]
        cut_decoder = [B.CutOutDecoder(decoder_in_ch, erase=self.train_opt['erase'])
                       for _ in range(self.train_opt['cutout'])]
        context_m_decoder = [B.ContextMaskingDecoder(decoder_in_ch)
                             for _ in range(self.train_opt['context_masking'])]
        object_masking = [B.ObjectMaskingDecoder(decoder_in_ch)
                          for _ in range(self.train_opt['object_masking'])]
        feature_drop = [B.FeatureDropDecoder(decoder_in_ch)
                        for _ in range(self.train_opt['feature_drop'])]
        feature_noise = [B.FeatureNoiseDecoder(decoder_in_ch,
                                               uniform_range=self.train_opt['uniform_range'])
                         for _ in range(self.train_opt['feature_noise'])]
        #self.aux_decoders = nn.ModuleList([*vat_decoder, *drop_decoder, *feature_drop, *feature_noise])
        self.aux_decoders = nn.ModuleList([*vat_decoder, *drop_decoder,*context_m_decoder,*object_masking,
                                           *feature_drop, *feature_noise])


    def forward(self,img_s,img_t,use_warm=False):

        e0, e1, e2, e3 = self.encoder(img_s)
        seg = self.decoder(e0, e1, e2, e3)
        if self.train_opt['mode']=='supervised' or use_warm==True:

            return seg,e3
        else:
            e0_t, e1_t, e2_t, e3_t = self.encoder(img_t)
            seg_t = self.decoder(e0_t, e1_t, e2_t, e3_t)
            # =================for Cross-Consistency training=======
            # Get auxiliary predictions
            aux_seg_t = [self.decoder(e0_t, e1_t, e2_t, aux_decoder(e3_t, seg_t.detach())) for aux_decoder in self.aux_decoders]
            return seg,(seg_t,aux_seg_t)














class unet_2D_PreTrain256_ED2Ref(nn.Module):

    def __init__(self, feature_scale=4, n_classes=1, is_deconv=True, in_channels=3, is_batchnorm=True, use_att=False,
                 use_res=True, dblock_type='AS', use_rfnet=False, use_logits=False):
        super(unet_2D_PreTrain256_ED2Ref, self).__init__()

        self.encoder=unet_2D_Encoder2Ref(in_channels=in_channels,dblock_type=dblock_type)
        self.decoder=unet_2D_Decoder2Ref(n_classes=n_classes,use_att=use_att)

    def forward(self, x):
        input,e0, e1, e2, e3=self.encoder(x)
        seg=self.decoder(input,e0,e1,e2,e3)
        return e3,seg

class unet_2D_PreTrain256_ED2(nn.Module):

    def __init__(self, feature_scale=4, n_classes=1, is_deconv=True, in_channels=3, is_batchnorm=True, use_att=False,
                 use_res=True, dblock_type='AS', use_rfnet=False, use_logits=False):
        super(unet_2D_PreTrain256_ED2, self).__init__()

        self.encoder=unet_2D_Encoder2(in_channels=in_channels,dblock_type=dblock_type)
        self.decoder=unet_2D_Decoder2(n_classes=n_classes,use_att=use_att)

    def forward(self, x):
        e0, e1, e2, e3=self.encoder(x)
        seg=self.decoder(e0, e1, e2, e3)
        return e3,seg



class unet_2D_PreTrain256_ED2_noDrop(nn.Module):

    def __init__(self, feature_scale=4, n_classes=1, is_deconv=True, in_channels=3, is_batchnorm=True, use_att=False,
                 use_res=True, dblock_type='AS', use_rfnet=False, use_logits=False):
        super(unet_2D_PreTrain256_ED2_noDrop, self).__init__()

        self.encoder=unet_2D_Encoder2(in_channels=in_channels,dblock_type=dblock_type)
        self.decoder=unet_2D_Decoder2_Df(n_classes=n_classes,use_att=use_att)

    def forward(self, x):
        e0, e1, e2, e3=self.encoder(x)
        seg=self.decoder(e0, e1, e2, e3)
        return e3,seg





class unet_2D_PreTrain256_ED3(nn.Module):

    def __init__(self, feature_scale=4, n_classes=1, is_deconv=True, in_channels=3, is_batchnorm=True, use_att=False,
                 use_res=True, dblock_type='AS', use_rfnet=False, use_logits=False):
        super(unet_2D_PreTrain256_ED3, self).__init__()

        self.encoder=unet_2D_Encoder2(in_channels=in_channels,dblock_type=dblock_type)
        self.decoder=unet_2D_Decoder2New(n_classes=n_classes,use_att=use_att)

    def forward(self, x):
        e0, e1, e2, e3=self.encoder(x)
        seg1,seg2=self.decoder(e0, e1, e2, e3)
        return e3,seg1,seg2



class unet_2D_Encoder3(nn.Module):#downsample 4 times
    def __init__(self, feature_scale=4, n_classes=1, is_deconv=True, in_channels=3, is_batchnorm=True, use_att=False,
                 use_res=True,
                 dblock_type='AS'):
        super(unet_2D_Encoder3, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale
        self.n_class = n_classes
        self.use_att = use_att



        # downsampling
        from torchvision import models
        filters = [64, 128, 256]
        resnet = models.resnet34(pretrained=True)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3

        if dblock_type=='AS':
           self.dblock = Dblock2(256,conv_num=2)
        else:
            self.dblock=ASPP(256)

    def forward(self, inputs):

        x = self.firstconv(inputs)  # [2,64,64,64]   [4,3,128,128]==>[4,64,64,64]
        x = self.firstbn(x)
        x = self.firstrelu(x)# [2,64,64,64]
        e0 = self.firstmaxpool(x)  # [2,64,64,64]    [4,64,32,32]
        e1 = self.encoder1(e0)  ##[2,64,64,64]      [4,64,32,32]
        e2 = self.encoder2(e1)  # [2,128,32,32]     [4,128,16,16]
        e3 = self.encoder3(e2)  # [2,256,16,16]     [4,256,8,8]
        #e4 = self.encoder4(e3)  # [2,512,8,8]

        # # Center
        e3 = self.dblock(e3)  # [2,512,8,8]   in decoder
        return  e0, e1, e2, e3
class unet_2D_Decoder3(nn.Module):#upsample 4 times
    def __init__(self, feature_scale=4, n_classes=1, is_deconv=True, in_channels=3, is_batchnorm=True, use_att=False,
                 use_res=True,
                 dblock_type='AS', use_rfnet=False):
        super(unet_2D_Decoder3, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale
        self.n_class = n_classes
        self.use_att = use_att
        if use_rfnet:
            self.RefNet = Dblock(1)
            logger.info('With refine module.')
        else:
            self.RefNet = None
            logger.info('Without Refine module.')


        filters = [64, 128, 256]

        if self.use_att:

            self.GAU3 = B.GAU_Res2(filters[2], filters[1])
            self.GAU2 = B.GAU_Res2(filters[1], filters[0])
            self.GAU1 = B.GAU_Res2(filters[0], filters[0])
            logger.info("using GAU attetnion...")
        else:
            #self.decoder4 = B.DecoderBlock(filters[3], filters[2])
            self.decoder3 = B.DecoderBlock(filters[2], filters[1])
            self.decoder2 = B.DecoderBlock(filters[1], filters[0])
            self.decoder1 = B.DecoderBlock(filters[0], filters[0])
            logger.info("without GAU attetnion...")
        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32,4,2,1)  # size*2
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, n_classes, 3, padding=1)

    def forward(self,e0, e1, e2, e3):

        # Center
        # e0:[4,64,32,32] e1:[4,64,32,32] e2:[4,64,16,16], e3:[4,64,8,8]

        if self.use_att:
            #d4 = self.GAU4(e4, e3)  # 16
            d3 = self.GAU3(e3, e2)  # 16
            d2 = self.GAU2(d3, e1)  # 32
            d1 = self.GAU1(d2, e0)  # 64
        else:
            #d4 = self.decoder4(e4) + e3  # [2,256,16,16]
            d3 = self.decoder3(e3) + e2  # [2,128,32,32]
            d2 = self.decoder2(d3) + e1  # [2,64,64,64]
            d1 = self.decoder1(d2)  # [2,64,128,128]
        out = self.finaldeconv1(d1)  # [2,64,256,256]
        out = self.finalrelu1(out)
        out = self.finalconv2(out)  # [2,32,256,256]
        out = self.finalrelu2(out)
        final = self.finalconv3(out)  # [2,1,256,256]

        if self.RefNet:
            final = self.RefNet(final)
        if self.n_class == 1:
            return F.sigmoid(final)  # for binary only
        else:
            return final
# class unet_2D_PreTrain256_ED3(nn.Module):
#
#     def __init__(self, feature_scale=4, n_classes=1, is_deconv=True, in_channels=3, is_batchnorm=True, use_att=False,
#                  use_res=True, dblock_type='AS', use_rfnet=False, use_logits=False):
#         super(unet_2D_PreTrain256_ED3, self).__init__()
#
#         self.encoder=unet_2D_Encoder3(in_channels=in_channels,dblock_type=dblock_type)
#         self.decoder=unet_2D_Decoder3(n_classes=n_classes,use_rfnet=use_rfnet)
#
#     def forward(self, x):
#         e0, e1, e2, e3=self.encoder(x)
#         seg=self.decoder(e0, e1, e2, e3)
#         return e3,seg



class unet_2D_Encoder4(nn.Module):#downsample 4 times
    def __init__(self, feature_scale=4, n_classes=1, is_deconv=True, in_channels=3, is_batchnorm=True, use_att=False,
                 use_res=True,
                 dblock_type='AS'):
        super(unet_2D_Encoder4, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale
        self.n_class = n_classes
        self.use_att = use_att



        # downsampling
        from torchvision import models
        filters = [64, 128, 256,512]
        resnet = models.resnet34(pretrained=True)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        if dblock_type=='AS':
           self.dblock = Dblock2(512,conv_num=3)
        elif dblock_type=='ASS':
           self.dblock = Dblock_Smooth(512,conv_num=3)
        elif dblock_type=='ASPP':
            self.dblock=ASPP(512)
        elif dblock_type=='Ladder-ASPP':
            self.dblock=Ladder_ASPP(512)
        else:
            self.dblock=LKPP(in_chan=512, out_chan=128, mode='parallel', with_gp=False)

    def forward(self, inputs):

        x = self.firstconv(inputs)  # [2,64,64,64]   [4,3,128,128]==>[4,64,64,64]
        x = self.firstbn(x)
        x = self.firstrelu(x)# [2,64,64,64]
        e0 = self.firstmaxpool(x)  # [2,64,64,64]    [4,64,32,32]
        e1 = self.encoder1(e0)  ##[2,64,64,64]      [4,64,32,32]
        e2 = self.encoder2(e1)  # [2,128,32,32]     [4,128,16,16]
        e3 = self.encoder3(e2)  # [2,256,16,16]     [4,256,8,8]
        e4 = self.encoder4(e3)  # [2,512,8,8]

        # # Center
        e4 = self.dblock(e4)  # [2,512,8,8]   in decoder
        return  x, e1, e2, e3,e4
class unet_2D_Decoder4(nn.Module):  # upsample 4 times
    def __init__(self, feature_scale=4, n_classes=1, is_deconv=True, in_channels=3, is_batchnorm=True, use_att=False,
                 use_res=True,
                 dblock_type='AS', use_rfnet=False):
        super(unet_2D_Decoder4, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale
        self.n_class = n_classes
        self.use_att = use_att
        self.use_rfnet = use_rfnet

        filters = [64, 128, 256, 512]

        if self.use_att:
            self.GAU4 = B.GAU_Res2(filters[3], filters[2])
            self.GAU3 = B.GAU_Res2(filters[2], filters[1])
            self.GAU2 = B.GAU_Res2(filters[1], filters[0])
            self.GAU1 = B.GAU_Res2(filters[0], filters[0])
            logger.info("using GAU attetnion...")
        else:
            # self.decoder4 = B.DecoderBlock(filters[3], filters[2])
            self.decoder3 = B.DecoderBlock(filters[2], filters[1])
            self.decoder2 = B.DecoderBlock(filters[1], filters[0])
            self.decoder1 = B.DecoderBlock(filters[0], filters[0])
            logger.info("without GAU attetnion...")
        if use_rfnet:
            self.final_conv = nn.Sequential(
                nn.Conv2d(filters[0], 32, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, n_classes, 3, padding=1)
            )
            self.RefNet = SpatialRefine(67, 64)
            logger.info('With refine module.')
        else:
            logger.info('Without Refine module.')
            self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)  # size*2
            self.finalrelu1 = nonlinearity
            self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
            self.finalrelu2 = nonlinearity
            self.finalconv3 = nn.Conv2d(32, n_classes, 3, padding=1)

    def forward(self, e0, e1, e2, e3, e4):

        # e4: [4,512,16,16] e3:[4,256,32,32] e2:[4,128,64,64], e1:[4,64,128,128] e0:[2,64,256,256]

        if self.use_att:
            d4 = self.GAU4(e4, e3)  # 16
            d3 = self.GAU3(d4, e2)  # 32
            d2 = self.GAU2(d3, e1)  # 64
            d1 = self.GAU1(d2, e0)  # 128
        else:
            d4 = self.decoder4(e4) + e3  # [2,256,16,16]
            d3 = self.decoder3(e3) + e2  # [2,128,32,32]
            d2 = self.decoder2(d3) + e1  # [2,64,64,64]
            d1 = self.decoder1(d2)  # [2,64,128,128]

        out = self.finaldeconv1(d1)  # [2,64,256,256]
        out = self.finalrelu1(out)
        out = self.finalconv2(out)  # [2,32,256,256]
        out = self.finalrelu2(out)
        final = self.finalconv3(out)  # [2,1,256,256]


        # output
        if self.n_class == 1:
            return F.sigmoid(final)  # for binary only
        else:
            return final

class unet_2D_Decoder4_drop(nn.Module):  # upsample 4 times
    def __init__(self, feature_scale=4, n_classes=1, is_deconv=True, in_channels=3, is_batchnorm=True, use_att=False,
                 use_res=True,
                 dblock_type='AS', use_rfnet=False):
        super(unet_2D_Decoder4_drop, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale
        self.n_class = n_classes
        self.use_att = use_att
        self.use_rfnet = use_rfnet

        filters = [64, 128, 256, 512]
        self.use_drop = True
        drop_rate = 0.1
        self.dropblock = LinearScheduler(
            DropBlock2D(drop_prob=drop_rate, block_size=7),
            start_value=0.,
            stop_value=drop_rate,
            nr_steps=10000
        )

        if self.use_att:
            self.GAU4 = B.GAU_Res2(filters[3], filters[2])
            self.GAU3 = B.GAU_Res2(filters[2], filters[1])
            self.GAU2 = B.GAU_Res2(filters[1], filters[0])
            self.GAU1 = B.GAU_Res2(filters[0], filters[0])
            logger.info("using GAU attetnion...")
        else:
            # self.decoder4 = B.DecoderBlock(filters[3], filters[2])
            self.decoder3 = B.DecoderBlock(filters[2], filters[1])
            self.decoder2 = B.DecoderBlock(filters[1], filters[0])
            self.decoder1 = B.DecoderBlock(filters[0], filters[0])
            logger.info("without GAU attetnion...")
        if use_rfnet:
            self.final_conv = nn.Sequential(
                nn.Conv2d(filters[0], 32, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, n_classes, 3, padding=1)
            )
            self.RefNet = SpatialRefine(67, 64)
            logger.info('With refine module.')
        else:
            logger.info('Without Refine module.')
            self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)  # size*2
            self.finalrelu1 = nonlinearity
            self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
            self.finalrelu2 = nonlinearity
            self.finalconv3 = nn.Conv2d(32, n_classes, 3, padding=1)

    def forward(self, e0, e1, e2, e3, e4):

        # e4: [4,512,16,16] e3:[4,256,32,32] e2:[4,128,64,64], e1:[4,64,128,128] e0:[2,64,256,256]
        if self.use_drop:
            self.dropblock.step()

        if self.use_att:
            d4 = self.GAU4(e4, e3)  # 16
            d3 = self.GAU3(d4, e2)  # 32
            d2 = self.GAU2(d3, e1)  # 64
            d1 = self.GAU1(d2, e0)  # 128
        else:
            d4 = self.decoder4(e4) + e3  # [2,256,16,16]
            d3 = self.decoder3(e3) + e2  # [2,128,32,32]
            d2 = self.decoder2(d3) + e1  # [2,64,64,64]
            d1 = self.decoder1(d2)  # [2,64,128,128]

        out = self.finaldeconv1(d1)  # [2,64,256,256]
        out = self.finalrelu1(out)
        out = self.finalconv2(out)  # [2,32,256,256]
        out = self.finalrelu2(out)
        out = self.dropblock(out)
        final = self.finalconv3(out)  # [2,1,256,256]


        # output
        if self.n_class == 1:
            return F.sigmoid(final)  # for binary only
        else:
            return final

class unet_2D_Decoder4_MS(nn.Module):  # upsample 4 times
    def __init__(self, feature_scale=4, n_classes=1, is_deconv=True, in_channels=3, is_batchnorm=True, use_att=False,
                 use_res=True,
                 dblock_type='AS', use_rfnet=False,use_DS=False):
        super(unet_2D_Decoder4_MS, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale
        self.n_class = n_classes
        self.use_att = use_att
        self.use_DS = use_DS

        filters = [64, 128, 256, 512]
        # self.use_drop = True
        # drop_rate = 0.1
        # self.dropblock = LinearScheduler(
        #     DropBlock2D(drop_prob=drop_rate, block_size=7),
        #     start_value=0.,
        #     stop_value=drop_rate,
        #     nr_steps=10000
        # )

        if self.use_att:
            self.GAU4 = B.GAU_Res2(filters[3], filters[2])
            self.GAU3 = B.GAU_Res2(filters[2], filters[1])
            self.GAU2 = B.GAU_Res2(filters[1], filters[0])
            self.GAU1 = B.GAU_Res2(filters[0], filters[0])
            logger.info("using GAU attetnion...")
        else:
            # self.decoder4 = B.DecoderBlock(filters[3], filters[2])
            self.decoder3 = B.DecoderBlock(filters[2], filters[1])
            self.decoder2 = B.DecoderBlock(filters[1], filters[0])
            self.decoder1 = B.DecoderBlock(filters[0], filters[0])
            logger.info("without GAU attetnion...")
        self.class_conv=nn.Sequential(nn.ConvTranspose2d(filters[0], 32, 4, 2, 1),  # size*2
             nn.ReLU(inplace=True),
             nn.Conv2d(32, 32, 3, padding=1),
             nn.ReLU(inplace=True),
             nn.Conv2d(32, n_classes, 3, padding=1),
             nn.Sigmoid())
        if self.use_DS:

            logger.info("using deep supervsion...")
            self.class_conv1 = nn.Sequential(nn.Conv2d(filters[2], 32, 3, 1, 1),  # 256
                                             nn.ReLU(inplace=True),
                                             # nn.Conv2d(32, 32, 3, padding=1),#will cause val_f1_score=nan
                                             # conv_act,
                                             #self.dropblock,
                                             nn.Conv2d(32, n_classes, 3, padding=1),
                                             nn.Sigmoid()
                                             )

            self.class_conv2 = nn.Sequential(nn.Conv2d(filters[1], 32, 3, 1, 1),  # 256
                                             nn.ReLU(inplace=True),
                                             # nn.Conv2d(32, 32, 3, padding=1),#will cause val_f1_score=nan
                                             # conv_act,
                                             #self.dropblock,
                                             nn.Conv2d(32, n_classes, 3, padding=1),
                                             nn.Sigmoid()
                                             )

            self.class_conv3 = nn.Sequential(nn.Conv2d(filters[0], 32, 3, 1, 1),  # 256
                                             nn.ReLU(inplace=True),
                                             # nn.Conv2d(32, 32, 3, padding=1),#will cause val_f1_score=nan
                                             # conv_act,
                                             #self.dropblock,
                                             nn.Conv2d(32, n_classes, 3, padding=1),
                                             nn.Sigmoid()
                                             )
        # if use_rfnet:
        #     self.final_conv = nn.Sequential(
        #         nn.Conv2d(filters[0], 32, 3, padding=1),
        #         nn.ReLU(inplace=True),
        #         nn.Conv2d(32, n_classes, 3, padding=1)
        #     )
        #     self.RefNet = SpatialRefine(67, 64)
        #     logger.info('With refine module.')
        # else:
        #     logger.info('Without Refine module.')
        #     self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)  # size*2
        #     self.finalrelu1 = nonlinearity
        #     self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        #     self.finalrelu2 = nonlinearity
        #     self.finalconv3 = nn.Conv2d(32, n_classes, 3, padding=1)



    def forward(self, e0, e1, e2, e3, e4):

        # e4: [4,512,16,16] e3:[4,256,32,32] e2:[4,128,64,64], e1:[4,64,128,128] e0:[2,64,256,256]
        # if self.use_drop:
        #     self.dropblock.step()

        if self.use_att:
            d4 = self.GAU4(e4, e3)  # 16
            d3 = self.GAU3(d4, e2)  # 32
            d2 = self.GAU2(d3, e1)  # 64
            d1 = self.GAU1(d2, e0)  # 128
        else:
            d4 = self.decoder4(e4) + e3  # [2,256,16,16]
            d3 = self.decoder3(e3) + e2  # [2,128,32,32]
            d2 = self.decoder2(d3) + e1  # [2,64,64,64]
            d1 = self.decoder1(d2)  # [2,64,128,128]

        # out = self.finaldeconv1(d1)  # [2,64,256,256]
        # out = self.finalrelu1(out)
        # out = self.finalconv2(out)  # [2,32,256,256]
        # out = self.finalrelu2(out)
        # out = self.dropblock(out)
        # final = self.finalconv3(out)  # [2,1,256,256]
        final=self.class_conv(d1)
        if self.use_DS:
            out1=self.class_conv1(d4)
            out2 = self.class_conv2(d3)
            out3 = self.class_conv3(d2)

            return out1,out2,out3,final
        else:
            return final



class unet_2D_Encoder4Ref(nn.Module):#downsample 4 times
    def __init__(self, feature_scale=4, n_classes=1, is_deconv=True, in_channels=3, is_batchnorm=True, use_att=False,
                 use_res=True,
                 dblock_type='AS'):
        super(unet_2D_Encoder4, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale
        self.n_class = n_classes
        self.use_att = use_att



        # downsampling
        from torchvision import models
        filters = [64, 128, 256,512]
        resnet = models.resnet34(pretrained=True)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        if dblock_type=='AS':
           self.dblock = Dblock2(512,conv_num=3)
        if dblock_type=='ASS':
           self.dblock = Dblock_Smooth(512,conv_num=3)
        elif dblock_type=='ASPP':
            self.dblock=ASPP(512)
        else:
            self.dblock=LKPP(in_chan=512, out_chan=128, mode='parallel', with_gp=False)

    def forward(self, inputs):

        x = self.firstconv(inputs)  # [2,64,64,64]   [4,3,128,128]==>[4,64,64,64]
        x = self.firstbn(x)
        x = self.firstrelu(x)# [2,64,64,64]
        e0 = self.firstmaxpool(x)  # [2,64,64,64]    [4,64,32,32]
        e1 = self.encoder1(e0)  ##[2,64,64,64]      [4,64,32,32]
        e2 = self.encoder2(e1)  # [2,128,32,32]     [4,128,16,16]
        e3 = self.encoder3(e2)  # [2,256,16,16]     [4,256,8,8]
        e4 = self.encoder4(e3)  # [2,512,8,8]

        # # Center
        e4 = self.dblock(e4)  # [2,512,8,8]   in decoder
        return  input, x, e1, e2, e3,e4

class unet_2D_Decoder4Ref(nn.Module):#upsample 4 times
    def __init__(self, feature_scale=4, n_classes=1, is_deconv=True, in_channels=3, is_batchnorm=True, use_att=False,
                 use_res=True,
                 dblock_type='AS', use_rfnet=True):
        super(unet_2D_Decoder4, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale
        self.n_class = n_classes
        self.use_att = use_att
        self.use_rfnet=use_rfnet



        filters = [64, 128, 256,512]

        if self.use_att:
            self.GAU4 = B.GAU_Res2(filters[3], filters[2])
            self.GAU3 = B.GAU_Res2(filters[2], filters[1])
            self.GAU2 = B.GAU_Res2(filters[1], filters[0])
            self.GAU1 = B.GAU_Res2(filters[0], filters[0])
            logger.info("using GAU attetnion...")
        else:
            #self.decoder4 = B.DecoderBlock(filters[3], filters[2])
            self.decoder3 = B.DecoderBlock(filters[2], filters[1])
            self.decoder2 = B.DecoderBlock(filters[1], filters[0])
            self.decoder1 = B.DecoderBlock(filters[0], filters[0])
            logger.info("without GAU attetnion...")
        if use_rfnet:
            self.final_conv=nn.Sequential(
                nn.Conv2d(filters[0], 32, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, n_classes, 3, padding=1)
            )
            self.RefNet = SpatialRefine(67, 64)
            logger.info('With refine module.')
        else:
            logger.info('Without Refine module.')
            self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)  # size*2
            self.finalrelu1 = nonlinearity
            self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
            self.finalrelu2 = nonlinearity
            self.finalconv3 = nn.Conv2d(32, n_classes, 3, padding=1)

    def forward(self,input,e0, e1, e2, e3,e4):

        #e4: [4,512,16,16] e3:[4,256,32,32] e2:[4,128,64,64], e1:[4,64,128,128] e0:[2,64,256,256]


        if self.use_att:
            d4 = self.GAU4(e4, e3)  # 16
            d3 = self.GAU3(d4, e2)  # 32
            d2 = self.GAU2(d3, e1)  # 64
            d1 = self.GAU1(d2, e0)  # 128
        else:
            d4 = self.decoder4(e4) + e3  # [2,256,16,16]
            d3 = self.decoder3(e3) + e2  # [2,128,32,32]
            d2 = self.decoder2(d3) + e1  # [2,64,64,64]
            d1 = self.decoder1(d2)  # [2,64,128,128]

        if self.use_rfnet:
            out=self.final_conv(d1)
            _,_,H0, W0 = e0.size()
            _,_,H,W=input.size()
            fea0=F.interpolate(input,(H0, W0), mode='bilinear', align_corners=True)
            fea=torch.cat([fea0,e0],dim=1)
            out=self.RefNet(out,fea)
            final=F.interpolate(out, (H, W), mode='bilinear', align_corners=True)

        else:
            out = self.finaldeconv1(d1)  # [2,64,256,256]
            out = self.finalrelu1(out)
            out = self.finalconv2(out)  # [2,32,256,256]
            out = self.finalrelu2(out)
            final = self.finalconv3(out)  # [2,1,256,256]


        #output
        if self.n_class == 1:
            return F.sigmoid(final)  # for binary only
        else:
            return final

class unet_2D_PreTrain512_ED(nn.Module):

    def __init__(self, feature_scale=4, n_classes=1, is_deconv=True, in_channels=3, is_batchnorm=True, use_att=False,
                 use_res=True, dblock_type='AS', use_rfnet=False, use_logits=False):
        super(unet_2D_PreTrain512_ED, self).__init__()

        self.encoder=unet_2D_Encoder4(in_channels=in_channels,dblock_type=dblock_type)
        self.decoder=unet_2D_Decoder4(n_classes=n_classes,use_att=use_att)

    def forward(self, x):
        e0, e1, e2, e3,e4=self.encoder(x)
        seg=self.decoder(e0, e1, e2, e3,e4)
        return e4,seg






class unet_2D_PreTrain512_ED2(nn.Module):

    def __init__(self, feature_scale=4, n_classes=1, is_deconv=True, in_channels=3, is_batchnorm=True, use_att=False,
                 use_res=True, dblock_type='AS', use_rfnet=False, use_logits=False):
        super(unet_2D_PreTrain512_ED2, self).__init__()

        self.encoder=unet_2D_Encoder4(in_channels=in_channels,dblock_type=dblock_type)
        self.decoder=unet_2D_Decoder4(n_classes=n_classes,use_att=use_att)

    def forward(self, x):
        e0, e1, e2, e3,e4=self.encoder(x)
        seg=self.decoder(e0, e1, e2, e3,e4)
        return e4,seg

class unet_2D_PreTrain512_ED2_drop(nn.Module):

    def __init__(self, feature_scale=4, n_classes=1, is_deconv=True, in_channels=3, is_batchnorm=True, use_att=False,
                 use_res=True, dblock_type='AS', use_rfnet=False, use_logits=False):
        super(unet_2D_PreTrain512_ED2_drop, self).__init__()

        self.encoder=unet_2D_Encoder4(in_channels=in_channels,dblock_type=dblock_type)
        self.decoder=unet_2D_Decoder4_drop(n_classes=n_classes,use_att=use_att)

    def forward(self, x):
        e0, e1, e2, e3,e4=self.encoder(x)
        seg=self.decoder(e0, e1, e2, e3,e4)
        return e4,seg

class unet_2D_PreTrain512_MS(nn.Module):

    def __init__(self, feature_scale=4, n_classes=1, is_deconv=True, in_channels=3, is_batchnorm=True, use_att=False,
                 use_res=True, dblock_type='AS', use_rfnet=False, use_DS=False):
        super(unet_2D_PreTrain512_MS, self).__init__()

        self.encoder=unet_2D_Encoder4(in_channels=in_channels,dblock_type=dblock_type)
        self.decoder=unet_2D_Decoder4_MS(n_classes=n_classes,use_att=use_att,use_DS=use_DS)
        self.use_DS=use_DS

    def forward(self, x):
        e0, e1, e2, e3,e4=self.encoder(x)
        seg=self.decoder(e0, e1, e2, e3,e4)
        if self.use_DS:
            out1,out2,out3,final=self.decoder(e0,e1,e2,e3,e4)
            return out1,out2,out3,final
        else:
            return self.decoder(e0,e1,e2,e3,e4)


class unet_2D_PreTrain512_ED2Ref(nn.Module):

    def __init__(self, feature_scale=4, n_classes=1, is_deconv=True, in_channels=3, is_batchnorm=True, use_att=False,
                 use_res=True, dblock_type='AS', use_rfnet=False, use_logits=False):
        super(unet_2D_PreTrain512_ED2Ref, self).__init__()

        self.encoder=unet_2D_Encoder4Ref(in_channels=in_channels,dblock_type=dblock_type)
        self.decoder=unet_2D_Decoder4Ref(n_classes=n_classes,use_att=use_att)

    def forward(self, x):
        x,e0, e1, e2, e3,e4=self.encoder(x)
        seg=self.decoder(x,e0, e1, e2, e3,e4)
        return e4,seg



class unet_2D_PreTrainC2(nn.Module):

    def __init__(self, feature_scale=4, n_classes=1, is_deconv=True, in_channels=3, is_batchnorm=True, use_att=False,
                 use_res=True, dblock_type='AS', use_rfnet=False, use_logits=False):
        super(unet_2D_PreTrainC2, self).__init__()

        self.encoder=unet_2D_Encoder(in_channels=in_channels)
        self.decoder1=unet_2D_Decoder(n_classes=n_classes,use_att=use_att)
        self.decoder2 = unet_2D_Decoder(n_classes=n_classes,use_att=use_att)
    def forward(self, x):
        x_conv, e1, e2, e3, e4=self.encoder(x)
        _,seg1=self.decoder1(x_conv,e1,e2,e3,e4)
        _,seg2 = self.decoder2(x_conv, e1, e2, e3, e4)
        return e4,seg1,seg2





class unet_2D_PreTrainED(nn.Module):

    def __init__(self, feature_scale=4, n_classes=1, is_deconv=True, in_channels=3, is_batchnorm=True, use_att=False,
                 use_res=True,
                 dblock_type='AS', use_rfnet=False, use_logits=False):
        super(unet_2D_PreTrainED, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale
        self.n_class = n_classes
        self.use_att = use_att

        if use_res:
            conv_block = unetConv2_res
            logger.info("using resblock conv...")
        else:
            conv_block = unetConv2
            logger.info("using basic conv...")

        # downsampling
        from torchvision import models
        filters = [64, 128, 256, 512]
        resnet = models.resnet34(pretrained=True)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4
        self.dblock = Dblock(512)

        if self.use_att:
            self.GAU4 = B.GAU(filters[3], filters[2], use_res=False)
            self.GAU3 = B.GAU(filters[2], filters[1], use_res=False)
            self.GAU2 = B.GAU(filters[1], filters[0], use_res=False)
            self.GAU1 = B.GAU(filters[0], filters[0], use_res=False)
            logger.info("using GAU attetnion...")
        else:
            self.decoder4 = B.DecoderBlock(filters[3], filters[2])
            self.decoder3 = B.DecoderBlock(filters[2], filters[1])
            self.decoder2 = B.DecoderBlock(filters[1], filters[0])
            self.decoder1 = B.DecoderBlock(filters[0], filters[0])
            logger.info("without GAU attetnion...")
        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)  # size*2
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, n_classes, 3, padding=1)

        # self.encoder=nn.Sequential(
        # resnet.conv1,
        # resnet.bn1,
        # resnet.relu,
        # resnet.maxpool,
        # resnet.layer1,
        # resnet.layer2,
        # resnet.layer3,
        # resnet.layer4,
        # Dblock(512)
        # )
        # self.decoder=nn.Sequential(
        # B.GAU(filters[3], filters[2], use_res=False),
        # B.GAU(filters[2], filters[1], use_res=False),
        # B.GAU(filters[1], filters[0], use_res=False),
        # B.GAU(filters[0], filters[0], use_res=False),
        # nn.ConvTranspose2d(filters[0], 32, 4, 2, 1),  # size*2
        # nn.ReLU(inplace=True),
        # nn.Conv2d(32, 32, 3, padding=1),
        # nn.ReLU(inplace=True),
        # nn.Conv2d(32, n_classes, 3, padding=1)
        # )

        # if use_rfnet:
        #     self.RefNet = Dblock(1)
        #     logger.info('With refine module.')
        # else:
        #     self.RefNet = None
        #     logger.info('Without Refine module.')
        # self.use_logits = use_logits

    def forward(self, inputs):

        x = self.firstconv(inputs)  # [2,64,128,128]
        x = self.firstbn(x)
        x_conv = self.firstrelu(x)
        x1 = self.firstmaxpool(x_conv)  # [2,64,64,64]
        e1 = self.encoder1(x1)  ##[2,64,64,64]
        e2 = self.encoder2(e1)  # [2,128,32,32]
        e3 = self.encoder3(e2)  # [2,256,16,16]
        e4 = self.encoder4(e3)  # [2,512,8,8]

        # Center
        e4 = self.dblock(e4)  # [2,512,8,8]

        # Decoder
        if self.use_att:
            d4 = self.GAU4(e4, e3)  # 16
            d3 = self.GAU3(d4, e2)  # 32
            d2 = self.GAU2(d3, e1)  # 64
            d1 = self.GAU1(d2, x_conv)  # 128
        else:
            d4 = self.decoder4(e4) + e3  # [2,256,16,16]
            d3 = self.decoder3(d4) + e2  # [2,128,32,32]
            d2 = self.decoder2(d3) + e1  # [2,64,64,64]
            d1 = self.decoder1(d2)  # [2,64,128,128]
        out = self.finaldeconv1(d1)  # [2,64,256,256]
        out = self.finalrelu1(out)
        out = self.finalconv2(out)  # [2,32,256,256]
        out = self.finalrelu2(out)
        final = self.finalconv3(out)  # [2,1,256,256]
        #
        # if self.RefNet:
        #     final = self.RefNet(final)
        # if self.use_logits:
        #     return final
        # else:
        #     if self.n_class == 1:
        #         return F.sigmoid(final)  # for binary only
        #     else:
        #         return final

        return e4,final




class unet_2D_PreTrainMC(nn.Module):

    def __init__(self, feature_scale=4, n_classes=1, is_deconv=True, in_channels=3, is_batchnorm=True,use_att=False,use_res=True,
        dblock_type='AS',use_rfnet=False,use_logits=True):
        super(unet_2D_PreTrainMC, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale
        self.n_class=n_classes
        self.use_att=use_att

        if use_res:
            conv_block=unetConv2_res
            logger.info("using resblock conv...")
        else:
            conv_block=unetConv2
            logger.info("using basic conv...")


        # downsampling
        from torchvision import models
        filters = [64, 128, 256, 512]
        resnet = models.resnet34(pretrained=True)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        #self.dblock = Dblock(512)

        if self.use_att:
            self.GAU4 = B.GAU(filters[3], filters[2], use_res=False)
            self.GAU3 = B.GAU(filters[2], filters[1], use_res=False)
            self.GAU2 = B.GAU(filters[1], filters[0], use_res=False)
            self.GAU1 = B.GAU(filters[0], filters[0], use_res=False)
            logger.info("using GAU attetnion...")
        else:
            self.decoder4 = B.DecoderBlock(filters[3], filters[2])
            self.decoder3 = B.DecoderBlock(filters[2], filters[1])
            self.decoder2 = B.DecoderBlock(filters[1], filters[0])
            self.decoder1 = B.DecoderBlock(filters[0], filters[0])
            logger.info("without GAU attetnion...")

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)  # size*2
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, n_classes, 3, padding=1)

        self.seg1=nn.Sequential(nn.ConvTranspose2d(filters[0], 32, 4, 2, 1),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(32, 32, 3, padding=1),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(32, n_classes, 3, padding=1),
                                Dblock(n_classes)

        )
        self.seg2 = nn.Sequential(nn.ConvTranspose2d(filters[0], 32, 4, 2, 1),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(32, 32, 3, padding=1),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(32, n_classes, 3, padding=1),
                                  Dblock(n_classes)

                                  )


        #self.dblock_typ=dblock_type
        if dblock_type=='ASPP':
           self.Dblock = ASPP(in_channel=filters[3])
           logger.info('With ASPP module.')
        elif dblock_type=='AS':
           self.Dblock=Dblock(filters[3])
           logger.info('With AS module.')
        else:
            logger.info('Without dblock module.')
            self.Dblock=None

        # if use_rfnet:
        #     self.RefNet=Dblock(1)
        #     logger.info('With refine module.')
        # else:
        #     self.RefNet = None
        #     logger.info('Without Refine module.')



        self.use_logits=use_logits

    def forward(self, inputs):


        x = self.firstconv(inputs)  # [2,64,128,128]
        x = self.firstbn(x)
        x_conv = self.firstrelu(x)
        x1 = self.firstmaxpool(x_conv)  # [2,64,64,64]
        e1 = self.encoder1(x1)  ##[2,64,64,64]
        e2 = self.encoder2(e1)  # [2,128,32,32]
        e3 = self.encoder3(e2)  # [2,256,16,16]
        e4 = self.encoder4(e3)  # [2,512,8,8]

        # Center
        e4 = self.Dblock(e4)  # [2,512,8,8]

        # Decoder
        if self.use_att:
            d4=self.GAU4(e4,e3)#16
            d3=self.GAU3(d4,e2)#32
            d2=self.GAU2(d3,e1)#64
            d1=self.GAU1(d2,x_conv)#128
        else:
            d4 = self.decoder4(e4) + e3  # [2,256,16,16]
            d3 = self.decoder3(d4) + e2  # [2,128,32,32]
            d2 = self.decoder2(d3) + e1  # [2,64,64,64]
            d1 = self.decoder1(d2)  # [2,64,128,128]
        # out = self.finaldeconv1(d1)  # [2,64,256,256]
        # out = self.finalrelu1(out)
        # out = self.finalconv2(out)  # [2,32,256,256]
        # out = self.finalrelu2(out)
        # final = self.finalconv3(out)  # [2,1,256,256]

        final1=self.seg1(d1)
        final2=self.seg2(d1)

        # if self.RefNet:
        #     final=self.RefNet(final)
        if self.use_logits:
            return final1,final2
        else:
            if self.n_class == 1:
                return F.sigmoid(final1), F.sigmoid(final2)  # for binary only
            else:
                return final1,final2












class unet_2D_PreTrainSCRef(nn.Module):

    def __init__(self, feature_scale=4, n_classes=1, is_deconv=True, in_channels=3, is_batchnorm=True,use_att=False,use_res=True,
        dblock_type='AS',use_rfnet=False,use_logits=False):
        super(unet_2D_PreTrainSCRef, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale
        self.n_class=n_classes
        self.use_att=use_att

        # filters = [64, 128, 256, 512, 1024]
        # filters = [int(x / self.feature_scale) for x in filters]#[16,32,64,128,256]
        if use_res:
            conv_block=unetConv2_res
            logger.info("using resblock conv...")
        else:
            conv_block=unetConv2
            logger.info("using basic conv...")


        # downsampling
        from torchvision import models
        filters = [64, 128, 256, 512]
        resnet = models.resnet34(pretrained=True)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.dblock = Dblock(512)

        if self.use_att:
            self.GAU4 = B.GAU_CS(filters[3], filters[2], use_res=False)
            self.GAU3 = B.GAU_CS(filters[2], filters[1], use_res=False)
            self.GAU2 = B.GAU_CS(filters[1], filters[0], use_res=False)
            self.GAU1 = B.GAU_CS(filters[0], filters[0], use_res=False)
            logger.info("using GAU attetnion...")
        else:
            self.decoder4 = B.DecoderBlock(filters[3], filters[2])
            self.decoder3 = B.DecoderBlock(filters[2], filters[1])
            self.decoder2 = B.DecoderBlock(filters[1], filters[0])
            self.decoder1 = B.DecoderBlock(filters[0], filters[0])
            logger.info("without GAU attetnion...")
        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)  # size*2
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, n_classes, 3, padding=1)




        # #self.dblock_typ=dblock_type
        # if dblock_type=='ASPP':
        #    self.Dblock = ASPP(in_channel=filters[4])
        #    logger.info('With ASPP module.')
        # elif dblock_type=='AS':
        #    self.Dblock=Dblock(filters[4])
        #    logger.info('With AS module.')
        # else:
        #     logger.info('Without dblock module.')
        #     self.Dblock=None

        if use_rfnet:
            self.RefNet=Dblock(1)
            logger.info('With refine module.')
        else:
            self.RefNet = None
            logger.info('Without Refine module.')


        self.use_logits=use_logits

    def forward(self, inputs):


        x = self.firstconv(inputs)  # [2,64,128,128]
        x = self.firstbn(x)
        x_conv = self.firstrelu(x)
        x1 = self.firstmaxpool(x_conv)  # [2,64,64,64]
        e1 = self.encoder1(x1)  ##[2,64,64,64]
        e2 = self.encoder2(e1)  # [2,128,32,32]
        e3 = self.encoder3(e2)  # [2,256,16,16]
        e4 = self.encoder4(e3)  # [2,512,8,8]

        # Center
        e4 = self.dblock(e4)  # [2,512,8,8]

        # Decoder
        if self.use_att:
            d4=self.GAU4(e4,e3)#16
            d3=self.GAU3(d4,e2)#32
            d2=self.GAU2(d3,e1)#64
            d1=self.GAU1(d2,x_conv)#128
        else:
            d4 = self.decoder4(e4) + e3  # [2,256,16,16]
            d3 = self.decoder3(d4) + e2  # [2,128,32,32]
            d2 = self.decoder2(d3) + e1  # [2,64,64,64]
            d1 = self.decoder1(d2)  # [2,64,128,128]
        out = self.finaldeconv1(d1)  # [2,64,256,256]
        out = self.finalrelu1(out)
        out = self.finalconv2(out)  # [2,32,256,256]
        out = self.finalrelu2(out)
        final = self.finalconv3(out)  # [2,1,256,256]


        if self.RefNet:
            final=self.RefNet(final)
        if self.use_logits:
            return final
        else:
            if self.n_class == 1:
                return F.sigmoid(final)  # for binary only
            else:
                return final

class unet_2D_PreTrainSC(nn.Module):

    def __init__(self, feature_scale=4, n_classes=1, is_deconv=True, in_channels=3, is_batchnorm=True,use_att=False,use_res=True,
        dblock_type='AS',use_rfnet=False,use_logits=False):
        super(unet_2D_PreTrainSC, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale
        self.n_class=n_classes
        self.use_att=use_att

        # filters = [64, 128, 256, 512, 1024]
        # filters = [int(x / self.feature_scale) for x in filters]#[16,32,64,128,256]
        if use_res:
            conv_block=unetConv2_res
            logger.info("using resblock conv...")
        else:
            conv_block=unetConv2
            logger.info("using basic conv...")


        # downsampling
        from torchvision import models
        filters = [64, 128, 256, 512]
        resnet = models.resnet34(pretrained=True)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        #self.dblock_typ=dblock_type
        if dblock_type=='ASPP':
           self.Dblock = ASPP(in_channel=filters[-1])
           logger.info('With ASPP module.')
        elif dblock_type=='AS':
           self.Dblock=Dblock(filters[-1])
           logger.info('With AS module.')
        else:
            logger.info('Without dblock module.')
            self.Dblock=None

        if use_rfnet:
            self.RefNet=Dblock(1)
            logger.info('With refine module.')
        else:
            self.RefNet = None
            logger.info('Without Refine module.')

        if self.use_att:
            self.GAU4 = B.GAU(filters[3], filters[2], use_res=False)
            self.GAU3 = B.GAU(filters[2], filters[1], use_res=False)
            self.GAU2 = B.GAU(filters[1], filters[0], use_res=False)
            self.GAU1 = B.GAU(filters[0], filters[0], use_res=False)
            logger.info("using GAU attetnion...")
        else:
            self.decoder4 = B.DecoderBlock(filters[3], filters[2])
            self.decoder3 = B.DecoderBlock(filters[2], filters[1])
            self.decoder2 = B.DecoderBlock(filters[1], filters[0])
            self.decoder1 = B.DecoderBlock(filters[0], filters[0])
            logger.info("without GAU attetnion...")

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)  # size*2
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, n_classes, 3, padding=1)





        # if use_rfnet:
        #     self.RefNet=Dblock(1)
        #     logger.info('With refine module.')
        # else:
        #     self.RefNet = None
        #     logger.info('Without Refine module.')

        self.use_logits=use_logits

        # self.Dblock3=ASPP(filters[1])
        # self.Dblock4=ASPP(filters[2])
        # self.cam34 = ChannelAttention(filters[2]+filters[1])
        self.pam12_1 = nn.Sequential(
            nn.Conv2d(filters[0], filters[0]//2, (1, 9), padding=(0, 4)),
            nn.BatchNorm2d(filters[0]//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(filters[0]//2, 1, (9, 1), padding=(4, 0)),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True)
        )
        self.pam12_2 = nn.Sequential(
            nn.Conv2d(filters[0], filters[0]//2, (9, 1), padding=(4, 0)),
            nn.BatchNorm2d(filters[0]//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(filters[0]//2, 1, (1, 9), padding=(0, 4)),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True)
        )
        # self.conv12 = nn.Conv2d(2 * filters[0], filters[0], 3, padding=1)
        # self.conv1234 = nn.Conv2d(filters[0]+filters[1]+filters[2], filters[0], 3, padding=1)
    def forward(self, inputs):


        x = self.firstconv(inputs)  # [2,64,128,128]
        x = self.firstbn(x)
        x_conv = self.firstrelu(x)
        x1 = self.firstmaxpool(x_conv)  # [2,64,64,64]
        e1 = self.encoder1(x1)  ##[2,64,64,64]
        e2 = self.encoder2(e1)  # [2,128,32,32]
        e3 = self.encoder3(e2)  # [2,256,16,16]
        e4 = self.encoder4(e3)  # [2,512,8,8]

        # Center
        #e4 = self.Dblock(e4)  # [2,512,8,8]

        # Decoder
        if self.use_att:
            d4=self.GAU4(e4,e3)#16
            d3=self.GAU3(d4,e2)#32
            d2=self.GAU2(d3,e1)#64
            d1=self.GAU1(d2,x_conv)#128
        else:
            d4 = self.decoder4(e4) + e3  # [2,256,16,16]
            d3 = self.decoder3(d4) + e2  # [2,128,32,32]
            d2 = self.decoder2(d3) + e1  # [2,64,64,64]
            d1 = self.decoder1(d2)  # [2,64,128,128]

        #========================using SA for finer feature aggregation and CA for coaser feaure aggreration===============
        # d3=self.Dblock3(d3)
        # d4=self.Dblock4(d4)
        # d34=torch.cat([F.upsample_bilinear(d4,scale_factor=2),d3],dim=1)
        # d34=self.cam34(d34)*d34

        #d12=self.conv12(torch.cat([d1,F.upsample_bilinear(d2,scale_factor=2)],dim=1))
        att_1 = self.pam12_1(d1)
        att_2 = self.pam12_2(d1)
        att12 = F.sigmoid(att_1 + att_2)
        d1 = att12 * d1#[4,64,256,256]

        #d1234=self.conv1234(torch.cat([d12,F.upsample_bilinear(d34,scale_factor=4)],dim=1))






        out = self.finaldeconv1(d1)  # [2,64,256,256]
        out = self.finalrelu1(out)
        out = self.finalconv2(out)  # [2,32,256,256]
        out = self.finalrelu2(out)
        final = self.finalconv3(out)  # [2,1,256,256]


        if self.RefNet:
            final=self.RefNet(final)
        if self.use_logits:
            return final
        else:
            if self.n_class == 1:
                return F.sigmoid(final)  # for binary only
            else:
                return final


class ResNet_PreTrainSC(nn.Module):

    def __init__(self, feature_scale=4, n_classes=1, is_deconv=True, in_channels=3, is_batchnorm=True,use_att=False,use_res=True,
        dblock_type='AS',use_rfnet=False,use_logits=False):
        super(ResNet_PreTrainSC, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale
        self.n_class=n_classes
        self.use_att=use_att

        # filters = [64, 128, 256, 512, 1024]
        # filters = [int(x / self.feature_scale) for x in filters]#[16,32,64,128,256]
        if use_res:
            conv_block=unetConv2_res
            logger.info("using resblock conv...")
        else:
            conv_block=unetConv2
            logger.info("using basic conv...")


        # downsampling
        from torchvision import models
        filters = [64, 128, 256, 512]
        resnet = models.resnet34(pretrained=True)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        #self.dblock_typ=dblock_type
        # if dblock_type=='ASPP':
        #    self.Dblock = ASPP(in_channel=filters[-1])
        #    logger.info('With ASPP module.')
        # elif dblock_type=='AS':
        #    self.Dblock=Dblock(filters[-1])
        #    logger.info('With AS module.')
        # else:
        #     logger.info('Without dblock module.')
        #     self.Dblock=None



        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)  # size*2
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, n_classes, 3, padding=1)





        if use_rfnet:
            self.RefNet=Dblock(1)
            logger.info('With refine module.')
        else:
            self.RefNet = None
            logger.info('Without Refine module.')

        self.use_logits=use_logits

        self.Dblock2=GALD(filters[1])
        self.Dblock3=GALD(filters[2])
        self.Dblock4=GALD(filters[3])
        self.cam234 = ChannelAttention(filters[3]+filters[2]+filters[1])
        self.conv234=unetConv2(filters[3]+filters[2]+filters[1],filters[0],n=2)

        self.pam12_1 = nn.Sequential(
            nn.Conv2d(filters[0], filters[0]//2, (1, 9), padding=(0, 4)),
            nn.BatchNorm2d(filters[0]//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(filters[0]//2, 1, (9, 1), padding=(4, 0)),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True)
        )
        self.pam12_2 = nn.Sequential(
            nn.Conv2d(filters[0], filters[0]//2, (9, 1), padding=(4, 0)),
            nn.BatchNorm2d(filters[0]//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(filters[0]//2, 1, (1, 9), padding=(0, 4)),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True)
        )
        self.conv12 = nn.Conv2d(2 * filters[0], filters[0], 3, padding=1)
        # self.conv1234 = nn.Conv2d(filters[0]+filters[1]+filters[2], filters[0], 3, padding=1)
        self.conv1234=unetConv2(2*filters[0],filters[0],n=2)
    def forward(self, inputs):


        x = self.firstconv(inputs)  # [2,64,128,128]
        x = self.firstbn(x)
        x_conv = self.firstrelu(x)
        x1 = self.firstmaxpool(x_conv)  # [2,64,64,64]
        e1 = self.encoder1(x1)  ##[2,64,64,64]
        e2 = self.encoder2(e1)  # [2,128,32,32]
        e3 = self.encoder3(e2)  # [2,256,16,16]
        e4 = self.encoder4(e3)  # [2,512,8,8]

        # Center
        #e4 = self.Dblock(e4)  # [2,512,8,8]

        # Decoder
        # if self.use_att:
        #     d4=self.GAU4(e4,e3)#16
        #     d3=self.GAU3(d4,e2)#32
        #     d2=self.GAU2(d3,e1)#64
        #     d1=self.GAU1(d2,x_conv)#128
        # else:
        #     d4 = self.decoder4(e4) + e3  # [2,256,16,16]
        #     d3 = self.decoder3(d4) + e2  # [2,128,32,32]
        #     d2 = self.decoder2(d3) + e1  # [2,64,64,64]
        #     d1 = self.decoder1(d2)  # [2,64,128,128]

        #========================using SA for finer feature aggregation and CA for coaser feaure aggreration===============
        d2=self.Dblock2(e2)
        d3=self.Dblock3(e3)
        d4=self.Dblock4(e4)
        d234=torch.cat([F.upsample_bilinear(d4,scale_factor=4),F.upsample_bilinear(d3,scale_factor=2),d2],dim=1)
        d234=self.cam234(d234)*d234
        d234=self.conv234(d234)


        d12=self.conv12(torch.cat([x_conv,F.upsample_bilinear(e1,scale_factor=2)],dim=1))
        att_1 = self.pam12_1(d12)
        att_2 = self.pam12_2(d12)
        att12 = F.sigmoid(att_1 + att_2)
        d12 = att12 * d12#[4,64,256,256]

        d1234=self.conv1234(torch.cat([d12,F.upsample_bilinear(d234,scale_factor=4)],dim=1))

        out = self.finaldeconv1(d1234)  # [2,64,256,256]
        out = self.finalrelu1(out)
        out = self.finalconv2(out)  # [2,32,256,256]
        out = self.finalrelu2(out)
        final = self.finalconv3(out)  # [2,1,256,256]


        if self.RefNet:
            final=self.RefNet(final)
        if self.use_logits:
            return final
        else:
            if self.n_class == 1:
                return F.sigmoid(final)  # for binary only
            else:
                return final


#========using GFF module and DFP module https://github.com/nhatuan84/GFF-Gated-Fully-Fusion-for-Semantic-Segmentation/blob/master/GFF.py============
class unet_2D_PreTrainGFF(nn.Module):

    def __init__(self, feature_scale=4, n_classes=1, is_deconv=True, in_channels=3, is_batchnorm=True,use_att=False,use_res=True,
        dblock_type='AS',use_rfnet=False,use_logits=False):
        super(unet_2D_PreTrainGFF, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale
        self.n_class=n_classes
        self.use_att=use_att


        if use_res:
            conv_block=unetConv2_res
            logger.info("using resblock conv...")
        else:
            conv_block=unetConv2
            logger.info("using basic conv...")


        # downsampling
        from torchvision import models
        filters = [64, 128, 256, 512]
        resnet = models.resnet34(pretrained=True)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        #self.dblock_typ=dblock_type
        if dblock_type=='ASPP':
           self.Dblock = ASPP(in_channel=filters[-1])
           logger.info('With ASPP module.')
        elif dblock_type=='AS':
           self.Dblock=Dblock(filters[-1])
           logger.info('With AS module.')
        else:
            logger.info('Without dblock module.')
            self.dblock=None

        self.conv1_gff=nn.Sequential(
            nn.Conv2d(filters[0],filters[1],kernel_size=1,stride=2),
            nn.Conv2d(filters[1], filters[1], kernel_size=1),
            nn.Sigmoid()
        )
        self.conv1_gff0=nn.Conv2d(filters[0],filters[1],kernel_size=1,stride=2)
        self.conv1_gff1=nn.Conv2d(filters[1], filters[1], kernel_size=1)
        self.conv2_gff0 = nn.Conv2d(filters[1], filters[1], kernel_size=1)
        self.conv2_gff1 = nn.Conv2d(filters[1], filters[1], kernel_size=1)
        self.conv3_gff0 = nn.ConvTranspose2d(filters[2], filters[1], 4, 2, 1)
        self.conv3_gff1 = nn.Conv2d(filters[1], filters[1], kernel_size=1)
        self.conv4_gff0 = nn.Sequential(
            nn.ConvTranspose2d(filters[3], filters[1], 4, 2, 1),
            nn.ConvTranspose2d(filters[1], filters[1], 4, 2, 1))
        self.conv4_gff1 = nn.Conv2d(filters[1], filters[1], kernel_size=1)




        self.conv_fuse=unetConv2(filters[1],filters[1],n=2)
        self.d5_conv=nn.Conv2d(5*filters[1],filters[0],kernel_size=1)
        self.d4_conv = nn.Conv2d(4*filters[1], filters[0],
                                 kernel_size=1)
        self.d3_conv = nn.Conv2d( 3*filters[1] , filters[0],
                                 kernel_size=1)
        self.d2_conv = nn.Conv2d(2*filters[1] , filters[0],
                                 kernel_size=1)
        self.d1_conv = nn.Conv2d(filters[1], filters[0],
                                 kernel_size=1)

        self.d12345_conv=unetConv2(filters[0]*5,filters[0],n=1)


        # if self.use_att:
        #     self.GAU4 = B.GAU(filters[3], filters[2], use_res=False)
        #     self.GAU3 = B.GAU(filters[2], filters[1], use_res=False)
        #     self.GAU2 = B.GAU(filters[1], filters[0], use_res=False)
        #     self.GAU1 = B.GAU(filters[0], filters[0], use_res=False)
        #     logger.info("using GAU attetnion...")
        # else:
        #     self.decoder4 = B.DecoderBlock(filters[3], filters[2])
        #     self.decoder3 = B.DecoderBlock(filters[2], filters[1])
        #     self.decoder2 = B.DecoderBlock(filters[1], filters[0])
        #     self.decoder1 = B.DecoderBlock(filters[0], filters[0])
        #     logger.info("without GAU attetnion...")

        self.finaldeconv1 = nn.Sequential(nn.ConvTranspose2d(filters[0], 32, 4, 2, 1) ,
                                          nn.ConvTranspose2d(32, 32, 4, 2, 1),
                                          nn.ConvTranspose2d(32, 32, 4, 2, 1) )# size*2
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, n_classes, 3, padding=1)





        if use_rfnet:
            self.RefNet=Dblock(1)
            logger.info('With refine module.')
        else:
            self.RefNet = None
            logger.info('Without Refine module.')


        self.use_logits=use_logits

    def forward(self, inputs):


        x = self.firstconv(inputs)  # [2,64,1,128]
        x = self.firstbn(x)
        x_conv = self.firstrelu(x)
        x = self.firstmaxpool(x_conv)  # [2,64,64,64]
        e1 = self.encoder1(x)  ##[2,64,64,64]
        e2 = self.encoder2(e1)  # [2,128,32,32]
        e3 = self.encoder3(e2)  # [2,256,16,16]
        e4 = self.encoder4(e3)  # [2,512,8,8]


        e1n=self.conv1_gff0(e1)
        g1=F.sigmoid(self.conv1_gff1(e1n))
        e2n = self.conv2_gff0(e2)
        g2 = F.sigmoid(self.conv2_gff1(e2n))
        e3n = self.conv3_gff0(e3)
        g3 = F.sigmoid(self.conv3_gff1(e3n))
        e4n = self.conv4_gff0(e4)
        g4 = F.sigmoid(self.conv4_gff1(e4n))




        e1_gff = self.conv_fuse((1 + g1) *e1n + (1 - g1) * (g2 * e2n + g3 * e3n + g4 * e4n))
        e2_gff = self.conv_fuse((1 + g2) * e2n + (1 - g2) * (g1 * e1n + g3 * e3n + g4 * e4n))
        e3_gff = self.conv_fuse((1 + g3) * e3n + (1 - g3) * (g2 * e2n + g1 * e1n + g4 * e4n))
        e4_gff = self.conv_fuse((1 + g4) * e4n + (1 - g4) * (g2 * e2n + g3 * e3n + g1 * e1n))

        # Center
        psp = self.Dblock(e4)  # [2,512,8,8]
        psp=self.conv4_gff0(psp)

        d5=self.d5_conv(torch.cat([psp,e1_gff,e2_gff,e3_gff,e4_gff],dim=1))
        d4 = self.d4_conv(torch.cat([e1_gff, e2_gff, e3_gff, e4_gff], dim=1))
        d3 = self.d3_conv(torch.cat([e1_gff, e2_gff, e3_gff], dim=1))
        d2 = self.d2_conv(torch.cat([e1_gff, e2_gff], dim=1))
        d1 = self.d1_conv(e1_gff)

        d12345=self.d12345_conv(torch.cat([d5,d4,d3,d2,d1],dim=1))




        out = self.finaldeconv1(d12345)  # [2,64,256,256]
        out = self.finalrelu1(out)
        out = self.finalconv2(out)  # [2,32,256,256]
        out = self.finalrelu2(out)
        final = self.finalconv3(out)  # [2,1,256,256]


        if self.RefNet:
            final=self.RefNet(final)
        if self.use_logits:
            return final
        else:
            if self.n_class == 1:
                return F.sigmoid(final)  # for binary only
            else:
                return final

#=======================using unet3+-like dense connections=====================================
class unet_2D_PreTrainDC(nn.Module):

    def __init__(self, feature_scale=4, n_classes=1, is_deconv=True, in_channels=3, is_batchnorm=True,use_att=False,use_res=True,
        dblock_type='AS',use_rfnet=False,use_logits=False):
        super(unet_2D_PreTrainDC, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale
        self.n_class=n_classes
        self.use_att=use_att

        # filters = [64, 128, 256, 512, 1024]
        # filters = [int(x / self.feature_scale) for x in filters]#[16,32,64,128,256]
        if use_res:
            conv_block=unetConv2_res
            logger.info("using resblock conv...")
        else:
            conv_block=unetConv2
            logger.info("using basic conv...")


        # downsampling
        from torchvision import models
        filters = [64, 128, 256, 512]
        resnet = models.resnet34(pretrained=True)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        #self.dblock_typ=dblock_type
        if dblock_type=='ASPP':
           self.Dblock = ASPP(in_channel=filters[-1])
           logger.info('With ASPP module.')
        elif dblock_type=='AS':
           self.Dblock=Dblock(filters[-1])
           logger.info('With AS module.')
        else:
            logger.info('Without dblock module.')
            self.dblock=None

        #self.CatChannels = filters[0]
        self.encoder05=nn.Sequential(nn.MaxPool2d(8, 8, ceil_mode=True),
                                     nn.Conv2d(filters[0], filters[2], 1),
                                     nn.BatchNorm2d(filters[2]),
                                     nn.ReLU(inplace=True))
        self.encoder06 = nn.Sequential(nn.MaxPool2d(4, 4, ceil_mode=True),
                                       nn.Conv2d(filters[0], filters[1], 1),
                                       nn.BatchNorm2d(filters[1]),
                                       nn.ReLU(inplace=True))
        self.encoder07 = nn.Sequential(nn.MaxPool2d(2, 2, ceil_mode=True),
                                       nn.Conv2d(filters[0], filters[0], 1),
                                       nn.BatchNorm2d(filters[0]),
                                       nn.ReLU(inplace=True))

        self.encoder15 = nn.Sequential(nn.MaxPool2d(4, 4, ceil_mode=True),
                                       nn.Conv2d(filters[0], filters[2], 1),
                                       nn.BatchNorm2d(filters[2]),
                                       nn.ReLU(inplace=True))
        self.encoder16 = nn.Sequential(nn.MaxPool2d(2, 2, ceil_mode=True),
                                       nn.Conv2d(filters[0], filters[1], 1),
                                       nn.BatchNorm2d(filters[1]),
                                       nn.ReLU(inplace=True))

        self.encoder25 = nn.Sequential(nn.MaxPool2d(2, 2, ceil_mode=True),
                                       nn.Conv2d(filters[1], filters[2], 1),
                                       nn.BatchNorm2d(filters[2]),
                                       nn.ReLU(inplace=True))

        self.decoder45 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear'),
                                       nn.Conv2d(filters[3], filters[2], 1),
                                       nn.BatchNorm2d(filters[2]),
                                       nn.ReLU(inplace=True))

        self.decoder46 = nn.Sequential(nn.Upsample(scale_factor=4, mode='bilinear'),
                                       nn.Conv2d(filters[3], filters[1], 1),
                                       nn.BatchNorm2d(filters[1]),
                                       nn.ReLU(inplace=True))
        self.decoder47 = nn.Sequential(nn.Upsample(scale_factor=8, mode='bilinear'),
                                       nn.Conv2d(filters[3], filters[0], 1),
                                       nn.BatchNorm2d(filters[0]),
                                       nn.ReLU(inplace=True))
        self.decoder48 = nn.Sequential(nn.Upsample(scale_factor=16, mode='bilinear'),
                                       nn.Conv2d(filters[3], filters[0], 1),
                                       nn.BatchNorm2d(filters[0]),
                                       nn.ReLU(inplace=True))

        self.decoder56 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear'),
                                       nn.Conv2d(filters[2], filters[1], 1),
                                       nn.BatchNorm2d(filters[1]),
                                       nn.ReLU(inplace=True))
        self.decoder57 = nn.Sequential(nn.Upsample(scale_factor=4, mode='bilinear'),
                                       nn.Conv2d(filters[2], filters[0], 1),
                                       nn.BatchNorm2d(filters[0]),
                                       nn.ReLU(inplace=True))
        self.decoder58 = nn.Sequential(nn.Upsample(scale_factor=8, mode='bilinear'),
                                       nn.Conv2d(filters[2], filters[0], 1),
                                       nn.BatchNorm2d(filters[0]),
                                       nn.ReLU(inplace=True))
        self.decoder67 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear'),
                                       nn.Conv2d(filters[1], filters[0], 1),
                                       nn.BatchNorm2d(filters[0]),
                                       nn.ReLU(inplace=True))
        self.decoder68 = nn.Sequential(nn.Upsample(scale_factor=4, mode='bilinear'),
                                       nn.Conv2d(filters[1], filters[0], 1),
                                       nn.BatchNorm2d(filters[0]),
                                       nn.ReLU(inplace=True))

        self.decoder78 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear'),
                                       nn.Conv2d(filters[0], filters[0], 1),
                                       nn.BatchNorm2d(filters[0]),
                                       nn.ReLU(inplace=True))


        self.dconv5=unetConv2(5*filters[2], filters[2],n=2, is_batchnorm=True)
        self.dconv6 = unetConv2(5 * filters[1], filters[1], n=2, is_batchnorm=True)
        self.dconv7 = unetConv2(5 * filters[0], filters[0], n=2, is_batchnorm=True)
        self.dconv8 = unetConv2(5 * filters[0], filters[0], n=2, is_batchnorm=True)

        # if self.use_att:
        #     self.GAU4 = B.GAU(filters[3], filters[2], use_res=False)
        #     self.GAU3 = B.GAU(filters[2], filters[1], use_res=False)
        #     self.GAU2 = B.GAU(filters[1], filters[0], use_res=False)
        #     self.GAU1 = B.GAU(filters[0], filters[0], use_res=False)
        #     logger.info("using GAU attetnion...")
        # else:
        #     self.decoder4 = B.DecoderBlock(filters[3], filters[2])
        #     self.decoder3 = B.DecoderBlock(filters[2], filters[1])
        #     self.decoder2 = B.DecoderBlock(filters[1], filters[0])
        #     self.decoder1 = B.DecoderBlock(filters[0], filters[0])
        #     logger.info("without GAU attetnion...")

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)  # size*2
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, n_classes, 3, padding=1)




        # #self.dblock_typ=dblock_type
        # if dblock_type=='ASPP':
        #    self.Dblock = ASPP(in_channel=filters[4])
        #    logger.info('With ASPP module.')
        # elif dblock_type=='AS':
        #    self.Dblock=Dblock(filters[4])
        #    logger.info('With AS module.')
        # else:
        #     logger.info('Without dblock module.')
        #     self.Dblock=None

        if use_rfnet:
            self.RefNet=Dblock(1)
            logger.info('With refine module.')
        else:
            self.RefNet = None
            logger.info('Without Refine module.')


        self.use_logits=use_logits

    def forward(self, inputs):


        x = self.firstconv(inputs)  # [2,64,128,128]
        x = self.firstbn(x)
        x_conv = self.firstrelu(x)
        x1 = self.firstmaxpool(x_conv)  # [2,64,64,64]
        e1 = self.encoder1(x1)  ##[2,64,64,64]
        e2 = self.encoder2(e1)  # [2,128,32,32]
        e3 = self.encoder3(e2)  # [2,256,16,16]
        e4 = self.encoder4(e3)  # [2,512,8,8]

        # Center
        e4 = self.Dblock(e4)  # [2,512,8,8]

        e05=self.encoder05(x_conv)
        e15=self.encoder15(e1)
        e25=self.encoder25(e2)
        #d5 = self.GAU4(e4, e3 + e05 + e15 + e25)  # 16
        d45=self.decoder45(e4)
        d5=self.dconv5(torch.cat([e05,e15,e25,e3,d45],dim=1))


        e06=self.encoder06(x_conv)
        e16=self.encoder16(e1)
        d46=self.decoder46(e4)
        #d6 = self.GAU3(d5 + d45, e2 + e06 + e16)  # 32
        d56=self.decoder56(d5)
        d6 = self.dconv6(torch.cat([e06, e16, e2,d46,d56], dim=1))

        e07=self.encoder07(x_conv)
        d47=self.decoder47(e4)
        d57=self.decoder57(d5)
        d67=self.decoder67(d6)

        #d7 = self.GAU2(d6 + d46 + d56, e1 + e07)  # 64
        d7 = self.dconv7(torch.cat([e07, e1, d47,d57,d67], dim=1))

        d48=self.decoder48(e4)
        d58=self.decoder58(d5)
        d68=self.decoder68(d6)
        d78=self.decoder78(d7)
        #d8=self.GAU1(d7+d47+d57+d67,x_conv)
        d8 = self.dconv8(torch.cat([x_conv, d48, d58, d68, d78], dim=1))


        out = self.finaldeconv1(d8)  # [2,64,256,256]
        out = self.finalrelu1(out)
        out = self.finalconv2(out)  # [2,32,256,256]
        out = self.finalrelu2(out)
        final = self.finalconv3(out)  # [2,1,256,256]


        if self.RefNet:
            final=self.RefNet(final)
        if self.use_logits:
            return final
        else:
            if self.n_class == 1:
                return F.sigmoid(final)  # for binary only
            else:
                return final











class Att_Head(nn.Module):
    def __init__(self, in_channels=2):
        """
        :param
        """
        super(Att_Head, self).__init__()

        self.conv_block=nn.Sequential(
            nn.Conv2d(in_channels,in_channels*4,kernel_size=3,padding=1),
            nn.BatchNorm2d(in_channels*4),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels*4, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=1)
        )
        #self.act=nn.Sigmoid()
        self.act = nn.Softmax()

    def forward(self, logits):
        """
        :param
        :return: att score map same size with seg_logits
        """
        out=self.conv_block(logits)
        out=self.act(out)

        return  out

class unet_2D_small(nn.Module):

    def __init__(self, feature_scale=4, n_classes=1, is_deconv=True, in_channels=5, is_batchnorm=True,use_res=False,
                 dblock_type='ASPP'):
        super(unet_2D_small, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm

        self.n_class=n_classes


        filters = [64, 128, 256]
        filters = [int(x / 2) for x in filters]#[16,32,64,128,256]
        if use_res:
            conv_block=unetConv2_res

        else:
            conv_block=unetConv2

        # downsampling
        self.conv1 = conv_block(self.in_channels, filters[0], self.is_batchnorm)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = conv_block(filters[0], filters[1], self.is_batchnorm)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.center = conv_block(filters[1], filters[2], self.is_batchnorm)
        #self.dblock_typ=dblock_type
        if dblock_type=='ASPP':
           self.Dblock = ASPP(in_channel=filters[2])
           logger.info('With ASPP module.')
        elif dblock_type=='AS':
           self.Dblock=Dblock(filters[2])
           logger.info('With AS module.')
        else:
            logger.info('Without dblock module.')
            self.Dblock=None

        # upsampling
        self.up_concat2 = unetUp(filters[2], filters[1], self.is_deconv,use_res=use_res)
        self.up_concat1 = unetUp(filters[1], filters[0], self.is_deconv,use_res=use_res)


        # final conv (without any concat)
        self.final = nn.Conv2d(filters[0], n_classes, 1)




    def forward(self, inputs):
        conv1 = self.conv1(inputs)#[1,6,256,256]==>[1,16,256,256]
        maxpool1 = self.maxpool1(conv1)#[1,16,256,256]==>[1,16,128,128]

        conv2 = self.conv2(maxpool1)#[1,16,128,128]==>[1,32,128,128]
        maxpool2 = self.maxpool2(conv2)#[1,32,128,128]==>[1,32,64,64]

        center = self.center(maxpool2)#[1,128,16,16]==>[1,256,16,16]
        if self.Dblock:
            center=self.Dblock(center)
        up2 = self.up_concat2(conv2, center)  # [1,256,16,16]+[1,128,32,32]+conv==>[1,128,32,32]
        up1 = self.up_concat1(conv1, up2)  #


        final = self.final(up1)#[1,16,256,256]==>[1,1,256,256]

        if self.n_class==1:
            return F.sigmoid(final)# for binary only
        else:
            return final

class MS_Refine(nn.Module):
    def __init__(self,in_channels=5):
        """
        :param: using multi-scale refinement by cascading seg map from coarse scale to fine scale
        """
        super(MS_Refine, self).__init__()

        self.ref_block=unet_2D_small()



    def forward(self, inputs,outs=[]):
        """
        :param: inputs+ upsampled seg maps at 4 scales
        :return: output map at 4 scales
        """
        scale_num=len(outs)
        outs_ref=[]
        for i in range(scale_num-1):
            cur_inputs=F.interpolate(inputs,scale_factor=1/(2**(3-i)), mode='bilinear', align_corners=True)
            if i==0:
                unet_inputs=torch.cat([cur_inputs,outs[i],outs[i]],dim=1)
                cur_outs=self.ref_block(unet_inputs)
                outs_ref.append(cur_outs)
            else:
                cur_outs=F.interpolate(cur_outs,scale_factor=2, mode='bilinear', align_corners=True)
                unet_inputs = torch.cat([cur_inputs, outs[i], cur_outs], dim=1)
                cur_outs = self.ref_block(unet_inputs)
                outs_ref.append(cur_outs)
        cur_outs = F.interpolate(cur_outs, scale_factor=2, mode='bilinear', align_corners=True)
        unet_inputs = torch.cat([inputs, outs[-1], cur_outs], dim=1)
        cur_outs = self.ref_block(unet_inputs)
        outs_ref.append(cur_outs)


        return  outs_ref

class unet_2D_PreTrain_MSRef(nn.Module):

    def __init__(self, feature_scale=4, n_classes=1, is_deconv=True, in_channels=3, is_batchnorm=True,use_att=False,use_res=False,
        dblock_type='AS',use_rfnet=False,use_logits=False):
        super(unet_2D_PreTrain_MSRef, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale
        self.n_class=n_classes
        self.use_att=use_att
        self.RefNet=MS_Refine(in_channels=5)
        # # filters = [64, 128, 256, 512, 1024]
        # # filters = [int(x / self.feature_scale) for x in filters]#[16,32,64,128,256]
        # if use_res:
        #     conv_block=unetConv2_res
        #     logger.info("using resblock conv...")
        # else:
        #     conv_block=unetConv2
        #     logger.info("using basic conv...")

        # downsampling
        from torchvision import models
        filters = [64, 128, 256, 512]
        resnet = models.resnet34(pretrained=True)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.dblock = Dblock(512)

        if self.use_att:
            self.GAU4 = B.GAU(filters[3], filters[2], use_res=False)
            self.GAU3 = B.GAU(filters[2], filters[1], use_res=False)
            self.GAU2 = B.GAU(filters[1], filters[0], use_res=False)
            self.GAU1 = B.GAU(filters[0], filters[0], use_res=False)
            logger.info("using GAU attetnion...")
        else:
            self.decoder4 = B.DecoderBlock(filters[3], filters[2])
            self.decoder3 = B.DecoderBlock(filters[2], filters[1])
            self.decoder2 = B.DecoderBlock(filters[1], filters[0])
            self.decoder1 = B.DecoderBlock(filters[0], filters[0])
            logger.info("without GAU attetnion...")
        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)  # size*2
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity

        self.outconv1 = nn.Conv2d(32, n_classes, 3, padding=1)
        self.outconv2 = nn.Conv2d(64, n_classes, 3, padding=1)
        self.outconv3 = nn.Conv2d(filters[0], n_classes, 3, padding=1)
        self.outconv4 = nn.Conv2d(filters[1], n_classes, 3, padding=1)



    def forward(self, inputs):


        x = self.firstconv(inputs)  # [2,64,128,128]
        x = self.firstbn(x)
        x_conv = self.firstrelu(x)
        x1 = self.firstmaxpool(x_conv)  # [2,64,64,64]
        e1 = self.encoder1(x1)  ##[2,64,64,64]
        e2 = self.encoder2(e1)  # [2,128,32,32]
        e3 = self.encoder3(e2)  # [2,256,16,16]
        e4 = self.encoder4(e3)  # [2,512,8,8]

        # Center
        e4 = self.dblock(e4)  # [2,512,8,8]

        # Decoder
        if self.use_att:
            d4=self.GAU4(e4,e3)#[4,256,32,32]
            d3=self.GAU3(d4,e2)#[4,128,64,64]
            d2=self.GAU2(d3,e1)#[4,64,128,128]
            d1=self.GAU1(d2,x_conv)#[4,64,256,256]
        else:
            d4 = self.decoder4(e4) + e3  # [2,256,16,16]
            d3 = self.decoder3(d4) + e2  # [2,128,32,32]
            d2 = self.decoder2(d3) + e1  # [2,64,64,64]
            d1 = self.decoder1(d2)  # [2,64,128,128]
        out = self.finaldeconv1(d1)  # [2,64,256,256]
        out = self.finalrelu1(out)
        out = self.finalconv2(out)  # [2,32,256,256]
        out = self.finalrelu2(out)
        # final4 = self.finalconv3(out)  # [2,1,256,256]
        # final3=self
        out1=self.outconv1(out)
        out2=self.outconv2(d1)
        out3=self.outconv3(d2)
        out4=self.outconv4(d3)
        outs=[out4,out3,out2,out1]
        outs_ref = self.RefNet(inputs,outs)
        preds0=F.interpolate(outs_ref[0],scale_factor=8, mode='bilinear', align_corners=True)
        preds1 = F.interpolate(outs_ref[1], scale_factor=4, mode='bilinear', align_corners=True)
        preds2 = F.interpolate(outs_ref[2], scale_factor=2, mode='bilinear', align_corners=True)
        preds3=outs_ref[3]
        return preds0,preds1,preds2,preds3

        # if self.RefNet:
        #     final=self.RefNet(final)
        # if self.use_logits:
        #     return final
        # else:
        #     if self.n_class == 1:
        #         return F.sigmoid(final)  # for binary only
        #     else:
        #         return final





class unet_2D_PreTrainScale(nn.Module):

    def __init__(self, feature_scale=4, n_classes=1, is_deconv=True, in_channels=3, is_batchnorm=True,use_att=True,use_res=True,
        dblock_type='AS',use_rfnet=False):
        super(unet_2D_PreTrainScale, self).__init__()

        self.segModel=unet_2D_PreTrain(n_classes=n_classes,use_att=use_att,use_rfnet=use_rfnet,use_logits=True)

        self.scale_head=Att_Head(in_channels=n_classes*2)

    def forward(self, inputs):
        inputs0 = F.interpolate(inputs, scale_factor=0.5, mode='bilinear', align_corners=True)
        out_logits0 = self.segModel(inputs0)
        out_logits1 = self.segModel(inputs)

        out_logits00=F.interpolate(out_logits0, scale_factor=2, mode='bilinear', align_corners=True)
        out_logits01=torch.cat([out_logits00,out_logits1],dim=1)

        att_map01 = self.scale_head(out_logits01)

        return out_logits00,out_logits1,att_map01

        # inputs0 = F.interpolate(inputs, scale_factor=0.5, mode='bilinear', align_corners=True)
        # out_logits0 = self.segModel(inputs0)
        # out_logits1 = self.segModel(inputs)
        # att_map0 = self.att_head(out_logits0)
        # out_seg0 = F.interpolate(out_logits0 * att_map0, scale_factor=2, mode='bilinear', align_corners=True)
        # att_map0 = F.interpolate(att_map0, scale_factor=2, mode='bilinear', align_corners=True)
        # out_seg1 = out_logits1 * (1 - att_map0)
        # out= F.sigmoid(out_seg0 + out_seg1)
        # return out




class unet_2D(nn.Module):

    def __init__(self, feature_scale=4, n_classes=1, is_deconv=True, in_channels=3, is_batchnorm=True,use_att=False,use_res=True,
        dblock_type='AS',use_rfnet=False):
        super(unet_2D, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale
        self.n_class=n_classes
        self.use_att=False
        # self.use_PSP=False#

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]#[16,32,64,128,256]
        if use_res:
            conv_block=unetConv2_res

        else:
            conv_block=unetConv2

        # downsampling
        self.conv1 = conv_block(self.in_channels, filters[0], self.is_batchnorm)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = conv_block(filters[0], filters[1], self.is_batchnorm)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = conv_block(filters[1], filters[2], self.is_batchnorm)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4 = conv_block(filters[2], filters[3], self.is_batchnorm)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)

        self.center = conv_block(filters[3], filters[4], self.is_batchnorm)
        #self.dblock_typ=dblock_type
        if dblock_type=='ASPP':
           self.Dblock = ASPP(in_channel=filters[4])
           logger.info('With ASPP module.')
        elif dblock_type=='AS':
           self.Dblock=Dblock(filters[4])
           logger.info('With AS module.')
        else:
            logger.info('Without dblock module.')
            self.Dblock=None
        if use_rfnet:
            self.RefNet=Dblock(1)
            logger.info('With refine module.')
        else:
            logger.info('Without Refine module.')
            self.RefNet=None
        # upsampling
        self.up_concat4 = unetUp(filters[4], filters[3], self.is_deconv,use_res=use_res)
        self.up_concat3 = unetUp(filters[3], filters[2], self.is_deconv,use_res=use_res)
        self.up_concat2 = unetUp(filters[2], filters[1], self.is_deconv,use_res=use_res)
        self.up_concat1 = unetUp(filters[1], filters[0], self.is_deconv,use_res=use_res)


        # final conv (without any concat)
        self.final = nn.Conv2d(filters[0], n_classes, 1)

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')


    def forward(self, inputs):
        conv1 = self.conv1(inputs)#[1,6,256,256]==>[1,16,256,256]
        maxpool1 = self.maxpool1(conv1)#[1,16,256,256]==>[1,16,128,128]

        conv2 = self.conv2(maxpool1)#[1,16,128,128]==>[1,32,128,128]
        maxpool2 = self.maxpool2(conv2)#[1,32,128,128]==>[1,32,64,64]

        conv3 = self.conv3(maxpool2)#[1,32,64,64]==>[1,64,64,64]
        maxpool3 = self.maxpool3(conv3)#[1,64,64,64]==>[1,64,32,32]

        conv4 = self.conv4(maxpool3)#[1,64,32,32]==>[1,128,32,32]
        maxpool4 = self.maxpool4(conv4)#[1,128,32,32]==>[1,128,16,16]

        center = self.center(maxpool4)#[1,128,16,16]==>[1,256,16,16]
        if self.Dblock:
            center=self.Dblock(center)
        if self.use_att:

            up4 = self.up_concat4_att(conv4, center)  # [1,256,16,16]+[1,128,32,32]+conv==>[1,128,32,32]
            up3 = self.up_concat3_att(conv3, up4)  # [1,64,64,64]+[1,128,32,32]+conv==>[1,64,64,64]
            up2 = self.up_concat2_att(conv2, up3)  # [1,32,128,128]+[1,64,64,64]+conv==>[1,32,128,128]
            up1 = self.up_concat1_att(conv1, up2)
            #pass
        else:

           up4 = self.up_concat4(conv4, center)#[1,256,16,16]+[1,128,32,32]+conv==>[1,128,32,32]
           up3 = self.up_concat3(conv3, up4)#[1,64,64,64]+[1,128,32,32]+conv==>[1,64,64,64]
           up2 = self.up_concat2(conv2, up3)#[1,32,128,128]+[1,64,64,64]+conv==>[1,32,128,128]
           up1 = self.up_concat1(conv1, up2)#[1,16,256,256]+[1,32,128,128]+conv==>[1,16,256,256]

        final = self.final(up1)#[1,16,256,256]==>[1,1,256,256]
        if self.RefNet:
            final=self.RefNet(final)
        if self.n_class==1:
            return F.sigmoid(final)# for binary only
        else:
            return final

######################################################################
#########################Style Generator  ############################
######################################################################
class AdaIN_Net(nn.Module):
    def __init__(self, in_channels=3, is_batchnorm=True,use_res=True):

        super(AdaIN_Net, self).__init__()
        filters=[32,64,128]
        encoder=[
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels,filters[0],kernel_size=3),
            nn.InstanceNorm2d(filters[0]),
            nn.ReLU(inplace=True)
        ]
        encoder+= [
            nn.ReflectionPad2d(1),
            nn.Conv2d(filters[0], filters[1], kernel_size=4,stride=2),
            nn.InstanceNorm2d(filters[1]),
            nn.ReLU(inplace=True)
        ]
        encoder+= [
            nn.ReflectionPad2d(1),
            nn.Conv2d(filters[1], filters[2], kernel_size=4, stride=2),
            nn.InstanceNorm2d(filters[2]),
            nn.ReLU(inplace=True)
        ]
        encoder+=[ResidualBlock(filters[2]),
                  ResidualBlock(filters[2]),
                  ResidualBlock(filters[2])]

        decoder=[
            # nn.ReflectionPad2d(1),
            # nn.Conv2d(filters[2], filters[1], kernel_size=3),
            # #nn.InstanceNorm2d(filters[0]),
            # nn.ReLU(inplace=True),
            # nn.Upsample(scale_factor=2, mode='bilinear')

            nn.ConvTranspose2d(filters[2], filters[1], 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True)
        ]
        decoder+= [
            # nn.ReflectionPad2d(1),
            # nn.Conv2d(filters[1], filters[0], kernel_size=3),
            # nn.ReLU(inplace=True),
            # nn.Upsample(scale_factor=2, mode='bilinear')

            nn.ConvTranspose2d(filters[1], filters[0], 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True)
        ]
        decoder+= [
            nn.ReflectionPad2d(1),
            nn.Conv2d(filters[0], in_channels, kernel_size=3)
            #,nn.Tanh()
        ]
        self.encoder=nn.Sequential(*encoder)
        self.decoder=nn.Sequential(*decoder)
    # def forward(self, imgs_content,imgs_style,alpha=1.0):
    #     assert 0 <= alpha <= 1
    #     content_feats=self.encoder(imgs_content)
    #     style_feats=self.encoder(imgs_style)
    #     from .utils import adaptive_instance_normalization as adain
    #     t = adain(content_feats, style_feats)  # [2,512,16,16]
    #     t = alpha * t + (1 - alpha) * content_feats
    #
    #     g_t = self.decoder(t)  # [2,3,128,128]
    #
    #     return g_t
    def forward(self, imgs_content,style_mean,style_std,alpha=1.0):
        assert 0 <= alpha <= 1
        content_feats=self.encoder(imgs_content)
        #style_feats=self.encoder(imgs_style)
        from .utils import adaptive_instance_normalization2 as adain
        t = adain(content_feats, style_mean,style_std)  # [2,512,16,16]
        t = alpha * t + (1 - alpha) * content_feats

        g_t = self.decoder(t)  # [2,3,128,128]

        return g_t

################################################################################################
#=====================================for ResNet_DeepLab=======================================#
################################################################################################

# import torch.nn as nn
# import numpy as np
# import torch

























######################################################################
#########################Discriminator################################
######################################################################
#ref:Classes Matter: A Fine-grained Adversarial Approach to Cross-domain Semantic Segmentation
class Discriminator_Pix(nn.Module):
    def __init__(self, input_nc, ndf=512, num_classes=2):
        super(Discriminator_Pix, self).__init__()

        self.D = nn.Sequential(
            nn.Conv2d(input_nc, ndf, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(ndf, ndf//2, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
		)
        self.cls1 = nn.Conv2d(ndf//2, num_classes, kernel_size=3, stride=1, padding=1)
        self.cls2 = nn.Conv2d(ndf//2, num_classes, kernel_size=3, stride=1, padding=1)

    def forward(self, x, size=None):
        out = self.D(x)
        src_out = self.cls1(out)
        tgt_out = self.cls2(out)
        out = torch.cat((src_out, tgt_out), dim=1)
        if size is not None:
            out = F.interpolate(out, size=size, mode='bilinear', align_corners=True)
        return out





class Discriminator_FC_512(nn.Module):

    def __init__(self, num_classes, ndf=64,act=False):
        super(Discriminator_FC_512, self).__init__()

        self.conv1 = nn.Conv2d(num_classes, ndf, kernel_size=4, stride=2, padding=1)  # 160 x 160
        self.conv2 = nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1)  # 80 x 80
        self.conv3 = nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1)  # 40 x 40
        self.conv4 = nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1)  # 20 x 20
        # if dataset == 'pascal_voc' or dataset == 'pascal_context':
        #     self.avgpool = nn.AvgPool2d((20, 20))
        # elif dataset == 'cityscapes':
        #     self.avgpool = nn.AvgPool2d((16, 32))
        self.avgpool = nn.AvgPool2d((32, 32))
        self.fc = nn.Linear(ndf * 8, 1)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.drop = nn.Dropout2d(0.5)
        self.sigmoid = nn.Sigmoid()
        self.act=act
    def forward(self, x):

        x = self.conv1(x)
        x = self.leaky_relu(x)
        x = self.drop(x)

        x = self.conv2(x)
        x = self.leaky_relu(x)
        x = self.drop(x)

        x = self.conv3(x)
        x = self.leaky_relu(x)
        x = self.drop(x)

        x = self.conv4(x)
        x = self.leaky_relu(x)

        maps = self.avgpool(x)
        #conv4_maps = maps
        out = maps.view(maps.size(0), -1)  # [4,512,1,1]==>[4,512]
        out=self.fc(out)
        if self.act:
          out = self.sigmoid(out)  ##[4,512]==>[4,1]
        return out
class Discriminator_FC_256(nn.Module):

    def __init__(self, num_classes, ndf=64,act=True):
        super(Discriminator_FC_256, self).__init__()

        self.conv1 = nn.Conv2d(num_classes, ndf, kernel_size=4, stride=2, padding=1)  # 160 x 160
        self.conv2 = nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1)  # 80 x 80
        self.conv3 = nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1)  # 40 x 40
        self.conv4 = nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1)  # 20 x 20
        # if dataset == 'pascal_voc' or dataset == 'pascal_context':
        #     self.avgpool = nn.AvgPool2d((20, 20))
        # elif dataset == 'cityscapes':
        #     self.avgpool = nn.AvgPool2d((16, 32))
        self.avgpool = nn.AvgPool2d((16, 16))
        self.fc = nn.Linear(ndf * 8, 1)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.drop = nn.Dropout2d(0.5)
        self.sigmoid = nn.Sigmoid()
        self.act=act
    def forward(self, x):

        x = self.conv1(x)
        x = self.leaky_relu(x)
        x = self.drop(x)

        x = self.conv2(x)
        x = self.leaky_relu(x)
        x = self.drop(x)

        x = self.conv3(x)
        x = self.leaky_relu(x)
        x = self.drop(x)

        x = self.conv4(x)
        x = self.leaky_relu(x)

        maps = self.avgpool(x)
        conv4_maps = maps
        out = maps.view(maps.size(0), -1)  # [4,512,1,1]==>[4,512]
        out=self.fc(out)
        if self.act:
          out = self.sigmoid(out)  ##[4,512]==>[4,1]

        return out

class Discriminator_FC_128(nn.Module):

    def __init__(self, num_classes, ndf=64,act=True):
        super(Discriminator_FC_128, self).__init__()

        self.conv1 = nn.Conv2d(num_classes, ndf, kernel_size=4, stride=2, padding=1)  # 160 x 160
        self.conv2 = nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1)  # 80 x 80
        self.conv3 = nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1)  # 40 x 40
        self.conv4 = nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1)  # 20 x 20
        # if dataset == 'pascal_voc' or dataset == 'pascal_context':
        #     self.avgpool = nn.AvgPool2d((20, 20))
        # elif dataset == 'cityscapes':
        #     self.avgpool = nn.AvgPool2d((16, 32))
        self.avgpool = nn.AvgPool2d((8, 8))
        self.fc = nn.Linear(ndf * 8, 1)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.drop = nn.Dropout2d(0.5)
        self.sigmoid = nn.Sigmoid()
        self.act=act
    def forward(self, x):

        x = self.conv1(x)
        x = self.leaky_relu(x)
        x = self.drop(x)

        x = self.conv2(x)
        x = self.leaky_relu(x)
        x = self.drop(x)

        x = self.conv3(x)
        x = self.leaky_relu(x)
        x = self.drop(x)

        x = self.conv4(x)
        x = self.leaky_relu(x)

        maps = self.avgpool(x)
        conv4_maps = maps
        out = maps.view(maps.size(0), -1)  # [4,512,1,1]==>[4,512]
        out=self.fc(out)
        if self.act:
          out = self.sigmoid(out)  ##[4,512]==>[4,1]

        return out

class Discriminator_FC_64(nn.Module):

    def __init__(self, num_classes, ndf=64,act=True):
        super(Discriminator_FC_64, self).__init__()

        self.conv1 = nn.Conv2d(num_classes, ndf, kernel_size=4, stride=2, padding=1)  # 32*32
        self.conv2 = nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1)  # 16*16
        self.conv3 = nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1)  #8*8
        self.conv4 = nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1)  # 4*4

        self.avgpool = nn.AvgPool2d((4, 4))
        self.fc = nn.Linear(ndf * 8, 1)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.drop = nn.Dropout2d(0.5)
        self.sigmoid = nn.Sigmoid()
        self.act=act
    def forward(self, x):

        x = self.conv1(x)
        x = self.leaky_relu(x)
        x = self.drop(x)

        x = self.conv2(x)
        x = self.leaky_relu(x)
        x = self.drop(x)

        x = self.conv3(x)
        x = self.leaky_relu(x)
        x = self.drop(x)

        x = self.conv4(x)
        x = self.leaky_relu(x)

        maps = self.avgpool(x)
        conv4_maps = maps
        out = maps.view(maps.size(0), -1)  # [4,512,1,1]==>[4,512]
        out=self.fc(out)
        if self.act:
          out = self.sigmoid(out)  ##[4,512]==>[4,1]

        return out

class Discriminator_FC_16(nn.Module):

    def __init__(self, num_classes, ndf=256,act=True):
        super(Discriminator_FC_16, self).__init__()
        nf_filters=[ndf,int(ndf/2),int(ndf/4),int(ndf/8)]
        self.conv1 = nn.Conv2d(num_classes, ndf, kernel_size=3, stride=1, padding=1)  #8*4
        self.conv2 = nn.Conv2d(ndf, nf_filters[1], kernel_size=3, stride=1, padding=1)  # 4*4
        self.conv3 = nn.Conv2d(nf_filters[1], nf_filters[2], kernel_size=4, stride=2, padding=1)  #8*8
        self.conv4 = nn.Conv2d(nf_filters[2], nf_filters[3], kernel_size=4, stride=2, padding=1)  # 4*4

        self.avgpool = nn.AvgPool2d((4, 4))
        self.fc = nn.Linear(nf_filters[3], 1)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.drop = nn.Dropout2d(0.5)
        self.sigmoid = nn.Sigmoid()
        self.act=act
    def forward(self, x):

        x = self.conv1(x)
        x = self.leaky_relu(x)
        x = self.drop(x)

        x = self.conv2(x)
        x = self.leaky_relu(x)
        x = self.drop(x)

        x = self.conv3(x)
        x = self.leaky_relu(x)
        x = self.drop(x)

        x = self.conv4(x)
        x = self.leaky_relu(x)

        maps = self.avgpool(x)
        conv4_maps = maps
        out = maps.view(maps.size(0), -1)  # [4,512,1,1]==>[4,512]
        out=self.fc(out)
        if self.act:
          out = self.sigmoid(out)  ##[4,512]==>[4,1]

        return out


class Discriminator_FC_4(nn.Module):

    def __init__(self, num_classes, ndf=256, act=False):
        super(Discriminator_FC_4, self).__init__()
        nf_filters = [ndf, int(ndf / 2), int(ndf / 4), int(ndf / 8)]
        self.conv1 = nn.Conv2d(num_classes, ndf, kernel_size=3, stride=1, padding=1)  #
        self.conv2 = nn.Conv2d(ndf, nf_filters[1], kernel_size=3, stride=1, padding=1)  # 4
        self.conv3 = nn.Conv2d(nf_filters[1], nf_filters[2], kernel_size=4, stride=2, padding=1)  # 2*2
        #self.conv4 = nn.Conv2d(nf_filters[2], nf_filters[3], kernel_size=4, stride=2, padding=1)  # 4*4

        self.avgpool = nn.AvgPool2d((2, 2))
        self.fc = nn.Linear(nf_filters[2], 1)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.drop = nn.Dropout2d(0.5)
        self.sigmoid = nn.Sigmoid()
        self.act = act

    def forward(self, x):
        x = self.conv1(x)
        x = self.leaky_relu(x)
        x = self.drop(x)

        x = self.conv2(x)
        x = self.leaky_relu(x)
        x = self.drop(x)

        # x = self.conv3(x)
        # x = self.leaky_relu(x)
        # x = self.drop(x)

        x = self.conv3(x)
        x = self.leaky_relu(x)

        maps = self.avgpool(x)
        conv4_maps = maps
        out = maps.view(maps.size(0), -1)  # [4,512,1,1]==>[4,512]
        out = self.fc(out)
        if self.act:
            out = self.sigmoid(out)  ##[4,512]==>[4,1]

        return out


class Discriminator_FC_8(nn.Module):

    def __init__(self, num_classes, ndf=256,act=False):
        super(Discriminator_FC_8, self).__init__()
        nf_filters=[ndf,int(ndf/2),int(ndf/4),int(ndf/8)]
        self.conv1 = nn.Conv2d(num_classes, ndf, kernel_size=3, stride=1, padding=1)  #8*8
        self.conv2 = nn.Conv2d(ndf, nf_filters[1], kernel_size=3, stride=1, padding=1)  # 8*8
        self.conv3 = nn.Conv2d(nf_filters[1], nf_filters[2], kernel_size=4, stride=2, padding=1)  #4*4
        self.conv4 = nn.Conv2d(nf_filters[2], nf_filters[3], kernel_size=4, stride=2, padding=1)  # 2*2

        self.avgpool = nn.AvgPool2d((2, 2))
        self.fc = nn.Linear(nf_filters[3], 1)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.drop = nn.Dropout2d(0.5)
        self.sigmoid = nn.Sigmoid()
        self.act=act
    def forward(self, x):

        x = self.conv1(x)
        x = self.leaky_relu(x)
        #x = self.drop(x)

        x = self.conv2(x)
        x = self.leaky_relu(x)
        #x = self.drop(x)

        x = self.conv3(x)
        x = self.leaky_relu(x)
        #x = self.drop(x)

        x = self.conv4(x)
        x = self.leaky_relu(x)

        maps = self.avgpool(x)
        #conv4_maps = maps
        out = maps.view(maps.size(0), -1)  # [4,512,1,1]==>[4,512]
        out=self.fc(out)
        if self.act:
          out = self.sigmoid(out)  ##[4,512]==>[4,1]

        return out

class Discriminator_ASPP(nn.Module):
    def __init__(self, in_channel=512):
        super(Discriminator_ASPP, self).__init__()
        # global average pooling : init nn.AdaptiveAvgPool2d ;also forward torch.mean(,,keep_dim=True)
        depth = int(in_channel)
        #self.mean = nn.AdaptiveAvgPool2d((1, 1))
        #self.conv = nn.Conv2d(in_channel, depth, 1, 1)
        # k=1 s=1 no pad
        self.atrous_block1 = nn.Conv2d(in_channel, depth, 1, 1)# default dilation=1
        self.atrous_block2 = nn.Conv2d(in_channel, depth, 3, 1, padding=2, dilation=2)
        self.atrous_block4 = nn.Conv2d(in_channel, depth, 3, 1, padding=3, dilation=3)
        self.atrous_block8 = nn.Conv2d(in_channel, depth, 3, 1, padding=4, dilation=4)
        self.Norm=nn.BatchNorm2d(depth,momentum=0.0003)

        self.conv_1x1_output = nn.Conv2d(depth * 4, depth, 1, 1)
        self.ReLU = nn.ReLU(inplace=True)
        self.att = False


    def forward(self, x):
        '''
        using sigle conv , thus the ROF is 1, 7, 15,31 for d=1,3,7,15
        :param x:
        :return:
        '''
        #size = x.shape[2:]

        # image_features = self.mean(x)
        # image_features = self.conv(image_features)
        # image_features = F.upsample(image_features, size=size, mode='bilinear')

        atrous_block1 = self.ReLU(self.Norm(self.atrous_block1(x)))

        atrous_block2 = self.ReLU(self.Norm(self.atrous_block2(x)))

        atrous_block4 = self.ReLU(self.Norm(self.atrous_block4(x)))

        atrous_block8 = self.ReLU(self.Norm(self.atrous_block8(x)))


        net = self.conv_1x1_output(torch.cat([atrous_block1, atrous_block2,
                                              atrous_block4, atrous_block8], dim=1))

        return torch.sigmoid(net)






class Discriminator_FC_Pix(nn.Module):
    def __init__(self, input_nc):
        super(Discriminator_FC_Pix, self).__init__()

        # A bunch of convolutions one after another
        model = [   nn.Conv2d(input_nc, 64, 4, stride=2, padding=1),
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(64, 128, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(128),
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(128, 256, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(256),
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(256, 512, 4, padding=1),
                    nn.InstanceNorm2d(512),
                    nn.LeakyReLU(0.2, inplace=True) ]

        # FCN classification layer
        model += [nn.Conv2d(512, 1, 4, padding=1)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        x =  self.model(x)
        # Average pooling and flatten
        return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)

#must consider th output size of the feature map before feding to discriminator
class Discriminator_FC_Pix4(nn.Module):
    def __init__(self, input_nc):
        super(Discriminator_FC_Pix4, self).__init__()

        # A bunch of convolutions one after another
        model = [   nn.Conv2d(input_nc, 64, 4, stride=2, padding=1),
                    nn.LeakyReLU(0.2, inplace=True) ]
        # model += [nn.Conv2d(64, 32, 3, stride=1, padding=1),
        #          nn.LeakyReLU(0.2, inplace=True)]
        # model += [nn.Conv2d(32, 32, 3, stride=1, padding=1),
        #           nn.LeakyReLU(0.2, inplace=True)]
        # model += [nn.Conv2d(32, 32, 3, stride=1, padding=1),
        #           nn.LeakyReLU(0.2, inplace=True)]

        # model += [  nn.Conv2d(64, 128, 4, stride=2, padding=1),
        #
        #             nn.LeakyReLU(0.2, inplace=True) ]

        # model += [  nn.Conv2d(128, 256, 4, stride=2, padding=1),
        #             nn.InstanceNorm2d(256),
        #             nn.LeakyReLU(0.2, inplace=True) ]
        #
        # model += [  nn.Conv2d(256, 512, 4, padding=1),
        #             nn.InstanceNorm2d(512),
        #             nn.LeakyReLU(0.2, inplace=True) ]

        # FCN classification layer
        model += [nn.Conv2d(64, 1, 3, padding=1)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        x =  self.model(x)
        # Average pooling and flatten
        return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)


class Discriminator_FC_Pix16(nn.Module):
    def __init__(self, input_nc):
        super(Discriminator_FC_Pix16, self).__init__()

        # A bunch of convolutions one after another
        model = [   nn.Conv2d(input_nc, 64, 4, stride=2, padding=1),
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(64, 128, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(128),
                    nn.LeakyReLU(0.2, inplace=True) ]

        # model += [  nn.Conv2d(128, 256, 4, stride=2, padding=1),
        #             nn.InstanceNorm2d(256),
        #             nn.LeakyReLU(0.2, inplace=True) ]
        #
        # model += [  nn.Conv2d(256, 512, 4, padding=1),
        #             nn.InstanceNorm2d(512),
        #             nn.LeakyReLU(0.2, inplace=True) ]

        # FCN classification layer

        model += [nn.Conv2d(128, 1, 3, padding=1)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        x =  self.model(x)
        # Average pooling and flatten
        return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)






'''
ref:TopoAL: An Adversarial Learning Approach for
Topology-Aware Road Segmentation
using a multi-scale outputs discriminator to preserve the good predictions and suppress the bad
'''
class Discriminator_FCN_Pix128_MS(nn.Module):
    def __init__(self, in_nc=4):
        super(Discriminator_FCN_Pix128_MS, self).__init__()

        # A bunch of convolutions one after another
        self.first_conv = nn.Sequential(nn.Conv2d(in_nc, 64, 4, stride=2, padding=1),
                    nn.LeakyReLU(0.2, inplace=True))   #64*64
        self.conv1=nn.Sequential( nn.Conv2d(64, 128, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(128),
                    nn.LeakyReLU(0.2, inplace=True))#32*32
        self.conv2 = nn.Sequential(nn.Conv2d(128, 256, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(256),
                    nn.LeakyReLU(0.2, inplace=True))  # 16*16

        self.conv3 = nn.Sequential(nn.Conv2d(256, 512, 4, stride=2, padding=1),
                                   nn.InstanceNorm2d(512),
                                   nn.LeakyReLU(0.2, inplace=True))  # 8*8
        self.conv46 = nn.Sequential(nn.Conv2d(512, 512, 4, stride=2, padding=1),
                                   nn.InstanceNorm2d(512),
                                   nn.LeakyReLU(0.2, inplace=True))  # 8*8
        # self.conv5 = nn.Sequential(nn.Conv2d(512, 512, 4, stride=2, padding=1),
        #                            nn.InstanceNorm2d(512),
        #                            nn.LeakyReLU(0.2, inplace=True))  # 4*4
        # self.conv6 = nn.Sequential(nn.Conv2d(512, 512, 4, stride=2, padding=1),
        #                            nn.InstanceNorm2d(512),
        #                            nn.LeakyReLU(0.2, inplace=True))  # 2*2
        # self.conv7 = nn.Sequential(nn.Conv2d(512, 512, 4, stride=2, padding=1),
        #                            nn.InstanceNorm2d(512),
        #                            nn.LeakyReLU(0.2, inplace=True))  # 1*1


        self.classifier=nn.Conv2d(512,1,3,padding=1)

    def forward(self, x):
        x=self.first_conv(x)#[16,64,256,256]
        x1=self.conv1(x)#
        x2=self.conv2(x1)
        x3=self.conv3(x2)
        out4 = self.classifier(x3)

        x4=self.conv46(x3)
        out3=self.classifier(x4)

        x5 = self.conv46(x4)
        out2 = self.classifier(x5)

        x6 = self.conv46(x5)
        out1 = self.classifier(x6)



        return out4,out3,out2,out1
        #return F.sigmoid(out4),F.sigmoid(out3),F.sigmoid(out2),F.sigmoid(out1)

class Discriminator_FCN_Pix256_MS(nn.Module):
    def __init__(self, in_nc=4):
        super(Discriminator_FCN_Pix256_MS, self).__init__()

        # A bunch of convolutions one after another
        self.first_conv = nn.Sequential(nn.Conv2d(in_nc, 64, 4, stride=2, padding=1),
                    nn.LeakyReLU(0.2, inplace=True))   #128*128
        self.conv1=nn.Sequential( nn.Conv2d(64, 128, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(128),
                    nn.LeakyReLU(0.2, inplace=True))#64*64
        self.conv2 = nn.Sequential(nn.Conv2d(128, 256, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(256),
                    nn.LeakyReLU(0.2, inplace=True))  # 32*32

        self.conv3 = nn.Sequential(nn.Conv2d(256, 512, 4, stride=2, padding=1),
                                   nn.InstanceNorm2d(512),
                                   nn.LeakyReLU(0.2, inplace=True))  # 16*16
        self.conv47 = nn.Sequential(nn.Conv2d(512, 512, 4, stride=2, padding=1),
                                   nn.InstanceNorm2d(512),
                                   nn.LeakyReLU(0.2, inplace=True))  # 8*8

        # self.conv5 = nn.Sequential(nn.Conv2d(512, 512, 4, stride=2, padding=1),
        #                            nn.InstanceNorm2d(512),
        #                            nn.LeakyReLU(0.2, inplace=True))  # 4*4
        # self.conv6 = nn.Sequential(nn.Conv2d(512, 512, 4, stride=2, padding=1),
        #                            nn.InstanceNorm2d(512),
        #                            nn.LeakyReLU(0.2, inplace=True))  # 2*2
        # self.conv7 = nn.Sequential(nn.Conv2d(512, 512, 4, stride=2, padding=1),
        #                            nn.InstanceNorm2d(512),
        #                            nn.LeakyReLU(0.2, inplace=True))  # 1*1


        self.classifier=nn.Conv2d(512,1,3,padding=1)

    def forward(self, x):
        x=self.first_conv(x)#[16,64,256,256]
        x1=self.conv1(x)#
        x2=self.conv2(x1)
        x3=self.conv3(x2)


        x4=self.conv47(x3)
        out4=self.classifier(x4)

        x5 = self.conv47(x4)
        out3 = self.classifier(x5)

        x6 = self.conv47(x5)
        out2 = self.classifier(x6)

        x7 = self.conv47(x6)
        out1 = self.classifier(x7)


        return out4,out3,out2,out1

class Discriminator_FCN_129(nn.Module):
    """
        Disriminator Network for the Adversarial Training.
        for input with not 2^n
    """
    def __init__(self,in_channels=2,negative_slope = 0.2):# in_channels=num_class
        super(Discriminator_FCN_129, self).__init__()
        self._in_channels = in_channels
        self._negative_slope = negative_slope
        self.feature_scale = 2
        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        self.conv1 = nn.Conv2d(in_channels=self._in_channels,out_channels=filters[0],kernel_size=4,stride=2,padding=2)
        self.relu1 = nn.LeakyReLU(self._negative_slope,inplace=True)
        self.conv2 = nn.Conv2d(in_channels=filters[0],out_channels=filters[1],kernel_size=4,stride=2,padding=2)
        self.relu2 = nn.LeakyReLU(self._negative_slope,inplace=True)
        self.conv3 = nn.Conv2d(in_channels=filters[1],out_channels=filters[2],kernel_size=4,stride=2,padding=2)
        self.relu3 = nn.LeakyReLU(self._negative_slope,inplace=True)
        self.conv4 = nn.Conv2d(in_channels=filters[2],out_channels=filters[3],kernel_size=4,stride=2,padding=2)
        self.relu4 = nn.LeakyReLU(self._negative_slope,inplace=True)
        self.conv5 = nn.Conv2d(in_channels=filters[3],out_channels=2,kernel_size=4,stride=2,padding=2)
        # initialise weights
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         init_weights(m, init_type='kaiming')
        #     elif isinstance(m, nn.BatchNorm2d):
        #         init_weights(m, init_type='kaiming')


    def forward(self,x):
        x= self.conv1(x) # [16,2,128,128]==>[16,32,65,65]
        x = self.relu1(x)
        x= self.conv2(x) # [16,32,65,65]==>[16,64,33,33]
        x = self.relu2(x)
        x= self.conv3(x) # [16,64,33,33]==>[16,128,17,17]
        x = self.relu3(x)
        x= self.conv4(x) # [16,128,17,17]==>[16,256,9,9]
        x = self.relu4(x)
        x = self.conv5(x) # [16,256,9,9]==>[16,2,5,5]
        # upsample
        x = F.upsample_bilinear(x,scale_factor=2)#[16,2,5,5]==>[16,2,10,10]
        x = x[:,:,:-1,:-1] # #[16,2,10,10]==>[16,2,9,9]

        x = F.upsample_bilinear(x,scale_factor=2)#[16,2,9,9]==>[16,2,18,18]
        x = x[:,:,:-1,:-1] # [16,2,18,18]==>[16,2,17,17]

        x = F.upsample_bilinear(x,scale_factor=2)
        x = x[:,:,:-1,:-1] #[16,2,34,34]==>[16,2,33,33]

        x = F.upsample_bilinear(x,scale_factor=2)
        x = x[:,:,:-1,:-1] #[16,2,66,66]==>[16,2,65,65]

        x = F.upsample_bilinear(x,scale_factor=2)
        x = x[:,:,:-1,:-1] # [16,2,130,130]==>[16,2,129,129]

        return x

class Discriminator_FCN_256(nn.Module):
    """
        Disriminator Network for the Adversarial Training.
        for input with 2^n,the output channel is 1
    """
    def __init__(self, in_channels=2, out_channels=1,negative_slope=0.2):  # in_channels=num_class
        super(Discriminator_FCN_256, self).__init__()
        self._in_channels = in_channels
        self._negative_slope = negative_slope
        self.feature_scale = 2
        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        self.conv1 = nn.Conv2d(in_channels=self._in_channels, out_channels=filters[0], kernel_size=3, stride=2,
                               padding=1)
        self.norm1=nn.InstanceNorm2d(filters[0])
        self.relu1 = nn.LeakyReLU(self._negative_slope, inplace=True)

        self.conv2 = nn.Conv2d(in_channels=filters[0], out_channels=filters[1], kernel_size=3, stride=2, padding=1)
        self.norm2 = nn.InstanceNorm2d(filters[1])
        self.relu2 = nn.LeakyReLU(self._negative_slope, inplace=True)

        self.conv3 = nn.Conv2d(in_channels=filters[1], out_channels=filters[2], kernel_size=3, stride=2, padding=1)
        self.norm3 = nn.InstanceNorm2d(filters[2])
        self.relu3 = nn.LeakyReLU(self._negative_slope, inplace=True)

        self.conv4 = nn.Conv2d(in_channels=filters[2], out_channels=filters[3], kernel_size=3, stride=2, padding=1)
        self.norm4 = nn.InstanceNorm2d(filters[3])
        self.relu4 = nn.LeakyReLU(self._negative_slope, inplace=True)

        self.conv5 = nn.Conv2d(in_channels=filters[3], out_channels=out_channels, kernel_size=3, stride=2, padding=1)



    def forward(self, x):
        x = self.conv1(x)  # [16,2,128,128]==>[16,32,64,64]
        x=self.norm1(x)
        x = self.relu1(x)

        x = self.conv2(x)  # [16,32,65,65]==>[16,64,33,33]
        x = self.norm2(x)
        x = self.relu2(x)

        x = self.conv3(x)  # [16,64,33,33]==>[16,128,17,17]
        x = self.norm3(x)
        x = self.relu3(x)

        x = self.conv4(x)  # [16,128,17,17]==>[16,256,9,9]
        x = self.norm4(x)
        x = self.relu4(x)

        x = self.conv5(x)  #
        # upsample
        # x = F.upsample_bilinear(x, scale_factor=2)  # [1,1,8,8]==>[1,1,16,16]
        #
        # x = F.upsample_bilinear(x, scale_factor=2)  # [1,1,16,16]==>[1,1,32,32]
        #
        # x = F.upsample_bilinear(x, scale_factor=2)  # [1,1,32,32]==>[1,1,64,64]
        #
        # x = F.upsample_bilinear(x, scale_factor=2)  # [1,1,64,64]==>[1,1,128,128]
        #
        # x = F.upsample_bilinear(x, scale_factor=2)  # [1,1,128,128]==>[1,1,256,256]


        return x


class unetUp_nest(nn.Module):
    def __init__(self, in_size, out_size, is_deconv, n_concat=2):
        super(unetUp_nest, self).__init__()
        self.conv = unetConv2_res(in_size + (n_concat - 2) * out_size, out_size, False)
        if is_deconv:
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2, padding=0)
        else:
            self.up = nn.Sequential(
                nn.UpsamplingBilinear2d(scale_factor=2),
                nn.Conv2d(in_size, out_size, 1))

        # initialise the blocks
        # for m in self.children():
        #     if m.__class__.__name__.find('unetConv2') != -1: continue
        #     init_weights(m, init_type='kaiming')

    def forward(self, high_feature, *low_feature):
        outputs0 = self.up(high_feature)
        for feature in low_feature:
            outputs0 = torch.cat([outputs0, feature], 1)
        return self.conv(outputs0)

class UNet_Nested2(nn.Module):

    def __init__(self, in_channels=3, n_classes=1, feature_scale=2, is_deconv=True, is_batchnorm=True, is_ds=False,dblock_type='AS'):
        super(UNet_Nested2, self).__init__()
        self.in_channels = in_channels
        self.feature_scale = feature_scale
        self.is_deconv = is_deconv
        self.is_batchnorm = is_batchnorm
        self.is_ds = is_ds
        self.n_classes=n_classes


        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.conv00 = unetConv2_res(self.in_channels, filters[0], self.is_batchnorm)
        self.conv10 = unetConv2_res(filters[0], filters[1], self.is_batchnorm)
        self.conv20 = unetConv2_res(filters[1], filters[2], self.is_batchnorm)
        self.conv30 = unetConv2_res(filters[2], filters[3], self.is_batchnorm)
        self.conv40 = unetConv2_res(filters[3], filters[4], self.is_batchnorm)

        # upsampling
        self.up_concat01 = unetUp_nest(filters[1], filters[0], self.is_deconv)
        self.up_concat11 = unetUp_nest(filters[2], filters[1], self.is_deconv)
        self.up_concat21 = unetUp_nest(filters[3], filters[2], self.is_deconv)
        self.up_concat31 = unetUp_nest(filters[4], filters[3], self.is_deconv)

        self.up_concat02 = unetUp_nest(filters[1], filters[0], self.is_deconv, 3)
        self.up_concat12 = unetUp_nest(filters[2], filters[1], self.is_deconv, 3)
        self.up_concat22 = unetUp_nest(filters[3], filters[2], self.is_deconv, 3)

        self.up_concat03 = unetUp_nest(filters[1], filters[0], self.is_deconv, 4)
        self.up_concat13 = unetUp_nest(filters[2], filters[1], self.is_deconv, 4)

        self.up_concat04 = unetUp_nest(filters[1], filters[0], self.is_deconv, 5)

        # final conv (without any concat)
        self.final_1 = nn.Conv2d(filters[0], n_classes, 1)
        self.final_2 = nn.Conv2d(filters[0], n_classes, 1)
        self.final_3 = nn.Conv2d(filters[0], n_classes, 1)
        self.final_4 = nn.Conv2d(filters[0], n_classes, 1)

        if dblock_type == 'AS':
            self.dblock = Dblock2(512, conv_num=3)

        elif dblock_type == 'ASS':  # must not if if dblock_type=='ASS', will lead to always LKPP!
            self.dblock = Dblock_Smooth(512, conv_num=3)
        elif dblock_type == 'ASPP':
            self.dblock = ASPP(512)

        else:
            self.dblock = LKPP(in_chan=512, out_chan=128, mode='parallel', with_gp=False)
        # self.final_1_ae = nn.Conv2d(filters[0], ae_out_channels, 1)
        # self.final_2_ae = nn.Conv2d(filters[0], ae_out_channels, 1)
        # self.final_3_ae = nn.Conv2d(filters[0], ae_out_channels, 1)
        # self.final_4_ae = nn.Conv2d(filters[0], ae_out_channels, 1)

        # self.final_conv=nn.Conv2d(4*n_classes,n_classes,1)
        # self.final_conv_ae=nn.Conv2d(4*ae_out_channels,ae_out_channels,1)

    #     # initialise weights
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d):
    #             init_weights(m, init_type='kaiming')
    #         elif isinstance(m, nn.BatchNorm2d):
    #             init_weights(m, init_type='kaiming')
    # def set_supervised(self,value):
    #     self.use_ae=not value

    def forward(self, inputs):
        # column : 0
        X_00 = self.conv00(inputs)  # 16*512*512
        maxpool0 = self.maxpool(X_00)  # 16*256*256
        X_10 = self.conv10(maxpool0)  # 32*256*256
        maxpool1 = self.maxpool(X_10)  # 32*128*128
        X_20 = self.conv20(maxpool1)  # 64*128*128
        maxpool2 = self.maxpool(X_20)  # 64*64*64
        X_30 = self.conv30(maxpool2)  # 128*64*64
        maxpool3 = self.maxpool(X_30)  # 128*32*32
        X_40 = self.conv40(maxpool3)  # 256*32*32

        X_40=self.dblock(X_40)
        # column : 1
        X_01 = self.up_concat01(X_10, X_00)
        X_11 = self.up_concat11(X_20, X_10)
        X_21 = self.up_concat21(X_30, X_20)
        X_31 = self.up_concat31(X_40, X_30)
        # column : 2
        X_02 = self.up_concat02(X_11, X_00, X_01)
        X_12 = self.up_concat12(X_21, X_10, X_11)
        X_22 = self.up_concat22(X_31, X_20, X_21)
        # column : 3
        X_03 = self.up_concat03(X_12, X_00, X_01, X_02)
        X_13 = self.up_concat13(X_22, X_10, X_11, X_12)
        # column : 4
        X_04 = self.up_concat04(X_13, X_00, X_01, X_02, X_03)

        # if self.use_ae:
        #     final_1 = self.final_1_ae(X_01)
        #     final_2 = self.final_2_ae(X_02)
        #     final_3 = self.final_3_ae(X_03)
        #     final_4 = self.final_4_ae(X_04)
        #
        #     final = (final_1 + final_2 + final_3 + final_4) / 4
        #
        #     return final
        # else:
            # final layer
        final_1 = self.final_1(X_01)
        final_2 = self.final_2(X_02)
        final_3 = self.final_3(X_03)
        final_4 = self.final_4(X_04)

        final = (final_1 + final_2 + final_3 + final_4) / 4

        if self.n_classes == 1:
            if self.is_ds:
                return F.sigmoid(final)
            else:
                return F.sigmoid(final_4)
        else:
            if self.is_ds:
                return final
            else:
                return final_4












if __name__ == '__main__':
    print('#### Test Case ###')
    '''
    if __name__ == '__main__':
    print('#### Test Case ###')
    from torch.autograd import Variable
    import sys

    sys.path.append('..')
    x = Variable(torch.rand(2, 3, 320, 320))
    model = UNet_3Plus_DeepSup_CGM()
    y = model(x)
    print(y.size())
    '''
    from torch.autograd import Variable
    import sys
    sys.path.append('..')
    from utils.utils import print_model_para, print_model_parm_nums

    #num_classes=2
    #x = Variable(torch.rand(2, 3, 256, 256))
    input = Variable(torch.rand(2, 3, 128, 128))
    model = UNet_3Plus()
    output=model(input)
    print_model_parm_nums(model)
    print(output[0].size())


