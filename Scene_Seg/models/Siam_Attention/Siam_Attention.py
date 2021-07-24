import torch
import torch.nn.functional as F
from torch import nn
import pdb
from models.networks_other import init_weights
from torchvision import models
import numpy as np
from models.utils import unetConv2


class Siam_Attention(nn.Module):
    def __init__(self,in_channels,num_class):
        super(Siam_Attention,self).__init__()
        # self.in_channels=in_channels
        # self.num_class=num_class
        self.is_batchnorm=True
        #resnet = models.resnet34(pretrained=True)  # using pretrained model, not work for CD task, for input must be 3-channel
        filters=[32,64,128,256,512]
        #===============using pretrained  model==========================
        # self.layer0=nn.Sequential( resnet.conv1,resnet.bn1,
        #                            resnet.relu
        # )
        # self.layer1= resnet.layer1
        # self.layer2 = resnet.layer2
        # self.layer3 = resnet.layer3
        # self.layer4 = resnet.layer4
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # self.atten0=self.channel_attention(filters[1])
        # self.atten1 = self.channel_attention(filters[1])
        # self.atten2 = self.channel_attention(filters[2])
        # self.atten3 = self.channel_attention(filters[3])
        # self.atten4 = self.channel_attention(filters[4])
        #===================using unet-like backbone=======================
        self.layer0 = nn.Sequential(
            unetConv2(in_channels, filters[0], self.is_batchnorm),
            nn.MaxPool2d(kernel_size=2)
        )
        self.layer1 = nn.Sequential(
            unetConv2(filters[0], filters[1], self.is_batchnorm),
            nn.MaxPool2d(kernel_size=2)
        )
        self.layer2 = nn.Sequential(
            unetConv2(filters[1], filters[2], self.is_batchnorm),
            nn.MaxPool2d(kernel_size=2)
        )
        self.layer3 = nn.Sequential(
            unetConv2(filters[2], filters[3], self.is_batchnorm),
            nn.MaxPool2d(kernel_size=2)
        )
        self.layer4 = nn.Sequential(
            unetConv2(filters[3], filters[4], self.is_batchnorm),
            nn.MaxPool2d(kernel_size=2)
        )
        self.atten0=self.channel_attention(filters[0])
        self.atten1 = self.channel_attention(filters[1])
        self.atten2 = self.channel_attention(filters[2])
        self.atten3 = self.channel_attention(filters[3])
        self.atten4 = self.channel_attention(filters[4])


        self.deconv4 = nn.Sequential(
            nn.ConvTranspose2d(filters[4], filters[3], kernel_size=2, stride=2, bias=False),
            # nn.ConvTranspose2d(filters[3], filters[2], kernel_size=3, stride=2,padding=1,output_padding=1, bias=False),
            nn.BatchNorm2d(filters[3])
        )
        self.deconv3=nn.Sequential(
                nn.ConvTranspose2d(filters[3],filters[2],kernel_size=2,stride=2,bias=False),
                #nn.ConvTranspose2d(filters[3], filters[2], kernel_size=3, stride=2,padding=1,output_padding=1, bias=False),
                nn.BatchNorm2d(filters[2])
        )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(filters[2], filters[1], kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(filters[1])
        )
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(filters[1], filters[0], kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(filters[0])
        )
        self.deconv0 = nn.Sequential(
            nn.ConvTranspose2d(filters[0], filters[0], kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(filters[0])
        )


        self.out4_conv=nn.Sequential(
            nn.Conv2d(filters[3],num_class,kernel_size=1,stride=1,bias=True),
            #nn.Conv2d(filters[4], num_class, kernel_size=3, padding=1, bias=True),
            nn.Sigmoid()
        )
        self.out3_conv = nn.Sequential(
            nn.Conv2d(filters[2], num_class, kernel_size=1, stride=1, bias=True),
            # nn.Conv2d(filters[4], num_class, kernel_size=3, padding=1, bias=True),
            nn.Sigmoid()
        )
        self.out2_conv = nn.Sequential(
            nn.Conv2d(filters[1], num_class, kernel_size=1, stride=1, bias=True),
            # nn.Conv2d(filters[4], num_class, kernel_size=3, padding=1, bias=True),
            nn.Sigmoid()
        )
        self.out1_conv = nn.Sequential(
            nn.Conv2d(filters[0], num_class, kernel_size=1, stride=1, bias=True),
            # nn.Conv2d(filters[4], num_class, kernel_size=3, padding=1, bias=True),
            nn.Sigmoid()
        )
        self.finalconv=nn.Sequential(
              nn.Conv2d(filters[0],filters[0],kernel_size=3,padding=1),
              nn.BatchNorm2d(filters[0]),nn.ReLU(),
              nn.Conv2d(filters[0], filters[0], kernel_size=3, padding=1),
              nn.BatchNorm2d(filters[0]), nn.ReLU()
        )
        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')

    def channel_attention(self, num_channel, ablation=False):
        # todo add convolution here
        pool = nn.AdaptiveAvgPool2d(1)
        conv = nn.Conv2d(num_channel, num_channel, kernel_size=1)
        # bn = nn.BatchNorm2d(num_channel)
        activation = nn.Sigmoid() # todo modify the activation function

        return nn.Sequential(*[pool, conv, activation])

    def forward(self, x1,x2):
        #==========encoder block0=======================
        layer1_0 = self.layer0(x1)  # [1,3,128,128]==>[1,32,64,64]
        layer2_0 = self.layer0(x2)
        atten1_0=self.atten0(layer1_0)#[1,64,1,1]
        atten2_0=self.atten0(layer2_0)
        layer_m0=layer1_0.mul(atten1_0)+layer2_0.mul(atten2_0)#[1,64,64,64]

        #=========encoder block1=======================
        layer1_1=self.layer1(layer1_0)# [1,64,64,64]==>[1,64,64,64]
        layer2_1 = self.layer1(layer2_0)
        layer_m1=self.layer1(layer_m0)

        atten1_1 = self.atten1(layer1_1)
        atten2_1 = self.atten1(layer2_1)
        layer_m1 = layer_m1+layer1_1.mul(atten1_1) + layer2_1.mul(atten2_1)#[1,64,64,64]
        # =========encoder block2=======================
        layer1_2 = self.layer2(layer1_1)  # [1,64,64,64]==>[1,128,32,32]
        layer2_2 = self.layer2(layer2_1)
        layer_m2 = self.layer2(layer_m1)

        atten1_2 = self.atten2(layer1_2)
        atten2_2 = self.atten2(layer2_2)
        layer_m2 = layer_m2 + layer1_2.mul(atten1_2) + layer2_2.mul(atten2_2)#[1,128,32,32]
        # =========encoder block3=======================
        layer1_3 = self.layer3(layer1_2)  # # [1,128,32,32]==>[1,256,16,16]
        layer2_3 = self.layer3(layer2_2)
        layer_m3 = self.layer3(layer_m2)

        atten1_3 = self.atten3(layer1_3)
        atten2_3 = self.atten3(layer2_3)
        layer_m3 = layer_m3 + layer1_3.mul(atten1_3) + layer2_3.mul(atten2_3)#[1,256,16,16]
        # =========encoder block4=======================
        layer1_4 = self.layer4(layer1_3)  # [1,256,16,16]==>[1,512,8,8]
        layer2_4 = self.layer4(layer2_3)
        layer_m4 = self.layer4(layer_m3)

        atten1_4 = self.atten4(layer1_4)
        atten2_4 = self.atten4(layer2_4)
        layer_m4 = layer_m4 + layer1_4.mul(atten1_4) + layer2_4.mul(atten2_4)#[1,512,8,8]

        #==========decoder block========================
        up4=self.deconv4(layer_m4)#  [1,512,4,4]==>[1,256,8,8]
        out4=self.out4_conv(up4)
        up4=up4+layer_m3

        up3 = self.deconv3(up4)  # [1,256,8,8]==>[1,128,16,16]
        out3 = self.out3_conv(up3)
        up3 = up3 + layer_m2

        up2 = self.deconv2(up3)  # [1,128,16,16]==>[1,64,32,32]
        out2 = self.out2_conv(up2)
        up2 = up2 + layer_m1#[1,64,64,64]

        up1 = self.deconv1(up2)  # [1,64,32,32]==>[1,32,64,64]
        out1 = self.out1_conv(up1)
        up1 = up1 +layer_m0

        #==============for resnet backbone================
        # up0 = self.deconv0(up1)  # [1,32,128,128]==>[1,32,256,256]
        # out0 = self.out1_conv(up0)
        # up0=self.finalconv(up1)
        # out0=self.out1_conv(up0)

        #=============for unet backbone==================
        up0 = self.deconv0(up1)  #[1,32,64,64]==>[1,32,128,128]
        #up0 = up0 + self.deconv0(up1)
        up0 = self.finalconv(up0)
        out0 = self.out1_conv(up0)

        return out0

if __name__ == '__main__':
    #xs = torch.randn(size=(1, 3, 256, 256))
    #net  = Multi_Attention(in_channels=6,num_class=1)  # init only input model config
    #output = net(xs)  # forward input
    x1=torch.randn(size=(1, 3, 128, 128))
    x2= torch.randn(size=(1, 3, 128, 128))
    net=Siam_Attention(in_channels=3,num_class=1)
    output=net(x1,x2)
    print(output.size())


