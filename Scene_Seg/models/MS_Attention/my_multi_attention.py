import torch
import torch.nn.functional as F
from torch import nn
from models.MS_Attention.attention import PAM_Module,CAM_Module,semanticModule
from models.utils import unetConv2
import pdb
from models.networks_other import init_weights
from torchvision import models
import numpy as np

class Multi_Attention(nn.Module):
    def __init__(self,in_channels,num_class):
        super(Multi_Attention,self).__init__()#super() 函数是用于调用父类(超类)的一个方法,不从父类继承则为空
        resnet = models.resnet34(pretrained=True)#using pretrained model, not work for CD task, for input must be 3-channel
        self.is_batchnorm = True
        self.in_channels=in_channels
        self.num_class=num_class
        self.feature_scale=2
        filters=[64,128,256,512,1024]
        filters=[int(x/self.feature_scale) for x in filters]
        self.training=False
        #=====================downsampling=============================
        # self.firstconv = resnet.conv1
        # self.firstbn = resnet.bn1
        # self.firstrelu = resnet.relu
        # self.firstmaxpool = resnet.maxpool
        self.encoder0=nn.Sequential(resnet.conv1,resnet.bn1,
                                    resnet.relu,resnet.maxpool
        )
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        #==========using unet-like======================================
        self.layer0=nn.Sequential(
                  unetConv2(self.in_channels,filters[0],self.is_batchnorm),
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

        self.down4=nn.Sequential(
            nn.Conv2d(filters[4], filters[0], kernel_size=1), nn.BatchNorm2d(filters[0]), nn.PReLU()
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(filters[3], filters[0], kernel_size=1), nn.BatchNorm2d(filters[0]), nn.PReLU()
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(filters[2], filters[0], kernel_size=1), nn.BatchNorm2d(filters[0]), nn.PReLU()
        )
        self.down1 = nn.Sequential(
            nn.Conv2d(filters[1], filters[0], kernel_size=1), nn.BatchNorm2d(filters[0]), nn.PReLU()
        )
        inter_channels=filters[0]
        out_channels=filters[0]
        self.conv_1=nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(filters[0], out_channels, 1))#每个通道按照概率0.1置为0
        self.conv_2=nn.Conv2d(filters[0],filters[0],1)

        self.pam_attention = nn.Sequential(
            nn.Conv2d(filters[1], filters[0], kernel_size=3, padding=1), nn.BatchNorm2d(filters[0]), nn.PReLU(),
            PAM_Module(filters[0]),
            nn.Conv2d(filters[0], filters[0], kernel_size=3, padding=1), nn.BatchNorm2d(filters[0]), nn.PReLU()
        )
        self.cam_attention = nn.Sequential(
            nn.Conv2d(filters[1], filters[0], kernel_size=3, padding=1), nn.BatchNorm2d(filters[0]), nn.PReLU(),
            CAM_Module(filters[0]),
            nn.Conv2d(filters[0], filters[0], kernel_size=3, padding=1), nn.BatchNorm2d(filters[0]), nn.PReLU()
        )
        self.semanticModule = semanticModule(filters[1])
        self.conv_sem = nn.Conv2d(filters[1], filters[0], kernel_size=3, padding=1)

        self.fuse=nn.Sequential(
            nn.Conv2d(filters[2], filters[0], kernel_size=3, padding=1), nn.BatchNorm2d(filters[0]), nn.PReLU(),
            nn.Conv2d(filters[0], filters[0], kernel_size=3, padding=1), nn.BatchNorm2d(filters[0]), nn.PReLU(),
            nn.Conv2d(filters[0], filters[0], kernel_size=1), nn.BatchNorm2d(filters[0]), nn.PReLU()
        )

        self.attention = nn.Sequential(
            nn.Conv2d(filters[1], filters[0], kernel_size=3, padding=1), nn.BatchNorm2d(filters[0]), nn.PReLU(),
            nn.Conv2d(filters[0], filters[0], kernel_size=3, padding=1), nn.BatchNorm2d(filters[0]), nn.PReLU(),
            nn.Conv2d(filters[0], filters[0], kernel_size=1), nn.Softmax2d()

        )

        self.refine = nn.Sequential(
            nn.Conv2d(filters[1], filters[0], kernel_size=3, padding=1), nn.BatchNorm2d(filters[0]), nn.PReLU(),
            nn.Conv2d(filters[0], filters[0], kernel_size=3, padding=1), nn.BatchNorm2d(filters[0]), nn.PReLU(),
            nn.Conv2d(filters[0], filters[0], kernel_size=1),nn.PReLU()
        )

        #self.predict=F.sigmoid()#没有激活，需要在train中使用softmax/sigmoid
        self.predict=nn.Sequential(
             nn.Conv2d(filters[0], self.num_class, 1),
             nn.Sigmoid()
         )
        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')

    def forward(self, x): #[1,3,256,256]
        layer0=self.layer0(x)#[1,3,256,256]==>[1,32,128,128]
        layer1=self.layer1(layer0)#[1,32,128,128]==>[1,64,64,64]
        layer2 = self.layer2(layer1)#[1,64,64,64]==>[1,128,32,32]
        layer3 = self.layer3(layer2)#[1,128,32,32]==>[1,256,16,16]
        layer4 = self.layer4(layer3)#[1,256,16,16]==>[1,512,8,8]

        # layer0 = self.encoder0(x)  # [1,3,256,256]==>[1,64,64,64]
        # layer1=self.encoder1(layer0)#[1,64,64,64]==>[1,64,64,64]
        # layer2 = self.encoder2(layer1)#[1,64,64,64]==>[1,128,32,32]
        # layer3 = self.encoder3(layer2)#[1,128,32,32]==>[1,256,16,16]
        # layer4 = self.encoder4(layer3)#[1,256,16,16]==>[1,512,8,8]

        down4 = F.upsample(self.down4(layer4), size=layer1.size()[2:],
                           mode='bilinear')  # [1,512,8,8]==>[1,32,8,8]==>[1,32,64,64]
        down3 = F.upsample(self.down3(layer3), size=layer1.size()[2:],
                           mode='bilinear')  # [1,256,16,16]==>[1,32,16,16]==>[1,32,64,64]
        down2 = F.upsample(self.down2(layer2), size=layer1.size()[2:],
                           mode='bilinear')  # [1,128,32,32]==>[1,32,32,32]==>[1,32,64,64]
        down1 = self.down1(layer1)  # [1,64,64,64]==>[1,32,64,64]

        predict4 = self.predict(down4)  # [1,32,64,64]==>[1,1,64,64]
        predict3 = self.predict(down3)  # [1,32,64,64]==>[1,1,64,64]
        predict2 = self.predict(down2)  # [1,32,64,64]==>[1,1,64,64]
        predict1 = self.predict(down1)  # [1,32,64,64]==>[1,1,64,64]

        fuse1 = self.fuse(torch.cat((down4, down3, down2, down1), 1)) #4[1,32,64,64]==>[1,128,64,64]==>[1,32,64,64]
        # compresses the input features F into a compacted representation in the latent space
        semVector_1_1, semanticModule_1_1 = self.semanticModule(
            torch.cat((down4, fuse1), 1))  # 2[1,32,64,64]==>[1,64,64,64]==> [65536],[1,64,64,64]
        # ==========================for encoder-decoder0============================
        attn_pam4 = self.pam_attention(torch.cat((down4, fuse1), 1))  # 2[1,32,64,64]==>[1,64,64,64]==>[1,32,64,64]
        attn_cam4 = self.cam_attention(torch.cat((down4, fuse1), 1))  # 2[1,32,64,64]==>[1,64,64,64]==>[1,32,64,64]
        attention1_4 = self.conv_2((attn_cam4 + attn_pam4) * self.conv_sem(
            semanticModule_1_1))  # [1,32,64,64]+[1,32,64,64]=[1,32,64,64] [1,32,64,64]*[1,32,64,64]=[1,32,64,64]

        semVector_1_2, semanticModule_1_2 = self.semanticModule(
            torch.cat((down3, fuse1), 1))  # 2[1,32,64,64]==>[1,64,64,64]==> [65536],[1,64,64,64]
        attn_pam3 = self.pam_attention(torch.cat((down3, fuse1), 1))  # 2[1,32,64,64]==>[1,64,64,64]==>[1,32,64,64]
        attn_cam3 = self.cam_attention(torch.cat((down3, fuse1), 1))  # 2[1,32,64,64]==>[1,64,64,64]==>[1,32,64,64]
        attention1_3 = self.conv_2((attn_cam3 + attn_pam3) * self.conv_sem(
            semanticModule_1_2))  # [1,32,64,64]+[1,32,64,64]=[1,32,64,64] [1,32,64,64]*[1,32,64,64]=[1,32,64,64]

        semVector_1_3, semanticModule_1_3 = self.semanticModule(torch.cat((down2, fuse1), 1))
        attn_pam2 = self.pam_attention(torch.cat((down2, fuse1), 1))
        attn_cam2 = self.cam_attention(torch.cat((down2, fuse1), 1))
        attention1_2 = self.conv_2((attn_cam2 + attn_pam2) * self.conv_sem(semanticModule_1_3))

        semVector_1_4, semanticModule_1_4 = self.semanticModule(torch.cat((down1, fuse1), 1))
        attn_pam1 = self.pam_attention(torch.cat((down1, fuse1), 1))
        attn_cam1 = self.cam_attention(torch.cat((down1, fuse1), 1))
        attention1_1 = self.conv_2((attn_cam1 + attn_pam1) * self.conv_sem(semanticModule_1_4))

        # ==========================for encoder-decoder1============================
        ##new design with stacked attention

        semVector_2_1, semanticModule_2_1 = self.semanticModule(torch.cat((down4, attention1_4 * fuse1),
                                                                              1))  # [1,32,64,64] cat [1,32,64,64]*[1,32,64,64]=[1,64,64,64]==>[65536] [1,64,64,64]

        refine4_1 = self.pam_attention(torch.cat((down4, attention1_4 * fuse1),
                                                     1))  ##[1,32,64,64] cat [1,32,64,64]*[1,32,64,64]=[1,64,64,64]==>[1,32,64,64]
        refine4_2 = self.cam_attention(torch.cat((down4, attention1_4 * fuse1),
                                                     1))  ##[1,32,64,64] cat [1,32,64,64]*[1,32,64,64]=[1,64,64,64]==>[1,32,64,64]
        refine4 = self.conv_2((refine4_1 + refine4_2) * self.conv_sem(
            semanticModule_2_1))  ##[1,32,64,64]+[1,32,64,64]=[1,32,64,64] [1,32,64,64]*[1,32,64,64]=[1,32,64,64]==>[1,32,64,64]

        semVector_2_2, semanticModule_2_2 = self.semanticModule(torch.cat((down3, attention1_3 * fuse1), 1))
        refine3_1 = self.pam_attention(torch.cat((down3, attention1_3 * fuse1), 1))
        refine3_2 = self.cam_attention(torch.cat((down3, attention1_3 * fuse1), 1))
        refine3 = self.conv_2((refine3_1 + refine3_2) * self.conv_sem(semanticModule_2_2))

        semVector_2_3, semanticModule_2_3 = self.semanticModule(torch.cat((down2, attention1_2 * fuse1), 1))
        refine2_1 = self.pam_attention(torch.cat((down2, attention1_2 * fuse1), 1))
        refine2_2 = self.cam_attention(torch.cat((down2, attention1_2 * fuse1), 1))
        refine2 = self.conv_2((refine2_1 + refine2_2) * self.conv_sem(semanticModule_2_3))

        semVector_2_4, semanticModule_2_4 = self.semanticModule(torch.cat((down1, attention1_1 * fuse1), 1))
        refine1_1 = self.pam_attention(torch.cat((down1, attention1_1 * fuse1), 1))
        refine1_2 = self.cam_attention(torch.cat((down1, attention1_1 * fuse1), 1))

        refine1 = self.conv_2((refine1_1 + refine1_2) * self.conv_sem(semanticModule_2_4))

        predict4_2 = self.predict(refine4)  # [1,32,64,64]==>[1,1,64,64]
        predict3_2 = self.predict(refine3)  # [1,32,64,64]==>[1,1,64,64]
        predict2_2 = self.predict(refine2)  # [1,32,64,64]==>[1,1,64,64]
        predict1_2 = self.predict(refine1)  # [1,32,64,64]==>[1,1,64,64]

        predict1 = F.upsample(predict1, size=x.size()[2:], mode='bilinear')  # [1,1,64,64]==>[1,1,256,256]
        predict2 = F.upsample(predict2, size=x.size()[2:], mode='bilinear')
        predict3 = F.upsample(predict3, size=x.size()[2:], mode='bilinear')
        predict4 = F.upsample(predict4, size=x.size()[2:], mode='bilinear')  # [1,5,64,64]==>[1,5,256,256]

        predict1_2 = F.upsample(predict1_2, size=x.size()[2:], mode='bilinear')  # [1,5,64,64]==>[1,5,256,256]
        predict2_2 = F.upsample(predict2_2, size=x.size()[2:], mode='bilinear')
        predict3_2 = F.upsample(predict3_2, size=x.size()[2:], mode='bilinear')
        predict4_2 = F.upsample(predict4_2, size=x.size()[2:], mode='bilinear')  # [1,5,64,64]==>[1,5,256,256]
        outputs=[]
        if self.training:

            #outputs = [semVector_1_1]
            outputs.append(semVector_1_1)#not [semVector_1_1] for the len of semVector_1_1 is batch_size
            outputs.append(semVector_2_1)
            outputs.append(semVector_1_2)
            outputs.append(semVector_2_2)
            outputs.append(semVector_1_3)
            outputs.append(semVector_2_3)
            outputs.append(semVector_1_4)
            outputs.append(semVector_2_4)

            outputs.append(torch.cat((down1, fuse1), 1))
            outputs.append(torch.cat((down2, fuse1), 1))
            outputs.append(torch.cat((down3, fuse1), 1))
            outputs.append(torch.cat((down4, fuse1), 1))

            outputs.append(torch.cat((down1, attention1_1 * fuse1), 1))
            outputs.append(torch.cat((down2, attention1_2 * fuse1), 1))
            outputs.append(torch.cat((down3, attention1_3 * fuse1), 1))
            outputs.append(torch.cat((down4, attention1_4 * fuse1), 1))

            outputs.append(semanticModule_1_4)
            outputs.append(semanticModule_1_3)
            outputs.append(semanticModule_1_2)
            outputs.append(semanticModule_1_1)
            outputs.append(semanticModule_2_4)
            outputs.append(semanticModule_2_3)
            outputs.append(semanticModule_2_2)
            outputs.append(semanticModule_2_1)

            outputs.append(predict1)
            outputs.append(predict2)
            outputs.append(predict3)
            outputs.append(predict4)
            outputs.append(predict1_2)
            outputs.append(predict2_2)
            outputs.append(predict3_2)
            outputs.append(predict4_2)

            return tuple(outputs)

        else:
            return ((predict1_2 + predict2_2 + predict3_2 + predict4_2) / 4)

            # outputs.append(semVector_1_1)  # not [semVector_1_1] for the len of semVector_1_1 is batch_size
            # outputs.append(semVector_2_1)
            # outputs.append(semVector_1_2)
            # outputs.append(semVector_2_2)
            # outputs.append(semVector_1_3)
            # outputs.append(semVector_2_3)
            # outputs.append(semVector_1_4)
            # outputs.append(semVector_2_4)
            #
            # outputs.append(torch.cat((down1, fuse1), 1))
            # outputs.append(torch.cat((down2, fuse1), 1))
            # outputs.append(torch.cat((down3, fuse1), 1))
            # outputs.append(torch.cat((down4, fuse1), 1))
            #
            # outputs.append(torch.cat((down1, attention1_1 * fuse1), 1))
            # outputs.append(torch.cat((down2, attention1_2 * fuse1), 1))
            # outputs.append(torch.cat((down3, attention1_3 * fuse1), 1))
            # outputs.append(torch.cat((down4, attention1_4 * fuse1), 1))
            #
            # outputs.append(semanticModule_1_4)
            # outputs.append(semanticModule_1_3)
            # outputs.append(semanticModule_1_2)
            # outputs.append(semanticModule_1_1)
            # outputs.append(semanticModule_2_4)
            # outputs.append(semanticModule_2_3)
            # outputs.append(semanticModule_2_2)
            # outputs.append(semanticModule_2_1)
            #
            # outputs.append(predict1)
            # outputs.append(predict2)
            # outputs.append(predict3)
            # outputs.append(predict4)
            # outputs.append(predict1_2)
            # outputs.append(predict2_2)
            # outputs.append(predict3_2)
            # outputs.append(predict4_2)
            #
            # return tuple(outputs)

class SemTest(nn.Module):
    def __init__(self):
        super(SemTest,self).__init__()
        self.semanticModule = semanticModule(64)
        self.layer0 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2)
        )

        self.features = nn.Sequential(
            nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
    def forward(self, x):
        down4=x
        fuse1=x
        semVector_1_1, semanticModule_1_1 = self.semanticModule(
            torch.cat((down4, fuse1), 1))  # 2[1,32,64,64]==>[1,64,64,64]==> [65536],[1,64,64,64]
        #layer0=self.layer0(x)
        # return layer0
        return semVector_1_1,semanticModule_1_1

        # x1 = self.features(x)
        # x2 = self.features(x)
        # return x1, x2



class SegMultiLoss(nn.Module):
    def __init__(self):
        super(SegMultiLoss, self).__init__()
        self.bce_loss = nn.BCELoss()
        self.mse_loss = nn.MSELoss()
    def forward(self,*inputs):
        # decodering params uysing tuple
        #*preds,true_masks=tuple(inputs)
        preds, true_masks = tuple(inputs)
        #preds=torch.from_numpy(np.array(preds))
        #sem=preds[0]
        semVector_1_1, \
        semVector_2_1, \
        semVector_1_2, \
        semVector_2_2, \
        semVector_1_3, \
        semVector_2_3, \
        semVector_1_4, \
        semVector_2_4, \
        inp_enc0, \
        inp_enc1, \
        inp_enc2, \
        inp_enc3, \
        inp_enc4, \
        inp_enc5, \
        inp_enc6, \
        inp_enc7, \
        out_enc0, \
        out_enc1, \
        out_enc2, \
        out_enc3, \
        out_enc4, \
        out_enc5, \
        out_enc6, \
        out_enc7, \
        outputs0, \
        outputs1, \
        outputs2, \
        outputs3, \
        outputs0_2, \
        outputs1_2, \
        outputs2_2, \
        outputs3_2=tuple(preds)


        loss0 = self.bce_loss(outputs0, true_masks)
        loss1 = self.bce_loss(outputs1, true_masks)
        loss2 = self.bce_loss(outputs2, true_masks)
        loss3 = self.bce_loss(outputs3, true_masks)
        loss0_2 = self.bce_loss(outputs0_2, true_masks)
        loss1_2 = self.bce_loss(outputs1_2, true_masks)
        loss2_2 = self.bce_loss(outputs2_2, true_masks)
        loss3_2 = self.bce_loss(outputs3_2, true_masks)
        # The objective is that the class information can be embedded in the second position-channel attention module by
        # forcing the semantic representation of both encoder-decoders to be close
        lossSemantic1 = self.mse_loss(semVector_1_1, semVector_2_1)
        lossSemantic2 = self.mse_loss(semVector_1_2, semVector_2_2)
        lossSemantic3 = self.mse_loss(semVector_1_3, semVector_2_3)
        lossSemantic4 = self.mse_loss(semVector_1_4, semVector_2_4)
        # to ensure that the reconstructed features correspond to the features at the input of the position-channel attention
        # modules, the output of the encoders are forced to be close to their input
        lossRec0 = self.mse_loss(inp_enc0, out_enc0)
        lossRec1 = self.mse_loss(inp_enc1, out_enc1)
        lossRec2 = self.mse_loss(inp_enc2, out_enc2)
        lossRec3 = self.mse_loss(inp_enc3, out_enc3)
        lossRec4 = self.mse_loss(inp_enc4, out_enc4)
        lossRec5 = self.mse_loss(inp_enc5, out_enc5)
        lossRec6 = self.mse_loss(inp_enc6, out_enc6)
        lossRec7 = self.mse_loss(inp_enc7, out_enc7)

        lossG = loss0 + loss1 + loss2 + loss3 + loss0_2 + loss1_2 + loss2_2 + loss3_2 + 0.25 * (
                lossSemantic1 + lossSemantic2 + lossSemantic3 + lossSemantic4) \
                + 0.1 * (
                        lossRec0 + lossRec1 + lossRec2 + lossRec3 + lossRec4 + lossRec5 + lossRec6 + lossRec7)  # CE_lossG
        seg_pred=(outputs0 + outputs1 + outputs2 + outputs3 + outputs0_2 + outputs1_2 + outputs2_2 + outputs3_2) / 8
        return lossG,seg_pred



if __name__ == '__main__':
    from torchsummary import summary
    import sys
    #xs = torch.randn(size=(1, 3, 256, 256))
    #net  = Multi_Attention(in_channels=6,num_class=1)  # init only input model config
    #output = net(xs)  # forward input
    x=torch.randn(size=(1, 32, 64, 64))
    net=SemTest()
    # semvector,semmodule=net(x)
    # print(semvector.size())
    # print(semmodule.size())
    # output=net(x)
    # print(output.size())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net=net.to(device)
    mylog = open('E:/info.txt', 'w')
    stdout_backup = sys.stdout
    sys.stdout = mylog  # 输出到文件
    #print(summary(net, (32, 64, 64)))
    summary(net, (32, 64, 64))

    #summary(net, [(32, 16, 16), (32, 28, 28)])#无需print？
