import functools
import logging
import torch
import torch.nn as nn
from torch.nn import init

import models.Satt_CD.modules.architecture as arch
#import models.Satt_CD.modules.DeepLab.deeplab as DL
logger = logging.getLogger('base')
####################
# initialize
####################


def weights_init_normal(m, std=0.02):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.normal_(m.weight.data, 0.0, std)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0.0, std)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, std)  # BN also uses norm
        init.constant_(m.bias.data, 0.0)


def weights_init_kaiming(m, scale=1):
    '''
    kaiming初始化方法，论文在《 Delving deep into rectifiers: Surpassing human-level performance on ImageNet classification》，
    公式推导同样从“方差一致性”出法，kaiming是针对xavier初始化方法在relu这一类激活函数表现不佳而提出的改进
    mode – either 'fan_in' (default) or 'fan_out'. Choosing 'fan_in' preserves the magnitude of the variance of the weights in the forward pass.
     Choosing 'fan_out' preserves the magnitudes in the backwards pass.
     nonlinearity – the non-linear function (nn.functional name), recommended to use only with 'relu' or 'leaky_relu' (default).
    :param m:
    :param scale:
    :return:
    '''
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        m.weight.data *= scale
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        m.weight.data *= scale
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        if m.affine != False:

            init.constant_(m.weight.data, 1.0)
            init.constant_(m.bias.data, 0.0)


def weights_init_orthogonal(m):
    ''''
    使得tensor是正交的，论文:Exact solutions to the nonlinear dynamics of learning in deep linear neural networks”
    '''
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.orthogonal_(m.weight.data, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.orthogonal_(m.weight.data, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.constant_(m.weight.data, 1.0)
        init.constant_(m.bias.data, 0.0)

def weights_init_xavier(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_normal_(m.weight.data, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.xavier_normal_(m.weight.data, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.constant_(m.weight.data, 1.0)
        init.constant_(m.bias.data, 0.0)



def init_weights(net, init_type='kaiming', scale=1, std=0.02):
    # scale for 'kaiming', std for 'normal'.
    logger.info('Initialization method [{:s}]'.format(init_type))
    if init_type == 'normal':
        weights_init_normal_ = functools.partial(weights_init_normal, std=std)
        net.apply(weights_init_normal_)
    elif init_type == 'kaiming':#在ReLU网络中，假定每一层有一半的神经元被激活，另一半为0。推荐在ReLU网络中使用。
        weights_init_kaiming_ = functools.partial(weights_init_kaiming, scale=scale)
        net.apply(weights_init_kaiming_)
    elif init_type == 'orthogonal':#主要用以解决深度网络下的梯度消失、梯度爆炸问题，在RNN中经常使用的参数初始化方法
        net.apply(weights_init_orthogonal)
    elif init_type == 'xavier':#Xavier初始化的基本思想是保持输入和输出的方差一致，这样就避免了所有输出值都趋向于0。这是通用的方法，适用于任何激活函数
        net.apply(weights_init_xavier)
    else:
        raise NotImplementedError('initialization method [{:s}] not implemented'.format(init_type))


####################
# define network
####################


# Generator
def define_G(opt, device=None):
    gpu_ids = False
    opt_net = opt['network_G']
    which_model = opt_net['which_model_G']
    # in_channels = 6, n_classes = 1, filters = [], feature_scale = 4, is_deconv = True, is_batchnorm = True, use_res = True,
    # use_dense = True, use_deep_sub = True, att_type = 'CA', dblock_type = 'ASPP', use_rfnet = False
    if which_model == 'UNet_2D':
        netG=arch.unet_2D(in_channels=opt_net['in_nc'], n_classes=opt_net['out_nc'],use_res=opt_net['use_res'],
                          dblock_type=opt_net['dblock_type'],use_rfnet=opt_net['use_rfnet'])




    elif which_model == 'UNet_2D_PreTrain256_ED2':
        netG = arch.unet_2D_PreTrain256_ED2(in_channels=opt_net['in_nc'], n_classes=opt_net['out_nc'],
                                        use_res=opt_net['use_res'],
                                        dblock_type=opt_net['dblock_type'], use_rfnet=opt_net['use_rfnet'],
                                        use_att=opt_net['use_att'])
    elif which_model == 'unet_2D_PreTrain256_ED2_noDrop':
        netG = arch.unet_2D_PreTrain256_ED2_noDrop(in_channels=opt_net['in_nc'], n_classes=opt_net['out_nc'],
                                        use_res=opt_net['use_res'],
                                        dblock_type=opt_net['dblock_type'], use_rfnet=opt_net['use_rfnet'],
                                        use_att=opt_net['use_att'])




    elif which_model == 'UNet_2D_PreTrain512_ED2':
        netG = arch.unet_2D_PreTrain512_ED(in_channels=opt_net['in_nc'], n_classes=opt_net['out_nc'],
                                        use_res=opt_net['use_res'],
                                        dblock_type=opt_net['dblock_type'], use_rfnet=opt_net['use_rfnet'],
                                        use_att=opt_net['use_att'])
    elif which_model == 'UNet_2D_PreTrain512_ED2_drop':
        netG = arch.unet_2D_PreTrain512_ED2_drop(in_channels=opt_net['in_nc'], n_classes=opt_net['out_nc'],
                                        use_res=opt_net['use_res'],
                                        dblock_type=opt_net['dblock_type'], use_rfnet=opt_net['use_rfnet'],
                                        use_att=opt_net['use_att'])








    else:
        raise NotImplementedError('Generator model [{:s}] not recognized'.format(which_model))

    # if opt['is_train']:
    #     init_weights(netG, init_type='kaiming', scale=0.1)#scale=0.1!

    #init_weights(netG, init_type='normal')
    #init_weights(netG, init_type='kaiming', scale=0.1)
    return netG







# Discriminator
def define_D(opt):

    opt_net = opt['network_D']
    which_model = opt_net['which_model_D']

    if which_model == 'discriminator_vgg_128':
        netD = arch.Discriminator_VGG_128(in_nc=opt_net['in_nc'], base_nf=opt_net['nf'], \
            norm_type=opt_net['norm_type'], mode=opt_net['mode'], act_type=opt_net['act_type'])
    elif which_model=='discriminator_fc_256':
        netD=arch.Discriminator_FC_256(opt_net['in_nc'],act=False)
    elif which_model=='discriminator_fc_pix':
        netD=arch.Discriminator_FC_Pix(opt_net["in_nc"])
    elif which_model=='discriminator_fc_512':
        netD=arch.Discriminator_FC_512(opt_net['in_nc'],act=False)
    elif which_model == 'discriminator_vgg_256':
        netD = arch.Discriminator_VGG_256(in_nc=opt_net['in_nc'], base_nf=opt_net['nf'], \
            norm_type=opt_net['norm_type'], mode=opt_net['mode'], act_type=opt_net['act_type'])
    elif which_model == 'discriminator_vgg_96':
        netD = arch.Discriminator_VGG_96(in_nc=opt_net['in_nc'], base_nf=opt_net['nf'], \
            norm_type=opt_net['norm_type'], mode=opt_net['mode'], act_type=opt_net['act_type'])
    elif which_model == 'discriminator_vgg_192':
        netD = arch.Discriminator_VGG_192(in_nc=opt_net['in_nc'], base_nf=opt_net['nf'], \
            norm_type=opt_net['norm_type'], mode=opt_net['mode'], act_type=opt_net['act_type'])
    elif which_model == 'discriminator_vgg_128_SN':
        netD = arch.Discriminator_VGG_128_SN()
    else:
        raise NotImplementedError('Discriminator model [{:s}] not recognized'.format(which_model))

    #init_weights(netD, init_type='kaiming', scale=1)
    init_weights(netD,init_type='normal')


    return netD

def define_D_MS(opt):
    opt_net = opt['network_D_MS']
    which_model = opt_net['which_model_D']

    if which_model == 'discriminator_fcn_128_ms':
        netD = arch.Discriminator_FCN_Pix128_MS(in_nc=opt_net['in_nc'])
    elif which_model == 'discriminator_fcn_256_ms':
        netD = arch.Discriminator_FCN_Pix256_MS(in_nc=opt_net['in_nc'])

    else:
        raise NotImplementedError('Discriminator model [{:s}] not recognized'.format(which_model))

    # init_weights(netD, init_type='kaiming', scale=1)
    init_weights(netD, init_type='normal')

    return netD




def define_D_grad(opt):
    opt_net = opt['network_D_fea']
    which_model = opt_net['which_model_D']

    if which_model == 'discriminator_vgg_128':
        netD = arch.Discriminator_VGG_128(in_nc=opt_net['in_nc'], base_nf=opt_net['nf'], \
                                          norm_type=opt_net['norm_type'], mode=opt_net['mode'],
                                          act_type=opt_net['act_type'])
    elif which_model == 'discriminator_fc_256':
        netD = arch.Discriminator_FC_256(opt_net['in_nc'], act=False)
    elif which_model == 'discriminator_fc_pix':
        netD=arch.Discriminator_FC_Pix(opt_net["in_nc"])
    elif which_model == 'discriminator_fc_pix4':
        netD=arch.Discriminator_FC_Pix4(opt_net["in_nc"])
    elif which_model == 'discriminator_fc_pix16':
        netD=arch.Discriminator_FC_Pix16(opt_net["in_nc"])
    elif which_model == 'discriminator_fc_4':
        netD=arch.Discriminator_FC_4(opt_net["in_nc"])
    elif which_model == 'discriminator_fc_512':
        netD = arch.Discriminator_FC_512(opt_net['in_nc'], act=False)
    elif which_model == 'discriminator_fc_256':
        netD = arch.Discriminator_FC_256(opt_net['in_nc'], act=False)
    elif which_model == 'discriminator_fc_128':
        netD = arch.Discriminator_FC_128(opt_net['in_nc'], act=False)
    elif which_model == 'discriminator_fc_64':
        netD = arch.Discriminator_FC_64(opt_net['in_nc'], act=False)
    elif which_model == 'discriminator_fc_16':
        netD = arch.Discriminator_FC_16(opt_net['in_nc'],act=False)
    elif which_model == 'discriminator_fc_8':
        netD = arch.Discriminator_FC_8(opt_net['in_nc'], act=False)
    elif which_model == 'discriminator_vgg_256':
        netD = arch.Discriminator_VGG_256(in_nc=opt_net['in_nc'], base_nf=opt_net['nf'], \
                                          norm_type=opt_net['norm_type'], mode=opt_net['mode'],
                                          act_type=opt_net['act_type'])
    elif which_model == 'discriminator_vgg_96':
        netD = arch.Discriminator_VGG_96(in_nc=opt_net['in_nc'], base_nf=opt_net['nf'], \
                                         norm_type=opt_net['norm_type'], mode=opt_net['mode'],
                                         act_type=opt_net['act_type'])
    elif which_model == 'discriminator_vgg_192':
        netD = arch.Discriminator_VGG_192(in_nc=opt_net['in_nc'], base_nf=opt_net['nf'], \
                                          norm_type=opt_net['norm_type'], mode=opt_net['mode'],
                                          act_type=opt_net['act_type'])
    elif which_model == 'discriminator_vgg_128_SN':
        netD = arch.Discriminator_VGG_128_SN()

    elif which_model == 'discriminator_ASPP':##
        netD = arch.Discriminator_ASPP(opt_net['in_nc'])

    else:
        raise NotImplementedError('Discriminator model [{:s}] not recognized'.format(which_model))

    # init_weights(netD, init_type='kaiming', scale=1)
    init_weights(netD, init_type='normal')

    return netD

