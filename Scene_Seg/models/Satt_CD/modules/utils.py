import torch
import torch.nn.functional as F
from math import pi
from math import cos
def get_cos_lr(initial_lr, iteration, epoch_per_cycle):#E:\TEST2020\DownLoadPrj\SemanticSeg\pytorch.snapshot.ensembles-master\pytorch.snapshot.ensembles-master\se.py
    # proposed learning late function
    return initial_lr * (cos(pi * iteration / epoch_per_cycle) + 1) / 2

def get_step_lr(initial_lr, iteration, epoch_per_cycle):#E:\TEST2020\DownLoadPrj\SemanticSeg\pytorch.snapshot.ensembles-master\pytorch.snapshot.ensembles-master\se.py
    # proposed learning late function
    #return initial_lr * (cos(pi * iteration / epoch_per_cycle) + 1) / 2
    steps=[int(0.25*epoch_per_cycle),int(0.5*epoch_per_cycle),int(0.75*epoch_per_cycle)]
    if iteration<steps[0]:
        return initial_lr
    elif iteration>=steps[0] and iteration<steps[1]:
        return initial_lr*0.5
    elif iteration>=steps[1] and iteration<steps[2]:
        return initial_lr*0.5*0.5
    else:
        return initial_lr * 0.5 * 0.5*0.5
def get_poly_lr(initial_lr, iteration, epoch_per_cycle):#E:\TEST2020\DownLoadPrj\SemanticSeg\pytorch.snapshot.ensembles-master\pytorch.snapshot.ensembles-master\se.py
    # proposed learning late function
    #return initial_lr * (cos(pi * iteration / epoch_per_cycle) + 1) / 2
    return initial_lr*((1-iteration*1.0/epoch_per_cycle)**0.9)

def freeze_bn_module(m):
    """ Freeze the module `m` if it is a batch norm layer.

    :param m: a torch module
    :param mode: 'eval' or 'no_grad'
    """
    classname = type(m).__name__
    if classname.find('BatchNorm') != -1:
        m.eval()


class LBSign(torch.autograd.Function):

    @staticmethod#该方法不强制要求传递参数，如下声明一个静态方法,静态方法无需实例化,也可以实例化后调用
    def forward(ctx, input):
        return torch.sign(input)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clamp_(-1, 1)






def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()#[2,512,16,16]
    assert (len(size) == 4)
    N, C = size[:2]#2,512
    feat_var = feat.view(N, C, -1).var(dim=2) + eps#[2,512]
    feat_std = feat_var.sqrt().view(N, C, 1, 1)#[2,512,1,1]
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)#[2,512,1,1]
    return feat_mean, feat_std


def adaptive_instance_normalization(content_feat, style_feat,alpha=1.0):
    assert (content_feat.size()[:2] == style_feat.size()[:2])
    size = content_feat.size()#[2,512,16,16]
    style_mean, style_std = calc_mean_std(style_feat)#[2,512,1,1]
    content_mean, content_std = calc_mean_std(content_feat)

    normalized_feat = (content_feat - content_mean.expand(
        size)) / content_std.expand(size)#[2,512,16,16]
    return (normalized_feat * style_std.expand(size) + style_mean.expand(size))*alpha+(1-alpha)*content_feat



def adaptive_instance_normalization2(content_feat, style_mean,style_std):
    #assert (content_feat.size()[:2] == style_feat.size()[:2])
    size = content_feat.size()#[2,512,16,16]
    #style_mean, style_std = calc_mean_std(style_feat)#[2,512,1,1]
    content_mean, content_std = calc_mean_std(content_feat)
    style_mean=style_mean.unsqueeze(-1).unsqueeze(-1)
    style_std=style_std.unsqueeze(-1).unsqueeze(-1)
    normalized_feat = (content_feat - content_mean.expand(
        size)) / content_std.expand(size)#[2,512,16,16]
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)



def _calc_feat_flatten_mean_std(feat):
    # takes 3D feat (C, H, W), return mean and std of array within channels
    assert (feat.size()[0] == 3)
    assert (isinstance(feat, torch.FloatTensor))
    feat_flatten = feat.view(3, -1)
    mean = feat_flatten.mean(dim=-1, keepdim=True)
    std = feat_flatten.std(dim=-1, keepdim=True)
    return feat_flatten, mean, std


def _mat_sqrt(x):
    U, D, V = torch.svd(x)
    return torch.mm(torch.mm(U, D.pow(0.5).diag()), V.t())


def coral(source, target):
    # assume both source and target are 3D array (C, H, W)
    # Note: flatten -> f

    source_f, source_f_mean, source_f_std = _calc_feat_flatten_mean_std(source)
    source_f_norm = (source_f - source_f_mean.expand_as(
        source_f)) / source_f_std.expand_as(source_f)
    source_f_cov_eye = \
        torch.mm(source_f_norm, source_f_norm.t()) + torch.eye(3)

    target_f, target_f_mean, target_f_std = _calc_feat_flatten_mean_std(target)
    target_f_norm = (target_f - target_f_mean.expand_as(
        target_f)) / target_f_std.expand_as(target_f)
    target_f_cov_eye = \
        torch.mm(target_f_norm, target_f_norm.t()) + torch.eye(3)

    source_f_norm_transfer = torch.mm(
        _mat_sqrt(target_f_cov_eye),
        torch.mm(torch.inverse(_mat_sqrt(source_f_cov_eye)),
                 source_f_norm)
    )

    source_f_transfer = source_f_norm_transfer * \
                        target_f_std.expand_as(source_f_norm) + \
                        target_f_mean.expand_as(source_f_norm)

    return source_f_transfer.view(source.size())


from PIL import Image
from torchvision.transforms import Compose, ToTensor, ToPILImage, CenterCrop, Resize
import numpy as np
import torch
import math
import torchvision.transforms.functional as TF

# set random seed for reproducibility
np.random.seed(0)


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])


def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)


def gaussian_noise(image, std_dev):
    noise = np.rint(np.random.normal(loc=0.0, scale=std_dev, size=np.shape(image)))
    return Image.fromarray(np.clip(image + noise, 0, 255).astype(np.uint8))


def display_transform():
    return Compose([
        ToPILImage(),
        Resize(400),
        CenterCrop(400),
        ToTensor()
    ])
###########################################################################################################
########################Optimizer for Mean-Teacher#########################################################
###########################################################################################################

class EMAWeightOptimizer (object):
    def __init__(self, target_net, source_net, ema_alpha):
        self.target_net = target_net
        self.source_net = source_net
        self.ema_alpha = ema_alpha
        self.target_params = [p for p in target_net.state_dict().values() if p.dtype == torch.float]
        self.source_params = [p for p in source_net.state_dict().values() if p.dtype == torch.float]

        for tgt_p, src_p in zip(self.target_params, self.source_params):
            tgt_p[...] = src_p[...]

        target_keys = set(target_net.state_dict().keys())
        source_keys = set(source_net.state_dict().keys())
        if target_keys != source_keys:
            raise ValueError('Source and target networks do not have the same state dict keys; do they have different architectures?')


    def step(self):
        one_minus_alpha = 1.0 - self.ema_alpha
        for tgt_p, src_p in zip(self.target_params, self.source_params):
            tgt_p.mul_(self.ema_alpha)
            tgt_p.add_(src_p * one_minus_alpha)

def update_ema_variables(model, ema_model, global_step,alpha=0.999):
    # Use the true average until the exponential average is more correct
    '''
    That is a.add_(10,b) equals a += 10 * b where a and b are tensors. https://github.com/pytorch/pytorch/issues/23786
    https://github.com/neuropoly/domainadaptation/issues/1
    :param model:
    :param ema_model:
    :param alpha:
    :param global_step:
    :return:
    '''
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add((1 - alpha)*param.data)
#================================================================================
#===========================ASS map for outspace alighment=======================
#================================================================================
# input = (torch.randn(2, 3, 128, 128))
# probs=F.softmax(input)
# aff_map=fourway_affinity_kld(probs)#[2,3*4,128,128]
# aff_map=eightway_affinity_kld(probs)#[2,3*8,128,128]
def eightway_affinity_kld(probs, size=1):
    b, c, h, w = probs.size()
    if probs.dim() != 4:
        raise Exception('Only support for 4-D tensors!')
    p = size
    probs_pad = F.pad(probs, [p]*4, mode='replicate')
    bot_epsilon = 1e-4
    top_epsilon = 1.0
    neg_probs_clamp = torch.clamp(1.0 - probs, bot_epsilon, top_epsilon)
    probs_clamp = torch.clamp(probs, bot_epsilon, top_epsilon)
    kldiv_groups = []
    for st_y in range(0, 2*size+1, size):  # 0, size, 2*size
        for st_x in range(0, 2*size+1, size):
            if st_y == size and st_x == size:
                # Ignore the center pixel/feature.
                continue
            probs_paired = probs_pad[:, :, st_y:st_y+h, st_x:st_x+w] * probs
            neg_probs_paired = torch.clamp(
                1.0-probs_paired, bot_epsilon, top_epsilon)
            probs_paired = torch.clamp(probs_paired, bot_epsilon, top_epsilon)
            kldiv = probs_paired * torch.log(probs_paired/probs_clamp) \
                + neg_probs_paired * \
                torch.log(neg_probs_paired/neg_probs_clamp)
            kldiv_groups.append(kldiv)
    return torch.cat(kldiv_groups, dim=1)


def fourway_affinity_kld(probs, size=1):
    b, c, h, w = probs.size()
    if probs.dim() != 4:
        raise Exception('Only support for 4-D tensors!')
    p = size
    probs_pad = F.pad(probs, [p]*4, mode='replicate')
    bot_epsilon = 1e-4
    top_epsilon = 1.0
    neg_probs_clamp = torch.clamp(1.0 - probs, bot_epsilon, top_epsilon)
    probs_clamp = torch.clamp(probs, bot_epsilon, top_epsilon)
    kldiv_groups = []
    for st_y in range(0, 2*size+1, size):  # 0, size, 2*size
        for st_x in range(0, 2*size+1, size):
            if abs(st_y - st_x) == size:
                probs_paired = probs_pad[:, :,
                                         st_y:st_y+h, st_x:st_x+w] * probs#p_x * p_n
                neg_probs_paired = torch.clamp(
                    1.0-probs_paired, bot_epsilon, top_epsilon)
                probs_paired = torch.clamp(
                    probs_paired, bot_epsilon, top_epsilon)
                kldiv = probs_paired * torch.log(probs_paired/probs_clamp) \
                    + neg_probs_paired * \
                    torch.log(neg_probs_paired/neg_probs_clamp)
                kldiv_groups.append(kldiv)
    return torch.cat(kldiv_groups, dim=1)





#################################################################################
# MATLAB imresize taken from ESRGAN (https://github.com/xinntao/BasicSR)
#################################################################################

def cubic(x):
    absx = torch.abs(x)
    absx2 = absx**2
    absx3 = absx**3
    return (1.5 * absx3 - 2.5 * absx2 + 1) * (
        (absx <= 1).type_as(absx)) + (-0.5 * absx3 + 2.5 * absx2 - 4 * absx + 2) * ((
            (absx > 1) * (absx <= 2)).type_as(absx))


def calculate_weights_indices(in_length, out_length, scale, kernel, kernel_width, antialiasing):
    if (scale < 1) and (antialiasing):
        # Use a modified kernel to simultaneously interpolate and antialias- larger kernel width
        kernel_width = kernel_width / scale

    # Output-space coordinates
    x = torch.linspace(1, out_length, out_length)

    # Input-space coordinates. Calculate the inverse mapping such that 0.5
    # in output space maps to 0.5 in input space, and 0.5+scale in output
    # space maps to 1.5 in input space.
    u = x / scale + 0.5 * (1 - 1 / scale)

    # What is the left-most pixel that can be involved in the computation?
    left = torch.floor(u - kernel_width / 2)

    # What is the maximum number of pixels that can be involved in the
    # computation?  Note: it's OK to use an extra pixel here; if the
    # corresponding weights are all zero, it will be eliminated at the end
    # of this function.
    P = math.ceil(kernel_width) + 2

    # The indices of the input pixels involved in computing the k-th output
    # pixel are in row k of the indices matrix.
    indices = left.view(out_length, 1).expand(out_length, P) + torch.linspace(0, P - 1, P).view(
        1, P).expand(out_length, P)

    # The weights used to compute the k-th output pixel are in row k of the
    # weights matrix.
    distance_to_center = u.view(out_length, 1).expand(out_length, P) - indices
    # apply cubic kernel
    if (scale < 1) and (antialiasing):
        weights = scale * cubic(distance_to_center * scale)
    else:
        weights = cubic(distance_to_center)
    # Normalize the weights matrix so that each row sums to 1.
    weights_sum = torch.sum(weights, 1).view(out_length, 1)
    weights = weights / weights_sum.expand(out_length, P)

    # If a column in weights is all zero, get rid of it. only consider the first and last column.
    weights_zero_tmp = torch.sum((weights == 0), 0)
    if not math.isclose(weights_zero_tmp[0], 0, rel_tol=1e-6):
        indices = indices.narrow(1, 1, P - 2)
        weights = weights.narrow(1, 1, P - 2)
    if not math.isclose(weights_zero_tmp[-1], 0, rel_tol=1e-6):
        indices = indices.narrow(1, 0, P - 2)
        weights = weights.narrow(1, 0, P - 2)
    weights = weights.contiguous()
    indices = indices.contiguous()
    sym_len_s = -indices.min() + 1
    sym_len_e = indices.max() - in_length
    indices = indices + sym_len_s - 1
    return weights, indices, int(sym_len_s), int(sym_len_e)


def imresize(img, scale, antialiasing=True):
    # Now the scale should be the same for H and W
    # input: img: CHW RGB [0,1]
    # output: CHW RGB [0,1] w/o round

    in_C, in_H, in_W = img.size()
    _, out_H, out_W = in_C, math.ceil(in_H * scale), math.ceil(in_W * scale)

    kernel_width = 4
    kernel = 'cubic'

    # Return the desired dimension order for performing the resize.  The
    # strategy is to perform the resize first along the dimension with the
    # smallest scale factor.
    # Now we do not support this.

    # get weights and indices
    weights_H, indices_H, sym_len_Hs, sym_len_He = calculate_weights_indices(
        in_H, out_H, scale, kernel, kernel_width, antialiasing)
    weights_W, indices_W, sym_len_Ws, sym_len_We = calculate_weights_indices(
        in_W, out_W, scale, kernel, kernel_width, antialiasing)
    # process H dimension
    # symmetric copying
    img_aug = torch.FloatTensor(in_C, in_H + sym_len_Hs + sym_len_He, in_W)
    img_aug.narrow(1, sym_len_Hs, in_H).copy_(img)

    sym_patch = img[:, :sym_len_Hs, :]
    inv_idx = torch.arange(sym_patch.size(1) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(1, inv_idx)
    img_aug.narrow(1, 0, sym_len_Hs).copy_(sym_patch_inv)

    sym_patch = img[:, -sym_len_He:, :]
    inv_idx = torch.arange(sym_patch.size(1) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(1, inv_idx)
    img_aug.narrow(1, sym_len_Hs + in_H, sym_len_He).copy_(sym_patch_inv)

    out_1 = torch.FloatTensor(in_C, out_H, in_W)
    kernel_width = weights_H.size(1)
    for i in range(out_H):
        idx = int(indices_H[i][0])
        out_1[0, i, :] = img_aug[0, idx:idx + kernel_width, :].transpose(0, 1).mv(weights_H[i])
        out_1[1, i, :] = img_aug[1, idx:idx + kernel_width, :].transpose(0, 1).mv(weights_H[i])
        out_1[2, i, :] = img_aug[2, idx:idx + kernel_width, :].transpose(0, 1).mv(weights_H[i])

    # process W dimension
    # symmetric copying
    out_1_aug = torch.FloatTensor(in_C, out_H, in_W + sym_len_Ws + sym_len_We)
    out_1_aug.narrow(2, sym_len_Ws, in_W).copy_(out_1)

    sym_patch = out_1[:, :, :sym_len_Ws]
    inv_idx = torch.arange(sym_patch.size(2) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(2, inv_idx)
    out_1_aug.narrow(2, 0, sym_len_Ws).copy_(sym_patch_inv)

    sym_patch = out_1[:, :, -sym_len_We:]
    inv_idx = torch.arange(sym_patch.size(2) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(2, inv_idx)
    out_1_aug.narrow(2, sym_len_Ws + in_W, sym_len_We).copy_(sym_patch_inv)

    out_2 = torch.FloatTensor(in_C, out_H, out_W)
    kernel_width = weights_W.size(1)
    for i in range(out_W):
        idx = int(indices_W[i][0])
        out_2[0, :, i] = out_1_aug[0, :, idx:idx + kernel_width].mv(weights_W[i])
        out_2[1, :, i] = out_1_aug[1, :, idx:idx + kernel_width].mv(weights_W[i])
        out_2[2, :, i] = out_1_aug[2, :, idx:idx + kernel_width].mv(weights_W[i])

    return torch.clamp(out_2, 0, 1)


def set_requires_grad(nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad



if __name__ == '__main__':
    import cv2
    print('#### Test Case ###')
    img_path='E:\TestData\lena\lena.jpg'
    img= np.array(Image.open(img_path).convert('RGB'))
    print(img.shape)
    img1=cv2.resize(img,(31,31),interpolation=cv2.INTER_CUBIC)
    print(img1.shape)
    timg=TF.to_tensor(img1)
    print(timg.size())
    timg_1=imresize(timg,scale=256/31)
    print(timg_1.size())


