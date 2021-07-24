import torch
import torch.nn as nn
import torch.nn.functional as F
#from models.networks_other import init_weights
import numpy as np
from PIL import Image
import copy
import random
from torch.autograd import Variable
#
def entropy_loss_ita(v,ita=2.0):
    """
        Entropy loss for probabilistic prediction vectors
        input: batch_size x channels x h x w
        output: #====batch_size x 1 x h x w====# 1

    """
    assert v.dim() == 4
    n, c, h, w = v.size()
    #vv=torch.sum(torch.mul(v, torch.log2(v + 1e-30)),dim=0)#[4,2,256,256]==>[2,256,256]
    #torch.sum(torch.mul(v, torch.log2(v + 1e-30)), dim=1)##[4,2,256,256]==>[4,256,256]
    if c>1:
       constant=np.log2(c)
    else:
       constant=1

    ent= -torch.sum(torch.mul(v, torch.log2(v + 1e-30))) / (n * h * w * constant)#c!=1!!!
    ent = ent ** 2.0 + 1e-8
    ent = ent ** ita
    loss_ent = ent.mean()

    return loss_ent



def entropy_loss(v):
    """
        Entropy loss for probabilistic prediction vectors
        input: batch_size x channels x h x w
        output: #====batch_size x 1 x h x w====# 1
    """
    assert v.dim() == 4
    n, c, h, w = v.size()

    return -torch.sum(torch.mul(v, torch.log2(v + 1e-30))) / (n * h * w * np.log2(c))


def prob_2_entropy(prob):
    """ convert probabilistic prediction maps to weighted self-information maps
        output.size()==input.size()
    """
    n, c, h, w = prob.size()
    return -torch.mul(prob, torch.log2(prob + 1e-30)) / np.log2(c)


def adjust_LR_iter(optimizer, i_iter,learning_rate,num_steps):
    def lr_poly(base_lr, iter, max_iter, power=0.9):
        return base_lr * ((1 - float(iter) / max_iter) ** (power))
    lr = lr_poly(learning_rate, i_iter, num_steps)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1 :
        optimizer.param_groups[1]['lr'] = lr * 10


def compute_argmax_map(output):
    output = output.detach().cpu().numpy()
    output = output.transpose((1, 2, 0))
    output = np.asarray(np.argmax(output, axis=2), dtype=np.int)
    output = torch.from_numpy(output).float()
    return output


def find_good_maps(D_outs, pred_all,thresh_ST=0.6,batch_size=8):
    count = 0
    for i in range(D_outs.size(0)):
        if D_outs[i] > thresh_ST:
            count += 1

    if count > 0:
        print('Above ST-Threshold : ', count, '/', batch_size)
        pred_sel = torch.Tensor(count, pred_all.size(1), pred_all.size(2), pred_all.size(3))
        label_sel = torch.Tensor(count, pred_sel.size(2), pred_sel.size(3))
        num_sel = 0
        for j in range(D_outs.size(0)):
            if D_outs[j] > thresh_ST:
                pred_sel[num_sel] = pred_all[j]
                label_sel[num_sel] = compute_argmax_map(pred_all[j])
                num_sel += 1
        return pred_sel.cuda(), label_sel.cuda(), count
    else:
        return 0, 0, count


def make_D_label(label, ignore_mask):
    ignore_mask = np.expand_dims(ignore_mask, axis=1)#[4,321,321]==>[4,1,321,321]
    D_label = np.ones(ignore_mask.shape)*label#[4,1,321,321]
    D_label[ignore_mask] = 255# if ignore_mask=false, then all elements of D_label is 1
    D_label = Variable(torch.FloatTensor(D_label)).cuda(0)##[4,1,321,321]

    return D_label

def find_good_regions(D_outs, pred_all,thresh_ST=0.6):
    '''
    generate high-confidence region based on D value for semi-supervised learning

    :param D_outs:
    :param pred_all:
    :param thresh_ST:
    :param batch_size:
    :return:
    '''
    semi_ignore_mask=D_outs.data.cpu().numpy().squeeze(axis=1)<thresh_ST
    #semi_ignore_count=D_outs.numel()#for torch tensor
    semi_gt=pred_all.data.cpu().numpy().argmax(axis=1)#generate the semi_gt based on the argmax of the pred tensor
    semi_gt[semi_ignore_mask]=255
    semi_ratio=1.0 - float(semi_ignore_mask.sum()*1.0/semi_ignore_mask.size)#semi_ignore_mask.size=size prod of ndarray(semi_ignore_mask)
    print('semi ratio: {:.4f}'.format(semi_ratio))
    if semi_ratio>0:
        return  semi_ratio, torch.FloatTensor(semi_gt).cuda(0)
    else:
        return semi_ratio,0


def find_good_regions2(pred_all,thresh_ST=0.6):
    '''
    generate high-confidence region based on prediction value for semi-supervised learning
    # can not generate labels by uisng pred, otherwise will cause self-referee problem
    :param D_outs:
    :param pred_all:
    :param thresh_ST:
    :param batch_size:
    :return:
    '''
    soft_label= F.softmax(pred_all,dim=1).detach()  # otherwise one of the variables needed for gradient computation has been modified by an inplace operation

    soft_label=soft_label[:, 1].unsqueeze(1)# can not generate labels by uisng pred, otherwise will cause self-referee problem
    semi_ignore_mask=soft_label.data.cpu().numpy().squeeze(axis=1)<thresh_ST#[16,128,128]
    #semi_ignore_count=D_outs.numel()#for torch tensor
    semi_gt=pred_all.data.cpu().numpy().argmax(axis=1)#[16,128,128] generate the semi_gt based on the argmax of the pred tensor
    semi_gt[semi_ignore_mask]=255#ignore value is 255
    semi_ratio=1.0 - float(semi_ignore_mask.sum()*1.0/semi_ignore_mask.size)#semi_ignore_mask.size=size prod of ndarray(semi_ignore_mask)

    if semi_ratio>0:
        print('semi ratio: {:.4f}'.format(semi_ratio))
        return  semi_ratio, torch.FloatTensor(semi_gt).cuda(0)
    else:
        return semi_ratio,0








def calpsnr(im1_tensor, im2_tensor,max255=False):
    '''
    pnsr:https://blog.csdn.net/u010886794/article/details/84784453
    :param im1_tensor:
    :param im2_tensor:
    :return:
    '''
    im1=im1_tensor.data.cpu().numpy()
    im2 = im2_tensor.data.cpu().numpy()
    if max255==False:
       im1=im1*255
       im2=im2*255
    diff = np.abs(im1 - im2)
    rmse = np.sqrt(np.mean(np.square(diff)))
    psnr = 20*np.log10(255/rmse)
    return psnr

def DeNormalization(rgb_mean,im_tensor):
    im=im_tensor.data.cpu().numpy()[0].transpose([1,2,0])
    im=(im+rgb_mean)*255
    im=im.astype('uint8')
    return Image.fromarray(im,'RGB')

def DeNormalization2(rgb_mean,im_tensor):
    im=im_tensor.data.cpu().numpy()[0].transpose([1,2,0])
    #im=(im+rgb_mean)*255
    im=im.astype('uint8')
    return Image.fromarray(im,'RGB')


def Acc(mask_probs,true_masks):
    masks_probs_flat = mask_probs.view(-1)
    true_masks_flat = true_masks.view(-1)
    masks_probs_flat = masks_probs_flat > 0.5
    correctsAcc = torch.sum(
        masks_probs_flat.int() == true_masks_flat.int()).item() * 1.0 / true_masks_flat.size(0)
    return correctsAcc

def IOU(mask_probs,true_masks):
    smooth = 1.0  # may change
    mask_probs=(mask_probs>0.6).float()
    i = torch.sum(true_masks)  # 对二维矩阵的所有元素求和
    j = torch.sum(mask_probs)
    intersection = torch.sum(true_masks * mask_probs)
    #score = (2. * intersection + smooth) / (i + j + smooth)#bce
    score = (intersection + smooth) / (i + j - intersection + smooth)#iou
    return score.mean()

def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) +
        label_pred[mask], minlength=n_class**2).reshape(n_class, n_class)
    return hist

def scores(label_trues, label_preds, n_class):
    """Returns accuracy score evaluation result.
      - overall accuracy
      - mean accuracy
      - mean IU
      - fwavacc
    """
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    mean_iu = np.nanmean(iu)
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    cls_iu = dict(zip(range(n_class), iu))

    return mean_iu, cls_iu

def set_grad(nets, requires_grad=False):
    for net in nets:
        for param in net.parameters():
            param.requires_grad = requires_grad
def one_hot(label,num_classes=2):
    label = label.numpy()
    one_hot = np.zeros((label.shape[0], num_classes, label.shape[1], label.shape[2]), dtype=label.dtype)
    for i in range(num_classes):
        one_hot[:,i,...] = (label==i)
    #handle ignore labels
    return torch.FloatTensor(one_hot)


def one_hot_raw(label,num_classes=7):
    #label = label.numpy()
    one_hot = np.zeros((label.shape[0],label.shape[1],  num_classes), dtype=label.dtype)
    for i in range(num_classes):
        one_hot[:,:,i] = (label==i)
    #handle ignore labels
    return one_hot




def one_hot_cuda(label,num_classes=2,label_smooth=False):
    label = label.data.cpu().numpy()
    one_hot = np.zeros((label.shape[0], num_classes, label.shape[1], label.shape[2]), dtype=label.dtype)
    for i in range(num_classes):
        one_hot[:,i,...] = (label==i)
    #handle ignore labels
    if label_smooth:
        #smooth_labels = (1.0 - label_smoothing) * one_hot_labels + label_smoothing / num_classes
        smooth_value=0.1
        one_hot=(1.0-smooth_value)*one_hot+smooth_value/num_classes

    return torch.FloatTensor(one_hot).cuda()




def make_one_hot(labels, num_class=2, gpu_id=0):
    '''
    Converts an integer label torch.autograd.Variable to a one-hot Variable.

    Parameters
    ----------
    labels : torch.autograd.Variable of torch.cuda.LongTensor
        N x 1 x H x W, where N is batch size.
        Each value is an integer representing correct classification.
    C : integer.
        number of classes in labels.

    Returns
    -------
    target : torch.autograd.Variable of torch.cuda.FloatTensor
        N x C x H x W, where C is class number. One-hot encoded.
    '''
    C=num_class
    labels = labels.long()
    try:
        one_hot = torch.FloatTensor(labels.size(0), C, labels.size(2), labels.size(3)).zero_()
        one_hot = one_hot.cuda(0)
    except:
        one_hot = torch.FloatTensor(labels.size(0), C, labels.size(2), labels.size(3)).zero_()
    target = one_hot.scatter_(1, labels.data, 1)

    return target

# To store 50 generated image in a pool and sample from it when it is full
# Shrivastava et al’s strategy
class Sample_from_Pool(object):
    def __init__(self, max_elements=50):
        self.max_elements = max_elements
        self.cur_elements = 0
        self.items = []

    def __call__(self, in_items):
        return_items = []
        for in_item in in_items:
            if self.cur_elements < self.max_elements:
                self.items.append(in_item)
                self.cur_elements = self.cur_elements + 1
                return_items.append(in_item)
            else:
                if np.random.ranf() > 0.5:
                    idx = np.random.randint(0, self.max_elements)
                    tmp = copy.copy(self.items[idx])
                    self.items[idx] = in_item
                    return_items.append(tmp)
                else:
                    return_items.append(in_item)
        return return_items

'''
To be used to apply gaussian noise in the input to the discriminator
'''

class LambdaLR():
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert ((n_epochs - decay_start_epoch) > 0), "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch)/(self.n_epochs - self.decay_start_epoch)

class ReplayBuffer():
    def __init__(self, max_size=50):
        assert (max_size > 0), 'Empty buffer or trying to create a black hole. Be careful.'
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0,1) > 0.5:
                    i = random.randint(0, self.max_size-1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return Variable(torch.cat(to_return))# transform list to tensor?

class GaussianNoise(nn.Module):
    """Gaussian noise regularizer.

    Args:
        sigma (float, optional): relative standard deviation used to generate the
            noise. Relative means that it will be multiplied by the magnitude of
            the value your are adding the noise to. This means that sigma can be
            the same regardless of the scale of the vector.
        is_relative_detach (bool, optional): whether to detach the variable before
            computing the scale of the noise. If `False` then the scale of the noise
            won't be seen as a constant but something to optimize: this will bias the
            network to generate vectors with smaller values.
    """

    def __init__(self, sigma=0.1, is_relative_detach=True):
        super().__init__()
        self.sigma = sigma
        self.is_relative_detach = is_relative_detach

    def forward(self, x):
        if self.training and self.sigma != 0:
            scale = self.sigma * x.detach() if self.is_relative_detach else self.sigma * x
            sampled_noise = torch.zeros(x.size()).normal_() * scale
            x = x + sampled_noise
        return x

class conv2DBatchNorm(nn.Module):
    def __init__(self, in_channels, n_filters, k_size,  stride, padding, bias=True):
        super(conv2DBatchNorm, self).__init__()

        self.cb_unit = nn.Sequential(nn.Conv2d(int(in_channels), int(n_filters), kernel_size=k_size,
                                               padding=padding, stride=stride, bias=bias),
                                 nn.BatchNorm2d(int(n_filters)),)

    def forward(self, inputs):
        outputs = self.cb_unit(inputs)
        return outputs


class deconv2DBatchNorm(nn.Module):
    def __init__(self, in_channels, n_filters, k_size,  stride, padding, bias=True):
        super(deconv2DBatchNorm, self).__init__()

        self.dcb_unit = nn.Sequential(nn.ConvTranspose2d(int(in_channels), int(n_filters), kernel_size=k_size,
                                               padding=padding, stride=stride, bias=bias),
                                 nn.BatchNorm2d(int(n_filters)),)

    def forward(self, inputs):
        outputs = self.dcb_unit(inputs)
        return outputs


class conv2DBatchNormRelu(nn.Module):
    def __init__(self, in_channels, n_filters, k_size,  stride, padding, bias=True):
        super(conv2DBatchNormRelu, self).__init__()

        self.cbr_unit = nn.Sequential(nn.Conv2d(int(in_channels), int(n_filters), kernel_size=k_size,
                                                padding=padding, stride=stride, bias=bias),
                                 nn.BatchNorm2d(int(n_filters)),
                                 nn.ReLU(inplace=True),)

    def forward(self, inputs):
        outputs = self.cbr_unit(inputs)
        return outputs


class deconv2DBatchNormRelu(nn.Module):
    def __init__(self, in_channels, n_filters, k_size, stride, padding, bias=True):
        super(deconv2DBatchNormRelu, self).__init__()

        self.dcbr_unit = nn.Sequential(nn.ConvTranspose2d(int(in_channels), int(n_filters), kernel_size=k_size,
                                                padding=padding, stride=stride, bias=bias),
                                 nn.BatchNorm2d(int(n_filters)),
                                 nn.ReLU(inplace=True),)

    def forward(self, inputs):
        outputs = self.dcbr_unit(inputs)
        return outputs


class unetConv2(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm=True, n=1, ks=3, stride=1, padding=1):
        super(unetConv2, self).__init__()
        self.n = n
        self.ks = ks
        self.stride = stride
        self.padding = padding
        s = stride
        p = padding
        if is_batchnorm:
            for i in range(1, n+1):
                conv = nn.Sequential(nn.Conv2d(in_size, out_size, ks, s, p),
                                     nn.BatchNorm2d(out_size),
                                     nn.ReLU(inplace=True),)
                setattr(self, 'conv%d'%i, conv)
                in_size = out_size

        else:
            for i in range(1, n+1):
                conv = nn.Sequential(nn.Conv2d(in_size, out_size, ks, s, p),
                                     nn.ReLU(inplace=True),)
                setattr(self, 'conv%d'%i, conv)
                in_size = out_size

        #====initialise the blocks=====
        # for m in self.children():
        #     init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        x = inputs
        for i in range(1, self.n+1):
            conv = getattr(self, 'conv%d'%i)
            x = conv(x)

        return x


class UnetConv3(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm, kernel_size=(3,3,1), padding_size=(1,1,0), init_stride=(1,1,1)):
        super(UnetConv3, self).__init__()

        if is_batchnorm:
            self.conv1 = nn.Sequential(nn.Conv3d(in_size, out_size, kernel_size, init_stride, padding_size),
                                       nn.BatchNorm3d(out_size),
                                       nn.ReLU(inplace=True),)
            self.conv2 = nn.Sequential(nn.Conv3d(out_size, out_size, kernel_size, 1, padding_size),
                                       nn.BatchNorm3d(out_size),
                                       nn.ReLU(inplace=True),)
        else:
            self.conv1 = nn.Sequential(nn.Conv3d(in_size, out_size, kernel_size, init_stride, padding_size),
                                       nn.ReLU(inplace=True),)
            self.conv2 = nn.Sequential(nn.Conv3d(out_size, out_size, kernel_size, 1, padding_size),
                                       nn.ReLU(inplace=True),)

        # initialise the blocks
        for m in self.children():
            init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        return outputs


class FCNConv3(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm, kernel_size=(3,3,1), padding_size=(1,1,0), init_stride=(1,1,1)):
        super(FCNConv3, self).__init__()

        if is_batchnorm:
            self.conv1 = nn.Sequential(nn.Conv3d(in_size, out_size, kernel_size, init_stride, padding_size),
                                       nn.BatchNorm3d(out_size),
                                       nn.ReLU(inplace=True),)
            self.conv2 = nn.Sequential(nn.Conv3d(out_size, out_size, kernel_size, 1, padding_size),
                                       nn.BatchNorm3d(out_size),
                                       nn.ReLU(inplace=True),)
            self.conv3 = nn.Sequential(nn.Conv3d(out_size, out_size, kernel_size, 1, padding_size),
                                       nn.BatchNorm3d(out_size),
                                       nn.ReLU(inplace=True),)
        else:
            self.conv1 = nn.Sequential(nn.Conv3d(in_size, out_size, kernel_size, init_stride, padding_size),
                                       nn.ReLU(inplace=True),)
            self.conv2 = nn.Sequential(nn.Conv3d(out_size, out_size, kernel_size, 1, padding_size),
                                       nn.ReLU(inplace=True),)
            self.conv3 = nn.Sequential(nn.Conv3d(out_size, out_size, kernel_size, 1, padding_size),
                                       nn.ReLU(inplace=True),)

        # initialise the blocks
        for m in self.children():
            init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        outputs = self.conv3(outputs)
        return outputs


class UnetGatingSignal3(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm):
        super(UnetGatingSignal3, self).__init__()
        self.fmap_size = (4, 4, 4)

        if is_batchnorm:
            self.conv1 = nn.Sequential(nn.Conv3d(in_size, in_size//2, (1,1,1), (1,1,1), (0,0,0)),
                                       nn.BatchNorm3d(in_size//2),
                                       nn.ReLU(inplace=True),
                                       nn.AdaptiveAvgPool3d(output_size=self.fmap_size),
                                       )
            self.fc1 = nn.Linear(in_features=(in_size//2) * self.fmap_size[0] * self.fmap_size[1] * self.fmap_size[2],
                                 out_features=out_size, bias=True)
        else:
            self.conv1 = nn.Sequential(nn.Conv3d(in_size, in_size//2, (1,1,1), (1,1,1), (0,0,0)),
                                       nn.ReLU(inplace=True),
                                       nn.AdaptiveAvgPool3d(output_size=self.fmap_size),
                                       )
            self.fc1 = nn.Linear(in_features=(in_size//2) * self.fmap_size[0] * self.fmap_size[1] * self.fmap_size[2],
                                 out_features=out_size, bias=True)

        # initialise the blocks
        for m in self.children():
            init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        batch_size = inputs.size(0)
        outputs = self.conv1(inputs)
        outputs = outputs.view(batch_size, -1)
        outputs = self.fc1(outputs)
        return outputs

class UnetGridGatingSignal2(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=(1,1), is_batchnorm=True):
        super(UnetGridGatingSignal2, self).__init__()

        if is_batchnorm:
            self.conv1 = nn.Sequential(nn.Conv2d(in_size, out_size, kernel_size),
                                       nn.BatchNorm2d(out_size),
                                       nn.ReLU(inplace=True),
                                       )
        else:
            self.conv1 = nn.Sequential(nn.Conv2d(in_size, out_size, kernel_size),
                                       nn.ReLU(inplace=True),
                                       )

        # initialise the blocks
        for m in self.children():
            init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        return outputs



class UnetGridGatingSignal3(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=(1,1,1), is_batchnorm=True):
        super(UnetGridGatingSignal3, self).__init__()

        if is_batchnorm:
            self.conv1 = nn.Sequential(nn.Conv3d(in_size, out_size, kernel_size, (1,1,1), (0,0,0)),
                                       nn.BatchNorm3d(out_size),
                                       nn.ReLU(inplace=True),
                                       )
        else:
            self.conv1 = nn.Sequential(nn.Conv3d(in_size, out_size, kernel_size, (1,1,1), (0,0,0)),
                                       nn.ReLU(inplace=True),
                                       )

        # initialise the blocks
        for m in self.children():
            init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        return outputs

class Attention_block(nn.Module):#addictive attention
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi
class unetUp_Att(nn.Module):
    def __init__(self, in_size, out_size, is_deconv):
        super(unetUp_Att, self).__init__()
        self.conv = unetConv2(in_size, out_size, False)
        if is_deconv:
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=4, stride=2, padding=1)
        else:
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)

        # initialise the blocks
        for m in self.children():
            if m.__class__.__name__.find('unetConv2') != -1: continue
            init_weights(m, init_type='kaiming')
        self.Att=Attention_block(F_g=out_size,F_l=out_size,F_int=int(out_size/2))
    def forward(self, inputs1, inputs2):#[1,128,32,32]+[1,256,16,16]+conv==>[1,128,32,32]
        outputs2 = self.up(inputs2)
        offset = outputs2.size()[2] - inputs1.size()[2]
        padding = 2 * [offset // 2, offset // 2]
        outputs1 = F.pad(inputs1, padding)
        outputs1_att=self.Att(g=outputs2,x=outputs1)


        return self.conv(torch.cat([outputs1_att, outputs2], 1))

class unetConv2_res(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm=True, n=1, ks=3, stride=1, padding=1, use_res=True,
                 use_att=False,act='relu'):
        super(unetConv2_res, self).__init__()
        self.n = n
        self.ks = ks
        self.stride = stride
        self.padding = padding
        s = stride
        p = padding
        self.use_res = use_res
        if act=='relu':
            activation=nn.ReLU(inplace=True)
        else:
            activation=nn.LeakyReLU(0.2,inplace=True)

        if is_batchnorm:
            self.conv0 = conv = nn.Sequential(  # nn.Conv2d(out_size, out_size, ks, s, p),
                nn.Conv2d(in_size, out_size, ks, s, p),
                nn.BatchNorm2d(out_size),
                #nn.ReLU(inplace=False)
                activation

            )
            self.conv_short = nn.Sequential(nn.Conv2d(in_size, out_size, ks, s, p),
                                            nn.BatchNorm2d(out_size),
                                            )
            for i in range(1, n + 1):
                conv = nn.Sequential(  # nn.Conv2d(out_size, out_size, ks, s, p),
                    nn.Conv2d(out_size, out_size, ks, s, p),
                    nn.BatchNorm2d(out_size),
                    #nn.ReLU(inplace=False)
                    activation

                )
                setattr(self, 'conv%d' % i, conv)
                # in_size = out_size
            self.conv_last=nn.Sequential(
                nn.Conv2d(out_size, out_size, ks, s, p),
                nn.BatchNorm2d(out_size)
            )
        else:
            self.conv_short = nn.Sequential(nn.Conv2d(in_size, out_size, ks, s, p))
            self.conv0 = conv = nn.Sequential(
                nn.Conv2d(in_size, out_size, ks, s, p),
                #nn.ReLU(inplace=False)
                activation
            )

            for i in range(1, n + 1):
                conv = nn.Sequential(  # nn.Conv2d(out_size, out_size, ks, s, p),
                    nn.Conv2d(out_size, out_size, ks, s, p),
                    #nn.ReLU(inplace=False)
                    activation
                )
                setattr(self, 'conv%d' % i, conv)
            self.conv_last = nn.Conv2d(out_size, out_size, ks, s, p)
        # initialise the blocks
        # for m in self.children():
        #     init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        x = inputs
        x = self.conv0(x)
        if self.n>1:
            for i in range(1, self.n + 1):
                conv = getattr(self, 'conv%d' % i)
                x = conv(x)  # [9,16,256,256]
        x_short = self.conv_short(inputs)
        x=self.conv_last(x)
        return x + x_short

class unetConv2R(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm=True, n=2, ks=3, stride=1, padding=1, use_res=True):
        super(unetConv2R, self).__init__()
        self.n = n
        self.ks = ks
        self.stride = stride
        self.padding = padding
        s = stride
        p = padding
        self.use_res = use_res
        self.use_att=use_res

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
            x = conv(x)#[9,16,256,256]

            if i == 1:
                x_identity = x

        if self.use_res:
            x = x_identity + x
        return x





class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [  nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features),
                        nn.ReLU(inplace=True),
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features)  ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)



class unetConv2_res_IN(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm=True, n=1, ks=3, stride=1, padding=1, use_res=True,
                 use_att=False,act='relu'):
        super(unetConv2_res_IN, self).__init__()
        self.n = n
        self.ks = ks
        self.stride = stride
        self.padding = padding
        s = stride
        p = padding
        self.use_res = use_res
        if act=='relu':
            activation=nn.ReLU(inplace=True)
        else:
            activation=nn.LeakyReLU(0.2,inplace=True)

        if is_batchnorm:
            self.conv0 = conv = nn.Sequential(  # nn.Conv2d(out_size, out_size, ks, s, p),
                nn.ReflectionPad2d(p),
                nn.Conv2d(in_size, out_size, ks, s),
                nn.InstanceNorm2d(out_size),
                activation

            )
            self.conv_short = nn.Sequential(
                nn.ReflectionPad2d(p),
                nn.Conv2d(in_size, out_size, ks, s),
                                            nn.BatchNorm2d(out_size),
                                            )
            for i in range(1, n + 1):
                conv = nn.Sequential(
                    nn.ReflectionPad2d(p),
                    nn.Conv2d(out_size, out_size, ks, s),
                    nn.InstanceNorm2d(out_size),
                    activation

                )
                setattr(self, 'conv%d' % i, conv)
                # in_size = out_size
            self.conv_last=nn.Sequential(
                nn.ReflectionPad2d(p),
                nn.Conv2d(out_size, out_size, ks, s),
                nn.InstanceNorm2d(out_size),
                activation
            )
        else:
            self.conv_short = nn.Sequential(nn.Conv2d(in_size, out_size, ks, s, p))
            self.conv0 = conv = nn.Sequential(
                nn.Conv2d(in_size, out_size, ks, s, p),
                #nn.ReLU(inplace=False)
                activation
            )

            for i in range(1, n + 1):
                conv = nn.Sequential(  # nn.Conv2d(out_size, out_size, ks, s, p),
                    nn.Conv2d(out_size, out_size, ks, s, p),
                    #nn.ReLU(inplace=False)
                    activation
                )
                setattr(self, 'conv%d' % i, conv)
            self.conv_last = nn.Conv2d(out_size, out_size, ks, s, p)


    def forward(self, inputs):
        x = inputs
        x = self.conv0(x)
        for i in range(1, self.n + 1):
            conv = getattr(self, 'conv%d' % i)
            x = conv(x)  # [9,16,256,256]
        x_short = self.conv_short(inputs)
        x=self.conv_last(x)
        return x + x_short







class unetConv2_res0(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm=True, n=1, ks=3, stride=1, padding=1, use_res=True,
                 use_att=False,act='relu'):
        super(unetConv2_res0, self).__init__()
        self.n = n
        self.ks = ks
        self.stride = stride
        self.padding = padding
        s = stride
        p = padding
        self.use_res = use_res
        # self.use_att = use_att
        # self.channel_Att = ChannelAttention(out_size)
        # self.spatial_Att = SpatialAttention()
        # self.SC_Att = SCAttention(out_size)
        if is_batchnorm:
            self.conv0 = conv = nn.Sequential(  # nn.Conv2d(out_size, out_size, ks, s, p),
                nn.Conv2d(in_size, out_size, ks, s, p),
                nn.BatchNorm2d(out_size),
                nn.ReLU(inplace=False)

            )
            self.conv_short = nn.Sequential(nn.Conv2d(in_size, out_size, ks, s, p),
                                            nn.BatchNorm2d(out_size),
                                            )
            for i in range(1, n + 1):
                conv = nn.Sequential(  # nn.Conv2d(out_size, out_size, ks, s, p),
                    nn.Conv2d(out_size, out_size, ks, s, p),
                    nn.BatchNorm2d(out_size),
                    nn.ReLU(inplace=False)

                )
                setattr(self, 'conv%d' % i, conv)
                # in_size = out_size
            self.conv_last=nn.Sequential(
                nn.Conv2d(out_size, out_size, ks, s, p),
                nn.BatchNorm2d(out_size)
            )
        else:
            self.conv_short = nn.Sequential(nn.Conv2d(in_size, out_size, ks, s, p))
            self.conv0 = conv = nn.Sequential(
                nn.Conv2d(in_size, out_size, ks, s, p),
                nn.ReLU(inplace=False)
            )

            for i in range(1, n + 1):
                conv = nn.Sequential(  # nn.Conv2d(out_size, out_size, ks, s, p),
                    nn.Conv2d(out_size, out_size, ks, s, p),
                    nn.ReLU(inplace=False)
                )
                setattr(self, 'conv%d' % i, conv)
            self.conv_last = nn.Conv2d(out_size, out_size, ks, s, p)
        # initialise the blocks
        for m in self.children():
            init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        x = inputs
        x = self.conv0(x)
        for i in range(1, self.n + 1):
            conv = getattr(self, 'conv%d' % i)
            x = conv(x)  # [9,16,256,256]
        x_short = self.conv_short(inputs)
        x=self.conv_last(x)
        return x + x_short

class unetUp(nn.Module):
    def __init__(self, in_size, out_size, is_deconv,use_res=False,act='relu'):
        super(unetUp, self).__init__()
        if use_res:
            self.conv=unetConv2_res(in_size, out_size, is_batchnorm=True,act=act)
        else:
            self.conv = unetConv2(in_size, out_size, is_batchnorm=True)
        if is_deconv:
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=4, stride=2, padding=1)
        else:
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)

        # initialise the blocks
        # for m in self.children():
        #     if m.__class__.__name__.find('unetConv2') != -1: continue
        #     init_weights(m, init_type='kaiming')

    def forward(self, inputs1, inputs2):
        outputs2 = self.up(inputs2)
        offset = outputs2.size()[2] - inputs1.size()[2]
        padding = 2 * [offset // 2, offset // 2]
        outputs1 = F.pad(inputs1, padding)
        return self.conv(torch.cat([outputs1, outputs2], 1))


class UnetUp3(nn.Module):
    def __init__(self, in_size, out_size, is_deconv, is_batchnorm=True):
        super(UnetUp3, self).__init__()
        if is_deconv:
            self.conv = UnetConv3(in_size, out_size, is_batchnorm)
            self.up = nn.ConvTranspose3d(in_size, out_size, kernel_size=(4,4,1), stride=(2,2,1), padding=(1,1,0))
        else:
            self.conv = UnetConv3(in_size+out_size, out_size, is_batchnorm)
            self.up = nn.Upsample(scale_factor=(2, 2, 1), mode='trilinear')

        # initialise the blocks
        for m in self.children():
            if m.__class__.__name__.find('UnetConv3') != -1: continue
            init_weights(m, init_type='kaiming')

    def forward(self, inputs1, inputs2):#[1,128,16,16,5]+[1,256,8,8,5]
        outputs2 = self.up(inputs2)#[1,256,8,8,5]==>[1,256,16,16,5]
        offset = outputs2.size()[2] - inputs1.size()[2]#0
        padding = 2 * [offset // 2, offset // 2, 0]#[0,0,0,0,0,0]
        outputs1 = F.pad(inputs1, padding)#[1,128,16,16,5]==>[1,128,16,16,5]
        return self.conv(torch.cat([outputs1, outputs2], 1))


class UnetUp3_CT(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm=True):
        super(UnetUp3_CT, self).__init__()
        self.conv = UnetConv3(in_size + out_size, out_size, is_batchnorm, kernel_size=(3,3,3), padding_size=(1,1,1))
        self.up = nn.Upsample(scale_factor=(2, 2, 2), mode='trilinear')

        # initialise the blocks
        for m in self.children():
            if m.__class__.__name__.find('UnetConv3') != -1: continue
            init_weights(m, init_type='kaiming')

    def forward(self, inputs1, inputs2):
        outputs2 = self.up(inputs2)
        offset = outputs2.size()[2] - inputs1.size()[2]
        padding = 2 * [offset // 2, offset // 2, 0]
        outputs1 = F.pad(inputs1, padding)
        return self.conv(torch.cat([outputs1, outputs2], 1))


# Squeeze-and-Excitation Network
class SqEx(nn.Module):

    def __init__(self, n_features, reduction=6):
        super(SqEx, self).__init__()

        if n_features % reduction != 0:
            raise ValueError('n_features must be divisible by reduction (default = 4)')

        self.linear1 = nn.Linear(n_features, n_features // reduction, bias=False)
        self.nonlin1 = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(n_features // reduction, n_features, bias=False)
        self.nonlin2 = nn.Sigmoid()

    def forward(self, x):

        y = F.avg_pool3d(x, kernel_size=x.size()[2:5])
        y = y.permute(0, 2, 3, 4, 1)
        y = self.nonlin1(self.linear1(y))
        y = self.nonlin2(self.linear2(y))
        y = y.permute(0, 4, 1, 2, 3)
        y = x * y
        return y

class UnetUp3_SqEx(nn.Module):
    def __init__(self, in_size, out_size, is_deconv, is_batchnorm):
        super(UnetUp3_SqEx, self).__init__()
        if is_deconv:
            self.sqex = SqEx(n_features=in_size+out_size)
            self.conv = UnetConv3(in_size, out_size, is_batchnorm)
            self.up = nn.ConvTranspose3d(in_size, out_size, kernel_size=(4,4,1), stride=(2,2,1), padding=(1,1,0))
        else:
            self.sqex = SqEx(n_features=in_size+out_size)
            self.conv = UnetConv3(in_size+out_size, out_size, is_batchnorm)
            self.up = nn.Upsample(scale_factor=(2, 2, 1), mode='trilinear')

        # initialise the blocks
        for m in self.children():
            if m.__class__.__name__.find('UnetConv3') != -1: continue
            init_weights(m, init_type='kaiming')

    def forward(self, inputs1, inputs2):
        outputs2 = self.up(inputs2)
        offset = outputs2.size()[2] - inputs1.size()[2]
        padding = 2 * [offset // 2, offset // 2, 0]
        outputs1 = F.pad(inputs1, padding)
        concat = torch.cat([outputs1, outputs2], 1)
        gated  = self.sqex(concat)
        return self.conv(gated)

class residualBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, n_filters, stride=1, downsample=None):
        super(residualBlock, self).__init__()

        self.convbnrelu1 = conv2DBatchNormRelu(in_channels, n_filters, 3,  stride, 1, bias=False)
        self.convbn2 = conv2DBatchNorm(n_filters, n_filters, 3, 1, 1, bias=False)
        self.downsample = downsample
        self.stride = stride
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x

        out = self.convbnrelu1(x)
        out = self.convbn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


class residualBottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, n_filters, stride=1, downsample=None):
        super(residualBottleneck, self).__init__()
        self.convbn1 = nn.Conv2DBatchNorm(in_channels,  n_filters, k_size=1, bias=False)
        self.convbn2 = nn.Conv2DBatchNorm(n_filters,  n_filters, k_size=3, padding=1, stride=stride, bias=False)
        self.convbn3 = nn.Conv2DBatchNorm(n_filters,  n_filters * 4, k_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.convbn1(x)
        out = self.convbn2(out)
        out = self.convbn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out




class SeqModelFeatureExtractor(nn.Module):
    def __init__(self, submodule, extracted_layers):
        super(SeqModelFeatureExtractor, self).__init__()

        self.submodule = submodule
        self.extracted_layers = extracted_layers

    def forward(self, x):
        outputs = []
        for name, module in self.submodule._modules.items():
            x = module(x)
            if name in self.extracted_layers:
                outputs += [x]
        return outputs + [x]


class HookBasedFeatureExtractor(nn.Module):
    def __init__(self, submodule, layername, upscale=False):
        super(HookBasedFeatureExtractor, self).__init__()

        self.submodule = submodule
        self.submodule.eval()
        self.layername = layername
        self.outputs_size = None
        self.outputs = None
        self.inputs = None
        self.inputs_size = None
        self.upscale = upscale

    def get_input_array(self, m, i, o):
        if isinstance(i, tuple):
            self.inputs = [i[index].data.clone() for index in range(len(i))]
            self.inputs_size = [input.size() for input in self.inputs]
        else:
            self.inputs = i.data.clone()
            self.inputs_size = self.input.size()
        print('Input Array Size: ', self.inputs_size)

    def get_output_array(self, m, i, o):
        if isinstance(o, tuple):
            self.outputs = [o[index].data.clone() for index in range(len(o))]
            self.outputs_size = [output.size() for output in self.outputs]
        else:
            self.outputs = o.data.clone()
            self.outputs_size = self.outputs.size()
        print('Output Array Size: ', self.outputs_size)

    def rescale_output_array(self, newsize):
        us = nn.Upsample(size=newsize[2:], mode='bilinear')
        if isinstance(self.outputs, list):
            for index in range(len(self.outputs)): self.outputs[index] = us(self.outputs[index]).data()
        else:
            self.outputs = us(self.outputs).data()

    def forward(self, x):
        target_layer = self.submodule._modules.get(self.layername)

        # Collect the output tensor
        h_inp = target_layer.register_forward_hook(self.get_input_array)
        h_out = target_layer.register_forward_hook(self.get_output_array)
        self.submodule(x)
        h_inp.remove()
        h_out.remove()

        # Rescale the feature-map if it's required
        if self.upscale: self.rescale_output_array(x.size())

        return self.inputs, self.outputs


class UnetDsv3(nn.Module):
    def __init__(self, in_size, out_size, scale_factor):
        super(UnetDsv3, self).__init__()
        self.dsv = nn.Sequential(nn.Conv3d(in_size, out_size, kernel_size=1, stride=1, padding=0),
                                 nn.Upsample(scale_factor=scale_factor, mode='trilinear'), )

    def forward(self, input):
        return self.dsv(input)


def print_model_parm_nums(model):            #得到模型参数总量

    total = sum([param.nelement() for param in model.parameters()])
    print('  + Number of params: %.2fM' % (total / 1e6))     #每一百万为一个单位
    return total/1e6

### compute model params
def count_param(model):
    param_count = 0
    for param in model.parameters():
        param_count += param.view(-1).size()[0]
    return param_count

### initalize the module
from torch.nn import init
def init_weights(net, init_type='normal'):
    #print('initialization method [%s]' % init_type)
    if init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    #print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)