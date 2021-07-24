import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from torch.nn import NLLLoss2d
from models.Satt_CD.modules.ramps import  *
#=======Define GAN loss: [vanilla | lsgan | wgan-gp]=====
class GANLoss(nn.Module):
    def __init__(self, gan_type='gan', real_label_val=1.0, fake_label_val=0.0):
        super(GANLoss, self).__init__()
        self.gan_type = gan_type.lower()
        self.real_label_val = real_label_val
        self.fake_label_val = fake_label_val

        if self.gan_type == 'gan' or self.gan_type == 'ragan':
            self.loss = nn.BCEWithLogitsLoss()
        elif self.gan_type == 'lsgan':
            self.loss = nn.MSELoss()
        elif self.gan_type == 'wgan-gp':

            def wgan_loss(input, target):
                # target is boolean
                return -1 * input.mean() if target else input.mean()
            self.loss = wgan_loss
        else:
            raise NotImplementedError('GAN type [{:s}] is not found'.format(self.gan_type))

    def get_target_label(self, input, target_is_real):
        if self.gan_type == 'wgan-gp':
            return target_is_real
        if target_is_real:
            return torch.empty_like(input).fill_(self.real_label_val)
        else:
            return torch.empty_like(input).fill_(self.fake_label_val)

    def forward(self, input, target_is_real):
        target_label = self.get_target_label(input, target_is_real)
        loss = self.loss(input, target_label)
        return loss

class GradientPenaltyLoss(nn.Module):
    def __init__(self, device=torch.device('cpu')):
        super(GradientPenaltyLoss, self).__init__()
        self.register_buffer('grad_outputs', torch.Tensor())
        self.grad_outputs = self.grad_outputs.to(device)

    def get_grad_outputs(self, input):
        if self.grad_outputs.size() != input.size():
            self.grad_outputs.resize_(input.size()).fill_(1.0)
        return self.grad_outputs

    def forward(self, interp, interp_crit):
        grad_outputs = self.get_grad_outputs(interp_crit)
        grad_interp = torch.autograd.grad(outputs=interp_crit, inputs=interp,
                                          grad_outputs=grad_outputs, create_graph=True,
                                          retain_graph=True, only_inputs=True)[0]
        grad_interp = grad_interp.view(grad_interp.size(0), -1)
        grad_interp_norm = grad_interp.norm(2, dim=1)

        loss = ((grad_interp_norm - 1)**2).mean()
        return loss

#=====================consistency weight===================================
from models.Satt_CD.modules import ramps

class consistency_weight(object):
    """
    ramp_types = ['sigmoid_rampup', 'linear_rampup', 'cosine_rampup', 'log_rampup', 'exp_rampup']
    """
    def __init__(self, final_w, iters_per_epoch, rampup_starts=0, rampup_ends=7, ramp_type='sigmoid_rampup'):
        self.final_w = final_w
        self.iters_per_epoch = iters_per_epoch
        self.rampup_starts = rampup_starts * iters_per_epoch
        self.rampup_ends = rampup_ends * iters_per_epoch
        self.rampup_length = (self.rampup_ends - self.rampup_starts)
        self.rampup_func = getattr(ramps, ramp_type)#用于返回一个对象属性值
        self.current_rampup = 0

    # def __call__(self, epoch, curr_iter):
    #     cur_total_iter = self.iters_per_epoch * epoch + curr_iter
    #     if cur_total_iter < self.rampup_starts:
    #         return 0
    #     self.current_rampup = self.rampup_func(cur_total_iter - self.rampup_starts, self.rampup_length)
    #     return self.final_w * self.current_rampup
    def __call__(self, cur_total_iter):
        #cur_total_iter = self.iters_per_epoch * epoch + curr_iter
        if cur_total_iter < self.rampup_starts:
            return 0
        self.current_rampup = self.rampup_func(cur_total_iter - self.rampup_starts, self.rampup_length)

        return self.final_w * self.current_rampup



def sigmoid_mse_loss(inputs, targets, conf_mask=False, threshold=None, use_softmax=False):
    #assert inputs.requires_grad == True and targets.requires_grad == False
    assert inputs.size() == targets.size()
    # inputs = F.softmax(inputs, dim=1)
    # if use_softmax:
    #     targets = F.softmax(targets, dim=1)

    # if threshold:
    #     loss_mat = F.mse_loss(inputs, targets, reduction='none')
    #     mask = (targets.max(1)[0] > threshold)
    #     loss_mat = loss_mat[mask.unsqueeze(1).expand_as(loss_mat)]
    #     if loss_mat.shape.numel() == 0: loss_mat = torch.tensor([0.]).to(inputs.device)
    #     return loss_mat.mean()
    # else:
    #     return F.mse_loss(inputs, targets, reduction='mean')
    #method2
    # num_classes = inputs.size()[1]
    # return F.mse_loss(inputs, targets, size_average=False) / num_classes
    #method3===========
    mse_loss = (inputs-targets)**2
    return mse_loss


def sigmoid_kl_loss(inputs, targets, conf_mask=False, threshold=None, use_softmax=False):
    #assert inputs.requires_grad == True and targets.requires_grad == False
    assert inputs.size() == targets.size()

    if threshold:
        loss_mat = F.kl_div(input.log(), targets, reduction='none')
        mask = (targets.max(1)[0] > threshold)
        loss_mat = loss_mat[mask.unsqueeze(1).expand_as(loss_mat)]
        if loss_mat.shape.numel() == 0: loss_mat = torch.tensor([0.]).to(inputs.device)
        return loss_mat.sum() / mask.shape.numel()
    else:
        return F.kl_div(input.log(), targets, reduction='mean')



def softmax_mse_loss(inputs, targets, conf_mask=False, threshold=None, use_softmax=False):
    assert inputs.requires_grad == True and targets.requires_grad == False
    assert inputs.size() == targets.size()
    inputs = F.softmax(inputs, dim=1)
    if use_softmax:
        targets = F.softmax(targets, dim=1)

    if conf_mask:
        loss_mat = F.mse_loss(inputs, targets, reduction='none')
        mask = (targets.max(1)[0] > threshold)
        loss_mat = loss_mat[mask.unsqueeze(1).expand_as(loss_mat)]
        if loss_mat.shape.numel() == 0: loss_mat = torch.tensor([0.]).to(inputs.device)
        return loss_mat.mean()
    else:
        return F.mse_loss(inputs, targets, reduction='mean')


def softmax_kl_loss(inputs, targets, conf_mask=False, threshold=None, use_softmax=False):
    assert inputs.requires_grad == True and targets.requires_grad == False
    assert inputs.size() == targets.size()
    input_log_softmax = F.log_softmax(inputs, dim=1)
    if use_softmax:
        targets = F.softmax(targets, dim=1)

    if conf_mask:
        loss_mat = F.kl_div(input_log_softmax, targets, reduction='none')
        mask = (targets.max(1)[0] > threshold)
        loss_mat = loss_mat[mask.unsqueeze(1).expand_as(loss_mat)]
        if loss_mat.shape.numel() == 0: loss_mat = torch.tensor([0.]).to(inputs.device)
        return loss_mat.sum() / mask.shape.numel()
    else:
        return F.kl_div(input_log_softmax, targets, reduction='mean')

#################################Metric learning loss========================
class BCL(nn.Module):
    """
    batch-balanced contrastive loss
    no-change，1
    change，-1
    """

    def __init__(self, margin=2.0):
        super(BCL, self).__init__()
        self.margin = margin

    def forward(self, distance, label):
        label[label==255] = 1
        mask = (label != 255).float()
        distance = distance * mask
        pos_num = torch.sum((label==1).float())+0.0001
        neg_num = torch.sum((label==-1).float())+0.0001

        loss_1 = torch.sum((1+label) / 2 * torch.pow(distance, 2)) /pos_num
        loss_2 = torch.sum((1-label) / 2 *
            torch.pow(torch.clamp(self.margin - distance, min=0.0), 2)
        ) / neg_num
        loss = loss_1 + loss_2
        return loss
###############################################################################################
####################################multi-class seg############################################
###############################################################################################


try:
    from itertools import ifilterfalse
except ImportError:  # py3k
    from itertools import filterfalse

eps = 1e-6

def dice_round(preds, trues):
    preds = preds.float()
    return soft_dice_loss(preds, trues)


def iou_round(preds, trues):
    preds = preds.float()
    return jaccard(preds, trues)


def soft_dice_loss(outputs, targets, per_image=False):
    batch_size = outputs.size()[0]
    if not per_image:
        batch_size = 1
    dice_target = targets.contiguous().view(batch_size, -1).float()
    dice_output = outputs.contiguous().view(batch_size, -1)
    intersection = torch.sum(dice_output * dice_target, dim=1)
    union = torch.sum(dice_output, dim=1) + torch.sum(dice_target, dim=1) + eps
    loss = (1 - (2 * intersection + eps) / union).mean()
    return loss


def jaccard(outputs, targets, per_image=False):
    batch_size = outputs.size()[0]
    if not per_image:
        batch_size = 1
    dice_target = targets.contiguous().view(batch_size, -1).float()
    dice_output = outputs.contiguous().view(batch_size, -1)
    intersection = torch.sum(dice_output * dice_target, dim=1)
    union = torch.sum(dice_output, dim=1) + torch.sum(dice_target, dim=1) - intersection + eps
    losses = 1 - (intersection + eps) / union
    return losses.mean()


class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True, per_image=False):
        super().__init__()
        self.size_average = size_average
        self.register_buffer('weight', weight)
        self.per_image = per_image

    def forward(self, input, target):
        return soft_dice_loss(input, target, per_image=self.per_image)


class JaccardLoss(nn.Module):
    def __init__(self, weight=None, size_average=True, per_image=False):
        super().__init__()
        self.size_average = size_average
        self.register_buffer('weight', weight)
        self.per_image = per_image

    def forward(self, input, target):
        return jaccard(input, target, per_image=self.per_image)


class StableBCELoss(nn.Module):
    def __init__(self):
        super(StableBCELoss, self).__init__()

    def forward(self, input, target):
        input = input.float().view(-1)
        target = target.float().view(-1)
        neg_abs = - input.abs()
        # todo check correctness
        loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()
        return loss.mean()


class ComboLoss(nn.Module):
    def __init__(self, weights, per_image=False):
        super().__init__()
        self.weights = weights
        self.bce = StableBCELoss()
        self.dice = DiceLoss(per_image=False)
        self.jaccard = JaccardLoss(per_image=False)
        self.lovasz = LovaszLoss(per_image=per_image)
        self.lovasz_sigmoid = LovaszLossSigmoid(per_image=per_image)
        self.focal = FocalLoss2d()
        self.mapping = {'bce': self.bce,
                        'dice': self.dice,
                        'focal': self.focal,
                        'jaccard': self.jaccard,
                        'lovasz': self.lovasz,
                        'lovasz_sigmoid': self.lovasz_sigmoid}
        self.expect_sigmoid = {'dice', 'focal', 'jaccard', 'lovasz_sigmoid'}
        self.values = {}

    def forward(self, outputs, targets):
        loss = 0
        weights = self.weights
        sigmoid_input = torch.sigmoid(outputs)
        for k, v in weights.items():
            if not v:
                continue
            val = self.mapping[k](sigmoid_input if k in self.expect_sigmoid else outputs, targets)
            self.values[k] = val
            loss += self.weights[k] * val
        return loss


def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts.float() - gt_sorted.float().cumsum(0)
    union = gts.float() + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1:  # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def lovasz_hinge(logits, labels, per_image=True, ignore=None):
    """
    Binary Lovasz hinge loss
      logits: [B, H, W] Variable, logits at each pixel (between -\infty and +\infty)
      labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
      per_image: compute the loss per image instead of per batch
      ignore: void class id
    """
    if per_image:
        loss = mean(lovasz_hinge_flat(*flatten_binary_scores(log.unsqueeze(0), lab.unsqueeze(0), ignore))
                    for log, lab in zip(logits, labels))
    else:
        loss = lovasz_hinge_flat(*flatten_binary_scores(logits, labels, ignore))
    return loss


def lovasz_hinge_flat(logits, labels):
    """
    Binary Lovasz hinge loss
      logits: [P] Variable, logits at each prediction (between -\infty and +\infty)
      labels: [P] Tensor, binary ground truth labels (0 or 1)
      ignore: label to ignore
    """
    if len(labels) == 0:
        # only void pixels, the gradients should be 0
        return logits.sum() * 0.
    signs = 2. * labels.float() - 1.
    errors = (1. - logits * Variable(signs))
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    perm = perm.data
    gt_sorted = labels[perm]
    grad = lovasz_grad(gt_sorted)
    loss = torch.dot(F.relu(errors_sorted), Variable(grad))
    return loss


def flatten_binary_scores(scores, labels, ignore=None):
    """
    Flattens predictions in the batch (binary case)
    Remove labels equal to 'ignore'
    """
    scores = scores.view(-1)
    labels = labels.view(-1)
    if ignore is None:
        return scores, labels
    valid = (labels != ignore)
    vscores = scores[valid]
    vlabels = labels[valid]
    return vscores, vlabels


def lovasz_sigmoid(probas, labels, per_image=False, ignore=None):
    """
    Multi-class Lovasz-Softmax loss
      probas: [B, C, H, W] Variable, class probabilities at each prediction (between 0 and 1)
      labels: [B, H, W] Tensor, ground truth labels (between 0 and C - 1)
      only_present: average only on classes present in ground truth
      per_image: compute the loss per image instead of per batch
      ignore: void class labels
    """
    if per_image:
        loss = mean(lovasz_sigmoid_flat(*flatten_binary_scores(prob.unsqueeze(0), lab.unsqueeze(0), ignore))
                          for prob, lab in zip(probas, labels))
    else:
        loss = lovasz_sigmoid_flat(*flatten_binary_scores(probas, labels, ignore))
    return loss


def lovasz_sigmoid_flat(probas, labels):
    """
    Multi-class Lovasz-Softmax loss
      probas: [P, C] Variable, class probabilities at each prediction (between 0 and 1)
      labels: [P] Tensor, ground truth labels (between 0 and C - 1)
      only_present: average only on classes present in ground truth
    """
    fg = labels.float()
    errors = (Variable(fg) - probas).abs()
    errors_sorted, perm = torch.sort(errors, 0, descending=True)
    perm = perm.data
    fg_sorted = fg[perm]
    loss = torch.dot(errors_sorted, Variable(lovasz_grad(fg_sorted)))
    return loss


def mean(l, ignore_nan=False, empty=0):
    """
    nanmean compatible with generators.
    """
    l = iter(l)
    if ignore_nan:
        l = ifilterfalse(np.isnan, l)
    try:
        n = 1
        acc = next(l)
    except StopIteration:
        if empty == 'raise':
            raise ValueError('Empty mean')
        return empty
    for n, v in enumerate(l, 2):
        acc += v
    if n == 1:
        return acc
    return acc / n


class LovaszLoss(nn.Module):
    def __init__(self, ignore_index=255, per_image=True):
        super().__init__()
        self.ignore_index = ignore_index
        self.per_image = per_image

    def forward(self, outputs, targets):
        outputs = outputs.contiguous()
        targets = targets.contiguous()
        return lovasz_hinge(outputs, targets, per_image=self.per_image, ignore=self.ignore_index)


class LovaszLossSigmoid(nn.Module):
    def __init__(self, ignore_index=255, per_image=True):
        super().__init__()
        self.ignore_index = ignore_index
        self.per_image = per_image

    def forward(self, outputs, targets):
        outputs = outputs.contiguous()
        targets = targets.contiguous()
        return lovasz_sigmoid(outputs, targets, per_image=self.per_image, ignore=self.ignore_index)


class FocalLoss2d(nn.Module):
    def __init__(self, gamma=2, ignore_index=255):
        super().__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, outputs, targets):
        outputs = outputs.contiguous()
        targets = targets.contiguous()
        # eps = 1e-8
        non_ignored = targets.view(-1) != self.ignore_index
        targets = targets.view(-1)[non_ignored].float()
        outputs = outputs.contiguous().view(-1)[non_ignored]
        outputs = torch.clamp(outputs, eps, 1. - eps)
        targets = torch.clamp(targets, eps, 1. - eps)
        pt = (1 - targets) * (1 - outputs) + targets * outputs
        return (-(1. - pt) ** self.gamma * torch.log(pt)).mean()


#===========fork from E:\TEST2020\DownLoadPrj\CD\xview2_solution-master\xview2_solution-master\losses.py==========
class StableBCELoss2(nn.Module):
    def __init__(self):
        super(StableBCELoss2, self).__init__()

    def forward(self, input, target):
        return bce_loss(input, target).mean()


def bce_loss(input, target):
    input = input.float().view(-1)
    target = target.float().view(-1)
    neg_abs = - input.abs()
    loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()
    return loss


def bce_loss_sigmoid(input, target):
    eps = 1e-6
    outputs = torch.clamp(input, eps, 1. - eps)
    targets = torch.clamp(target, eps, 1. - eps)
    pt = (1 - targets) * (1 - outputs) + targets * outputs
    return -torch.log(pt)

class FocalLossWithDice(nn.Module):
    def __init__(self, num_classes, ignore_index=255, gamma=2, ce_weight=1., d_weight=0.1, weight=None,
                 size_average=True, ohpm=False, ohpm_pixels=128 * 128):
        super().__init__()
        self.num_classes = num_classes
        self.d_weight = d_weight
        self.ce_w = ce_weight
        self.gamma = gamma
        if weight is not None:
            weight = torch.Tensor(weight).float()
        self.nll_loss = NLLLoss2d(weight, size_average, ignore_index=ignore_index)
        self.ignore_index = ignore_index
        self.ohpm = ohpm
        self.ohpm_pixels = ohpm_pixels

    def forward(self, outputs, targets):
        probas = F.softmax(outputs, dim=1)
        ce_loss = self.nll_loss((1 - probas) ** self.gamma * F.log_softmax(outputs, dim=1), targets)
        d_loss = soft_dice_loss_mc(outputs, targets, self.num_classes, ignore_index=self.ignore_index, ohpm=self.ohpm,
                                   ohpm_pixels=self.ohpm_pixels)
        non_ignored = targets != 255
        loc = soft_dice_loss(1 - probas[:, 0, ...][non_ignored], (targets[non_ignored] > 0) * 1.)

        return self.ce_w * ce_loss + self.d_weight * d_loss + self.d_weight * loc


def soft_dice_loss_mc(outputs, targets, num_classes, per_image=False, only_existing_classes=False, ignore_index=255,
                      minimum_class_pixels=10, reduce_batch=True, ohpm=True, ohpm_pixels=16384):
    batch_size = outputs.size()[0]
    eps = 1e-5
    outputs = F.softmax(outputs, dim=1)

    def _soft_dice_loss(outputs, targets):
        loss = 0
        non_empty_classes = 0
        for cls in range(1, num_classes):
            non_ignored = targets.view(-1) != ignore_index
            dice_target = (targets.view(-1)[non_ignored] == cls).float()
            dice_output = outputs[:, cls].contiguous().view(-1)[non_ignored]
            if ohpm:
                loss_b = torch.abs(dice_target - dice_output)
                px, indc = loss_b.topk(ohpm_pixels)
                dice_target = dice_target[indc]
                dice_output = dice_output[indc]

            intersection = (dice_output * dice_target).sum()
            if dice_target.sum() > minimum_class_pixels:
                union = dice_output.sum() + dice_target.sum() + eps
                loss += (1 - (2 * intersection + eps) / union)
                non_empty_classes += 1
        if only_existing_classes:
            loss /= (non_empty_classes + eps)
        else:
            loss /= (num_classes - 1)
        return loss

    if per_image:
        if reduce_batch:
            loss = 0
            for i in range(batch_size):
                loss += _soft_dice_loss(torch.unsqueeze(outputs[i], 0), torch.unsqueeze(targets[i], 0))
            loss /= batch_size
        else:
            loss = torch.Tensor(
                [_soft_dice_loss(torch.unsqueeze(outputs[i], 0), torch.unsqueeze(targets[i], 0)) for i in
                 range(batch_size)])
    else:
        loss = _soft_dice_loss(outputs, targets)

    return loss

#============================for label smooth========================================
#=====https://blog.csdn.net/Guo_Python/article/details/105953076?utm_medium=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-1.channel_param&depth_1-utm_source=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-1.channel_param

# import torch
# import torch.nn as nn
class LSR(nn.Module):

    def __init__(self, e=0.1, reduction='mean'):
        super().__init__()

        self.log_softmax = nn.LogSoftmax(dim=1)
        self.e = e
        self.reduction = reduction

    def _one_hot(self, labels, classes, value=1):
        """
            Convert labels to one hot vectors

        Args:
            labels: torch tensor in format [label1, label2, label3, ...]
            classes: int, number of classes
            value: label value in one hot vector, default to 1

        Returns:
            return one hot format labels in shape [batchsize, classes]
        """

        one_hot = torch.zeros(labels.size(0), classes)

        # labels and value_added  size must match
        labels = labels.view(labels.size(0), -1)
        value_added = torch.Tensor(labels.size(0), 1).fill_(value)
        #value_added = torch.FloatTensor(labels.size(0), 1).fill_(value)

        value_added = value_added.to(labels.device)
        one_hot = one_hot.to(labels.device)

        one_hot.scatter_add_(1, labels, value_added)

        return one_hot

    def _smooth_label(self, target, length, smooth_factor=0.1):
        """convert targets to one-hot format, and smooth
        them.
        Args:
            target: target in form with [label1, label2, label_batchsize]
            length: length of one-hot format(number of classes)
            smooth_factor: smooth factor for label smooth

        Returns:
            smoothed labels in one hot format
        """
        one_hot = self._one_hot(target, length, value=1 - smooth_factor)
        one_hot += smooth_factor / length

        return one_hot.to(target.device)

    def forward(self, x, target):

        if x.size(0) != target.size(0):
            raise ValueError('Expected input batchsize ({}) to match target batch_size({})'
                             .format(x.size(0), target.size(0)))

        if x.dim() < 2:
            raise ValueError('Expected input tensor to have least 2 dimensions(got {})'
                             .format(x.size(0)))

        if x.dim() != 2:
            raise ValueError('Only 2 dimension tensor are implemented, (got {})'
                             .format(x.size()))

        smoothed_target = self._smooth_label(target, x.size(1), self.e)
        x = self.log_softmax(x)
        loss = torch.sum(- x * smoothed_target, dim=1)

        if self.reduction == 'none':
            return loss

        elif self.reduction == 'sum':
            return torch.sum(loss)

        elif self.reduction == 'mean':
            return torch.mean(loss)

        else:
            raise ValueError('unrecognized option, expect reduction to be one of none, mean, sum')

#==========for label_smooth crossentropy loss=========================
class LabelSmoothLoss(nn.Module):
    def __init__(self, smoothing=0.0):
        super(LabelSmoothLoss, self).__init__()
        self.smoothing = smoothing


    def forward(self, input, target):
        log_prob = F.log_softmax(input, dim=-1)#[3,5]
        weight = input.new_ones(input.size()) * \
                 self.smoothing / (input.size(-1) - 1.)#[3,5] y = x.new_ones(3, 2) 所得到的y会保持 x 原有的属性，比如dtype
        weight.scatter_(-1, target.unsqueeze(-1), (1. - self.smoothing))#[3,5] scatter_(input, dim, index, src)将src中数据根据index中的索引按照dim的方向填进input中
        loss = (-weight * log_prob).sum(dim=-1).mean()
        return loss

#=========================================================abCELoss============================================
from models.Satt_CD.modules import  ramps
class abCE_loss(nn.Module):
    """
    Annealed-Bootstrapped cross-entropy loss
    假设太容易的样本无益于训练，因此只训练对当前模型有一定难度的样本
    https://blog.csdn.net/bea_tree/article/details/59480787
    https://zhuanlan.zhihu.com/p/86610982
    ramp_type='log_rampup'

    sup_loss = abCE_loss(iters_per_epoch=iter_per_epoch, epochs=config['trainer']['epochs'],
                                num_classes=val_loader.dataset.num_classes)

    """

    def __init__(self, iters_per_epoch, epochs, num_classes, weight=None,
                 reduction='mean', thresh=0.7, min_kept=1, ramp_type=None):
        super(abCE_loss, self).__init__()
        self.weight = torch.FloatTensor(weight) if weight is not None else weight
        self.reduction = reduction
        self.thresh = thresh
        self.min_kept = min_kept
        self.ramp_type = ramp_type

        if ramp_type is not None:
            self.rampup_func = getattr(ramps, ramp_type)
            self.iters_per_epoch = iters_per_epoch
            self.num_classes = num_classes
            self.start = 1 / num_classes
            self.end = 0.9
            self.total_num_iters = (epochs - (0.6 * epochs)) * iters_per_epoch

    def threshold(self, curr_iter):
        #cur_total_iter = self.iters_per_epoch * epoch + curr_iter
        current_rampup = self.rampup_func(curr_iter, self.total_num_iters)
        return current_rampup * (self.end - self.start) + self.start

    def forward(self, predict, target, ignore_index=255, curr_iter=100):
        batch_kept = self.min_kept * target.size(0)#1*4=4
        prob_out = F.softmax(predict, dim=1)#[4,7,32,32]
        tmp_target = target.clone()#[4,32,32]
        tmp_target[tmp_target == ignore_index] = 0
        '''
        gather在one-hot为输出的多分类问题中，可以把最大值坐标作为index传进去，然后提取到每一行的正确预测结果，这也是gather可能的一个作用。
        https://blog.csdn.net/edogawachia/article/details/80515038
        '''
        prob = prob_out.gather(1, tmp_target.unsqueeze(1))#[4,1,32,32] torch.gather(b, dim=1, index=index_1)
        mask = target.contiguous().view(-1, ) != ignore_index#[4096]
        sort_prob, sort_indices = prob.contiguous().view(-1, )[mask].contiguous().sort()#[4096],[4096] sort_prob is from small to large

        if self.ramp_type is not None:
            thresh = self.threshold(curr_iter=curr_iter)
        else:
            thresh = self.thresh#0.7

        min_threshold = sort_prob[min(batch_kept, sort_prob.numel() - 1)] if sort_prob.numel() > 0 else 0.0#0.1179
        threshold = max(min_threshold, thresh)#0.7
        loss_matrix = F.cross_entropy(predict, target,
                                      weight=self.weight.to(predict.device) if self.weight is not None else None,
                                      ignore_index=ignore_index, reduction='none')
        loss_matirx = loss_matrix.contiguous().view(-1, )#[4096]
        sort_loss_matirx = loss_matirx[mask][sort_indices]#[4096] sort from large to small
        select_loss_matrix = sort_loss_matirx[sort_prob < threshold]#delete the element that has the most certain prob_out, thus concentrating on the hard samples
        if self.reduction == 'sum' or select_loss_matrix.numel() == 0:
            return select_loss_matrix.sum()
        elif self.reduction == 'mean':
            return select_loss_matrix.mean()
        else:
            raise NotImplementedError('Reduction Error!')
