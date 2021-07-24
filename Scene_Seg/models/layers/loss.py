import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
from torch.autograd import Function, Variable

class CrossEntropy2d(nn.Module):

    def __init__(self, size_average=True, ignore_label=255):
        super(CrossEntropy2d, self).__init__()
        self.size_average = size_average
        self.ignore_label = ignore_label

    def forward(self, predict, target, weight=None):
        """
            Args:
                predict:(n, c, h, w)
                target:(n, h, w)
                weight (Tensor, optional): a manual rescaling weight given to each class.
                                           If given, has to be a Tensor of size "nclasses"
        """
        assert not target.requires_grad
        assert predict.dim() == 4
        assert target.dim() == 3#[4,321,321]
        assert predict.size(0) == target.size(0), "{0} vs {1} ".format(predict.size(0), target.size(0))
        assert predict.size(2) == target.size(1), "{0} vs {1} ".format(predict.size(2), target.size(1))
        assert predict.size(3) == target.size(2), "{0} vs {1} ".format(predict.size(3), target.size(3))
        n, c, h, w = predict.size()  # predict=[4,21,321,321]
        # ==================================for weighted loss 直接使用此类weight效果很差======================================
        #log_p = predict.transpose(1, 2).transpose(2, 3).contiguous().view(1, -1)  # [1,1,384,209]==>[1,80256]
        # target_t = target.view(1, -1)
        # target_trans = target_t.clone()
        # pos_index = (target_t > 0)  # [1,80256]  dtype=torch.uint8 >0处为1，其他位置为0
        # neg_index = (target_t == 0)  # [1,80256]
        # target_trans[pos_index] = 1
        # target_trans[neg_index] = 0
        # pos_index = pos_index.data.cpu().numpy().astype(bool)
        # neg_index = neg_index.data.cpu().numpy().astype(bool)  # 转换为bool后统计正负样本值
        # weight = torch.Tensor(c).fill_(0)  # [1,80256]
        # weight = weight.numpy()
        # pos_num = pos_index.sum()  # 13061
        # neg_num = neg_index.sum()  # 67195
        # sum_num = pos_num + neg_num
        # weight[0] = 1.0-neg_num * 1.0 / sum_num
        # weight[1] = 1.0-pos_num * 1.0 / sum_num
        # weight = torch.from_numpy(weight)
        # weight = weight.cuda()
        # ==============================================================
        target_mask = (target >= 0) * (target != self.ignore_label)#[4,321,321] target>=0 ==>satisfy==1 or 0
        target = target[target_mask]#[412164]
        if not target.data.dim():
            return Variable(torch.zeros(1))
        predict = predict.transpose(1, 2).transpose(2, 3).contiguous()#[4,21,321,321]==>[4,321,21,321]==>[4,321,321,21] [8655444]
        predict = predict[target_mask.view(n, h, w, 1).repeat(1, 1, 1, c)].view(-1, c)#target_mask.view(n, h, w, 1): [4,321,321]==>[4,321,321,1] ==>[4,321,321,21]==>[8655444]==>[412164,21]
        loss = F.cross_entropy(predict, target, weight=weight, size_average=self.size_average)

        return loss

class CrossEntropy2dWeighted(nn.Module):

    def __init__(self, size_average=True, ignore_label=255):
        super(CrossEntropy2dWeighted, self).__init__()
        self.size_average = size_average
        self.ignore_label = ignore_label

    def forward(self, predict, target, weight=None):
        """
            Args:
                predict:(n, c, h, w)
                target:(n, h, w)
                weight (Tensor, optional): a manual rescaling weight given to each class.
                                           If given, has to be a Tensor of size "nclasses"
        """
        assert not target.requires_grad
        assert predict.dim() == 4
        assert target.dim() == 3#[4,321,321]
        assert predict.size(0) == target.size(0), "{0} vs {1} ".format(predict.size(0), target.size(0))
        assert predict.size(2) == target.size(1), "{0} vs {1} ".format(predict.size(2), target.size(1))
        assert predict.size(3) == target.size(2), "{0} vs {1} ".format(predict.size(3), target.size(3))
        n, c, h, w = predict.size()  # predict=[4,21,321,321]
        # ==================================for weighted loss 直接使用此类weight效果很差======================================
        log_p = predict.transpose(1, 2).transpose(2, 3).contiguous().view(1, -1)  # [3,2,128,128]==>[1,98304]
        target_t = target.view(1, -1)## [3,128,128]=[1,49152]
        target_trans = target_t.clone()
        pos_index = (target_t > 0)  # [1,80256]  dtype=torch.uint8 >0处为1，其他位置为0
        neg_index = (target_t == 0)  # [1,80256]
        target_trans[pos_index] = 1
        target_trans[neg_index] = 0
        pos_index = pos_index.data.cpu().numpy().astype(bool)
        neg_index = neg_index.data.cpu().numpy().astype(bool)  # 转换为bool后统计正负样本值
        weight = torch.Tensor(log_p.size()).fill_(0)  # [1,80256]
        weight = weight.numpy()
        pos_num = pos_index.sum()  # 13061
        neg_num = neg_index.sum()  # 67195
        sum_num = pos_num + neg_num
        weight[pos_index] = neg_num * 1.0 / sum_num
        weight[neg_index] = pos_num * 1.0 / sum_num
        weight = torch.from_numpy(weight)
        weight = weight.cuda()
        # ==============================================================
        target_mask = (target >= 0) * (target != self.ignore_label)#[4,321,321] target>=0 ==>satisfy==1 or 0
        target = target[target_mask]#[412164]
        if not target.data.dim():
            return Variable(torch.zeros(1))
        predict = predict.transpose(1, 2).transpose(2, 3).contiguous()#[4,21,321,321]==>[4,321,21,321]==>[4,321,321,21] [8655444]
        predict = predict[target_mask.view(n, h, w, 1).repeat(1, 1, 1, c)].view(-1, c)#target_mask.view(n, h, w, 1): [4,321,321]==>[4,321,321,1] ==>[4,321,321,21]==>[8655444]==>[412164,21]
        #loss = F.cross_entropy(predict, target, weight=weight, size_average=self.size_average)
        t=onehot_embedding(target.data.cpu(),c)
        t=Variable(t).cuda()
        loss=F.binary_cross_entropy(predict,t,weight)

        return loss



class BCEWithLogitsLoss2d(nn.Module):

    def __init__(self, size_average=True, ignore_label=255):
        super(BCEWithLogitsLoss2d, self).__init__()
        self.size_average = size_average
        self.ignore_label = ignore_label

    def forward(self, predict, target, weight=None):
        """
            Args:
                predict:(n, 1, h, w)
                target:(n, 1, h, w)
                weight (Tensor, optional): a manual rescaling weight given to each class.
                                           If given, has to be a Tensor of size "nclasses"
        """
        assert not target.requires_grad
        assert predict.dim() == 4
        assert target.dim() == 4
        assert predict.size(0) == target.size(0), "{0} vs {1} ".format(predict.size(0), target.size(0))
        assert predict.size(2) == target.size(2), "{0} vs {1} ".format(predict.size(2), target.size(2))
        assert predict.size(3) == target.size(3), "{0} vs {1} ".format(predict.size(3), target.size(3))
        n, c, h, w = predict.size()
        target_mask = (target >= 0) * (target != self.ignore_label)#[4,1,321,321]*[4,1,321,321]=[4,1,321,321] ignore_label=255
        target = target[target_mask]
        if not target.data.dim():#=4
            return Variable(torch.zeros(1))
        predict = predict[target_mask]#use target_mask to generate mask of predict and target
        loss = F.binary_cross_entropy_with_logits(predict, target, weight=weight, size_average=self.size_average)
        return loss

class BCELoss2d(nn.Module):

    def __init__(self, size_average=True, ignore_label=255):
        super(BCELoss2d, self).__init__()
        self.size_average = size_average
        self.ignore_label = ignore_label

    def forward(self, predict, target, weight=None):
        """
            Args:
                predict:(n, 1, h, w)
                target:(n, 1, h, w)
                weight (Tensor, optional): a manual rescaling weight given to each class.
                                           If given, has to be a Tensor of size "nclasses"
        """
        assert not target.requires_grad
        assert predict.dim() == 4
        assert target.dim() == 4
        assert predict.size(0) == target.size(0), "{0} vs {1} ".format(predict.size(0), target.size(0))
        assert predict.size(2) == target.size(2), "{0} vs {1} ".format(predict.size(2), target.size(2))
        assert predict.size(3) == target.size(3), "{0} vs {1} ".format(predict.size(3), target.size(3))
        n, c, h, w = predict.size()
        target_mask = (target >= 0) * (target != self.ignore_label)#[4,1,321,321]*[4,1,321,321]=[4,1,321,321] ignore_label=255
        target = target[target_mask]
        if not target.data.dim():#=4
            return Variable(torch.zeros(1))
        predict = predict[target_mask]#use target_mask to generate mask of predict and target
        loss = F.binary_cross_entropy(predict, target, weight=weight, size_average=self.size_average)
        return loss


def cross_entropy_2D(input, target, weight=None, size_average=True):
    n, c, h, w = input.size()
    log_p = F.log_softmax(input, dim=1)
    log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    target = target.view(target.numel())
    loss = F.nll_loss(log_p, target, weight=weight, size_average=False)
    if size_average:
        loss /= float(target.numel())
    return loss


def cross_entropy_3D(input, target, weight=None, size_average=True):
    n, c, h, w, s = input.size()
    log_p = F.log_softmax(input, dim=1)
    log_p = log_p.transpose(1, 2).transpose(2, 3).transpose(3, 4).contiguous().view(-1, c)
    target = target.view(target.numel())
    loss = F.nll_loss(log_p, target, weight=weight, size_average=False)
    if size_average:
        loss /= float(target.numel())
    return loss


class SoftDiceLoss(nn.Module):

    #dice_loss=1-2\xUY\/(X+Y)


    def __init__(self, n_classes):
        super(SoftDiceLoss, self).__init__()
        self.one_hot_encoder = One_Hot(n_classes).forward
        self.n_classes = n_classes

    def forward(self, input, target):
        smooth = 0.01
        batch_size = input.size(0)

        input = F.softmax(input, dim=1).view(batch_size, self.n_classes, -1)#[N,C,W*H]
        target = self.one_hot_encoder(target).contiguous().view(batch_size, self.n_classes, -1)##[N,C,W*H]

        inter = torch.sum(input * target, 2) + smooth##[N,C]
        union = torch.sum(input, 2) + torch.sum(target, 2) + smooth

        score = torch.sum(2.0 * inter / union)
        score = 1.0 - score / (float(batch_size) * float(self.n_classes))

        return score

class CustomSoftDiceLoss(nn.Module):
    def __init__(self, n_classes, class_ids):
        super(CustomSoftDiceLoss, self).__init__()
        self.one_hot_encoder = One_Hot(n_classes).forward
        self.n_classes = n_classes
        self.class_ids = class_ids

    def forward(self, input, target):
        smooth = 0.01
        batch_size = input.size(0)

        input = F.softmax(input[:,self.class_ids], dim=1).view(batch_size, len(self.class_ids), -1)
        target = self.one_hot_encoder(target).contiguous().view(batch_size, self.n_classes, -1)
        target = target[:, self.class_ids, :]

        inter = torch.sum(input * target, 2) + smooth
        union = torch.sum(input, 2) + torch.sum(target, 2) + smooth

        score = torch.sum(2.0 * inter / union)
        score = 1.0 - score / (float(batch_size) * float(self.n_classes))

        return score


class One_Hot(nn.Module):
    def __init__(self, depth):
        super(One_Hot, self).__init__()
        self.depth = depth
        self.ones = torch.sparse.torch.eye(depth).cuda()

    def forward(self, X_in):
        n_dim = X_in.dim()
        output_size = X_in.size() + torch.Size([self.depth])
        num_element = X_in.numel()
        X_in = X_in.data.long().view(num_element)
        out = Variable(self.ones.index_select(0, X_in)).view(output_size)
        return out.permute(0, -1, *range(1, n_dim)).squeeze(dim=2).float()

    def __repr__(self):
        return self.__class__.__name__ + "({})".format(self.depth)


def onehot_embedding(labels,num_classes):
    N = labels.size(0)
    D = num_classes
    y = torch.zeros(N,D)
    y[torch.arange(0,N).long(),labels] = 1
    return y

def focal_loss2d(input, target, start_cls_index=0,size_average=True):
    '''
    https://www.cnblogs.com/king-lps/p/9497836.html
    https://blog.csdn.net/qq_34914551/article/details/101644942 Balance binary cross entropy

    :param input:
    :param target:
    :param start_cls_index:
    :param size_average:
    :return:
    '''
    n, c, h, w = input.size()
    p = F.softmax(input) #[3,2,128,128]
    p = p.transpose(1, 2).transpose(2, 3).contiguous()#[3,128,128,2]
    p = p[target.view(n, h, w, 1).repeat(1, 1, 1, c) >= start_cls_index] #[98304]  exclude background example
    p = p.view(-1, c)#[49152,2]

    mask = target >=start_cls_index #[3,128,128] exclude background example
    target = target[mask]#[3,128,128]==>[45912]

    t = onehot_embedding(target.data.cpu(),c)
    t = Variable(t).cuda()
    #==============for cpu test========================
    t = onehot_embedding(target.data, c)#[45912,2]
    t = Variable(t)

    alpha = 0.25#平衡正负样本本身的比例不均
    gamma = 2#解决简单与困难样本的问题
    w = alpha* t + (1-alpha)*(1-t)#[45912,2]
    w = w * (1-p).pow(gamma)

    loss = F.binary_cross_entropy(p,t,w,size_average=False)#make sure p,t,w are of the same dim

    if size_average:
       loss /= mask.data.sum()
    return loss

def focal_loss2dWeighted(input, target, start_cls_index=0,size_average=True):
    '''
    https://www.cnblogs.com/king-lps/p/9497836.html
    https://blog.csdn.net/qq_34914551/article/details/101644942 Balance binary cross entropy

    :param input:
    :param target:
    :param start_cls_index:
    :param size_average:
    :return:
    '''
    assert input.dim()==4,"prediction dim must be 4"
    assert target.dim()==3,"target dim must be 3"
    n, c, h, w = input.size()
    p = F.softmax(input) #[3,2,128,128]
    p = p.transpose(1, 2).transpose(2, 3).contiguous()#[3,128,128,2]
    p = p[target.view(n, h, w, 1).repeat(1, 1, 1, c) >= start_cls_index] #[98304]  exclude background example
    p = p.view(-1, c)#[49152,2]

    mask = target >=start_cls_index #[3,128,128] exclude background example
    target = target[mask]#[3,128,128]==>[45912]

    t = onehot_embedding(target.data.cpu(),c)
    t = Variable(t).cuda()
    #==============for cpu test========================
    t = onehot_embedding(target.data, c)#[45912,2]
    t = Variable(t)

    alpha = 0.25#平衡正负样本本身的比例不均,此时alpha为一常数，
    gamma = 2#解决简单与困难样本的问题
    #w = alpha* t + (1-alpha)*(1-t)#[45912,2]

    pos = (target== 1).float()  # [3,2,128,128]
    neg = (target== 0).float()  # [3,2,128,128]
    num_pos = torch.sum(pos)  # 49321
    num_neg = torch.sum(neg)  # 48983
    num_total = num_pos + num_neg  # 98304
    alpha_pos = num_neg / num_total  # 0.4983
    alpha_neg = num_pos / num_total  # 0.5017
    #w = alpha_pos * pos + alpha_neg * neg


    w = w * (1-p).pow(gamma)

    loss = F.binary_cross_entropy(p,t,w,size_average=False)#make sure p,t,w are of the same dim

    if size_average:
       loss /= mask.data.sum()

    return loss

def BCE2dWeighted(pred, gt):
    '''
     for multi-label classification, gt must convert to one-hot encoder, so for each channel it works like binary-cross-entropy with
     single channel tensor
    :param pred:
    :param gt:only work for bianry output
    :return:
    '''
    assert pred.dim()==gt.dim()
    pos = (gt == 1).float()#[3,2,128,128]
    neg= (gt == 0).float()#[3,2,128,128]
    num_pos = torch.sum(pos)#49321
    num_neg = torch.sum(neg)#48983
    num_total = num_pos + num_neg#98304
    alpha_pos = num_neg / num_total#0.4983
    alpha_neg = num_pos / num_total#0.5017
    weights = alpha_pos * pos + alpha_neg * neg
    return F.binary_cross_entropy_with_logits(pred, target, weights)



def bin_clsloss(input, target, size_average=True):
    n, c = input.size()
    p = input
    target_emdding=torch.zeros((n,c))
    for i in range(n):
        nclasses = set(target.data.cpu().numpy()[i].flat)
        for nclass in nclasses:
            target_emdding[i][nclass]=1.0

    mask = target >= 0

    t = target_emdding[:,1:] #exclude background
    t = Variable(t).cuda()

    p = p[:,1:] #exclude background
    p = F.sigmoid(p) #binaray cls

    loss = F.binary_cross_entropy(p,t,size_average=size_average)

    return loss


if __name__ == '__main__':
    from torch.autograd import Variable
    depth=3
    batch_size=2
    # encoder = One_Hot(depth=depth).forward
    # y = Variable(torch.LongTensor(batch_size, 1, 1, 2 ,2).random_() % depth).cuda()  # 4 classes,1x3x3 img
    # y_onehot = encoder(y)
    # x = Variable(torch.randn(y_onehot.size()).float()).cuda()
    # dicemetric = SoftDiceLoss(n_classes=depth)
    # dicemetric(x,y)
    predict=torch.randn((3,2,128,128))
    target=torch.empty(3,128,128).random_(2).long()
    # predict1=torch.rand((3,2))
    # predict1_soft=F.softmax(predict1,dim=0)#default dim=-1
    # print(predict1)
    # print(predict1_soft)

    #print(target.size())
    loss_func=CrossEntropy2dWeighted()
    focal_loss=loss_func(predict,target)
    print(focal_loss.item())