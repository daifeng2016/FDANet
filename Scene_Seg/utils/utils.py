#!/usr/bin/env python
#  -- coding: utf-8 --
"""
Copyright (c) 2018. All rights reserved.
Created by D. F. Peng on 2018/6/18
"""
import os
import shutil
import matplotlib.pyplot as plt
import numpy as np
import glob
from ptflops import get_model_complexity_info
import logging
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F

class Get_gradient_nopadding(nn.Module):
    def __init__(self):
        super(Get_gradient_nopadding, self).__init__()
        # kernel_v = [[0, -1, 0],
        #             [0, 0, 0],
        #             [0, 1, 0]]
        # kernel_h = [[0, 0, 0],
        #             [-1, 0, 1],
        #             [0, 0, 0]]

        # =======================sobel kernel===================
        kernel_v = [[-1, -2, -1],
                    [0, 0, 0],
                    [1, 2, 1]]
        kernel_h = [[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]]


        kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0).cuda()
        kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0).cuda()
        self.weight_h = nn.Parameter(data=kernel_h, requires_grad=False)

        self.weight_v = nn.Parameter(data=kernel_v, requires_grad=False)

    def forward(self, x):
        x_list = []
        for i in range(x.shape[1]):  # ==>channel num
            x_i = x[:, i]  # [2,32,32]
            x_i_v = F.conv2d(x_i.unsqueeze(1), self.weight_v, padding=1) # [2,1,32,32]  padding=2==>[2,1,34,34]
            x_i_h = F.conv2d(x_i.unsqueeze(1), self.weight_h, padding=1)
            x_i = torch.sqrt(torch.pow(x_i_v, 2) + torch.pow(x_i_h, 2) + 1e-6)  # [2,1,32,32]
            x_list.append(x_i)

        x_grad = torch.cat(x_list, dim=1)  # 3[2,1,32,32]==>[2,3,32,32]

        return x_grad
def weightmap(pred1, pred2):
    # output = 1.0 - torch.sum((pred1 * pred2), 1).view(1, 1, pred1.size(2), pred1.size(3)) / \
    # (torch.norm(pred1, 2, 1) * torch.norm(pred2, 2, 1)).view(1, 1, pred1.size(2), pred1.size(3))#only work for batchsize=1
    #torch.sum(a*b,[2,3]) torch.norm(torch.norm(b,2,-1),2,-1)
    # b=pred1.size(0)
    # output = 1.0 - torch.sum((pred1 * pred2), 1).view(b, 1, pred1.size(2), pred1.size(3)) / \
    #          (torch.norm(pred1, 2, 1) * torch.norm(pred2, 2, 1)).view(b, 1, pred1.size(2), pred1.size(3))
    output=1.0-torch.sum(pred1*pred2,[2,3])/(torch.norm(torch.norm(pred1,2,-1),2,-1)*torch.norm(torch.norm(pred2,2,-1),2,-1))
    return output#for computing weight for loss function of D



def PR_score_whole(y_true, y_pred):
    smooth = 1.0


    y_true = np.ravel(y_true)  # 1-dim vector [1825,256,256]==>[119603200]
    y_pred = np.ravel(y_pred)  # 1-dim vector [1825,256,256]==>[119603200]


    c1 = np.sum(np.round(
        np.clip(y_true * y_pred, 0, 1)))  # the clip opetration make sure y_pred y_true and y_true * y_pred are [0,1]
    c2 = np.sum(np.round(np.clip(y_pred, 0, 1)))
    c3 = np.sum(np.round(np.clip(y_true, 0, 1)))
    # =======================================================

    # How many selected items are relevant?
    precision = c1 * 1.0 / c2
    # How many relevant items are selected?
    recall = c1 * 1.0 / c3
    # Calculate f1_score
    eps=1e-7#not 1.0, otherwise cause acc error  or use
    f1_score = (2 * precision * recall) / (precision + recall+eps)  #
    #f1_score = (2 * precision * recall+1.0) / (precision + recall +1.0)#smooth=1.0
    y_true = np.array(y_true)
    y_pred = np.array(y_pred).astype('float')  # must be float
    cond1 = (y_true == 1)  # 119603200 [True,False,...]
    cond2 = (y_pred > 0.5)
    cond3 = (y_true == 0)
    cond4 = (y_pred < 0.5)
    idx_TP = np.where(cond1 & cond2)[0]  # not np.where( a and b) np.where(cond1&cond2)==>tuple
    idx_FP = np.where(cond3 & cond2)[0]
    idx_FN = np.where(cond1 & cond4)[0]
    idx_TN = np.where(cond3 & cond4)[0]
    # pix_number = (y_pred.shape[0] * y_pred.shape[1])
    pix_number = y_pred.shape[0]
    acc = (len(idx_TP) + len(idx_TN)) * 1.0 / pix_number

    nTPNum = len(idx_TP)
    nFPNum = len(idx_FP)
    nFNNum = len(idx_FN)
    nTNNum = len(idx_TN)
    temp1 = ((nTPNum + nFPNum) / 1e5) * ((nTPNum + nFNNum) / 1e5)
    temp2 = ((nFNNum + nTNNum) / 1e5) * ((nFPNum + nTNNum) / 1e5)
    temp3 = (pix_number / 1e5) * (pix_number / 1e5)
    dPre = (temp1 + temp2) * 1.0 / temp3
    kappa = (acc - dPre) / (1 - dPre)

    return precision, recall, f1_score, acc, kappa
def print_model_parm_nums(model):            #得到模型参数总量

    total = sum([param.nelement() for param in model.parameters()])
    print('  + Number of params: %.2fM' % (total / 1e6))     #每一百万为一个单位
    return total/1e6

def print_model_para(save_path,net):
    import sys
    mylog = open(save_path, 'w')
    stdout_backup = sys.stdout
    sys.stdout = mylog


    macs, params = get_model_complexity_info(net, (3, 256, 256),
                                             as_strings=True,
                                             print_per_layer_stat=True,
                                             ost=mylog)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    mylog.close()
    sys.stdout = stdout_backup

def get_timestamp():
    return datetime.now().strftime('%y%m%d-%H%M%S')
def setup_logger(logger_name, root, phase, level=logging.INFO, screen=False):
    '''set up logger'''
    l = logging.getLogger(logger_name)
    formatter = logging.Formatter(
        '%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s', datefmt='%y-%m-%d %H:%M:%S')
    log_file = os.path.join(root, phase + '_{}.log'.format(get_timestamp()))

    fh = logging.FileHandler(log_file, mode='w')
    fh.setFormatter(formatter)
    l.setLevel(level)
    l.addHandler(fh)
    if screen:
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        l.addHandler(sh)





#=============================for PR curve and F-score curve===============================
lineSylClr = ['r--', 'b--']  # curve style, same size with rs_dirs
linewidth = [1.5, 1.5]  # line width, same size with rs_dirs
def draw_PR_curve(data_dir,gt_dir,rs_dirs):
    ## 0. =======set the data path=======
    print("------0. set the data path------")

    # >>>>>>> Follows have to be manually configured <<<<<<< #
    data_name = 'WHU Building Dataset'  # this will be drawn on the bottom center of the figures
    #data_dir = './test_data/'  # set the data directory,
    # ground truth and results to-be-evaluated should be in this directory
    # the figures of PR and F-measure curves will be saved in this directory as well
    #gt_dir = 'gt'  # set the ground truth folder name
    #rs_dirs = ['rs1', 'rs2']  # set the folder names of different methods


    gt_name_list = glob.glob(data_dir + '/'+gt_dir + '/' + '*.tif')  # get the ground truth file name list

    ## get directory list of predicted maps
    rs_dir_lists = []
    for i in range(len(rs_dirs)):
        #rs_dir_lists.append(data_dir + '/'+rs_dirs[i] + '/remove_hole_area/')
        rs_dir_lists.append(data_dir + '/' + rs_dirs[i]+'/')
    print('\n')

    ## 2. =======compute the Precision, Recall and F-measure of methods=========
    from utils.measures import compute_PRE_REC_FM_of_methods, plot_save_pr_curves, plot_save_fm_curves

    print('\n')
    print("------2. Compute the Precision, Recall and F-measure of Methods------")
    PRE, REC, FM, gt2rs_fm = compute_PRE_REC_FM_of_methods(gt_name_list, rs_dir_lists, beta=1.0)
    for i in range(0, FM.shape[0]):
        print(">>", rs_dirs[i], ":", "num_rs/num_gt-> %d/%d," % (int(gt2rs_fm[i][0]), len(gt_name_list)),
              "maxF->%.3f, " % (np.max(FM, 1)[i]), "meanF->%.3f, " % (np.mean(FM, 1)[i]))
    print('\n')
    save_pr_curves_path=save_path=data_dir+'/'+rs_dirs[0]+'_'+rs_dirs[1]+"_pr_curves.png"
    save_fm_curves_path = save_path = data_dir + '/' + rs_dirs[0] + '_' + rs_dirs[1] + "_fm_curves.png"
    ## 3. =======Plot and save precision-recall curves=========
    print("------ 3. Plot and save precision-recall curves------")
    plot_save_pr_curves(PRE,  # numpy array (num_rs_dir,255), num_rs_dir curves will be drawn
                        REC,  # numpy array (num_rs_dir,255)
                        method_names=rs_dirs,  # method names, shape (num_rs_dir), will be included in the figure legend
                        lineSylClr=lineSylClr,  # curve styles, shape (num_rs_dir)
                        linewidth=linewidth,  # curve width, shape (num_rs_dir)
                        xrange=(0.5, 1.0),  # the showing range of x-axis
                        yrange=(0.5, 1.0),  # the showing range of y-axis
                        dataset_name=data_name,  # dataset name will be drawn on the bottom center position
                        save_dir=save_pr_curves_path,  # figure save directory
                        save_fmt='png')  # format of the to-be-saved figure
    print('\n')

    ## 4. =======Plot and save F-measure curves=========
    print("------ 4. Plot and save F-measure curves------")
    plot_save_fm_curves(FM,  # numpy array (num_rs_dir,255), num_rs_dir curves will be drawn
                        mybins=np.arange(0, 256),
                        method_names=rs_dirs,  # method names, shape (num_rs_dir), will be included in the figure legend
                        lineSylClr=lineSylClr,  # curve styles, shape (num_rs_dir)
                        linewidth=linewidth,  # curve width, shape (num_rs_dir)
                        xrange=(0.0, 1.0),  # the showing range of x-axis
                        yrange=(0.0, 1.0),  # the showing range of y-axis
                        dataset_name=data_name,  # dataset name will be drawn on the bottom center position
                        save_dir=save_fm_curves_path,  # figure save directory
                        save_fmt='png')  # format of the to-be-saved figure
    print('\n')

    print('Done!!!')


def poly_lr_scheduler(optimizer, init_lr, iter, lr_decay_iter=1, max_iter=20000, power=0.9):
    """Polynomial decay of learning rate
        :param init_lr is base learning rate
        :param iter is a current iteration
        :param lr_decay_iter how frequently decay occurs, default is 1
        :param max_iter is number of maximum iterations
        :param power is a polymomial power
    """
    if iter % lr_decay_iter or iter > max_iter:
        return optimizer

    for param_group in optimizer.param_groups:
        param_group['lr'] = init_lr*(1 - float(iter)/max_iter)**power
    return optimizer





def mkdir_if_not_exist(dir, is_delete=False):
    """
    创建文件夹
    :param dirs: 文件夹列表
    :param is_delete: 是否删除
    :return: 是否成功
    """
    try:
        if is_delete:
            if os.path.exists(dir):
                shutil.rmtree(dir)
                print (u'[INFO] 文件夹 "%s" 存在, 删除文件夹.' % dir)

        if not os.path.exists(dir):
            os.makedirs(dir)
            print (u'[INFO] 文件夹 "%s" 不存在, 创建文件夹.' % dir)
        return True
    except Exception as e:
        print ('[Exception] %s' % e)
        return False

def smooth_curve(points, factor=0.8):
        smoothed_points = []
        for point in points:
            if smoothed_points:
                previous = smoothed_points[-1]
                smoothed_points.append(previous * factor + point * (1 - factor))
            else:
                smoothed_points.append(point)
        return smoothed_points
def visualize_train(history,save_path, mode=2):
        acc = history.history['acc']
        val_acc = history.history['val_acc']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        if mode == 1:
            # =======================mode1=========================
            epochs = range(1, len(acc) + 1)
            plt.plot(epochs, acc, 'bo', label='Training acc')
            plt.plot(epochs, val_acc, 'b', label='Validation acc')
            plt.title('Training and validation accuracy')
            plt.legend()
            plt.figure()
            plt.plot(epochs, loss, 'bo', label='Training loss')
            plt.plot(epochs, val_loss, 'b', label='Validation loss')
            plt.title('Training and validation loss')
            plt.legend()
            plt.savefig(save_path+'m1_acc_loss.png')
            plt.show()
        elif mode == 2:
            # ========================model2=======================
            plt.plot(acc)
            plt.plot(val_acc)
            plt.title('model accuracy')
            plt.ylabel('accuracy')
            plt.xlabel('epoch')
            plt.legend(['train', 'test'], loc='upper left')
            plt.show()
            # ====for loss====
            plt.plot(loss)
            plt.plot(val_loss)
            plt.title('model loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['train', 'test'], loc='upper right')
            plt.show()
        elif mode == 3:
            # ========================mode 3====================
            plt.subplot(121)
            plt.plot(acc)
            plt.plot(val_acc)
            plt.title('model accuracy')
            plt.ylabel('accuracy')
            plt.xlabel('epoch')
            plt.legend(['train', 'test'], loc='upper left')
            plt.subplot(122)
            plt.plot(loss)
            plt.plot(val_loss)
            plt.title('model loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['train', 'test'], loc='upper right')

            #loss_path=self.config.img_dir + 'm3_acc_loss'+str(self.config.patch_size)+'.png'
            plt.savefig(save_path)
            plt.show()
        else:# using smoothed_points
            # ========================mode 4====================

            plt.subplot(121)
            plt.plot(smooth_curve(acc))
            plt.plot(smooth_curve(val_acc))
            plt.title('model accuracy')
            plt.ylabel('accuracy')
            plt.xlabel('epoch')
            plt.legend(['train', 'test'], loc='upper left')
            plt.subplot(122)
            plt.plot(smooth_curve(loss))
            plt.plot(smooth_curve(val_loss))
            plt.title('model loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['train', 'test'], loc='upper right')
            plt.show()

        return 'visualizing over...'
