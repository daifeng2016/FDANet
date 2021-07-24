import os
import logging
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.optim import lr_scheduler
#import kornia
from . import networks as networks
from .base_model import BaseModel
from models.Satt_CD.modules.loss import GANLoss, GradientPenaltyLoss
import itertools
logger = logging.getLogger('base')

import torch.nn.functional as F
import pdb
from losses.myLoss import bce_edge_loss
from torch.autograd import Variable
import numpy as np
from math import exp
from models.Satt_CD.modules import block as B
import albumentations as A
from models.Satt_CD.modules.utils import set_requires_grad
from models.Satt_CD.modules import loss
from models.Satt_CD.modules.AdamW2 import AdamW

class BRSeg_Model(BaseModel):
    def __init__(self, opt):
        super(BRSeg_Model, self).__init__(opt)
        train_opt = opt['train']
        # self.device='cpu'
        #=============define networks and load pretrained models==================
        # if train_opt["use_encoder_decoder"]:
        #     self.netEnc,self.netDec= networks.define_G(opt).to(self.device)  # G
        # else:
        self.netG = networks.define_G(opt).to(self.device)  # G (can has encoder, main_decoder,aux_decoder if SSL setting )

        #====================for deeplab========================
        self.use_DS=opt["network_G"]['use_DS']
        self.use_scaleRef=train_opt['use_scaleRef']
        self.use_scaleATT=train_opt['use_scaleATT']
        self.netD_seg=None
        self.netD_fea=None
        self.netG_AdaIN=None
        if train_opt['is_adv_train']:
            self.netD_seg = networks.define_D(opt).to(self.device)  # D
            self.netD_fea = networks.define_D_grad(opt).to(self.device)  # D_grad

            self.netD_fea.train()
            self.netD_seg.train()
            #self.netD_MS.train()

        if train_opt["use_SSL"]:
            self.cri_seg_loss = bce_edge_loss(use_mask=True,gamma=train_opt["ssl_gamma"]).to(self.device)#train_opt["ssl_gamma"] must be set to 0 when trainingwith single model
        else:
            self.cri_seg_loss = bce_edge_loss(use_edge=False).to(self.device)

        self.cri_seg0_loss=bce_edge_loss(use_edge=False).to(self.device)
        self.kl_loss = nn.KLDivLoss(reduction='mean')
        #self.cri_seg_loss = nn.BCELoss().to(self.device)
        self.D_update_ratio = train_opt['D_update_ratio'] if train_opt['D_update_ratio'] else 1  # 1
        self.D_init_iters = train_opt['D_init_iters'] if train_opt['D_init_iters'] else 0  # 0
        self.G_update_ratio = train_opt['G_update_ratio'] if train_opt['G_update_ratio'] else 1  # 1
        # optimizers
        #===================================G=================================
        wd_G = train_opt['weight_decay_G'] if train_opt['weight_decay_G'] else 0
        optim_params = []

        if self.netG:
            for k, v in self.netG.named_parameters():  # optimize part of the model

                if v.requires_grad:
                    optim_params.append(v)  # list, without name
                else:
                    logger.warning('Params [{:s}] will not optimize.'.format(k))
                    #torch.optim.AdamW#pytorch has its own AdamW
            if train_opt["opti_type"] == "Adam":
                logger.info('using Adam for G with lr_D_{:.4e}'.format(train_opt['lr_G']))
                self.optimizer_G = torch.optim.Adam(optim_params, lr=train_opt['lr_G'], \
                                                    weight_decay=wd_G, betas=(
                        train_opt['beta1_G'], 0.999))  # for Adam, no need to use weight_decay
            else:
                logger.info('using AdamW for G with lr_D_{:.4e}'.format(train_opt['lr_G']))
                self.optimizer_G = AdamW(optim_params, lr=train_opt['lr_G'], \
                                                weight_decay=wd_G)
            # self.optimizer_G = torch.optim.SGD(optim_params,
            #                       lr=train_opt['lr_G'], momentum=train_opt["momentum"], weight_decay=wd_G)#wd_G set to 1e-6
            self.optimizers.append(self.optimizer_G)  # self.netG.parameters()





        if opt["mean_teacher"]["use_stu_tea"]:
            self.netG_Tea=networks.define_G(opt).to(self.device)
            for p in self.netG_Tea.parameters():
                p.requires_grad = False
            # for param in self.netG_Tea.parameters():
            #     param.detach_()
            from models.Satt_CD.modules.utils import EMAWeightOptimizer
            self.optimizer_G_Tea =EMAWeightOptimizer(self.netG_Tea, self.netG, opt["mean_teacher"]["teacher_alpha"])
            #self.optimizers.append(self.optimizer_G_Tea)
            # from models.Satt_CD.modules.block import FeatureDropImg,FeatureNoiseImg,DropOutImg
            # ImgFeaDrop=[FeatureNoiseImg(conv_in_ch=3) for _ in range(opt["mean_teacher"]["perb_num"])]
            # ImgDrop=[DropOutImg(conv_in_ch=3) for _ in range(opt["mean_teacher"]["perb_num"]) ]
            # ImgGNoise=[FeatureNoiseImg(conv_in_ch=3) for _ in range(opt["mean_teacher"]["perb_num"])]
            # ImgUNoise=[FeatureNoiseImg(conv_in_ch=3,noise_type='uniform') for _ in range(opt["mean_teacher"]["perb_num"])]
            # self.perb1=nn.ModuleList([*ImgGNoise])
            # self.perb2=nn.ModuleList([*ImgUNoise])

            # self.perb1=DropOutImg()
            # self.perb2=FeatureNoiseImg(conv_in_ch=3)
        if opt["DA_method"] == "CycleFea_DA" or opt["DA_method"] == "Cycle_DA":
            from models.Satt_CD.GAN_model import Generator, Discriminator
            wd_D = train_opt['weight_decay_D'] if train_opt['weight_decay_D'] else 0
            self.netD_s=Discriminator(3).to(self.device)
            self.netD_t=Discriminator(3).to(self.device)
            self.netG_s=Generator(3,3).to(self.device)
            self.netG_t=Generator(3,3).to(self.device)
            self.optimizer_D_st = torch.optim.Adam(itertools.chain(self.netD_s.parameters(), self.netD_t.parameters()), lr=train_opt['lr_D'], \
                                                    weight_decay=wd_D, betas=(0.5, 0.999))
            self.optimizer_G_st = torch.optim.Adam(itertools.chain(self.netG_s.parameters(), self.netG_t.parameters()),
                                                   lr=train_opt['lr_G'], \
                                                   weight_decay=wd_G, betas=(0.5, 0.999))

            self.optimizers.append(self.optimizer_D_st)
            self.optimizers.append(self.optimizer_G_st)

        #################################################################################################
        #===================================D============================================================
        #################################################################################################
        if train_opt['is_adv_train']:
            # GD gan loss
           self.cri_gan = GANLoss(train_opt['gan_type'], 1.0, 0.0).to(self.device)
           self.l_gan_w = train_opt['gan_weight']
           wd_D = train_opt['weight_decay_D'] if train_opt['weight_decay_D'] else 0
           if train_opt["opti_type"]=="Adam":
               logger.info('using Adam for D with lr_D_{:.4e}'.format(train_opt['lr_D']))
               self.optimizer_D_fea = torch.optim.Adam(self.netD_fea.parameters(), lr=train_opt['lr_D'], \
                                                       weight_decay=wd_D, betas=(train_opt['beta1_D'], 0.999))
               self.optimizer_D_seg = torch.optim.Adam(self.netD_seg.parameters(), lr=train_opt['lr_D'], \
                                                       weight_decay=wd_D, betas=(train_opt['beta1_D'], 0.999))
           else:
               logger.info('using AdamW for D with lr_D_{:.4e}'.format(train_opt['lr_D']))
               self.optimizer_D_fea = AdamW(self.netD_fea.parameters(), lr=train_opt['lr_D'], \
                                                       weight_decay=wd_D, betas=(train_opt['beta1_D'], 0.999))
               self.optimizer_D_seg = AdamW(self.netD_seg.parameters(), lr=train_opt['lr_D'], \
                                                       weight_decay=wd_D, betas=(train_opt['beta1_D'], 0.999))



           self.optimizers.append(self.optimizer_D_fea)
           self.optimizers.append(self.optimizer_D_seg)
           #self.optimizers.append(self.optimizer_D_MS)





        #================================schedulers==========================
        if train_opt['lr_scheme'] == 'MultiStepLR':
            for optimizer in self.optimizers:
                self.schedulers.append(lr_scheduler.MultiStepLR(optimizer, \
                                                                train_opt['lr_steps'], train_opt['lr_gamma']))
        elif train_opt['lr_scheme'] == 'CosineLR':
            for optimizer in self.optimizers:
                self.schedulers.append(lr_scheduler.CosineAnnealingLR(optimizer, \
                                                                      T_max=train_opt["nepoch"]))
        else:

            raise NotImplementedError('MultiStepLR learning rate scheme is enough.')

        self.log_dict = OrderedDict()
        self.total_iters=train_opt['niter']
        if train_opt["use_SSL"]:
            self.PREHEAT_STEPS = 0
        else:
            #self.PREHEAT_STEPS=int(self.total_iters / 4)  # ===for warmup
            self.PREHEAT_STEPS = int(self.total_iters*train_opt["pre_steps"])


        #=======Labels for Adversarial Training===========
        source_label = 1
        target_label = 0
        self.bce_loss = torch.nn.BCEWithLogitsLoss()
        self.epsion=train_opt['epsion']
        self.lam_local=train_opt['lam_local']
        self.lam_adv=train_opt['lam_adv']
        self.lam_weight=train_opt['lam_weight']
        self.T=train_opt['temperature']
        self.config = opt
        self.LB=train_opt['LB']
        self.lam_ent=train_opt['lam_ent']
        self.ita=train_opt['ita']
        self.lam_KL=train_opt['lam_KL']
        self.train_opt=train_opt

    def feed_data(self, data):

        self.img,self.label=data["img"],data["label"]
        self.img,self.label=self.img.to(self.device),self.label.to(self.device)

    def feed_data_batch_st(self, batch_s,bacht_t):

        self.batch_s=batch_s
        self.batch_t=bacht_t















    def optimize_parameters_AdvSeg_PFO_SSL(self, step):
            '''
            align domain using pixel- feature- and outspace-level feature maps:
            using spatial-aware pixel features,
            ref:ROAD: Reality Oriented Adaptation for Semantic Segmentation of Urban Scenes
            SRDA-Net: Super-Resolution Domain Adaptation Networks for Semantic Segmentation

            using multi-scale feature alignment, ref:An End-to-End Network for Remote Sensing Imagery Semantic Segmentation via Joint
    Pixel- and Representation-Level Domain Adaptation

            :param step:
            :return:
            '''
            bce_loss = self.bce_loss
            # mse_loss=nn.MSELoss()

            source_label = 1
            target_label = 0
            from models.utils import entropy_loss_ita
            from losses.myLoss import N4ASCLoss, N8ASCLoss
            # ======================================================================================
            # train G
            # ======================================================================================
            if step < self.PREHEAT_STEPS:
                if step % self.G_update_ratio == 0:
                    # ===========Remove Grads in D==============
                    # for p in self.netD_seg.parameters():
                    #     p.requires_grad = False
                    # for p in self.netD_fea.parameters():
                    #     p.requires_grad = False
                    self.optimizer_G.zero_grad()
                    l_g_total = 0
                    # =======Train with Source=============
                    from models.utils import entropy_loss_ita
                    images_s, labels = self.batch_s['img'], self.batch_s['label']
                    images_s, labels = Variable(images_s.to(self.device)), Variable(labels.to(self.device))
                    images_t,labels_t = Variable(self.batch_t['img'].to(self.device)),Variable(self.batch_t['label'].to(self.device))

                    # =======transfer image_s to image_st using wallis filter======

                    from models.Satt_CD.modules.utils import adaptive_instance_normalization
                    images_st = adaptive_instance_normalization(images_s, images_t)#alpha=1.0
                    images_st = torch.clamp(images_st, 0.0, 1.0)

                    images_st1 = adaptive_instance_normalization(images_s, images_t,
                                                                 alpha=0.8)  # using wallis filter for image-level alignment
                    images_st1 = torch.clamp(images_st1, 0.0, 1.0)

                    fea_src, pred_src = self.netG(images_st)
                    fea_src1, pred_src1 = self.netG(images_st1)
                    fea_src, pred_src = (fea_src + fea_src1) * 0.5, (pred_src + pred_src1) * 0.5

                    loss_seg = self.cri_seg_loss(pred_src, labels)
                    #loss_ent = entropy_loss_ita(pred_src)
                    loss_ent=N8ASCLoss(pred_src)
                    l_g_total += loss_seg + self.lam_ent * loss_ent

                    # # ======Train with Target==============================
                    #
                    # fea_tgt, pred_tgt = self.netG(images_t)
                    #
                    # D_out_fea = self.netD_fea(fea_tgt)
                    # loss_adv_fea = bce_loss(D_out_fea,
                    #                         Variable(torch.FloatTensor(D_out_fea.data.size()).fill_(source_label)).cuda(
                    #                             0))
                    # loss_adv = self.lam_adv * (loss_adv_fea)
                    # l_g_total += loss_adv

                    l_g_total.backward()
                    self.optimizer_G.step()
                    self.log_dict['l_g_total'] = l_g_total.item()
                    self.log_dict['l_d_total'] = 0
            else:
                if step % self.G_update_ratio == 0:
                    #self.netG_Tea.train()
                    # ===========Remove Grads in D==============
                    # for p in self.netD_seg.parameters():
                    #     p.requires_grad = False
                    for p in self.netD_fea.parameters():
                        p.requires_grad = False
                    self.optimizer_G.zero_grad()
                    l_g_total = 0
                    # =======Train with Source=============

                    images_s, labels = self.batch_s['img'], self.batch_s['label']
                    images_s, labels = Variable(images_s.to(self.device)), Variable(labels.to(self.device))
                    images_t,labels_t = self.batch_t['img'],self.batch_t['label']
                    images_t = Variable(images_t.to(self.device))
                    labels_t = Variable(labels_t.to(self.device))

                    # =======transfer image_s to image_st using wallis filter======

                    from models.Satt_CD.modules.utils import adaptive_instance_normalization
                    images_st = adaptive_instance_normalization(images_s, images_t)
                    images_st = torch.clamp(images_st, 0.0, 1.0)

                    images_st1 = adaptive_instance_normalization(images_s, images_t, alpha=0.8)
                    images_st1 = torch.clamp(images_st1, 0.0, 1.0)

                    fea_src, pred_src = self.netG(images_st)
                    fea_src1, pred_src1 = self.netG(images_st1)
                    fea_src, pred_src = (fea_src + fea_src1) * 0.5, (
                                pred_src + pred_src1 ) * 0.5

                    loss_seg = self.cri_seg_loss(pred_src, labels)
                    #loss_ent = entropy_loss_ita(pred_src)
                    loss_ent=N8ASCLoss(pred_src)
                    l_g_total += loss_seg + self.lam_ent * loss_ent
                    # ==========================Train with Target==============================

                    fea_tgt, pred_tgt = self.netG(images_t)
                    loss_seg = self.cri_seg_loss(pred_tgt, labels_t)
                    #loss_ent = entropy_loss_ita(pred_tgt)
                    loss_ent=N8ASCLoss(pred_tgt)
                    l_g_total += loss_seg + self.lam_ent * loss_ent

                    D_out_fea = self.netD_fea(fea_tgt)
                    loss_adv_fea = bce_loss(D_out_fea,
                                            Variable(torch.FloatTensor(D_out_fea.data.size()).fill_(source_label)).cuda(
                                                0))
                    loss_adv = self.lam_adv * (loss_adv_fea)
                    l_g_total += loss_adv

                    l_g_total.backward()
                    self.optimizer_G.step()
                    # if self.optimizer_G_Tea is not None:
                    #     self.optimizer_G_Tea.step()

                if step % self.D_update_ratio == 0:
                    l_d_fea = 0
                    # ==============Bring back Grads in D================================
                    for param in self.netD_fea.parameters():
                        param.requires_grad = True
                    self.optimizer_D_fea.zero_grad()
                    # ==============================================================
                    fea_src = fea_src.detach()

                    D_out_s = self.netD_fea(fea_src)  # the input of D is probability map sum(dim=1)=1
                    loss_D_s = bce_loss(D_out_s,
                                        Variable(torch.FloatTensor(D_out_s.data.size()).fill_(source_label)).cuda(0))

                    fea_tgt = fea_tgt.detach()
                    D_out_t = self.netD_fea(fea_tgt)
                    loss_D_t = bce_loss(D_out_t,
                                        Variable(torch.FloatTensor(D_out_t.data.size()).fill_(target_label)).cuda(0))
                    l_d_fea = (loss_D_s + loss_D_t) / 2
                    l_d_fea.backward()
                    self.optimizer_D_fea.step()

                    # ================set log=======================================
                    self.log_dict['l_g_total'] = l_g_total.item()
                    self.log_dict['l_d_total'] = l_d_fea.item()

    def optimize_parameters_AdvSeg_PFO_SSL2(self, step,con_weight):
            '''
            align domain using pixel- feature- and outspace-level feature maps:
            using spatial-aware pixel features,
            ref:ROAD: Reality Oriented Adaptation for Semantic Segmentation of Urban Scenes
            SRDA-Net: Super-Resolution Domain Adaptation Networks for Semantic Segmentation

            using multi-scale feature alignment, ref:An End-to-End Network for Remote Sensing Imagery Semantic Segmentation via Joint
    Pixel- and Representation-Level Domain Adaptation

            :param step:
            :return:
            '''
            bce_loss = self.bce_loss
            # mse_loss=nn.MSELoss()

            source_label = 1
            target_label = 0
            from models.utils import entropy_loss_ita
            from losses.myLoss import N4ASCLoss, N8ASCLoss
            # ======================================================================================
            # train G
            # ======================================================================================
            images_t, labels_t = self.batch_t['img'], self.batch_t['label']
            images_t = Variable(images_t.to(self.device))
            labels_t = Variable(labels_t.to(self.device))
            self.optimizer_G.zero_grad()
            l_g_total = 0
            fea_tgt, pred_tgt = self.netG(images_t)
            loss_seg = self.cri_seg_loss(pred_tgt, labels_t)
            # loss_ent = entropy_loss_ita(pred_tgt)
            loss_ent = N8ASCLoss(pred_tgt)
            l_g_total += loss_seg + self.lam_ent * loss_ent
            #========using sparse-CRF=======================
            # from losses.sparseCRF import SparseCRFLoss
            # sparse_loss=SparseCRFLoss(num_classes=1)
            # crf_loss=sparse_loss(pred_tgt,labels_t,images_t)
            # l_g_total+=0.1*crf_loss

            # ==================for transforms consistency loss, imposed on both labeled and unlabeled data===============
            if self.config["train"]["con_type"] == 'tcsm2':
                self.netG_Tea.train()
            from data_loaders.transforms import transforms_back_rot, transforms_for_noise, transforms_for_scale, \
                transforms_for_rot, transforms_back_scale, postprocess_scale
            from models.Satt_CD.modules.utils import update_ema_variables
            from models.Satt_CD.modules.loss import sigmoid_mse_loss
            with torch.no_grad():
                # compute guessed labels of unlabel samples
                inputs_u = images_t.clone()
                inputs_u2 = images_t.clone()
                # tcsm
                # inputs_u2_noise = transforms_for_noise(inputs_u2, 0.5)  # add gaussian noise

                inputs_u2_noise, rot_mask, flip_mask = transforms_for_rot(inputs_u2)  # add rot and flip

                #  add scale
                #inputs_u2_noise, scale_mask = transforms_for_scale(inputs_u2_noise,self.config.patch_size)

                if self.config["train"]["con_type"] == 'tcsm':
                    _, outputs_u = self.netG(inputs_u)  # without transforms
                    _, outputs_u_ema = self.netG(inputs_u2_noise)  # with transforms
                else:
                    _, outputs_u = self.netG(inputs_u)
                    _, outputs_u_ema = self.netG_Tea(inputs_u2_noise)

                # if args.scale:
                # outputs_u_ema, scale_mask = transforms_back_scale(outputs_u_ema, scale_mask, self.config.patch_size)
                # outputs_u = postprocess_scale(outputs_u, scale_mask, self.config.patch_size)

                # tcsm back: modify ema output
                outputs_u_ema = transforms_back_rot(outputs_u_ema, rot_mask, flip_mask)
            # consistency_weight = get_current_consistency_weight(epoch)
            consistency_dist = sigmoid_mse_loss(outputs_u, outputs_u_ema)
            consistency_dist = torch.mean(consistency_dist)

            # Lu = consistency_weight * consistency_dist
            l_g_total += con_weight * consistency_dist
            # ============================================================================================================
            l_g_total.backward()
            self.optimizer_G.step()
            if self.config["train"]["con_type"] == 'tcsm2':
                update_ema_variables(self.netG, self.netG_Tea, step + 1)  # update teacher-model

            # ================set log=======================================
            self.log_dict['l_g_total'] = l_g_total.item()
            self.log_dict['l_d_total'] = 0

    def optimize_parameters_AdvSeg_PFO_SSL0(self, step):
            '''
            align domain using pixel- feature- and outspace-level feature maps:
            using spatial-aware pixel features,
            ref:ROAD: Reality Oriented Adaptation for Semantic Segmentation of Urban Scenes
            SRDA-Net: Super-Resolution Domain Adaptation Networks for Semantic Segmentation

            using multi-scale feature alignment, ref:An End-to-End Network for Remote Sensing Imagery Semantic Segmentation via Joint
    Pixel- and Representation-Level Domain Adaptation

            :param step:
            :return:
            '''
            bce_loss = self.bce_loss
            # mse_loss=nn.MSELoss()

            source_label = 1
            target_label = 0
            from models.utils import entropy_loss_ita
            from losses.myLoss import N4ASCLoss, N8ASCLoss
            # ======================================================================================
            # train G
            # ======================================================================================
            images_t, labels_t = self.batch_t['img'], self.batch_t['label']
            images_t = Variable(images_t.to(self.device))
            labels_t = Variable(labels_t.to(self.device))
            self.optimizer_G.zero_grad()
            l_g_total = 0
            fea_tgt, pred_tgt = self.netG(images_t)
            loss_seg = self.cri_seg_loss(pred_tgt, labels_t)
            # loss_ent = entropy_loss_ita(pred_tgt)
            loss_ent = N8ASCLoss(pred_tgt)
            l_g_total += loss_seg + self.lam_ent * loss_ent


            # ============================================================================================================
            l_g_total.backward()
            self.optimizer_G.step()

            # ================set log=======================================
            self.log_dict['l_g_total'] = l_g_total.item()
            self.log_dict['l_d_total'] = 0






    def optimize_parameters_AdvSeg_PFO_Con(self, step,total_iters,con_weight):
            '''


            :param step:
            :return:
            '''
            bce_loss = self.bce_loss
            # mse_loss=nn.MSELoss()

            source_label = 1
            target_label = 0
            from losses.myLoss import N4ASCLoss, N8ASCLoss
            from models.Satt_CD.modules.utils import eightway_affinity_kld
            # ======================================================================================
            # train G
            # ======================================================================================
            if step < self.PREHEAT_STEPS:
                if step % self.G_update_ratio == 0:

                    self.optimizer_G.zero_grad()
                    l_g_total = 0
                    # =======Train with Source=============
                    from models.utils import entropy_loss_ita
                    images_s, labels = self.batch_s['img'], self.batch_s['label']
                    images_s, labels = Variable(images_s.to(self.device)), Variable(labels.to(self.device))
                    images_t = self.batch_t['img']
                    images_t = Variable(images_t.to(self.device))


                    from models.Satt_CD.modules.utils import adaptive_instance_normalization
                    images_st = adaptive_instance_normalization(images_s, images_t)
                    images_st = torch.clamp(images_st, 0.0, 1.0)

                    images_st1 = adaptive_instance_normalization(images_s, images_t,
                                                                 alpha=0.8)  # using wallis filter for image-level alignment
                    images_st1 = torch.clamp(images_st1, 0.0, 1.0)

                    fea_src, pred_src = self.netG(images_st)
                    fea_src1, pred_src1 = self.netG(images_st1)
                    fea_src, pred_src = (fea_src + fea_src1) * 0.5, (pred_src + pred_src1) * 0.5

                    loss_seg = self.cri_seg_loss(pred_src, labels)
                    #loss_ent = entropy_loss_ita(pred_src)
                    loss_ent=0
                    loss_AS=N8ASCLoss(pred_src)

                    l_g_total += loss_seg + self.lam_ent * (loss_ent+loss_AS)


                    l_g_total.backward()
                    self.optimizer_G.step()
                    self.log_dict['l_g_total'] = l_g_total.item()
                    self.log_dict['l_d_total'] = 0
            else:
                if step % self.G_update_ratio == 0:
                    # ===========Remove Grads in D==============
                    # for p in self.netD_seg.parameters():
                    #     p.requires_grad = False
                    for p in self.netD_fea.parameters():
                        p.requires_grad = False
                    self.optimizer_G.zero_grad()
                    l_g_total = 0
                    # =======Train with Source=============
                    from models.utils import entropy_loss_ita
                    images_s, labels = self.batch_s['img'], self.batch_s['label']
                    images_s, labels = Variable(images_s.to(self.device)), Variable(labels.to(self.device))
                    images_t = self.batch_t['img']
                    images_t = Variable(images_t.to(self.device))

                    # =======transfer image_s to image_st using wallis filter======

                    from models.Satt_CD.modules.utils import adaptive_instance_normalization
                    images_st = adaptive_instance_normalization(images_s, images_t)
                    images_st = torch.clamp(images_st, 0.0, 1.0)

                    # images_st1 = adaptive_instance_normalization(images_s, images_t, alpha=0.8)
                    # images_st1 = torch.clamp(images_st1, 0.0, 1.0)



                    fea_src, pred_src = self.netG(images_st)
                    # fea_src1, pred_src1 = self.netG(images_st1)
                    # fea_src, pred_src = (fea_src + fea_src1) * 0.5, (pred_src + pred_src1) * 0.5

                    loss_seg = self.cri_seg_loss(pred_src, labels)
                    #loss_ent = entropy_loss_ita(pred_src)
                    loss_ent = 0
                    if self.config["train"]["use_aff"]:
                       loss_AS=N8ASCLoss(pred_src)
                    else:
                        loss_AS=0
                    l_g_total += loss_seg + self.lam_ent * (loss_ent + loss_AS)

                    # ==========================Train with Target==============================

                    fea_tgt, pred_tgt = self.netG(images_t)
                    #loss_ent =entropy_loss_ita(pred_tgt)
                    loss_ent=0#without loss_ent seems to work better
                    #loss_AS=N8ASCLoss(pred_tgt)
                    if self.config["train"]["use_aff"]:
                       loss_AS=N8ASCLoss(pred_tgt)
                    else:
                        loss_AS=0
                    l_g_total += self.lam_ent * (loss_ent + loss_AS)

                    D_out_fea = self.netD_fea(fea_tgt)
                    loss_adv_fea = bce_loss(D_out_fea,
                                            Variable(torch.FloatTensor(D_out_fea.data.size()).fill_(source_label)).cuda(
                                                0))

                    # D_out_s = self.netD_fea(fea_src)  # the input of D is probability map sum(dim=1)=1
                    # loss_D_s = bce_loss(D_out_s,
                    #                     Variable(torch.FloatTensor(D_out_s.data.size()).fill_(target_label)).cuda(0))

                    loss_adv = self.lam_adv * (loss_adv_fea)
                    l_g_total += loss_adv
                    #==================for transforms consistency loss, imposed on both labeled and unlabeled data===============
                    if self.config["train"]["con_type"] == 'tcsm2':
                        self.netG_Tea.train()
                    from data_loaders.transforms import transforms_back_rot,transforms_for_noise,transforms_for_scale,transforms_for_rot,transforms_back_scale,postprocess_scale
                    from models.Satt_CD.modules.utils import update_ema_variables
                    from models.Satt_CD.modules.loss import sigmoid_mse_loss
                    #=================for unkown target data, using uncertainty weight to only transfer reliable knowledge ref: https://github.com/yulequan/UA-MT



                    with torch.no_grad():
                        # compute guessed labels of unlabel samples
                        inputs_u=torch.cat([images_st,images_t])
                        inputs_u2 = torch.cat([images_st, images_t])

                        # inputs_u = images_t.clone()
                        # inputs_u2 = images_t.clone()


                        # tcsm
                        #inputs_u2_noise = transforms_for_noise(inputs_u2, 0.5)  # add gaussian noise

                        inputs_u2_noise, rot_mask, flip_mask = transforms_for_rot(inputs_u2)  # add rot and flip

                        #  add scale
                        #inputs_u2_noise, scale_mask = transforms_for_scale(inputs_u2_noise,self.config.patch_size)

                        if self.config["train"]["con_type"]=='tcsm':
                            _,outputs_u = self.netG(inputs_u)  # without transforms
                            _,outputs_u_ema = self.netG(inputs_u2_noise)  # with transforms
                        else:
                            _,outputs_u = self.netG(inputs_u)
                            _,outputs_u_ema = self.netG_Tea(inputs_u2_noise)

                        # if args.scale:
                        # outputs_u_ema, scale_mask = transforms_back_scale(outputs_u_ema, scale_mask, self.config.patch_size)
                        # outputs_u = postprocess_scale(outputs_u, scale_mask, self.config.patch_size)

                        # tcsm back: modify ema output
                        outputs_u_ema = transforms_back_rot(outputs_u_ema, rot_mask, flip_mask)
                    consistency_dist = sigmoid_mse_loss(outputs_u, outputs_u_ema)#[4,1,512,512]

                    if self.config["train"]["use_uncertain"]:
                        from models.Satt_CD.modules import ramps
                        T = 8
                        inputs_u2_noise = transforms_for_noise(inputs_u2, 0.5)
                        stride=inputs_u2_noise.shape[0]
                        preds=torch.zeros(stride * T, 1, self.config["patch_size"],self.config["patch_size"]).cuda()

                        # volume_batch_r = volume_batch.repeat(2, 1, 1, 1, 1)
                        # stride = volume_batch_r.shape[0] // 2
                        # preds = torch.zeros([stride * T, 2, 112, 112, 80]).cuda()
                        for i in range(T):
                            #ema_inputs = volume_batch_r + torch.clamp(torch.randn_like(volume_batch_r) * 0.1, -0.2, 0.2)
                            with torch.no_grad():
                                _,preds[stride * i:stride * (i + 1)]= self.netG_Tea(inputs_u2_noise)
                        # preds = F.softmax(preds, dim=1)
                        preds = preds.reshape(T, stride, 1, self.config["patch_size"], self.config["patch_size"])
                        preds = torch.mean(preds, dim=0)  # (8,4,1,512,512)==>(4,1,512,512)
                        uncertainty = -1.0 * torch.sum(preds * torch.log(preds + 1e-6), dim=1,
                                                       keepdim=True)  # (batch, 1, 512,512)
                        threshold = (0.75 + 0.25 * ramps.sigmoid_rampup(step, total_iters)) * np.log(2)
                        mask = (uncertainty < threshold).float()
                        consistency_dist = torch.sum(mask * consistency_dist) / (2 * torch.sum(mask) + 1e-16)

                    #consistency_weight = get_current_consistency_weight(epoch)
                    consistency_dist = torch.mean(consistency_dist)

                    #Lu = consistency_weight * consistency_dist
                    l_g_total+=con_weight * consistency_dist
                    #============================================================================================================
                    l_g_total.backward()
                    self.optimizer_G.step()
                    if self.config["train"]["con_type"] == 'tcsm2':
                       update_ema_variables(self.netG, self.netG_Tea, step+1)#update teacher-model


                if step % self.D_update_ratio == 0:
                    l_d_fea = 0
                    # ==============Bring back Grads in D================================
                    for param in self.netD_fea.parameters():
                        param.requires_grad = True
                    self.optimizer_D_fea.zero_grad()
                    # ==============================================================
                    fea_src = fea_src.detach()

                    D_out_s = self.netD_fea(fea_src)  # the input of D is probability map sum(dim=1)=1
                    loss_D_s = bce_loss(D_out_s,
                                        Variable(torch.FloatTensor(D_out_s.data.size()).fill_(source_label)).cuda(0))

                    fea_tgt = fea_tgt.detach()
                    D_out_t = self.netD_fea(fea_tgt)
                    loss_D_t = bce_loss(D_out_t,
                                        Variable(torch.FloatTensor(D_out_t.data.size()).fill_(target_label)).cuda(0))
                    l_d_fea = (loss_D_s + loss_D_t) / 2
                    l_d_fea.backward()
                    self.optimizer_D_fea.step()




                    # ================set log=======================================
                    self.log_dict['l_g_total'] = l_g_total.item()
                    self.log_dict['l_d_total'] = l_d_fea.item()

    def optimize_parameters_AdvSeg_PFO_Con_tune(self, step,total_iters,con_weight,config):
            '''


            :param step:
            :return:
            '''
            bce_loss = self.bce_loss
            # mse_loss=nn.MSELoss()

            source_label = 1
            target_label = 0
            from losses.myLoss import N4ASCLoss, N8ASCLoss
            from models.Satt_CD.modules.utils import eightway_affinity_kld
            # ======================================================================================
            # train G
            # ======================================================================================
            if step < self.PREHEAT_STEPS:
                if step % self.G_update_ratio == 0:

                    self.optimizer_G.zero_grad()
                    l_g_total = 0
                    # =======Train with Source=============
                    from models.utils import entropy_loss_ita
                    images_s, labels = self.batch_s['img'], self.batch_s['label']
                    images_s, labels = Variable(images_s.to(self.device)), Variable(labels.to(self.device))
                    images_t = self.batch_t['img']
                    images_t = Variable(images_t.to(self.device))


                    from models.Satt_CD.modules.utils import adaptive_instance_normalization
                    images_st = adaptive_instance_normalization(images_s, images_t)
                    images_st = torch.clamp(images_st, 0.0, 1.0)

                    images_st1 = adaptive_instance_normalization(images_s, images_t,
                                                                 alpha=0.8)  # using wallis filter for image-level alignment
                    images_st1 = torch.clamp(images_st1, 0.0, 1.0)

                    fea_src, pred_src = self.netG(images_st)
                    fea_src1, pred_src1 = self.netG(images_st1)
                    fea_src, pred_src = (fea_src + fea_src1) * 0.5, (pred_src + pred_src1) * 0.5

                    loss_seg = self.cri_seg_loss(pred_src, labels)
                    #loss_ent = entropy_loss_ita(pred_src)
                    loss_ent=0
                    loss_AS=N8ASCLoss(pred_src)

                    l_g_total += loss_seg + config["lam_ent"] * (loss_ent+loss_AS)


                    l_g_total.backward()
                    self.optimizer_G.step()
                    self.log_dict['l_g_total'] = l_g_total.item()
                    self.log_dict['l_d_total'] = 0
            else:
                if step % self.G_update_ratio == 0:
                    # ===========Remove Grads in D==============
                    # for p in self.netD_seg.parameters():
                    #     p.requires_grad = False
                    for p in self.netD_fea.parameters():
                        p.requires_grad = False
                    self.optimizer_G.zero_grad()
                    l_g_total = 0
                    # =======Train with Source=============
                    from models.utils import entropy_loss_ita
                    images_s, labels = self.batch_s['img'], self.batch_s['label']
                    images_s, labels = Variable(images_s.to(self.device)), Variable(labels.to(self.device))
                    images_t = self.batch_t['img']
                    images_t = Variable(images_t.to(self.device))

                    # =======transfer image_s to image_st using wallis filter======

                    from models.Satt_CD.modules.utils import adaptive_instance_normalization
                    images_st = adaptive_instance_normalization(images_s, images_t)
                    images_st = torch.clamp(images_st, 0.0, 1.0)

                    # images_st1 = adaptive_instance_normalization(images_s, images_t, alpha=0.8)
                    # images_st1 = torch.clamp(images_st1, 0.0, 1.0)



                    fea_src, pred_src = self.netG(images_st)
                    # fea_src1, pred_src1 = self.netG(images_st1)
                    # fea_src, pred_src = (fea_src + fea_src1) * 0.5, (pred_src + pred_src1) * 0.5

                    loss_seg = self.cri_seg_loss(pred_src, labels)
                    #loss_ent = entropy_loss_ita(pred_src)
                    loss_ent = 0
                    if self.config["train"]["use_aff"]:
                       loss_AS=N8ASCLoss(pred_src)
                    else:
                        loss_AS=0
                    l_g_total += loss_seg + config["lam_ent"] * (loss_ent + loss_AS)

                    # ==========================Train with Target==============================

                    fea_tgt, pred_tgt = self.netG(images_t)
                    #loss_ent =entropy_loss_ita(pred_tgt)
                    loss_ent=0#without loss_ent seems to work better
                    #loss_AS=N8ASCLoss(pred_tgt)
                    if self.config["train"]["use_aff"]:
                       loss_AS=N8ASCLoss(pred_tgt)
                    else:
                        loss_AS=0
                    l_g_total += config["lam_ent"] * (loss_ent + loss_AS)

                    D_out_fea = self.netD_fea(fea_tgt)
                    loss_adv_fea = bce_loss(D_out_fea,
                                            Variable(torch.FloatTensor(D_out_fea.data.size()).fill_(source_label)).cuda(
                                                0))

                    # D_out_s = self.netD_fea(fea_src)  # the input of D is probability map sum(dim=1)=1
                    # loss_D_s = bce_loss(D_out_s,
                    #                     Variable(torch.FloatTensor(D_out_s.data.size()).fill_(target_label)).cuda(0))

                    loss_adv = config["lam_adv"]* (loss_adv_fea)
                    l_g_total += loss_adv
                    #==================for transforms consistency loss, imposed on both labeled and unlabeled data===============
                    if self.config["train"]["con_type"] == 'tcsm2':
                        self.netG_Tea.train()
                    from data_loaders.transforms import transforms_back_rot,transforms_for_noise,transforms_for_scale,transforms_for_rot,transforms_back_scale,postprocess_scale
                    from models.Satt_CD.modules.utils import update_ema_variables
                    from models.Satt_CD.modules.loss import sigmoid_mse_loss
                    #=================for unkown target data, using uncertainty weight to only transfer reliable knowledge ref: https://github.com/yulequan/UA-MT



                    with torch.no_grad():
                        # compute guessed labels of unlabel samples
                        inputs_u=torch.cat([images_st,images_t])
                        inputs_u2 = torch.cat([images_st, images_t])

                        # inputs_u = images_t.clone()
                        # inputs_u2 = images_t.clone()


                        # tcsm
                        #inputs_u2_noise = transforms_for_noise(inputs_u2, 0.5)  # add gaussian noise

                        inputs_u2_noise, rot_mask, flip_mask = transforms_for_rot(inputs_u2)  # add rot and flip

                        #  add scale
                        #inputs_u2_noise, scale_mask = transforms_for_scale(inputs_u2_noise,self.config.patch_size)

                        if self.config["train"]["con_type"]=='tcsm':
                            _,outputs_u = self.netG(inputs_u)  # without transforms
                            _,outputs_u_ema = self.netG(inputs_u2_noise)  # with transforms
                        else:
                            _,outputs_u = self.netG(inputs_u)
                            _,outputs_u_ema = self.netG_Tea(inputs_u2_noise)

                        # if args.scale:
                        # outputs_u_ema, scale_mask = transforms_back_scale(outputs_u_ema, scale_mask, self.config.patch_size)
                        # outputs_u = postprocess_scale(outputs_u, scale_mask, self.config.patch_size)

                        # tcsm back: modify ema output
                        outputs_u_ema = transforms_back_rot(outputs_u_ema, rot_mask, flip_mask)
                    consistency_dist = sigmoid_mse_loss(outputs_u, outputs_u_ema)#[4,1,512,512]

                    if self.config["train"]["use_uncertain"]:
                        from models.Satt_CD.modules import ramps
                        T = 8
                        inputs_u2_noise = transforms_for_noise(inputs_u2, 0.5)
                        stride=inputs_u2_noise.shape[0]
                        preds=torch.zeros(stride * T, 1, self.config["patch_size"],self.config["patch_size"]).cuda()

                        # volume_batch_r = volume_batch.repeat(2, 1, 1, 1, 1)
                        # stride = volume_batch_r.shape[0] // 2
                        # preds = torch.zeros([stride * T, 2, 112, 112, 80]).cuda()
                        for i in range(T):
                            #ema_inputs = volume_batch_r + torch.clamp(torch.randn_like(volume_batch_r) * 0.1, -0.2, 0.2)
                            with torch.no_grad():
                                _,preds[stride * i:stride * (i + 1)]= self.netG_Tea(inputs_u2_noise)
                        # preds = F.softmax(preds, dim=1)
                        preds = preds.reshape(T, stride, 1, self.config["patch_size"], self.config["patch_size"])
                        preds = torch.mean(preds, dim=0)  # (8,4,1,512,512)==>(4,1,512,512)
                        uncertainty = -1.0 * torch.sum(preds * torch.log(preds + 1e-6), dim=1,
                                                       keepdim=True)  # (batch, 1, 512,512)
                        threshold = (0.75 + 0.25 * ramps.sigmoid_rampup(step, total_iters)) * np.log(2)
                        mask = (uncertainty < threshold).float()
                        consistency_dist = torch.sum(mask * consistency_dist) / (2 * torch.sum(mask) + 1e-16)

                    #consistency_weight = get_current_consistency_weight(epoch)
                    consistency_dist = torch.mean(consistency_dist)

                    #Lu = consistency_weight * consistency_dist
                    l_g_total+=con_weight * consistency_dist
                    #============================================================================================================
                    l_g_total.backward()
                    self.optimizer_G.step()
                    if self.config["train"]["con_type"] == 'tcsm2':
                       update_ema_variables(self.netG, self.netG_Tea, step+1)#update teacher-model


                if step % self.D_update_ratio == 0:
                    l_d_fea = 0
                    # ==============Bring back Grads in D================================
                    for param in self.netD_fea.parameters():
                        param.requires_grad = True
                    self.optimizer_D_fea.zero_grad()
                    # ==============================================================
                    fea_src = fea_src.detach()

                    D_out_s = self.netD_fea(fea_src)  # the input of D is probability map sum(dim=1)=1
                    loss_D_s = bce_loss(D_out_s,
                                        Variable(torch.FloatTensor(D_out_s.data.size()).fill_(source_label)).cuda(0))

                    fea_tgt = fea_tgt.detach()
                    D_out_t = self.netD_fea(fea_tgt)
                    loss_D_t = bce_loss(D_out_t,
                                        Variable(torch.FloatTensor(D_out_t.data.size()).fill_(target_label)).cuda(0))
                    l_d_fea = (loss_D_s + loss_D_t) / 2
                    l_d_fea.backward()
                    self.optimizer_D_fea.step()




                    # ================set log=======================================
                    self.log_dict['l_g_total'] = l_g_total.item()
                    self.log_dict['l_d_total'] = l_d_fea.item()





    def optimize_parameters_AdvSeg_PFO_Con0(self, step,con_weight):#wallis filter+Df

            bce_loss = self.bce_loss
            source_label = 1
            target_label = 0
            from losses.myLoss import N4ASCLoss, N8ASCLoss
            from models.Satt_CD.modules.utils import eightway_affinity_kld
            # ======================================================================================
            # train G
            # ======================================================================================
            if step < self.PREHEAT_STEPS:
                if step % self.G_update_ratio == 0:

                    self.optimizer_G.zero_grad()
                    l_g_total = 0
                    # =======Train with Source=============
                    from models.utils import entropy_loss_ita
                    images_s, labels = self.batch_s['img'], self.batch_s['label']
                    images_s, labels = Variable(images_s.to(self.device)), Variable(labels.to(self.device))
                    images_t = self.batch_t['img']
                    images_t = Variable(images_t.to(self.device))


                    from models.Satt_CD.modules.utils import adaptive_instance_normalization
                    images_st = adaptive_instance_normalization(images_s, images_t)
                    images_st = torch.clamp(images_st, 0.0, 1.0)

                    # images_st1 = adaptive_instance_normalization(images_s, images_t,
                    #                                              alpha=0.8)  # using wallis filter for image-level alignment
                    # images_st1 = torch.clamp(images_st1, 0.0, 1.0)

                    fea_src, pred_src = self.netG(images_st)
                    # fea_src1, pred_src1 = self.netG(images_st1)
                    # fea_src, pred_src = (fea_src + fea_src1) * 0.5, (pred_src + pred_src1) * 0.5

                    loss_seg = self.cri_seg_loss(pred_src, labels)
                    #loss_ent = entropy_loss_ita(pred_src)
                    loss_ent=N8ASCLoss(pred_src)

                    l_g_total += loss_seg + self.lam_ent * loss_ent


                    l_g_total.backward()
                    self.optimizer_G.step()
                    self.log_dict['l_g_total'] = l_g_total.item()
                    self.log_dict['l_d_total'] = 0
            else:
                if step % self.G_update_ratio == 0:
                    # ===========Remove Grads in D==============
                    # for p in self.netD_seg.parameters():
                    #     p.requires_grad = False
                    for p in self.netD_fea.parameters():
                        p.requires_grad = False
                    self.optimizer_G.zero_grad()
                    l_g_total = 0
                    # =======Train with Source=============
                    from models.utils import entropy_loss_ita
                    images_s, labels = self.batch_s['img'], self.batch_s['label']
                    images_s, labels = Variable(images_s.to(self.device)), Variable(labels.to(self.device))
                    images_t = self.batch_t['img']
                    images_t = Variable(images_t.to(self.device))

                    # =======transfer image_s to image_st using wallis filter======

                    from models.Satt_CD.modules.utils import adaptive_instance_normalization
                    images_st = adaptive_instance_normalization(images_s, images_t)
                    images_st = torch.clamp(images_st, 0.0, 1.0)

                    # images_st1 = adaptive_instance_normalization(images_s, images_t, alpha=0.8)
                    # images_st1 = torch.clamp(images_st1, 0.0, 1.0)

                    fea_src, pred_src = self.netG(images_st)
                    # fea_src1, pred_src1 = self.netG(images_st1)
                    # fea_src, pred_src = (fea_src + fea_src1) * 0.5, (pred_src + pred_src1) * 0.5

                    loss_seg = self.cri_seg_loss(pred_src, labels)
                    #loss_ent = entropy_loss_ita(pred_src)
                    #loss_ent=N8ASCLoss(pred_src)
                    loss_ent=0
                    l_g_total += loss_seg + self.lam_ent * loss_ent

                    # ==========================Train with Target==============================

                    fea_tgt, pred_tgt = self.netG(images_t)
                    # loss_ent = N8ASCLoss(pred_tgt)
                    # l_g_total += self.lam_ent * loss_ent

                    D_out_fea = self.netD_fea(fea_tgt)
                    loss_adv_fea = bce_loss(D_out_fea,
                                            Variable(torch.FloatTensor(D_out_fea.data.size()).fill_(source_label)).cuda(
                                                0))


                    loss_adv = self.lam_adv * (loss_adv_fea)
                    l_g_total += loss_adv
                    #==================for transforms consistency loss, imposed on both labeled and unlabeled data===============
                    # if self.config["train"]["con_type"] == 'tcsm2':
                    #     self.netG_Tea.train()
                    # from data_loaders.transforms import transforms_back_rot,transforms_for_noise,transforms_for_scale,transforms_for_rot,transforms_back_scale,postprocess_scale
                    # from models.Satt_CD.modules.utils import update_ema_variables
                    # from models.Satt_CD.modules.loss import sigmoid_mse_loss
                    # with torch.no_grad():
                    #
                    #
                    #     inputs_u = images_t.clone()
                    #     inputs_u2 = images_t.clone()
                    #
                    #
                    #     inputs_u2_noise, rot_mask, flip_mask = transforms_for_rot(inputs_u2)  # add rot and flip
                    #
                    #     #  add scale
                    #     #inputs_u2_noise, scale_mask = transforms_for_scale(inputs_u2_noise,self.config.patch_size)
                    #
                    #     if self.config["train"]["con_type"]=='tcsm':
                    #         _,outputs_u = self.netG(inputs_u)  # without transforms
                    #         _,outputs_u_ema = self.netG(inputs_u2_noise)  # with transforms
                    #     else:
                    #         _,outputs_u = self.netG(inputs_u)
                    #         _,outputs_u_ema = self.netG_Tea(inputs_u2_noise)
                    #
                    #     # if args.scale:
                    #     # outputs_u_ema, scale_mask = transforms_back_scale(outputs_u_ema, scale_mask, self.config.patch_size)
                    #     # outputs_u = postprocess_scale(outputs_u, scale_mask, self.config.patch_size)
                    #
                    #     # tcsm back: modify ema output
                    #     outputs_u_ema = transforms_back_rot(outputs_u_ema, rot_mask, flip_mask)
                    # #consistency_weight = get_current_consistency_weight(epoch)
                    # consistency_dist = sigmoid_mse_loss(outputs_u, outputs_u_ema)
                    # consistency_dist = torch.mean(consistency_dist)
                    #
                    # #Lu = consistency_weight * consistency_dist
                    # l_g_total+=con_weight * consistency_dist
                    #============================================================================================================
                    l_g_total.backward()
                    self.optimizer_G.step()
                    # if self.config["train"]["con_type"] == 'tcsm2':
                    #    update_ema_variables(self.netG, self.netG_Tea, step+1)#update teacher-model


                if step % self.D_update_ratio == 0:
                    l_d_fea = 0
                    # ==============Bring back Grads in D================================
                    for param in self.netD_fea.parameters():
                        param.requires_grad = True
                    self.optimizer_D_fea.zero_grad()
                    # ==============================================================
                    fea_src = fea_src.detach()

                    D_out_s = self.netD_fea(fea_src)  # the input of D is probability map sum(dim=1)=1
                    loss_D_s = bce_loss(D_out_s,
                                        Variable(torch.FloatTensor(D_out_s.data.size()).fill_(source_label)).cuda(0))

                    fea_tgt = fea_tgt.detach()
                    D_out_t = self.netD_fea(fea_tgt)
                    loss_D_t = bce_loss(D_out_t,
                                        Variable(torch.FloatTensor(D_out_t.data.size()).fill_(target_label)).cuda(0))
                    l_d_fea = (loss_D_s + loss_D_t) / 2
                    l_d_fea.backward()
                    self.optimizer_D_fea.step()




                    # ================set log=======================================
                    self.log_dict['l_g_total'] = l_g_total.item()
                    self.log_dict['l_d_total'] = l_d_fea.item()


















    def optimize_parameters_CycleFea(self, step):
            '''
            DA using cycle-gan and feature-level alignment ref: An End-to-End Network for Remote Sensing
Imagery Semantic Segmentation via Joint Pixel- and Representation-Level Domain Adaptation
            step1: style-transfer using cyclegan with res-block generator and vgg-like discriminators
            step2: feature-level alignment using ASPP-like discriminator
            step3: generate segmentation result using pre-trained unet
            :param step:
            :return:
            '''
            bce_loss = self.bce_loss
            criterionGAN = loss.GANLoss().to(self.device)
            #criterionGAN1 = networks.GANLoss().to(device)
            criterionCycle = torch.nn.L1Loss()
            criterionIdt = torch.nn.L1Loss()
            #criterionSeg = cross_entropy2d()

            source_label = 1
            target_label = 0
            from losses.myLoss import N4ASCLoss, N8ASCLoss
            # ======================================================================================
            # train G
            # ======================================================================================
            # for p in self.netD_fea.parameters():
            #     p.requires_grad = False
            set_requires_grad([self.netD_fea,self.netD_s,self.netD_t],False)
            self.optimizer_G.zero_grad()
            self.optimizer_G_st.zero_grad()
            l_g_total = 0
            # =======Train with Cyclegan===================
            images_s, labels = self.batch_s['img'], self.batch_s['label']
            images_s, labels = Variable(images_s.to(self.device)), Variable(labels.to(self.device))
            images_t = self.batch_t['img']
            images_t = Variable(images_t.to(self.device))
            #A==>s B==>t
            #==========Identity loss=========
            self.lam_idt=self.config["train"]["lam_idt"]
            loss_idt=0
            same_t=self.netG_t(images_t)
            loss_idt+=criterionIdt(same_t,images_t)*self.lam_idt
            same_s = self.netG_s(images_s)
            loss_idt+= criterionIdt(same_s, images_s) * self.lam_idt
            #==========GAN loss==============
            loss_GAN=0
            fake_t = self.netG_t(images_s)
            pred_fake = self.netD_t(fake_t)  # [1,1]
            loss_GAN+= criterionGAN(pred_fake, True)# target=[1.]

            fake_s = self.netG_s(images_t)
            pred_fake = self.netD_s(fake_s)
            loss_GAN+= criterionGAN(pred_fake, False)
            #=========Cycle loss===========
            self.lam_rec=10.0
            loss_cycle=0
            rec_s = self.netG_s(fake_t)
            loss_cycle+= criterionCycle(rec_s, images_s) * self.lam_rec

            rec_t = self.netG_t(fake_s)
            loss_cycle+= criterionCycle(rec_t, images_t) * self.lam_rec

            # Total cycleGAN loss
            l_g_total+= loss_idt+loss_GAN+loss_cycle
            #
            fea_src, pred_src = self.netG(fake_t)
            fea_tgt,_=self.netG(images_t)
            # feature-level discriminator loss
            l_g_total+=self.lam_adv*(criterionGAN(self.netD_fea(fea_src), True) + criterionGAN(self.netD_fea(fea_tgt),
                                                                                   False)) * 0.5
            #seg loss
            l_g_total+=self.cri_seg_loss(pred_src,labels)

            l_g_total.backward()
            self.optimizer_G.step()
            self.optimizer_G_st.step()

            #=================training Ds==================================
            l_d_total=0
            set_requires_grad([self.netD_fea,self.netD_s,self.netD_t],True)
            self.optimizer_D_fea.zero_grad()
            self.optimizer_D_st.zero_grad()

            loss_d_fea=criterionGAN(self.netD_fea(fea_src.detach()), False) + criterionGAN(self.netD_fea(fea_tgt.detach()),                                                                       True)*0.5
            loss_d_fea.backward()

            loss_d_s=criterionGAN(self.netD_s(fake_s.detach()), False) + criterionGAN(self.netD_s(images_s),
                                                                                   True)*0.5
            loss_d_s.backward()

            loss_d_t = criterionGAN(self.netD_t(fake_t.detach()), False) + criterionGAN(self.netD_t(images_t),
                                                                                        True) * 0.5
            loss_d_t.backward()

            self.optimizer_D_st.step()
            self.optimizer_D_fea.step()

            # ================set log=======================================
            self.log_dict['l_g_total'] = l_g_total.item()
            self.log_dict['l_d_total'] = loss_d_fea.item()+loss_d_s.item()+loss_d_t.item()

    def optimize_parameters_Cycle(self, step):
        '''
        DA using cycle-gan and feature-level alignment ref: An End-to-End Network for Remote Sensing
Imagery Semantic Segmentation via Joint Pixel- and Representation-Level Domain Adaptation
        step1: style-transfer using cyclegan with res-block generator and vgg-like discriminators
        step2: feature-level alignment using ASPP-like discriminator
        step3: generate segmentation result using pre-trained unet
        :param step:
        :return:
        '''
        bce_loss = self.bce_loss
        criterionGAN = loss.GANLoss().to(self.device)
        # criterionGAN1 = networks.GANLoss().to(device)
        criterionCycle = torch.nn.L1Loss()
        criterionIdt = torch.nn.L1Loss()
        # criterionSeg = cross_entropy2d()

        source_label = 1
        target_label = 0
        from losses.myLoss import N4ASCLoss, N8ASCLoss
        # ======================================================================================
        # train G
        # ======================================================================================
        # for p in self.netD_fea.parameters():
        #     p.requires_grad = False
        set_requires_grad([self.netD_s, self.netD_t], False)
        self.optimizer_G.zero_grad()
        self.optimizer_G_st.zero_grad()
        l_g_total = 0
        # =======Train with Cyclegan===================
        images_s, labels = self.batch_s['img'], self.batch_s['label']
        images_s, labels = Variable(images_s.to(self.device)), Variable(labels.to(self.device))
        images_t = self.batch_t['img']
        images_t = Variable(images_t.to(self.device))
        # A==>s B==>t
        # ==========Identity loss=========
        self.lam_idt = self.config["train"]["lam_idt"]
        loss_idt = 0
        same_t = self.netG_t(images_t)
        loss_idt += criterionIdt(same_t, images_t) * self.lam_idt
        same_s = self.netG_s(images_s)
        loss_idt += criterionIdt(same_s, images_s) * self.lam_idt
        # ==========GAN loss==============
        loss_GAN = 0
        fake_t = self.netG_t(images_s)
        pred_fake = self.netD_t(fake_t)  # [1,1]
        loss_GAN += criterionGAN(pred_fake, True)  # target=[1.]

        fake_s = self.netG_s(images_t)
        pred_fake = self.netD_s(fake_s)
        loss_GAN += criterionGAN(pred_fake, False)
        # =========Cycle loss===========
        self.lam_rec = 10.0
        loss_cycle = 0
        rec_s = self.netG_s(fake_t)
        loss_cycle += criterionCycle(rec_s, images_s) * self.lam_rec

        rec_t = self.netG_t(fake_s)
        loss_cycle += criterionCycle(rec_t, images_t) * self.lam_rec

        # Total cycleGAN loss
        l_g_total += loss_idt + loss_GAN + loss_cycle
        #
        fea_src, pred_src = self.netG(fake_t)
        fea_tgt, _ = self.netG(images_t)
        # feature-level discriminator loss
        # l_g_total += self.lam_adv * (criterionGAN(self.netD_fea(fea_src), True) + criterionGAN(self.netD_fea(fea_tgt),
        #                                                                                        False)) * 0.5
        # seg loss
        l_g_total += self.cri_seg_loss(pred_src, labels)

        l_g_total.backward()
        self.optimizer_G.step()
        self.optimizer_G_st.step()

        # =================training Ds==================================
        l_d_total = 0
        set_requires_grad([self.netD_s, self.netD_t], True)
        #self.optimizer_D_fea.zero_grad()
        self.optimizer_D_st.zero_grad()

        # loss_d_fea = criterionGAN(self.netD_fea(fea_src.detach()), False) + criterionGAN(
        #     self.netD_fea(fea_tgt.detach()), True) * 0.5
        # loss_d_fea.backward()

        loss_d_s = criterionGAN(self.netD_s(fake_s.detach()), False) + criterionGAN(self.netD_s(images_s),
                                                                                    True) * 0.5
        loss_d_s.backward()

        loss_d_t = criterionGAN(self.netD_t(fake_t.detach()), False) + criterionGAN(self.netD_t(images_t),
                                                                                    True) * 0.5
        loss_d_t.backward()

        self.optimizer_D_st.step()
        #self.optimizer_D_fea.step()

        # ================set log=======================================
        self.log_dict['l_g_total'] = l_g_total.item()
        self.log_dict['l_d_total'] = loss_d_s.item() + loss_d_t.item()










    def optimize_parameters_AdvSeg_PFO(self,step):
        '''
        align domain using pixel- feature- and outspace-level feature maps:
        using spatial-aware pixel features,
        ref:ROAD: Reality Oriented Adaptation for Semantic Segmentation of Urban Scenes
        SRDA-Net: Super-Resolution Domain Adaptation Networks for Semantic Segmentation

        using multi-scale feature alignment, ref:An End-to-End Network for Remote Sensing Imagery Semantic Segmentation via Joint
Pixel- and Representation-Level Domain Adaptation

        :param step:
        :return:
        '''
        bce_loss = self.bce_loss
        #mse_loss=nn.MSELoss()

        source_label = 1
        target_label = 0
        from losses.myLoss import N4ASCLoss, N8ASCLoss
        # ======================================================================================
        # train G
        # ======================================================================================
        if step<self.PREHEAT_STEPS:
            if step % self.G_update_ratio == 0:
                # ===========Remove Grads in D==============
                # for p in self.netD_seg.parameters():
                #     p.requires_grad = False
                # for p in self.netD_fea.parameters():
                #     p.requires_grad = False
                self.optimizer_G.zero_grad()
                l_g_total = 0
                # =======Train with Source=============
                from models.utils import entropy_loss_ita
                images_s, labels = self.batch_s['img'], self.batch_s['label']
                images_s, labels = Variable(images_s.to(self.device)), Variable(labels.to(self.device))
                images_t = self.batch_t['img']
                images_t = Variable(images_t.to(self.device))
                #
                # meanB = np.load(self.config.meanB)
                # stdB = np.load(self.config.stdB)
                # meanB = torch.from_numpy(meanB).float().to(self.device)
                # stdB = torch.from_numpy(stdB).float().to(self.device)
                # # self.netG_AdaIN.eval()
                # # self.netG_AdaIN.load_state_dict(torch.load(self.config.model_style_path))
                # with torch.no_grad():
                #     images_st = self.netG_AdaIN(images_s, meanB, stdB)
                # images_st = torch.clamp(images_st, 0.0, 1.0)
                # =======transfer image_s to image_st using wallis filter======

                from models.Satt_CD.modules.utils import adaptive_instance_normalization
                images_st = adaptive_instance_normalization(images_s, images_t)
                images_st = torch.clamp(images_st, 0.0, 1.0)

                images_st1 = adaptive_instance_normalization(images_s, images_t, alpha=0.8)#using wallis filter for image-level alignment
                images_st1 = torch.clamp(images_st1, 0.0, 1.0)

                fea_src, pred_src = self.netG(images_st)
                fea_src1, pred_src1 = self.netG(images_st1)
                fea_src, pred_src = (fea_src + fea_src1) * 0.5, (pred_src + pred_src1) * 0.5

                loss_seg = self.cri_seg_loss(pred_src, labels)
                loss_ent = entropy_loss_ita(pred_src)
                #loss_ent=N8ASCLoss(pred_src)

                l_g_total += loss_seg + self.lam_ent * loss_ent

                # # ======Train with Target==============================
                #
                # fea_tgt, pred_tgt = self.netG(images_t)
                #
                # D_out_fea = self.netD_fea(fea_tgt)
                # loss_adv_fea = bce_loss(D_out_fea,
                #                         Variable(torch.FloatTensor(D_out_fea.data.size()).fill_(source_label)).cuda(
                #                             0))
                # loss_adv = self.lam_adv * (loss_adv_fea)
                # l_g_total += loss_adv

                l_g_total.backward()
                self.optimizer_G.step()
                self.log_dict['l_g_total'] = l_g_total.item()
                self.log_dict['l_d_total'] = 0
        else:
            if step % self.G_update_ratio == 0:
                # ===========Remove Grads in D==============
                # for p in self.netD_seg.parameters():
                #     p.requires_grad = False
                for p in self.netD_fea.parameters():
                    p.requires_grad = False
                self.optimizer_G.zero_grad()
                l_g_total = 0
                # =======Train with Source=============
                from models.utils import entropy_loss_ita
                images_s, labels = self.batch_s['img'], self.batch_s['label']
                images_s, labels = Variable(images_s.to(self.device)), Variable(labels.to(self.device))
                images_t = self.batch_t['img']
                images_t = Variable(images_t.to(self.device))

                # =======transfer image_s to image_st using wallis filter======

                from models.Satt_CD.modules.utils import adaptive_instance_normalization
                images_st = adaptive_instance_normalization(images_s, images_t)
                images_st = torch.clamp(images_st, 0.0, 1.0)

                images_st1 = adaptive_instance_normalization(images_s, images_t, alpha=0.8)
                images_st1 = torch.clamp(images_st1, 0.0, 1.0)

                # images_st2 = adaptive_instance_normalization(images_s, images_t, alpha=0.9)
                # images_st2 = torch.clamp(images_st2, 0.0, 1.0)

                fea_src, pred_src = self.netG(images_st)
                fea_src1, pred_src1 = self.netG(images_st1)
                #fea_src2, pred_src2 = self.netG(images_st2)
                fea_src, pred_src = (fea_src + fea_src1) * 0.5, (pred_src + pred_src1) * 0.5

                loss_seg = self.cri_seg_loss(pred_src, labels)
                loss_ent = entropy_loss_ita(pred_src)
                #loss_ent=N8ASCLoss(pred_src)
                l_g_total += loss_seg + self.lam_ent * loss_ent

                # ==========================Train with Target==============================

                fea_tgt, pred_tgt = self.netG(images_t)

                D_out_fea = self.netD_fea(fea_tgt)
                loss_adv_fea = bce_loss(D_out_fea,
                                        Variable(torch.FloatTensor(D_out_fea.data.size()).fill_(source_label)).cuda(
                                            0))


                loss_adv = self.lam_adv * (loss_adv_fea)
                l_g_total += loss_adv
                #================using class-mix for domain adaption in the output space=================
                #=====ref:DACS: DOMAIN ADAPTATION VIA CROSS-DOMAIN MIXED SAMPLING========
                #====1,forward source with student model, 2, forward target with teacher model, 3, generate classmix mask, 4 train the mixmask using student model









                l_g_total.backward()
                self.optimizer_G.step()
            if step % self.D_update_ratio == 0:
                l_d_fea = 0
                # ==============Bring back Grads in D================================
                for param in self.netD_fea.parameters():
                    param.requires_grad = True
                self.optimizer_D_fea.zero_grad()
                # ==============================================================
                fea_src = fea_src.detach()

                D_out_s = self.netD_fea(fea_src)  # the input of D is probability map sum(dim=1)=1
                loss_D_s = bce_loss(D_out_s,
                                    Variable(torch.FloatTensor(D_out_s.data.size()).fill_(source_label)).cuda(0))

                fea_tgt = fea_tgt.detach()
                D_out_t = self.netD_fea(fea_tgt)
                loss_D_t = bce_loss(D_out_t,
                                    Variable(torch.FloatTensor(D_out_t.data.size()).fill_(target_label)).cuda(0))
                l_d_fea = (loss_D_s + loss_D_t) / 2
                l_d_fea.backward()
                self.optimizer_D_fea.step()

                # ================set log=======================================
                self.log_dict['l_g_total'] = l_g_total.item()
                self.log_dict['l_d_total'] = l_d_fea.item()









        #======for fea and outspace alighment=================

        # fea_tgt, pred_tgt = self.netG(images_t)
        #
        # D_out_fea=self.netD_fea(fea_tgt)
        # loss_adv_fea = bce_loss(D_out_fea, Variable(torch.FloatTensor(D_out_fea.data.size()).fill_(source_label)).cuda(
        #     0))
        # D_out_seg = self.netD_seg(torch.cat([pred_tgt,images_t],dim=1))
        # loss_adv_seg= bce_loss(D_out_seg, Variable(torch.FloatTensor(D_out_seg.data.size()).fill_(source_label)).cuda(
        #     0))  # source_label=0
        # =========for pixel alighment============================================

        # loss_adv =self.lam_adv*(loss_adv_fea)
        # l_g_total += loss_adv
        #=========================================================================

        # ======================================================================================
        #================================train D=================================================
        # ======================================================================================


        # for param in self.netD_seg.parameters():
        #     param.requires_grad = True
        # self.optimizer_D_seg.zero_grad()
        #========Train with Source=======================
        # pred_src = pred_src.detach()
        #
        # D_out_s = self.netD_seg(torch.cat([pred_src,images_s],dim=1))  # the input of D is probability map sum(dim=1)=1
        # loss_D_s = bce_loss(D_out_s, Variable(torch.FloatTensor(D_out_s.data.size()).fill_(source_label)).cuda(0))
        #
        # pred_tgt=pred_tgt.detach()
        # D_out_t = self.netD_seg(torch.cat([pred_tgt,images_t],dim=1))
        # loss_D_t = bce_loss(D_out_t,
        #                         Variable(torch.FloatTensor(D_out_t.data.size()).fill_(target_label)).cuda(0))
        # l_d_seg = (loss_D_s + loss_D_t) / 2
        # l_d_seg.backward()
        # self.optimizer_D_seg.step()

































    def get_current_log(self):
        return self.log_dict



    def print_network(self):
        # Generator
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel):
            net_struc_str = '{} - {}'.format(self.netG.__class__.__name__,
                                             self.netG.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)

        logger.info('Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
        logger.info(s)
        if self.is_train:
            # Discriminator
            s, n = self.get_network_description(self.netD)
            if isinstance(self.netD, nn.DataParallel):
                net_struc_str = '{} - {}'.format(self.netD.__class__.__name__,
                                                 self.netD.module.__class__.__name__)
            else:
                net_struc_str = '{}'.format(self.netD.__class__.__name__)

            logger.info('Network D structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
            logger.info(s)

            if self.cri_fea:  # F, Perceptual Network
                s, n = self.get_network_description(self.netF)
                if isinstance(self.netF, nn.DataParallel):
                    net_struc_str = '{} - {}'.format(self.netF.__class__.__name__,
                                                     self.netF.module.__class__.__name__)
                else:
                    net_struc_str = '{}'.format(self.netF.__class__.__name__)

                logger.info('Network F structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
                logger.info(s)

    def load(self):
        load_path_G = self.opt['path']['pretrain_model_G']
        if load_path_G is not None:
            logger.info('Loading pretrained model for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netG)
        load_path_D = self.opt['path']['pretrain_model_D']
        if self.opt['is_train'] and load_path_D is not None:
            logger.info('Loading pretrained model for D [{:s}] ...'.format(load_path_D))
            self.load_network(load_path_D, self.netD)

    def save_best(self,ratio_name=None):
        #model_path=self.config.model_name
        if ratio_name:
            #model_path = self.config.model_dir + '/' + self.config.pred_name + '_'+ratio_name+'_best_acc.pth'
            model_path = self.config.model_dir + '/' + self.config.pred_name + '_best_acc.pth'
        else:
            model_path = self.config.model_dir + '/' + self.config.pred_name + '_best_acc.pth'

        torch.save(self.netG.state_dict(),model_path)


    def save_style(self):
        if self.config["DA_method"] == "CycleFea":
            model_path0=self.config.model_dir + '/' + self.config.pred_style_name + '_final_iter.pth'
            torch.save(self.netG_t.state_dict(), model_path0)
            self.save_final()
        if self.config["DA_method"] == "AdaIn":
            model_path0 = self.config.model_dir + '/' + self.config.pred_style_name + '_final_iter.pth'
            torch.save(self.netG_AdaIN.state_dict(), model_path0)
        if self.config["DA_method"] == "Atk":
            model_path0 = self.config.model_dir + '/' + self.config.pred_style_name + '_final_iter.pth'
            torch.save(self.netG_atk.state_dict(), model_path0)
        if self.config["DA_method"] == "CycleFea_DA":
            model_path0 = self.config.model_dir + '/' + self.config.pred_style_name + '_final_iter.pth'
            torch.save(self.netG_t.state_dict(), model_path0)
            self.save_final()

    def save_final(self,ratio_name=None):
        #model_path=self.config.model_name
        if ratio_name:
            #model_path = self.config.model_dir + '/' + self.config.pred_name +'_'+ ratio_name+'_final_iter.pth'
            model_path = self.config.model_dir + '/' + self.config.pred_name + '_final_iter.pth'
        else:
            model_path = self.config.model_dir + '/' + self.config.pred_name + '_final_iter.pth'

        torch.save(self.netG.state_dict(),model_path)


    def save(self, iter_step,save_atk=False,save_style=False):
        # if save_atk:
        #     self.save_network(self.netG_atk, 'Gatk', iter_step)
        # elif save_style:
        #     self.save_network(self.netG_AdaIN, 'Gstyle', iter_step)
        # else:
        #     #self.save_network(self.netG, 'G', iter_step)

        model_path = self.config.model_dir + '/' + self.config.pred_name + '_final_iter.pth'
        torch.save(self.netG.state_dict(), model_path)
        if self.config["DA_method"] == "CycleFea_DA" or self.config["DA_method"] == "Cycle_DA":
            model_path0 = self.config.model_dir + '/' + self.config.pred_style_name + '_final_iter.pth'
            torch.save(self.netG_t.state_dict(), model_path0)


if __name__ == '__main__':
    print('#### Test Case ###')
    from torch.autograd import Variable
    import sys

    sys.path.append('..')
    # num_classes=2
    # x = Variable(torch.rand(2, 3, 256, 256))
    from configs.config_utils import process_config, get_train_args

    opt_path = r'configs/config.json'
    opt = process_config(opt_path)
    model = BRSeg_Model(opt)
    img = Variable(torch.rand(2, 3, 256, 256)).cuda(0)
    label = Variable(torch.rand(2, 1, 256, 256)).cuda(0)
    train_data = img.label
    model.feed_data(train_data)
    seg_map=model.netG(img)
    print(seg_map[0].size())




