# -- coding: utf-8 --
"""
Copyright (c) 2018. All rights reserved.
Created by C. L. Wang on 2018/4/18
"""

from scipy.io import loadmat
import scipy.misc as smi
from utils.utils import mkdir_if_not_exist

import numpy as np
import random
import cv2
import math
from tqdm import tqdm
import glob
import os
import json


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)


class RSCD_DL(object):
    def __init__(self, config=None):

        #====================load RS data========================

        config.img_dir = config.data_dir + '/result/img'
        config.model_dir = config.data_dir + '/result/model'
        config.log_dir = config.data_dir + '/result/log'
        config.ray_dir = config.data_dir + '/result/ray'
        #==============================for train===================================================

        config.train_dir = config.data_dir + '/train'
        #=========================for src_tgt==============================================

        str0 = '\\'
        str1 = 'file'
        idx0 = config.data_dir.find(str0, 14)
        idx1 = config.data_dir.find(str1, 14)
        src_name = (config.data_dir[idx0 + 1:idx1 - 1])
        idx0 = config.data_dir_tgt.find(str0, 14)
        idx1 = config.data_dir_tgt.find(str1, 14)
        tgt_name = (config.data_dir_tgt[idx0 + 1:idx1 - 1])
        #if config["train"]["visit_tgt"]:
        if config.use_DA:
            #config["train"]["visit_tgt"]=True
            src_tgt_name = src_name + '_' + tgt_name
        else:
            #config["train"]["visit_tgt"] = False
            src_tgt_name = src_name + '_' + src_name
        config.src_name=src_name
        config.tgt_name=tgt_name
        config.src_tgt_name=src_tgt_name
        src_src_name=src_name + '_' + src_name

        if config.use_DA:
            if config.DA_method=='CycleFea_DA':
                if src_name == 'Massachusett_Building' or tgt_name == 'Massachusett_Building':
                    config["patch_size"] = 128
                    config["batch_size"] = 8
                else:
                    config["patch_size"] = 512
                    config["batch_size"] = 2

            else:
                if src_name == 'Massachusett_Building' or tgt_name == 'Massachusett_Building':
                    config["patch_size"] = 128
                    config["batch_size"] = 16
                    config["network_G"]["which_model_G"]="UNet_2D_PreTrain256_ED2"
                    config["network_D_fea"]["which_model_D"]="discriminator_fc_8"
                    config["network_D_fea"]["in_nc"] =256

                else:
                    config["patch_size"] = 512
                    config["batch_size"] = 4
                    config["network_G"]["which_model_G"] = "UNet_2D_PreTrain512_ED2_drop"
                    config["network_D_fea"]["which_model_D"] = "discriminator_fc_16"
                    config["network_D_fea"]["in_nc"] = 512


            if config.use_resample == False:
                config["patch_size"] = 512
                config["batch_size"] = 4

        else:
            if src_name == 'Massachusett_Building' or tgt_name == 'Massachusett_Building':
                config["patch_size"] = 512#128 for DA
                config["batch_size"] = 8#16 for DA
            else:
                config["patch_size"] = 512
                config["batch_size"] = 8




        if config["train"]["use_SSL"]:

            config["train"]["lr_G"] = 1.5e-5
            config["train"]["ssl_gamma"] = 4.0
            config["ramp_up_end"] = 0.0
            config["train"]["nepoch"] = 10
            if src_name == 'Massachusett_Building' or tgt_name == 'Massachusett_Building':
                config["batch_size"] = 32
            else:
                config["batch_size"] = 8
        else:

            config["train"]["lr_G"] = 1e-4
            config["train"]["ssl_gamma"] = 0.0
            #config["ramp_up_end"] = 0.4
            config["train"]["nepoch"] = 20
            if config.DA_method == 'CycleFea_DA':
                config["train"]["nepoch"] = 20#for mas, FCN-128,else 10
            if config.DA_method == 'Atk':
                config["train"]["nepoch"] = 20#

        if config.use_DA:
            config["train"]["visit_tgt"] = True
            config.pred_name = 'netG_AS2_Tea2R1_{}_{}_SSL_{}_aff_{}_edge_{}_contype_{}_pre_steps_{}_ramp_{}_gamma_{}_delta_{}_lamE_{}_lamF_{}_{}_{}_patch_{}_batch_{}_nepoch_{}'.format(
                config["train"]["opti_type"],
                config["DA_method"],
                config["train"]["use_SSL"],config["train"]["use_aff"],config["train"]["use_edge"], config["train"]["con_type"],config["train"]["pre_steps"],
                config["ramp_up_end"],
                config["train"]["ssl_gamma"],
                config["unsupervised_w"],
                config["train"]["lam_ent"],
                config["train"]["lam_adv"],
                src_tgt_name,

                config["network_G"][
                    "which_model_G"],

                config.patch_size,
                config.batch_size,
                config["train"][
                    "nepoch"])



        else:
            config["train"]["visit_tgt"]=False
            if not config["network_G"]["use_DS"]:
                config["network_G"]["multi_outputs"]==False

            config.pred_name = 'netG_{}_{}_dblock_{}_useDCN2_{}_patch_{}_batch_{}_nepoch_{}'.format(

                src_tgt_name,

                config["network_G"][
                    "which_model_G"],
                config["network_G"][
                    "dblock_type"],
                config["network_G"][
                    "use_DCN"],

                config.patch_size,
                config.batch_size,
                config["train"][
                    "nepoch"],
                )


        config.pred_style_name='netG_{}_{}_epoch_{}'.format(config["DA_method"],src_tgt_name,config["train"][
                "nepoch"])



        if config["DA_method"]=='Atk_DA' or config["DA_method"]=='Atk':
            config.pred_name0 = 'netG5_{}_{}_patch_{}_batch_{}_nepoch_{}'.format(

                src_src_name,

                config["network_G"][
                    "which_model_G"],

                config.patch_size,
                #16,#for mas
                8,
                20)

            config.pred_style_name0 = 'netG_{}_{}_epoch_{}'.format(
                # "AdaIn",
                "Atk",
                src_tgt_name,
                20
            )
        else:
            config.pred_name0 = 'netG5_aug_SSL_{}_contype_{}_ramp_{}_gamma_{}_{}_{}_patch_{}_batch_{}_nepoch_{}'.format(
                'false', config["train"]["con_type"],
                0.4,
                0.0,
                src_tgt_name,

                config["network_G"][
                    "which_model_G"],

                config.patch_size,
                4,
                20)  # for SSL
            config.pred_style_name0 = 'netG_{}_{}_epoch_{}'.format(
                 "AdaIn",
                src_tgt_name,
                #config["train"]["nepoch"],  # 10 for Atk
                 20#for AdaIn
            )

        #===============for tgt=======================
        if config["train"]["visit_tgt"]:
            config.test_dir = config.data_dir_tgt + '/test'
            config.test_pred_dir = config.data_dir_tgt  + '/test/pred'
            #for pseudo label dir
            config.pseudo_label_dir=config.data_dir_tgt+'/train/plabel'+'_'+config.src_tgt_name
            #mkdir_if_not_exist(config.pseudo_label_dir)

        else:

            config.test_dir = config.data_dir + '\\test'
            config.test_pred_dir = config.data_dir + '\\test\\pred'

        mkdir_if_not_exist(config.test_pred_dir)

        config.pretrained_model_path =config.model_dir+'/'+config.pred_name0+'_best_acc.pth'# for Atk, SSL
        #config.style_model_path = config.model_dir + '/' + config.pred_style_name0 + '_final_iter.pth'#for AdaIn,Atk

        if config.mode=="Test":
           config["train"]["mode"]='supervised'

           config.pred_dir = config.test_pred_dir + '/pred_' + config.pred_name

           mkdir_if_not_exist(config.pred_dir)
           mkdir_if_not_exist(config.pred_dir + '/Binary')
           mkdir_if_not_exist(config.pred_dir + '/ReA')
           config.precision_path = config.pred_dir + '/precision.txt'

        print("pred_model is {}".format(config.pred_name))



        config.model_name = config.model_dir + '/'+config.pred_name+'.pth'

        #==============for log=========================
        config.json_name = config.model_dir + '/min_max.json'
        config.loss_path = config.img_dir + '/' + config.pred_name + '.png'
        config.log_path = config.model_dir + '/' + config.pred_name + '.txt'

        #=============================================
        mkdir_if_not_exist(config.model_dir)
        mkdir_if_not_exist(config.img_dir)
        mkdir_if_not_exist(config.log_dir)
        mkdir_if_not_exist(config.ray_dir)
        #===================================================

        self.config=config
        self.data_dir=config.data_dir
        self.train_dir=config.train_dir

        self.test_dir = config.test_dir
        self.val_dir=config.data_dir+'/val'
        self.json_name=config.json_name


    def get_train_data(self):
        return self.X_train, self.y_train

    def get_test_data(self):
        return self.X_test, self.y_test








