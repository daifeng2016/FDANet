import math
import os, time,sys
import numpy as np
from PIL import Image
import os.path as osp

# import torch modules
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from data_loaders.RSCD_dl import RSCD_DL
from torchsummary import summary
import matplotlib.pyplot as plt
import logging
from losses.myLoss import bce_edge_loss
from utils.utils import PR_score_whole

from tqdm import tqdm
import cv2

from utils.utils import mkdir_if_not_exist
import random
from sklearn.model_selection import train_test_split
from functools import partial
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from data_loaders.data_proc import ToTensor_BR2,RandomCrop,RandomFlip,RandomRotate
from sklearn.model_selection import train_test_split
from data_loaders.data_proc import ImagesetDatasetBR,ToTensor_BR
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

class TrainerOptimDA(object):
    # init function for class
    def __init__(self, config,trainDataloader_src, valDataloader_src,trainDataloader_tgt, valDataloader_tgt,train_list_tgt=None
                ):

        self.config = config
        self.model_path = config.model_name
        self.log_file = config.log_path
        self.lossImg_path=config.loss_path
        self.train_list_tgt=train_list_tgt

        # set the GPU flag

        self.device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


        self.valDataloader_src = valDataloader_src
        self.trainDataloader_src = trainDataloader_src
        self.valDataloader_tgt = valDataloader_tgt
        self.trainDataloader_tgt = trainDataloader_tgt



        self.pix_cri=bce_edge_loss(use_edge=config["train"]["use_edge"]).to(self.device)
        self.pix_cri0 = bce_edge_loss(use_edge=False).to(self.device)




    def train_optim_UDA(self):
        # create model
        start_time = time.perf_counter()
        from models.Satt_CD import create_model
        from utils.utils import setup_logger
        # model = create_model(self.config)
        # resume state??
        setup_logger(None, self.config.log_dir, 'train_' + self.config.pred_name, level=logging.INFO,
                     screen=True)  # print info in the terminal and write the log file
        # setup_logger('val', self.config.log_dir, 'val_'+self.config.pred_name, level=logging.INFO)
        logger = logging.getLogger('base')
        current_step = 0
        train_history = {"loss": [],
                         "acc": [],
                         "val_loss_src": [],
                         "val_acc_src": [],
                         "val_loss_tgt": [],
                         "val_acc_tgt": [],
                         }
        total_epochs = self.config['train']['nepoch']  #
        total_iters = int(total_epochs * max(len(self.trainDataloader_src),len(self.trainDataloader_tgt)))
        self.config['train']['niter'] = total_iters
        self.config["train"]["lr_steps"] = [int(0.35 * total_iters),int(0.65 * total_iters), int(0.85 * total_iters)]
        self.config['logger']['save_iter'] = [int(1.0 * total_iters)-1]
        model = create_model(self.config)  # create model after updating config
        multi_outputs = self.config["network_G"]["multi_outputs"]
        #==============using total iters to indicate training process==========================
        best_f1=0
        trainloader_s_iter = iter(self.trainDataloader_src)
        trainloader_t_iter = iter(self.trainDataloader_tgt)
        #===========================for ramps============================
        rampup_starts = int(self.config['ramp_up_start'] * self.config['train']['nepoch'])
        rampup_ends = int(self.config['ramp_up_end'] * self.config['train']['nepoch'])  # "ramp_up": 0.1, "epochs": 80
        pre_steps = int(self.config['train']['nepoch'] * self.config["train"]["pre_steps"])
        if pre_steps > 0:
            rampup_starts = pre_steps
            # rampup_ends+=pre_steps
        print("rampup_starts={},rampup_end={}".format(rampup_starts, rampup_ends))
        from models.Satt_CD.modules.loss import consistency_weight
        cons_w_unsup = consistency_weight(final_w=self.config['unsupervised_w'], iters_per_epoch=self.config["iter_per_epoch"],
                                          rampup_starts=rampup_starts,
                                          rampup_ends=rampup_ends)#ramp_val

        if self.config['train']['use_SSL']:
           model.netG.load_state_dict(torch.load(self.config.pretrained_model_path))
           print("loading pretrained moddel {}".format(self.config.pretrained_model_path))

        #=============load model==========================
        # if self.config["train"]["test_style"]:
        #   model.netG_AdaIN.eval()
        #   model.netG_AdaIN.load_state_dict(torch.load(self.config.model_style_path))
        #
        # model.netG.load_state_dict(torch.load(self.config.pretrained_model_path))

        for iter_step in tqdm(range(0,total_iters)):
            try:
                batch_s=next(trainloader_s_iter)
            except:
                trainloader_s_iter=iter(self.trainDataloader_src)
                batch_s = next(trainloader_s_iter)

            try:
                batch_t=next(trainloader_t_iter)
            except:
                trainloader_t_iter=iter(self.trainDataloader_tgt)
                batch_t = next(trainloader_t_iter)

            model.feed_data_batch_st(batch_s,batch_t)
            model.netG.train()
            #============for different optimze============



            if self.config["DA_method"] == "CycleFea_DA":  # ==========for CycleGAN Feature-level alignment DA=============
                model.netD_s.train()
                model.netD_t.train()
                model.netG_s.train()
                model.netG_t.train()
                model.optimize_parameters_CycleFea(iter_step)
            elif self.config[
                         "DA_method"] == "Cycle_DA":  # ==========for CycleGAN Feature-level alignment DA=============
                model.netD_s.train()
                model.netD_t.train()
                model.netG_s.train()
                model.netG_t.train()
                model.optimize_parameters_Cycle(iter_step)

            else:
                model.optimize_parameters_AdvSeg_PFO_Con(iter_step,total_iters,cons_w_unsup(iter_step))# cons_w_unsup(9*2000)   using transform consistency loss


            # update learning rate
            #model.update_learning_rate()
            current_step=iter_step
            if current_step % self.config['logger']['print_freq'] == 0:
                logs = model.get_current_log()
                message = '<iter:{:8,d}, lr:{:.3e}, lossG:{:.6f}, lossD:{:.6f}> '.format(
                    iter_step, model.get_current_learning_rate(),
                    logs['l_g_total'],logs['l_d_total'])

                logger.info(message)
                # =======for val test======================================
                # val_loss_src, val_acc_src, val_loss_tgt, val_acc_tgt= self.val_DA(model=model, multi_outputs=multi_outputs,out_map=self.config["network_G"]["out_map"])
                # message = '<val_loss_src:{:.6f},val_f1_score_src:{:.6f},val_loss_tgt:{:.6f},val_f1_score_tgt:{:.6f}>'.format(
                #     val_loss_src, val_acc_src, val_loss_tgt, val_acc_tgt)
                # logger.info(message)
                #============================================================
            model.save(current_step)
            if current_step in self.config['logger']['save_iter']:


                    if self.config['train']['train_style']:
                        logger.info('Saving netG_style...')
                        model.save_style()
                    else:
                        logger.info('Saving netG...')
                        model.save(current_step)

            #if current_step >=0:#====for debug test
            if current_step > 0 and current_step%self.config['train']['val_iter'] == 0 :
                    if self.config['train']['train_style']==False:
                        val_loss_src, val_acc_src, val_loss_tgt, val_acc_tgt = self.val_DA(model=model,
                                                                                           multi_outputs=multi_outputs,out_map=self.config["network_G"]["out_map"])
                        message = '<val_loss_src:{:.6f},val_f1_score_src:{:.6f},val_loss_tgt:{:.6f},val_f1_score_tgt:{:.6f}>'.format(
                            val_loss_src, val_acc_src, val_loss_tgt, val_acc_tgt)
                        logger.info(message)
                        logs = model.get_current_log()
                        train_history["loss"].append(logs['l_g_total'])

                        train_history["val_loss_src"].append(val_loss_src)
                        train_history["val_acc_src"].append(val_acc_src)
                        train_history["val_loss_tgt"].append(val_loss_tgt)
                        train_history["val_acc_tgt"].append(val_acc_tgt)
                        if val_acc_tgt > best_f1:#not reasonable, should choose from test_tgt?
                            best_f1 = val_acc_tgt
                            model.save_best()

                    else:
                        #model.save_best()
                        logs = model.get_current_log()
                        train_history["loss"].append(logs['l_g_total'])

        end_time = time.perf_counter()
        run_time = end_time - start_time
        # print(end_time - start_time, 'seconds')
        message = 'running time is {:.4f} seconds!'.format(run_time)
        logger.info(message)
        self.visualize_train_optim_DA(train_history)

    def train_optim_UDA_tune(self,config):
        # create model
        start_time = time.perf_counter()
        #======for parallel, dataset must be constructed at this function==================
        # ==========================using new dataloader for building and road extraction====================================
        #config=self.config
        data_dir_src = self.config.data_dir
        train_set_src = [pic for pic in
                         os.listdir(os.path.join(data_dir_src, 'train', 'img'))]  # for BR segmentation img.jpg
        data_dir_tgt = self.config.data_dir_tgt
        train_set_tgt = [pic for pic in os.listdir(os.path.join(data_dir_tgt, 'train', 'img'))]
        val_proportion = self.config.val_proportion
        batch_size = self.config.batch_size
        num_worker = self.config.num_worker
        train_list_src, val_list_src = train_test_split(train_set_src,
                                                        test_size=val_proportion,
                                                        random_state=1, shuffle=True)
        train_list_tgt, val_list_tgt = train_test_split(train_set_tgt,
                                                        test_size=val_proportion,
                                                        random_state=1, shuffle=True)
        train_transforms_LR = transforms.Compose([
            # RandomCrop(128, 128),#for whu-mas

            RandomFlip(),
            RandomRotate(),
            ToTensor_BR(),
            # Normalize_BR([0.5,0.5,0.5], [0.5,0.5,0.5])
        ])
        train_transforms_LR_128 = transforms.Compose([
            RandomCrop(128, 128),  # for whu-mas

            RandomFlip(),
            RandomRotate(),
            ToTensor_BR(),
            # Normalize_BR([0.5,0.5,0.5], [0.5,0.5,0.5])
        ])
        train_transforms_LR2 = transforms.Compose([
            # RandomCrop(128, 128),#for whu-mas

            RandomFlip(),
            RandomRotate(),
            ToTensor_BR2(),  # with ignore_label

        ])
        train_transforms_LR2_128 = transforms.Compose([
            RandomCrop(128, 128),  # for whu-mas

            RandomFlip(),
            RandomRotate(),
            ToTensor_BR2(),  # with ignore_label

        ])
        train_transforms_HR = transforms.Compose([
            # RandomCrop(384, 384),#for whu-mas
            # ReSize(128,128),#for whu-mas it is very important to make the resolution between the source and target consistent

            RandomFlip(),
            RandomRotate(),
            ToTensor_BR(),  # with normal label
            # Normalize_BR([0.5,0.5,0.5], [0.5,0.5,0.5])
        ])
        train_transforms_HR_128 = transforms.Compose([
            RandomCrop(384, 384),  # for whu-mas
            ReSize(128, 128),
            # for whu-mas it is very important to make the resolution between the source and target consistent

            RandomFlip(),
            RandomRotate(),
            ToTensor_BR(),  # with normal label
            # Normalize_BR([0.5,0.5,0.5], [0.5,0.5,0.5])
        ])
        if self.config.src_name == 'Massachusett_Building':
            src_transforms = train_transforms_LR_128
            tgt_transforms = train_transforms_HR_128
            tgt_transforms2 = train_transforms_LR2_128
        elif self.config.tgt_name == 'Massachusett_Building':
            src_transforms = train_transforms_HR_128
            tgt_transforms = train_transforms_LR_128
            tgt_transforms2 = train_transforms_LR2_128
        else:
            src_transforms = train_transforms_HR
            tgt_transforms = train_transforms_HR
            tgt_transforms2 = train_transforms_LR2
        # ===========for different resolution test=============
        if self.config.use_resample == False:
            src_transforms = train_transforms_HR
            tgt_transforms = train_transforms_HR
            tgt_transforms2 = train_transforms_LR2



        train_dataset=ImagesetDatasetBR2(imset_list_src=train_list_src, imset_list_tgt=train_list_tgt,
                                         config=self.config, mode='Train',
                                              transform=src_transforms)

        trainDataloader= DataLoader(train_dataset, batch_size=batch_size,
                                          shuffle=True, num_workers=0,
                                          # collate_fn=collateFunction(),  # 如何取样本的，我们可以定义自己的函数来准确地实现想要的功能
                                          pin_memory=True, drop_last=True)  # len(train_dataloader)

        val_dataset = ImagesetDatasetBR2(imset_list_src=val_list_src, imset_list_tgt=val_list_tgt,
                                           config=self.config, mode='Train',
                                           transform=src_transforms)

        valDataloader = DataLoader(val_dataset, batch_size=batch_size,
                                     shuffle=False, num_workers=0,
                                     # collate_fn=collateFunction(),  # 如何取样本的，我们可以定义自己的函数来准确地实现想要的功能
                                     pin_memory=True, drop_last=True)









        from models.Satt_CD import create_model
        from utils.utils import setup_logger

        setup_logger(None, self.config.log_dir, 'train_' + self.config.pred_name, level=logging.INFO,
                     screen=True)  # print info in the terminal and write the log file

        logger = logging.getLogger('base')
        iters_per_epoch = max(len(self.trainDataloader_src), len(self.trainDataloader_tgt))
        #self.config['train']['nepoch']=15
        total_epochs = self.config['train']['nepoch']  #
        total_iters = int(total_epochs * max(len(self.trainDataloader_src),len(self.trainDataloader_tgt)))
        self.config['train']['niter'] = total_iters
        self.config["train"]["lr_steps"] = [int(0.35 * total_iters),int(0.65 * total_iters), int(0.85 * total_iters)]
        self.config['logger']['save_iter'] = [int(1.0 * total_iters)-1]
        model = create_model(self.config)  # create model after updating config
        multi_outputs = self.config["network_G"]["multi_outputs"]
        #==============using total iters to indicate training process==========================
        best_f1=0

        #===========================for ramps============================
        rampup_starts = int(self.config['ramp_up_start'] * self.config['train']['nepoch'])
        rampup_ends = int(self.config['ramp_up_end'] * self.config['train']['nepoch'])  # "ramp_up": 0.1, "epochs": 80
        pre_steps = int(self.config['train']['nepoch'] * self.config["train"]["pre_steps"])
        if pre_steps > 0:
            rampup_starts = pre_steps
            # rampup_ends+=pre_steps
        print("rampup_starts={},rampup_end={}".format(rampup_starts, rampup_ends))
        from models.Satt_CD.modules.loss import consistency_weight
        cons_w_unsup = consistency_weight(final_w=config['unsupervised_w'], iters_per_epoch=iters_per_epoch,
                                          rampup_starts=rampup_starts,
                                          rampup_ends=rampup_ends)#ramp_val


        #=============load model==========================


        for epoch in tqdm(range(0,total_epochs)):

            for iter, batch in enumerate(trainDataloader):
                # try:
                #     batch_s = next(trainloader_s_iter)
                # except:
                #     trainloader_s_iter = iter(trainDataloader_src)
                #     batch_t = next(trainloader_s_iter)
                # try:
                #     batch_t = next(trainloader_t_iter)
                # except:
                #     trainloader_t_iter = iter(trainDataloader_tgt)
                #     batch_t = next(trainloader_t_iter)
                batch_s={"img":batch["img_src"],"label":batch["label_src"]}
                batch_t= {"img": batch["img_tgt"], "label": batch["label_tgt"]}
                model.feed_data_batch_st(batch_s, batch_t)
                model.netG.train()
                iter_step=iter+iters_per_epoch*epoch
                model.optimize_parameters_AdvSeg_PFO_Con_tune(iter_step, total_iters, cons_w_unsup(iter_step), config)

            #val
            val_loss_src, val_acc_src= self.val_DA_tune(model=model,
                                                                               multi_outputs=multi_outputs,
                                                                               out_map=self.config["network_G"][
                                                                                   "out_map"],valDataloader=valDataloader)
            message = '<val_loss_src:{:.6f},val_f1_score_src:{:.6f}>'.format(
                val_loss_src, val_acc_src)
            logger.info(message)
            tune.report(loss=val_loss_src, accuracy=val_acc_src)#report each epoch

        end_time = time.perf_counter()
        run_time = end_time - start_time
        message = 'running time for this trial is {:.4f} seconds!'.format(run_time)
        logger.info(message)




    def parameters_tune(self,num_samples=1, max_num_epochs=20, gpus_per_trial=0.25):
        '''
        when using grid search, num_samples=1 will generate 4*3*3 trails
        '''

        # # 全局文件路径
        # data_dir = DATA_PATH
        # # 加载训练数据
        # load_data(data_dir)
        # 配置超参数搜索空间
        # 每次实验，Ray Tune会随机采样超参数组合，并行训练模型，找到最优参数组合
        config = {
            # 自定义采样方法
            #"l1": tune.sample_from(lambda _: 2 ** np.random.randint(2, 9)),
            #"l2": tune.sample_from(lambda _: 2 ** np.random.randint(2, 9)),
            # 随机分布采样
            #"lr": tune.loguniform(1e-4, 1e-1),
            # 从类别型值中随机选择
            #"batch_size": tune.choice([2, 4, 8, 16])
            "unsupervised_w":tune.grid_search([0.1,1,5,10]),
            "lam_ent": tune.grid_search([0.001,0.005,0.01]),
            "lam_adv": tune.grid_search([0.001,0.005,0.01])
        }
        # ASHAScheduler会根据指定标准提前中止坏实验
        scheduler = ASHAScheduler(
            metric="accuracy",
            mode="max",
            max_t=max_num_epochs,
            grace_period=1,
            reduction_factor=2)
        # 在命令行打印实验报告
        reporter = CLIReporter(

            metric_columns=["loss", "accuracy","training_iteration"])
        # 执行训练过程
        result = tune.run(
            #partial(train_cifar, data_dir=data_dir),
            self.train_optim_UDA_tune,
            # 指定训练资源
            resources_per_trial={"cpu": 8, "gpu": gpus_per_trial},
            config=config,
            num_samples=num_samples,
            scheduler=scheduler,
            local_dir=os.path.join(self.config.ray_dir, 'ray_results'),
            progress_reporter=reporter)

        # 找出最佳实验
        best_trial = result.get_best_trial("accuracy", "max", "last")
        # 打印最佳实验的参数配置
        print("Best trial config: {}".format(best_trial.config))
        print("Best trial final validation loss: {}".format(
            best_trial.last_result["loss"]))
        print("Best trial final validation accuracy: {}".format(
            best_trial.last_result["accuracy"]))












    def generate_pseudo_labels2(self, model, model_path, pseudo_label_dir, label_ratio,pseudo_label_num, testDataloader,use_tmp=False):
        '''
        #===generate pseudo labels, then fine-tune the network using both source labels and target labels
        the value of output label map is: 0 non_building pixls, 1 building pixels, 255  ignore pixels
        :param targetloader:
        :return:
        '''

        predicted_label = np.zeros((pseudo_label_num, 512, 512), dtype='uint8')
        predicted_prob = np.zeros((pseudo_label_num, 512, 512))
        predicted_prob1 = np.zeros((pseudo_label_num, 512, 512))
        image_name = []

        model.eval()
        #model.load_state_dict(torch.load(model_path))  # 通过网络参数形式加载网络

        for index, batch in enumerate(tqdm(testDataloader, 0)):
            print("\rprocessing image %d" % index,end='')
            if index>=pseudo_label_num:
                break
            image, name = batch['img'], batch['name']
            #image = image.unsqueeze(0)
            H, W = image.shape[2], image.shape[3]
            with torch.no_grad():
                _, output1 = model(Variable(image).cuda())  # [1,1,H,W]
                output2 = torch.zeros(H, W, 2)
                output2[:, :, 0] = 1 - output1[0, 0, :, :]
                output2[:, :, 1] = output1[0, 0, :, :]
                output = output2.cpu().numpy()
                # output = output.transpose(1, 2, 0)  # [H,W,C]
                label, prob = np.argmax(output, axis=2), np.max(output, axis=2)  # np.unique(label)
                predicted_label[index] = label.copy()
                predicted_prob[index] = prob.copy()
                predicted_prob1[index]=output1[0].data.cpu().numpy()
                image_name.append(name)

        thres = []
        for i in range(2):  # compute a thresh for each class by using medium value of predicted_prob
            x = predicted_prob[predicted_label == i]
            if len(x) == 0:
                thres.append(0)
                continue
            x = np.sort(x)
            thres.append(x[np.int(np.round(len(x) * label_ratio))])

        thres = np.array(thres)
        print(thres)
        thres[thres > 0.9] = 0.9  # label soft?

        #===shuffle the img and only select the pseudo_num
        img_list=[]
        #random.shuffle(image_name)#wrong, orelse the image_name and the prob are not consistent
        # pseudo_label_array=np.arange(pseudo_label_num)
        # random.shuffle(pseudo_label_array)
        pseudo_label_dir1=pseudo_label_dir+'_tmp'
        mkdir_if_not_exist(pseudo_label_dir1)

        for index in range(pseudo_label_num):
            name = image_name[index]
            label = predicted_label[index]
            prob = predicted_prob[index]
            for i in range(2):
                label[(prob < thres[i]) * (label == i)] = 255  # label[0,0]
            print("\rwriting %d image" % index, end='')



            file_name = pseudo_label_dir + '/' + name[0] + '.png'  # cannot be jpg file, or else , label value is [0 ,1,255], while saved_img is [0,1,2,3,...255]
            cv2.imwrite(file_name, label)
            if use_tmp:
               file_name1=pseudo_label_dir1 + '/' + name[0] + '.png'
               prob1=predicted_prob1[index]*255
               cv2.imwrite(file_name1, prob1.astype('uint8'))

            img_list.append(name[0] + '.png')




        # for index in range(len(testDataloader)):
        #     name = image_name[index]
        #     label = predicted_label[index]
        #     prob = predicted_prob[index]
        #     for i in range(2):
        #         label[(prob < thres[i]) * (label == i)] = 255  # label[0,0]
        #     print("\rwriting %d image" % index, end='')
        #
        #     # output.save('%s/%s%s' % (self.config.pseudo_label_dir, name,'.jpg'))
        #     file_name = pseudo_label_dir + '/' + name + '.png'  # cannot be jpg file, or else , label value is [0 ,1,255], while saved_img is [0,1,2,3,...255]
        #     cv2.imwrite(file_name, label)
        return img_list

    def compute_entropy(self,arr, num_classes):
        tensor = torch.from_numpy(arr)
        predicted_entropy = torch.sum(torch.mul(tensor, torch.log(tensor)), dim=2) * (-1 / np.log(num_classes))
        return predicted_entropy.numpy()  # [H,W]

    def generate_pseudo_labels2_ESL(self, model, model_path, pseudo_label_dir, label_ratio,pseudo_label_num, testDataloader):
        '''
        #===generate pseudo labels, then fine-tune the network using both source labels and target labels
        the value of output label map is: 0 non_building pixls, 1 building pixels, 255  ignore pixels
        :param targetloader:
        :return:
        '''
        predicted_label = np.zeros((len(testDataloader), 512, 512), dtype='uint8')
        predicted_entropy = np.zeros((len(testDataloader), 512, 512))
        image_name = []

        model.eval()
        model.load_state_dict(torch.load(model_path))  # 通过网络参数形式加载网络

        for index, batch in enumerate(tqdm(testDataloader, 0)):
            print("\rprocessing image %d" % index,end='')
            #print("processing cur image %d" % index)
            # if (index )<len(targetloader):
            image, name = batch['img'], batch['name']
            #image = image.unsqueeze(0)
            H, W = image.shape[2], image.shape[3]
            with torch.no_grad():
                _, output1 = model(Variable(image).cuda())  # [1,1,H,W]
                output2 = torch.zeros(H, W, 2)
                output2[:, :, 0] = 1 - output1[0, 0, :, :]
                output2[:, :, 1] = output1[0, 0, :, :]
                output = output2.cpu().numpy()
                # output = output.transpose(1, 2, 0)  # [H,W,C]
                label, prob = np.argmax(output, axis=2), np.max(output, axis=2)  # np.unique(label)
                predicted_label[index] = label.copy()
                predicted_entropy[index] = self.compute_entropy(output,2)
                image_name.append(name)

        thres = []
        for i in range(2):  # compute a thresh for each class by using medium value of predicted_prob
            x = predicted_entropy[predicted_label == i]
            if len(x) == 0:
                thres.append(0)
                continue
            x = np.sort(x)
            thres.append(x[np.int(np.round(len(x) * label_ratio))])

        thres = np.array(thres)
        print(thres)
        #thres[thres > 0.9] = 0.9  # label soft?
        thres[thres < 0.1] = 0.1 #
        #===shuffle the img and only select the pseudo_num
        img_list=[]
        #random.shuffle(image_name)
        for index in range(pseudo_label_num):
            name = image_name[index]
            label = predicted_label[index]
            entropy = predicted_entropy[index]
            for i in range(2):
                label[(entropy > thres[i]) * (label == i)] = 255  # label[0,0]
            print("\rwriting %d image" % index, end='')

            # output.save('%s/%s%s' % (self.config.pseudo_label_dir, name,'.jpg'))
            file_name = pseudo_label_dir + '/' + name[0] + '.png'  # cannot be jpg file, or else , label value is [0 ,1,255], while saved_img is [0,1,2,3,...255]
            cv2.imwrite(file_name, label)
            img_list.append(name[0] + '.png')

        return img_list

    def generate_pseudo_labels2_ESL2(self, model, model_path, pseudo_label_dir, label_ratio,pseudo_label_num, testDataloader):
        '''
        #===generate pseudo labels, then fine-tune the network using both source labels and target labels
        the value of output label map is: 0 non_building pixls, 1 building pixels, 255  ignore pixels
        :param targetloader:
        :return:
        '''
        predicted_label = np.zeros((pseudo_label_num, 512, 512), dtype='uint8')
        predicted_entropy = np.zeros((pseudo_label_num, 512, 512))
        image_name = []

        model.eval()
        model.load_state_dict(torch.load(model_path))  # 通过网络参数形式加载网络
        #cur_img_name=0
        for index, batch in enumerate(tqdm(testDataloader, 0)):
            print("\rprocessing image %d" % index,end='')
            if index>=pseudo_label_num:
                break
            #print("processing cur image %d" % index)
            # if (index )<len(targetloader):
            image, name = batch['img'], batch['name']
            #image = image.unsqueeze(0)
            H, W = image.shape[2], image.shape[3]
            with torch.no_grad():
                _, output1 = model(Variable(image).cuda())  # [1,1,H,W]
                output2 = torch.zeros(H, W, 2)
                output2[:, :, 0] = 1 - output1[0, 0, :, :]
                output2[:, :, 1] = output1[0, 0, :, :]
                output = output2.cpu().numpy()
                # output = output.transpose(1, 2, 0)  # [H,W,C]
                label, prob = np.argmax(output, axis=2), np.max(output, axis=2)  # np.unique(label)
                predicted_label[index] = label.copy()
                predicted_entropy[index] = self.compute_entropy(output,2)
                image_name.append(name)
            #cur_img_name+=1

        thres = []
        for i in range(2):  # compute a thresh for each class by using medium value of predicted_prob
            x = predicted_entropy[predicted_label == i]
            if len(x) == 0:
                thres.append(0)
                continue
            x = np.sort(x)
            thres.append(x[np.int(np.round(len(x) * label_ratio))])

        thres = np.array(thres)
        print(thres)
        #thres[thres > 0.9] = 0.9  # label soft?
        thres[thres < 0.1] = 0.1 #
        #===shuffle the img and only select the pseudo_num
        img_list=[]
        random.shuffle(image_name)
        for index in range(pseudo_label_num):
            name = image_name[index]
            label = predicted_label[index]
            entropy = predicted_entropy[index]
            for i in range(2):
                label[(entropy > thres[i]) * (label == i)] = 255  # label[0,0]
            print("\rwriting %d image" % index, end='')

            # output.save('%s/%s%s' % (self.config.pseudo_label_dir, name,'.jpg'))
            file_name = pseudo_label_dir + '/' + name[0] + '.png'  # cannot be jpg file, or else , label value is [0 ,1,255], while saved_img is [0,1,2,3,...255]
            cv2.imwrite(file_name, label)
            img_list.append(name[0] + '.png')

        return img_list



    def train_optim_UDA_DT(self,use_tmp=False):
        '''
        use_tmp=True will output plabel_0.5_tmp file
        '''
        start_time = time.perf_counter()
        from models.Satt_CD import create_model
        from utils.utils import setup_logger

        setup_logger(None, self.config.log_dir, 'train_' + self.config.pred_name, level=logging.INFO,
                     screen=True)  # print info in the terminal and write the log file
        logger = logging.getLogger('base')
        current_step = 0
        train_history = {"loss": [],
                         "acc": [],
                         "val_loss_src": [],
                         "val_acc_src": [],
                         "val_loss_tgt": [],
                         "val_acc_tgt": [],
                         }
        # total_epochs = self.config['train']['nepoch']  #
        # total_iters = int(total_epochs * max(len(self.trainDataloader_src),len(self.trainDataloader_tgt)))
        # iters_per_epoch=max(len(self.trainDataloader_src),len(self.trainDataloader_tgt))
        # self.config['train']['niter'] = total_iters
        # self.config["train"]["lr_steps"] = [int(0.35 * total_iters),int(0.65 * total_iters), int(0.85 * total_iters)]
        # self.config['logger']['save_iter'] = [int(1.0 * total_iters)-1]
        model = create_model(self.config)  # create model after updating config
        multi_outputs = self.config["network_G"]["multi_outputs"]
        '''
        domain adaption using a curriculum-way training, namely selecting pseudo labels from easy to hard {20%, 40%, 60%, 80%, 100%},
        note that tgt_sample are the same between different rounds, but labeled pixels in each sample are different
        1) generate pseudo labels_{i} using saved best model,save in disk in the same dir with label
        2) generate tgt_dataloader with tgt images and pseudo labels
        3) restart the lr, fine-tune the netG using src_data and tgt_data during epoch_per_round
        4) save the best model for per round
        '''
        from data_loaders.data_proc import ImagesetDatasetBR,ImagesetDatasetBR2
        from torch.utils.data import Dataset, DataLoader
        from data_loaders.data_proc import ToTensor_BR, ToTensor_BR2,  RandomCrop, RandomFlip, RandomRotate,RandomScaleBR
        import torchvision.transforms as transforms

        # ===========================for ramps============================
        rampup_starts = int(self.config['ramp_up_start'] * self.config['train']['nepoch'])
        rampup_ends = int(self.config['ramp_up_end'] * self.config['train']['nepoch'])  # "ramp_up": 0.1, "epochs": 80
        from models.Satt_CD.modules.loss import consistency_weight
        cons_w_unsup = consistency_weight(final_w=self.config['unsupervised_w'],
                                          iters_per_epoch=self.config["iter_per_epoch"],
                                          rampup_starts=rampup_starts,
                                          rampup_ends=rampup_ends)  # ramp_val
        # train_transforms_LR = transforms.Compose([
        #     RandomFlip(),
        #     RandomRotate(),
        #     ToTensor_BR()
        # ])
        if self.config["batch_size"] >=16:
            train_transforms_LR2 = transforms.Compose([

                RandomCrop(128, 128),#for whu-mas can not fine-tune if use unet512 as the training model
                # RandomCrop(256, 256),#worse the perfmance
                RandomFlip(),
                RandomRotate(),
                # RandomScaleBR(), #seems to worse teh performance
                ToTensor_BR2()
            ])
        else:
            train_transforms_LR2 = transforms.Compose([

                # RandomCrop(128, 128),#for whu-mas can not fine-tune if use unet512 as the training model
                # RandomCrop(256, 256),#worse the perfmance
                RandomFlip(),
                RandomRotate(),
                # RandomScaleBR(), #seems to worse teh performance
                ToTensor_BR2()
            ])
        tgt_transforms2 = train_transforms_LR2

        test_transforms = transforms.Compose([
            ToTensor_BR()
        ])

        label_ratio=self.config["train"]["label_ratio"]
        round_num=len(label_ratio)

        train_list_tgt=self.train_list_tgt
        test_dataset = ImagesetDatasetBR2(imset_list=train_list_tgt, config=self.config, mode='train',
                                          transform=test_transforms,
                                          visit_tgt=self.config["train"]["visit_tgt"])  # for generate plabel
        test_dataloader = DataLoader(test_dataset, batch_size=1,
                                     shuffle=True, num_workers=self.config.num_worker,
                                     pin_memory=True
                                     )

        val_list_tgt=[pic for pic in os.listdir(os.path.join(self.config.data_dir_tgt, 'test', 'img'))]
        random.shuffle(val_list_tgt)
        val_list_num=min(500,len(val_list_tgt))
        cur_val_list_tgt=[]
        for i in range(val_list_num):
            cur_val_list_tgt.append(val_list_tgt[i])
        cur_val_dataset_tgt = ImagesetDatasetBR(imset_list=cur_val_list_tgt, config=self.config, mode='Test',
                                                transform=test_transforms, visit_tgt=True
                                                # use_SSL=self.config["train"]["use_SSL"],
                                                # plabel_dir=cur_pseudo_label_name
                                                )
        cur_val_dataloader_tgt = DataLoader(cur_val_dataset_tgt, batch_size=self.config.batch_size * 2,
                                            shuffle=True, num_workers=self.config.num_worker,
                                            pin_memory=True, drop_last=False)

        for cur_round in range((round_num)):
            cur_pseudo_label_dir = self.config.pseudo_label_dir + '_' + str(label_ratio[cur_round])
            cur_pseudo_label_name='plabel_'+self.config.src_tgt_name+'_' + str(label_ratio[cur_round])
            mkdir_if_not_exist(cur_pseudo_label_dir)

            #cur_pseudo_label_num = int(len(train_list_tgt) * label_ratio[cur_round])
            cur_pseudo_label_num =min(len(train_list_tgt),3000)
            if cur_round==0:
                cur_model_path=self.config.pretrained_model_path

                #model.netG.load_state_dict(torch.load(self.config.pretrained_model_path))
            else:
                cur_model_path = self.config.model_dir + '/' + self.config.pred_name + '_' + str(label_ratio[cur_round-1]) + '_best_acc.pth'
                #model.netG.load_state_dict(torch.load(self.pre_best_model_path))
            print("loading pretrained model {} for round {:2d}".format(cur_model_path, cur_round))
            model.netG.load_state_dict(torch.load(cur_model_path))
            #generate_pseudo_labels2(self, model, model_path, pseudo_label_dir, pseudo_label_num, testDataloader):
            cur_train_list_tgt = [pic for pic in os.listdir(os.path.join(cur_pseudo_label_dir))]

            #if os.path.exists(cur_pseudo_label_dir):
            if len(cur_train_list_tgt)==0:
                cur_train_list_tgt=self.generate_pseudo_labels2(model.netG,cur_model_path,cur_pseudo_label_dir,label_ratio[cur_round],cur_pseudo_label_num,test_dataloader,use_tmp=use_tmp)
                # cur_train_list_tgt = self.generate_pseudo_labels2_ESL2(model.netG, cur_model_path, cur_pseudo_label_dir,
                #                                                   label_ratio[cur_round], cur_pseudo_label_num,
                #                                                   test_dataloader)#
            # cur_train_list_tgt, cur_val_list_tgt = train_test_split(cur_data_list_tgt,
            #                                                         test_size=self.config.val_proportion,
            #                                                         random_state=1, shuffle=True)
            cur_train_dataset_tgt = ImagesetDatasetBR(imset_list=cur_train_list_tgt, config=self.config, mode='Train',
                                                  transform=tgt_transforms2, visit_tgt=True,
                                                  use_SSL=self.config["train"]["use_SSL"],plabel_dir=cur_pseudo_label_name
                                                  )

            cur_train_dataloader_tgt = DataLoader(cur_train_dataset_tgt, batch_size=self.config.batch_size,
                                              shuffle=True, num_workers=self.config.num_worker,
                                              pin_memory=True, drop_last=True)


            # ==============using total iters to indicate training process==========================
            best_f1 = 0
            trainloader_s_iter = iter(self.trainDataloader_src)
            trainloader_t_iter = iter(cur_train_dataloader_tgt)

            #===================train model=========================================================
            logger.info("SSL training of round {:6d} with {:6d} pseudo labels".format(cur_round,cur_pseudo_label_num))
            init_G_lr=self.config["train"]['lr_G']
            init_D_lr=self.config["train"]['lr_D']
            epochs_per_cycle=self.config["train"]["nepoch"]
            total_iters = int(epochs_per_cycle * len(cur_train_dataloader_tgt))
            iters_per_epoch = self.config.iter_per_epoch
            ratio_name = str(str(label_ratio[cur_round]))
            for iter_step in tqdm(range(0, total_iters)):
                try:
                    batch_s = next(trainloader_s_iter)
                except:
                    trainloader_s_iter = iter(self.trainDataloader_src)
                    batch_s = next(trainloader_s_iter)

                try:
                    batch_t = next(trainloader_t_iter)
                except:
                    trainloader_t_iter = iter(cur_train_dataloader_tgt)
                    batch_t = next(trainloader_t_iter)

                model.feed_data_batch_st(batch_s, batch_t)
                model.netG.train()
                cur_epoch=iter_step//iters_per_epoch
                cur_lr = model.update_learning_rate_cos(init_G_lr, cur_epoch, epochs_per_cycle)
                #model.optimize_parameters_AdvSeg_PFO_SSL(iter_step)#DA-like fine-tune
                #model.optimize_parameters_AdvSeg_PFO_SSL2(iter_step,cons_w_unsup(iter_step))# use seg_loss and con_loss from tgt
                model.optimize_parameters_AdvSeg_PFO_SSL0(iter_step)  # only use seg loss from tgt
                # update learning rate
                #model.update_learning_rate()

                current_step = iter_step
                if current_step % self.config['logger']['print_freq'] == 0:
                    logs = model.get_current_log()
                    message = '<iter:{:8,d}, lr:{:.3e}, lossG:{:.6f}, lossD:{:.6f}> '.format(
                        iter_step, cur_lr,
                        logs['l_g_total'], logs['l_d_total'])

                    logger.info(message)
                    # =======for val test======================================
                    # val_loss_tgt, val_acc_tgt = self.val_DA_SSL(model=model, multi_outputs=multi_outputs,valDataloader_tgt=cur_val_dataloader_tgt)
                    # message = '<val_loss_tgt:{:.6f},val_f1_score_tgt:{:.6f}>'.format(val_loss_tgt, val_acc_tgt)
                    # logger.info(message)
                    #==========================================================
                # if self.config['train']['train_style']:
                #     model.save_style()
                if current_step==total_iters-1:
                    logger.info('Saving models and training states.')
                    #model.save(current_step, save_style=self.config["train"]["train_style"])
                    model.save_final(ratio_name)
                    # if self.config['train']['train_style']:
                    #     model.save_style()


                if current_step > 0 and current_step % self.config['train']['val_iter'] == 0:
                    if self.config['train']['train_style'] == False:
                        val_loss_tgt, val_acc_tgt = self.val_DA_SSL(model=model, multi_outputs=multi_outputs,
                                                                    valDataloader_tgt=cur_val_dataloader_tgt)

                        message = '<val_loss_tgt:{:.6f},val_f1_score_tgt:{:.6f}>'.format(val_loss_tgt, val_acc_tgt)
                        logger.info(message)
                        logs = model.get_current_log()
                        train_history["loss"].append(logs['l_g_total'])

                        # train_history["val_loss_src"].append(val_loss_src)
                        # train_history["val_acc_src"].append(val_acc_src)
                        train_history["val_loss_tgt"].append(val_loss_tgt)
                        train_history["val_acc_tgt"].append(val_acc_tgt)
                        if val_acc_tgt > best_f1:
                            best_f1 = val_acc_tgt
                            model.save_best(ratio_name)
                    # else:
                    #     logs = model.get_current_log()

        end_time = time.perf_counter()
        run_time = end_time - start_time
        message = 'running time is {:.4f} seconds!'.format(run_time)
        logger.info(message)
        self.visualize_train_optim_DA(train_history)

#====================================================================================================
#=======================================util function================================================
#====================================================================================================


#===================================================================




    def visualize_train_optim_DA(self, history):
        train_loss = history["loss"]
        val_acc_src = history["val_acc_src"]
        val_acc_tgt = history["val_acc_tgt"]

        val_loss_src = history["val_loss_src"]
        val_loss_tgt = history["val_loss_tgt"]
        #loss = history["loss"]

        plt.subplot(121)
        #plt.plot(acc)#for the acc is much accurate to calculate using large batch, we currently do not compute it for each batch
        plt.plot(val_acc_src)
        plt.plot(val_acc_tgt)
        plt.title('model acc')
        plt.ylabel('acc')
        plt.xlabel('epoch')
        plt.legend(['val_src','val_tgt'], loc='upper left')

        plt.subplot(122)
        plt.plot(train_loss)
        plt.plot(val_loss_src)
        plt.plot(val_loss_tgt)
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train','val_src','val_tgt'], loc='upper right')

        plt.savefig(self.lossImg_path)
        plt.show()

    def print_info(self,history={},elapse_time=0.0,epochs=20):
        mylog = open(self.log_file, 'w')
        stdout_backup = sys.stdout
        sys.stdout = mylog  # 输出到文件


        print(summary(self.net,(3, 48, 48)))

        print("model train time is %.6f s" % elapse_time)
        print('model_name:', self.model_path)
        loss=history['loss']# equal to history["loss"]
        acc=history["acc"]
        val_loss = history["val_loss"]
        val_acc = history["val_acc"]
        for i in range(epochs):
            print('epoch: %d' % (i + 1))
            print('train_loss: %.5f' % loss[i], 'val_loss:%.5f' % val_loss[i])
            print('train_acc:%.5f' % acc[i], 'val_acc:%.5f' % val_acc[i])
            mylog.flush()
        print('Finish!')
        mylog.close()
        sys.stdout = stdout_backup

    def  val_DA_SSL(self, model=None,multi_outputs=False,valDataloader_tgt=None,out_map=2):
        # eval model on validation set
        '''
        need to add with torch_no_grad so as to alleviate the memory burst
        :param epoch:
        :param segmulti:
        :param multi_inputs:
        :return:
        '''
        print('=================Evaluation:======================')
        # convert to test mode

        #losses = []
        lossAcc_src = 0.0
        correctsAcc_src=0.0
        lossAcc_tgt = 0.0
        correctsAcc_tgt = 0.0
        # perform test inference
        if model==None:
            self.net.eval()
            val_model=self.net

        else:
            val_model=model.netG
            val_model.eval()


        for i, sample in (enumerate(tqdm(valDataloader_tgt, 0))):  # not tqdm(enumerate((self.valDataloader, 0)))
            # get the test sample
            with torch.no_grad():

                imgs, labels = sample['img'], sample['label']
                imgs, labels = imgs.cuda(), labels.cuda()
                if multi_outputs:
                    if self.config["network_G"]["out_nc"] > 1:
                        outputs0, outputs1 = val_model(imgs)
                        pred_prob = F.softmax(outputs1, dim=1)
                        outputs = pred_prob[:, 1].unsqueeze(1)
                    elif self.config["train"]["mode"] == "semi":
                        outputs, _ = val_model(imgs, imgs, use_warm=True)
                    else:
                        if out_map == 2:
                            _, outputs = val_model(imgs)
                        else:
                            _, pred1, pred2 = val_model(imgs)
                            outputs = F.sigmoid(0.5 * pred1 + pred2)



                else:
                    if self.config["network_G"]["out_nc"] > 1:
                        outputs0 = val_model(imgs)
                        pred_prob = F.softmax(outputs0, dim=1)
                        outputs = pred_prob[:, 1].unsqueeze(1)
                    else:
                        outputs = val_model(imgs)

                #self.cri_seg_loss = bce_edge_loss(use_mask=True, gamma=self.config["train"]["ssl_gamma"]).to(self.device)
                loss = self.pix_cri(outputs, labels)
                if np.isnan(float(loss.item())):
                    raise ValueError('loss is nan while training')
                lossAcc_tgt += loss.item()
                # ===========for f1-score metric===============
                # precision, recall, f1_score_value,acc,kappa = self.PR_score_whole(y_true, y_pred)
                _, _, f1_score, _, _ = PR_score_whole(labels.data.cpu().numpy(), outputs.data.cpu().numpy())
                correctsAcc_tgt += f1_score

        # val_loss_src = lossAcc_src * 1.0 / len(self.valDataloader_src)
        # val_acc_src = correctsAcc_src * 1.0 / len(self.valDataloader_src)
        val_loss_tgt=lossAcc_tgt*1.0/len(valDataloader_tgt)
        val_acc_tgt=correctsAcc_tgt*1.0/len(valDataloader_tgt)




        return  val_loss_tgt,val_acc_tgt



    def  val_DA(self, model=None,multi_outputs=False,use_scaleATT=False,use_scaleRef=False,out_map=2):
        # eval model on validation set
        '''
        need to add with torch_no_grad so as to alleviate the memory burst
        :param epoch:
        :param segmulti:
        :param multi_inputs:
        :return:
        '''
        print('=================Evaluation:======================')
        # convert to test mode

        #losses = []
        lossAcc_src = 0.0
        correctsAcc_src=0.0
        lossAcc_tgt = 0.0
        correctsAcc_tgt = 0.0
        # perform test inference
        if model==None:
            self.net.eval()
            val_model=self.net

        else:
            val_model=model.netG
            val_model.eval()
        for i, sample in (enumerate(tqdm(self.valDataloader_src, 0))):#not tqdm(enumerate((self.valDataloader, 0)))
            # get the test sample
            with torch.no_grad():

                imgs,labels=sample['img'],sample['label']
                imgs, labels = imgs.cuda(), labels.cuda()
                if multi_outputs:
                    if use_scaleATT:
                        out_logits0, out_logits1, att_map = val_model(imgs)
                        # out_seg0 = F.interpolate(out_logits0 * att_map0, scale_factor=2, mode='bilinear',
                        #                          align_corners=True)
                        # att_map0 = F.interpolate(att_map0, scale_factor=2, mode='bilinear', align_corners=True)
                        # out_seg1 = out_logits1 * (1 - att_map0)
                        # outputs = F.sigmoid(out_seg0 + out_seg1)

                        att_map0 = att_map[:, 0].unsqueeze(1)
                        att_map1 = att_map[:, 1].unsqueeze(1)
                        out_seg = out_logits0 * att_map0 + out_logits1 * att_map1
                        outputs = F.sigmoid(out_seg)
                    elif use_scaleRef:
                        preds0, preds1, preds2, preds3 = val_model(imgs)
                        outputs=preds3
                    else:
                        if self.config["network_G"]["out_nc"]>1:
                            outputs0, outputs1 = val_model(imgs)
                            pred_prob = F.softmax(outputs1, dim=1)
                            outputs = pred_prob[:, 1].unsqueeze(1)
                        elif self.config["train"]["mode"]=="semi":
                            outputs, _ = val_model(imgs,imgs,use_warm=True)
                        else:
                            if out_map==2:
                               _,outputs=val_model(imgs)
                            else:
                               _,pred1,pred2=val_model(imgs)
                               outputs=F.sigmoid(0.5*pred1+pred2)

                else:
                    if self.config["network_G"]["out_nc"] > 1:
                        outputs0 = val_model(imgs)
                        pred_prob = F.softmax(outputs0, dim=1)
                        outputs = pred_prob[:, 1].unsqueeze(1)
                    else:
                        outputs = val_model(imgs)
                if use_scaleRef:
                    loss= self.pix_cri0(preds0, labels) + self.pix_cri0(preds1, labels) + \
                               self.pix_cri(preds2, labels) + self.pix_cri(preds3, labels)
                else:
                    loss= self.pix_cri(outputs, labels)
                if np.isnan(float(loss.item())):
                   raise ValueError('loss is nan while training')
                lossAcc_src += loss.item()
            #===========for f1-score metric===============
                #precision, recall, f1_score_value,acc,kappa = self.PR_score_whole(y_true, y_pred)
                _,_,f1_score,_,_=PR_score_whole(labels.data.cpu().numpy(),outputs.data.cpu().numpy())
                correctsAcc_src+=f1_score

        for i, sample in (enumerate(tqdm(self.valDataloader_tgt, 0))):  # not tqdm(enumerate((self.valDataloader, 0)))
            # get the test sample
            with torch.no_grad():

                imgs, labels = sample['img'], sample['label']
                imgs, labels = imgs.cuda(), labels.cuda()
                if multi_outputs:
                    if use_scaleATT:
                        out_logits0, out_logits1, att_map = val_model(imgs)

                        att_map0 = att_map[:, 0].unsqueeze(1)
                        att_map1 = att_map[:, 1].unsqueeze(1)
                        out_seg = out_logits0 * att_map0 + out_logits1 * att_map1
                        outputs = F.sigmoid(out_seg)
                    elif use_scaleRef:
                        preds0, preds1, preds2, preds3 = val_model(imgs)
                        outputs = preds3
                    else:
                        if self.config["network_G"]["out_nc"]>1:
                            outputs0, outputs1 = val_model(imgs)
                            pred_prob = F.softmax(outputs1, dim=1)
                            outputs = pred_prob[:, 1].unsqueeze(1)
                        elif self.config["train"]["mode"]=="semi":
                            outputs, _ = val_model(imgs,imgs,use_warm=True)
                        else:
                            if out_map==2:
                               _,outputs=val_model(imgs)
                            else:
                               _,pred1,pred2=val_model(imgs)
                               outputs=F.sigmoid(0.5*pred1+pred2)


                else:
                    if self.config["network_G"]["out_nc"] > 1:
                        outputs0 = val_model(imgs)
                        pred_prob = F.softmax(outputs0, dim=1)
                        outputs = pred_prob[:, 1].unsqueeze(1)
                    else:
                        outputs = val_model(imgs)

                if use_scaleRef:
                    loss = self.pix_cri0(preds0, labels) + self.pix_cri0(preds1, labels) + \
                           self.pix_cri(preds2, labels) + self.pix_cri(preds3, labels)
                else:
                    loss = self.pix_cri(outputs, labels)
                if np.isnan(float(loss.item())):
                    raise ValueError('loss is nan while training')
                lossAcc_tgt += loss.item()
                # ===========for f1-score metric===============
                # precision, recall, f1_score_value,acc,kappa = self.PR_score_whole(y_true, y_pred)
                _, _, f1_score, _, _ = PR_score_whole(labels.data.cpu().numpy(), outputs.data.cpu().numpy())
                correctsAcc_tgt += f1_score

        val_loss_src = lossAcc_src * 1.0 / len(self.valDataloader_src)
        val_acc_src = correctsAcc_src * 1.0 / len(self.valDataloader_src)
        val_loss_tgt=lossAcc_tgt*1.0/len(self.valDataloader_tgt)
        val_acc_tgt=correctsAcc_tgt*1.0/len(self.valDataloader_tgt)


        # convert to train mode for next training
        # if model==None:
        #     self.net.train()

        # del outputs
        # torch.cuda.empty_cache()

        return  val_loss_src,val_acc_src,val_loss_tgt,val_acc_tgt

    def  val_DA_tune(self, model=None,multi_outputs=False,use_scaleATT=False,use_scaleRef=False,out_map=2,valDataloader=None):
        # eval model on validation set
        '''
        need to add with torch_no_grad so as to alleviate the memory burst
        :param epoch:
        :param segmulti:
        :param multi_inputs:
        :return:
        '''
        print('=================Evaluation:======================')
        # convert to test mode

        #losses = []
        lossAcc_src = 0.0
        correctsAcc_src=0.0
        lossAcc_tgt = 0.0
        correctsAcc_tgt = 0.0
        # perform test inference
        if model==None:
            self.net.eval()
            val_model=self.net

        else:
            val_model=model.netG
            val_model.eval()



        batch_num=0
        for i, sample in (enumerate(tqdm(valDataloader, 0))):#not tqdm(enumerate((self.valDataloader, 0)))
            # get the test sample
            # batch_s = {"img": batch["img_src"], "label": batch["label_src"]}
            # batch_t = {"img": batch["img_tgt"], "label": batch["label_tgt"]}
            batch_num+=1
            with torch.no_grad():

                #imgs,labels=sample['img'],sample['label']
                imgs, labels =sample["img_src"],sample["label_src"]
                imgs, labels = imgs.cuda(), labels.cuda()
                if multi_outputs:
                    if use_scaleATT:
                        out_logits0, out_logits1, att_map = val_model(imgs)
                        # out_seg0 = F.interpolate(out_logits0 * att_map0, scale_factor=2, mode='bilinear',
                        #                          align_corners=True)
                        # att_map0 = F.interpolate(att_map0, scale_factor=2, mode='bilinear', align_corners=True)
                        # out_seg1 = out_logits1 * (1 - att_map0)
                        # outputs = F.sigmoid(out_seg0 + out_seg1)

                        att_map0 = att_map[:, 0].unsqueeze(1)
                        att_map1 = att_map[:, 1].unsqueeze(1)
                        out_seg = out_logits0 * att_map0 + out_logits1 * att_map1
                        outputs = F.sigmoid(out_seg)
                    elif use_scaleRef:
                        preds0, preds1, preds2, preds3 = val_model(imgs)
                        outputs=preds3
                    else:
                        if self.config["network_G"]["out_nc"]>1:
                            outputs0, outputs1 = val_model(imgs)
                            pred_prob = F.softmax(outputs1, dim=1)
                            outputs = pred_prob[:, 1].unsqueeze(1)
                        elif self.config["train"]["mode"]=="semi":
                            outputs, _ = val_model(imgs,imgs,use_warm=True)
                        else:
                            if out_map==2:
                               _,outputs=val_model(imgs)
                            else:
                               _,pred1,pred2=val_model(imgs)
                               outputs=F.sigmoid(0.5*pred1+pred2)

                else:
                    if self.config["network_G"]["out_nc"] > 1:
                        outputs0 = val_model(imgs)
                        pred_prob = F.softmax(outputs0, dim=1)
                        outputs = pred_prob[:, 1].unsqueeze(1)
                    else:
                        outputs = val_model(imgs)
                if use_scaleRef:
                    loss= self.pix_cri0(preds0, labels) + self.pix_cri0(preds1, labels) + \
                               self.pix_cri(preds2, labels) + self.pix_cri(preds3, labels)
                else:
                    loss= self.pix_cri(outputs, labels)
                if np.isnan(float(loss.item())):
                   raise ValueError('loss is nan while training')
                lossAcc_src += loss.item()
            #===========for f1-score metric===============
                #precision, recall, f1_score_value,acc,kappa = self.PR_score_whole(y_true, y_pred)
                _,_,f1_score,_,_=PR_score_whole(labels.data.cpu().numpy(),outputs.data.cpu().numpy())
                correctsAcc_src+=f1_score



        val_loss_src = lossAcc_src * 1.0 / batch_num
        val_acc_src = correctsAcc_src * 1.0 / batch_num
        # val_loss_tgt=lossAcc_tgt*1.0/len(valDataloader_tgt)
        # val_acc_tgt=correctsAcc_tgt*1.0/len(valDataloader_tgt)


        return  val_loss_src,val_acc_src









    def  val_DA_ENT(self, model=None,multi_outputs=False,use_scaleATT=False,use_scaleRef=False,out_map=2):
        # eval model on validation set
        '''
        need to add with torch_no_grad so as to alleviate the memory burst
        :param epoch:
        :param segmulti:
        :param multi_inputs:
        :return:
        '''
        print('=================Evaluation:======================')
        # convert to test mode

        #losses = []
        lossAcc_src = 0.0
        correctsAcc_src=0.0
        lossAcc_tgt = 0.0
        correctsAcc_tgt = 0.0
        from models.utils import entropy_loss_ita
        # perform test inference
        if model==None:
            self.net.eval()
            val_model=self.net

        else:
            val_model=model.netG
            val_model.eval()
        for i, sample in (enumerate(tqdm(self.valDataloader_src, 0))):#not tqdm(enumerate((self.valDataloader, 0)))
            # get the test sample
            with torch.no_grad():

                imgs,labels=sample['img'],sample['label']
                imgs, labels = imgs.cuda(), labels.cuda()
                if multi_outputs:
                    if use_scaleATT:
                        out_logits0, out_logits1, att_map = val_model(imgs)
                        # out_seg0 = F.interpolate(out_logits0 * att_map0, scale_factor=2, mode='bilinear',
                        #                          align_corners=True)
                        # att_map0 = F.interpolate(att_map0, scale_factor=2, mode='bilinear', align_corners=True)
                        # out_seg1 = out_logits1 * (1 - att_map0)
                        # outputs = F.sigmoid(out_seg0 + out_seg1)

                        att_map0 = att_map[:, 0].unsqueeze(1)
                        att_map1 = att_map[:, 1].unsqueeze(1)
                        out_seg = out_logits0 * att_map0 + out_logits1 * att_map1
                        outputs = F.sigmoid(out_seg)
                    elif use_scaleRef:
                        preds0, preds1, preds2, preds3 = val_model(imgs)
                        outputs=preds3
                    else:
                        if self.config["network_G"]["out_nc"]>1:
                            outputs0, outputs1 = val_model(imgs)
                            pred_prob = F.softmax(outputs1, dim=1)
                            outputs = pred_prob[:, 1].unsqueeze(1)
                        elif self.config["train"]["mode"]=="semi":
                            outputs, _ = val_model(imgs,imgs,use_warm=True)
                        else:
                            if out_map==2:
                               _,outputs=val_model(imgs)
                            else:
                               _,pred1,pred2=val_model(imgs)
                               outputs=F.sigmoid(0.5*pred1+pred2)

                else:
                    if self.config["network_G"]["out_nc"] > 1:
                        outputs0 = val_model(imgs)
                        pred_prob = F.softmax(outputs0, dim=1)
                        outputs = pred_prob[:, 1].unsqueeze(1)
                    else:
                        outputs = val_model(imgs)
                if use_scaleRef:
                    loss= self.pix_cri0(preds0, labels) + self.pix_cri0(preds1, labels) + \
                               self.pix_cri(preds2, labels) + self.pix_cri(preds3, labels)
                else:
                    loss= self.pix_cri(outputs, labels)
                if np.isnan(float(loss.item())):
                   raise ValueError('loss is nan while training')
                lossAcc_src += loss.item()
            #===========for f1-score metric===============
                #precision, recall, f1_score_value,acc,kappa = self.PR_score_whole(y_true, y_pred)
                _,_,f1_score,_,_=PR_score_whole(labels.data.cpu().numpy(),outputs.data.cpu().numpy())
                correctsAcc_src+=f1_score
        #for tgt_image, use entropy due the absence of ground truth

        for i, sample in (enumerate(tqdm(self.valDataloader_tgt, 0))):  # not tqdm(enumerate((self.valDataloader, 0)))
            # get the test sample
            with torch.no_grad():

                imgs, labels = sample['img'], sample['label']
                imgs, labels = imgs.cuda(), labels.cuda()
                if multi_outputs:
                    if use_scaleATT:
                        out_logits0, out_logits1, att_map = val_model(imgs)

                        att_map0 = att_map[:, 0].unsqueeze(1)
                        att_map1 = att_map[:, 1].unsqueeze(1)
                        out_seg = out_logits0 * att_map0 + out_logits1 * att_map1
                        outputs = F.sigmoid(out_seg)
                    elif use_scaleRef:
                        preds0, preds1, preds2, preds3 = val_model(imgs)
                        outputs = preds3
                    else:
                        if self.config["network_G"]["out_nc"]>1:
                            outputs0, outputs1 = val_model(imgs)
                            pred_prob = F.softmax(outputs1, dim=1)
                            outputs = pred_prob[:, 1].unsqueeze(1)
                        elif self.config["train"]["mode"]=="semi":
                            outputs, _ = val_model(imgs,imgs,use_warm=True)
                        else:
                            if out_map==2:
                               _,outputs=val_model(imgs)
                            else:
                               _,pred1,pred2=val_model(imgs)
                               outputs=F.sigmoid(0.5*pred1+pred2)


                else:
                    if self.config["network_G"]["out_nc"] > 1:
                        outputs0 = val_model(imgs)
                        pred_prob = F.softmax(outputs0, dim=1)
                        outputs = pred_prob[:, 1].unsqueeze(1)
                    else:
                        outputs = val_model(imgs)

                # if use_scaleRef:
                #     loss = self.pix_cri0(preds0, labels) + self.pix_cri0(preds1, labels) + \
                #            self.pix_cri(preds2, labels) + self.pix_cri(preds3, labels)
                # else:
                #     loss = self.pix_cri(outputs, labels)
                # if np.isnan(float(loss.item())):
                #     raise ValueError('loss is nan while training')
                # lossAcc_tgt += loss.item()
                # ===========for f1-score metric===============
                # precision, recall, f1_score_value,acc,kappa = self.PR_score_whole(y_true, y_pred)
                #_, _, f1_score, _, _ = PR_score_whole(labels.data.cpu().numpy(), outputs.data.cpu().numpy())
                entropy=entropy_loss_ita(outputs)
                correctsAcc_tgt += 1-entropy

        val_loss_src = lossAcc_src * 1.0 / len(self.valDataloader_src)
        val_acc_src = correctsAcc_src * 1.0 / len(self.valDataloader_src)
        val_loss_tgt=0
        val_acc_tgt=correctsAcc_tgt*1.0/len(self.valDataloader_tgt)


        return  val_loss_src,val_acc_src,val_loss_tgt,val_acc_tgt


