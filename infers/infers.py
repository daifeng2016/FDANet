import math
import os, time,sys,cv2
import numpy as np
from PIL import Image
import os.path as osp
import matplotlib.pyplot as plt
# import torch modules
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from data_loaders.RSCD_dl import RSCD_DL
from models.utils import Acc
from utils.postprocessing import post_proc
from tqdm import tqdm
from utils.utils import mkdir_if_not_exist
from sklearn.metrics import roc_curve,auc,accuracy_score,precision_score,recall_score,f1_score,precision_recall_curve

class Infer(object):
    # init function for class
    def __init__(self, config,net, testDataloader,batchsize=1, cuda=True, gpuID=0
                ):
        dl=RSCD_DL(config)
        self.model_path=dl.config.model_name
        self.pred_dir=dl.config.pred_dir
        self.batchsize=batchsize#测试时保证imgsize/batchsize=int,使得测试网络时正好遍历完所有测试样本是最理想的情况
        self.batchnum=1
        self.output_size=(1,1,256,256)
        # set the GPU flag
        self.cuda = cuda
        self.gpuID = gpuID
        # define an optimizer
        #self.optimG = torch.optim.Adam(net.parameters(),lr=lr)
        # set the network
        self.net = net
        # set the data loaders
        self.testDataloader = testDataloader
        self.config=dl.config
        self.config.precision_path=dl.config.precision_path
        self.test_dir=dl.config.test_dir
        self.multi_outputs=self.config["network_G"]["multi_outputs"]
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    def grayTrans(self, img,use_batch=False):
        #img = img.data.cpu().numpy()[0][0]*255.0#equal to img.data.squeeze().cpu().numpy()[0]
        if use_batch==False:
            img = img[0] * 255.0
        else:
            img = img.cpu().numpy()[0][0] * 255.0
        img = (img).astype(np.uint8)
        img = Image.fromarray(img, 'L')
        return img
    def grayTrans_numpy(self, img):
        #img = img.data.cpu().numpy()[0][0]*255.0#equal to img.data.squeeze().cpu().numpy()[0]
        img = img[0] * 255.0
        img = (img).astype(np.uint8)
        img = Image.fromarray(img, 'L')
        return img

    def saveCycle_tensor(self,img_tensor):
        img_data=0.5*(img_tensor.squeeze(0)+1.)*255
        img_data = img_data.clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
        im = Image.fromarray(img_data)
        return im


    def predict_x(self,batch_x,net):
        use_3D=False
        with torch.no_grad():
            if use_3D==False:
            # ========================for unet 2D without sigmoid===================
            #   _,output = net.forward(batch_x)  # for unet_2D sigmoid
            #   output=F.sigmoid(output)
            #========================for unet 2D sigmoid,note simoid must be used====================
              # imgs=torch.cat((batch_x,batch_x),dim=1)
              #_,output=net.forward(batch_x)#for multi-unet with fea
              try:
                  output,_,_,_,_,_= net.forward(batch_x)#for unet_2D sigmoid deepsupervison
                  #output = net.forward(batch_x)  # for unet_2D sigmoid
                  #output = torch.argmax(output, dim=1).unsqueeze(1).float()#for softmax
                  #=================for multi-output prabability==================================
                  # output=F.softmax(output,dim=1)# for prabability output
                  # output=output[:,1,:,:].unsqueeze(1).float()# for prabability output
              except RuntimeError as exception:
                  if "out of memory" in str(exception):
                     print("WARNING: out of cuda memory, prediction is now switched using smaller batchsize")
                  if hasattr(torch.cuda, 'empty_cache'):
                     torch.cuda.empty_cache()
                  batch_size = batch_x.size(0)
                  pred_num = int(batch_size / self.batchnum)
                  mod_num = int(batch_size % self.batchnum)

                  output = torch.zeros(size=self.output_size).cuda(0)
                  for i in range(pred_num):
                    temp_out = net.forward(batch_x[i * self.batchnum:(i + 1) * self.batchnum,...])
                    #temp_out = torch.argmax(temp_out, dim=1).unsqueeze(1).float()#for unet_softmax
                    temp_out = temp_out[:, 1, :, :]  # for probability output
                    output = torch.cat((output, temp_out), dim=0)
                  if mod_num > 0:
                    temp_out = net.forward(batch_x[batch_size - mod_num:batch_size, ...])
                    #temp_out = torch.argmax(temp_out, dim=1).unsqueeze(1).float() # for unet_softmax, index output
                    temp_out=temp_out[:,1,:,:]#for probability output
                    output = torch.cat((output, temp_out), dim=0)
                  output = output[1:, ...]
            #==================for unet softmax=====================
            #    pred=net.forward(batch_x)
            # #output = torch.max(pred, dim=1)[0].unsqueeze(1)#返回最大概率值
            #    output=torch.argmax(pred,dim=1).unsqueeze(1)#返回标号值
            #==============for unet 3D=============================
            # output=net.forward(torch.unsqueeze(batch_x,dim=-1))#for unet_3D
            # output = torch.squeeze(output, -1)#for unet_3D
            #=====================for conv3d input=<n,c,d,w,h>================
            else:
               img1 = batch_x[:, 0:3, :, :]
               img2 = batch_x[:, 3:6, :, :]
               imgs_12 = torch.cat((img1.unsqueeze(dim=2), img2.unsqueeze(dim=2)), dim=2)
               pred = net.forward(imgs_12)
               output = torch.argmax(pred, dim=1).unsqueeze(1)  # 返回标号值
               #output=output[:,1,:,:].unsqueeze(1)


        return output
    def predict_xy(self, batch_x, net):
        with torch.no_grad():
            #output = net.forward(batch_x[:, 0:3, ...], batch_x[:, 3:6, ...])
           try:
              output=net.forward(batch_x[:,0:3,...],batch_x[:,3:6,...])
           except RuntimeError as exception:
               if "out of memory" in str(exception):
                   print("WARNING: out of cuda memory, prediction is now switched using smaller batchsize")
                   if hasattr(torch.cuda, 'empty_cache'):
                       torch.cuda.empty_cache()
                   batch_size=batch_x.size(0)
                   pred_num=int(batch_size/self.batchnum)
                   mod_num=int(batch_size%self.batchnum)
                   output=torch.zeros(size=self.output_size).cuda(0)
                   for i in range(pred_num):
                       temp_out=net.forward(batch_x[i*self.batchnum:(i+1)*self.batchnum,0:3,...],batch_x[i*self.batchnum:(i+1)*self.batchnum,3:6,...])
                       output=torch.cat((output,temp_out),dim=0)
                   if mod_num>0:
                       temp_out=net.forward(batch_x[batch_size-mod_num:batch_size,0:3,...],batch_x[batch_size-mod_num:batch_size,3:6,...])
                       output = torch.cat((output, temp_out), dim=0)
                   output=output[1:,...]
                   #====================single by single====================
                   # for i in range(batch_size):
                   #     x1=torch.unsqueeze(batch_x[i, 0:3, ...], dim=0)
                   #     x2=torch.unsqueeze(batch_x[i, 3:6, ...], dim=0)
                   #     temp_out = net.forward(x1, x2)
                   #     temp_out=torch.squeeze(temp_out,dim=0)
                   #     output.append(temp_out.cpu().numpy())
               else:
                   raise exception
        return output
    def predict_img_pad(self,x,target_size,predict,multi_inputs=False):
        '''
                滑动窗口预测图像。
               每次取target_size大小的图像预测，但只取中间的1/4，这样预测可以避免产生接缝。
                :param target_size:
                :return:
                '''
        # target window是正方形，target_size是边长
        #x_gpu=x
        x_cpu=x.cpu().numpy()
        x_cpu=x_cpu.reshape(x_cpu.shape[1],x_cpu.shape[2],x_cpu.shape[3])
        quarter_target_size = target_size // 4
        half_target_size = target_size // 2
        pad_width = (
            (0, 0),
            (quarter_target_size, target_size),#axis=0 填充后+quarter_target_size+target_size
            (quarter_target_size, target_size)#axis=1 填充后+quarter_target_size+target_size
            )
        #pad_x = np.pad(x, pad_width, 'constant', constant_values=0)#（448,784,6）==》（588,924,6）
        pad_x = np.pad(x_cpu, pad_width, 'reflect')
        pad_y = np.zeros(
            (1,pad_x.shape[1], pad_x.shape[2]),
            dtype=np.float32)

        def update_prediction_center(one_batch):
            """根据预测结果更新原图中的一个小窗口，只取预测结果正中间的1/4的区域"""
            wins = []
            for row_begin, row_end, col_begin, col_end in one_batch:
                win = pad_x[:,row_begin:row_end, col_begin:col_end]
                win = np.expand_dims(win, 0)
                wins.append(win)
            x_window = np.concatenate(wins, 0)#(836,256,256,6) for test0
            x_window=torch.from_numpy(x_window).cuda()
            #x_window = torch.from_numpy(x_window)
            if self.config.deep_supervision==True:
               y1, y2, y3, y4 = predict(x_window)
               y_window = y4
            else:
               # if multi_inputs:
               #    y_window=predict(x_window[:,0:3,...],x_window[:,3:6,...])
               # else:
               y_window = predict(x_window)  # 预测一个窗格
               if isinstance(y_window,list):
                  y_window=torch.from_numpy(np.array(y_window)).cuda()

            for k in range(len(wins)):
                row_begin, row_end, col_begin, col_end = one_batch[k]
                pred = y_window[k, ...]
                y_window_center = pred[:,
                                  quarter_target_size:target_size - quarter_target_size,
                                  quarter_target_size:target_size - quarter_target_size
                                  ]  # 只取预测结果中间区域 将正方形四等分，只取中间的1/4区域  pred(112,112,1)==>y_window_center(56,56,1)

                pad_y[:,
                row_begin + quarter_target_size:row_end - quarter_target_size,
                col_begin + quarter_target_size:col_end - quarter_target_size
                 ] = y_window_center.cpu().numpy()  # 更新也，

            # 每次移动半个窗格
        batchs = []
        batch = []
        for row_begin in range(0, pad_x.shape[1], half_target_size):
            for col_begin in range(0, pad_x.shape[2], half_target_size):
                row_end = row_begin + target_size
                col_end = col_begin + target_size
                if row_end <= pad_x.shape[1] and col_end <= pad_x.shape[2]:
                    batch.append((row_begin, row_end, col_begin, col_end))
        if len(batch) > 0:
            batchs.append(batch)
            batch = []
        for bat in tqdm(batchs, desc='Batch pred'):
            update_prediction_center(bat)
        y = pad_y[:,quarter_target_size:quarter_target_size + x_cpu.shape[1],
                quarter_target_size:quarter_target_size + x_cpu.shape[2]
                ]

        return y

    def compute_pred_evaluation_ave(self):


        start_time = time.perf_counter()


        #y_true, y_pred =self.inferSen(multi_inputs=False)
        precision, recall, f1_score, acc = self.inferSenBR(use_ave=True)


        end_time = time.perf_counter()
        run_time = (end_time - start_time)
        mylog = open(self.config.precison_path, 'w')
        stdout_backup = sys.stdout
        sys.stdout = mylog
        print("model is %s" % self.config.model_name)
        print("prediction time is %fs" % run_time)

        print("precision is %.4f" % precision)
        print("recall is %.4f" % recall)
        print("f1_score is %.4f" % f1_score)
        print("overall accuracy is %.4f" % acc)

        print('Finish!')
        mylog.close()
        sys.stdout = stdout_backup

    def DeNormalization(self,rgb_mean, im_tensor, min_max=(0, 1), max255=False):

        if max255 == False:
            im_tensor = im_tensor.squeeze().float().cpu().clamp_(*min_max)  # clamp
            im_tensor = (im_tensor - min_max[0]) / (min_max[1] - min_max[0])  # to range [0,1]
            im = im_tensor.numpy().transpose([1, 2, 0])
            im = (im + rgb_mean) * 255
        else:
            min_max = (0, 255)
            im_tensor = im_tensor.squeeze().float().cpu().clamp_(*min_max)  # clamp
            # im_tensor = (im_tensor - min_max[0]) / (min_max[1] - min_max[0])  # to range [0,1]
            im = im_tensor.numpy().transpose([1, 2, 0])

        im = im.astype('uint8')
        return Image.fromarray(im, 'RGB')

    def compute_pred_evaluation(self,use_TTA=False,use_CRF=False,use_scaleATT=False,mode='_final_iter'):



        start_time = time.perf_counter()


        y_true, y_pred, y_pred_p = self.inferSenBR(multi_outputs=self.multi_outputs,use_TTA=use_TTA,use_CRF=use_CRF,use_scaleATT=use_scaleATT,mode=mode)
        # ========================================
        img_num=y_true.shape[0]
        end_time = time.perf_counter()
        run_time = (end_time - start_time)
        mylog = open(self.config.precision_path, 'w')
        stdout_backup = sys.stdout
        sys.stdout = mylog
        print("model is %s" % self.config.model_name)
        print("prediction time is %fs" % run_time)

        precision=0.0
        recall=0.0
        acc=0
        f1=0.0
        test_num=0
        precision, recall, f1_score_value,acc,kappa,iou_score = self.PR_score_whole(y_true, y_pred)


        print("precision is %.4f" % precision)
        print("recall is %.4f" % recall)
        print("f1_score is %.4f" % f1_score_value)
        print("overall accuracy is %.4f" % acc)
        print("kappa is %.4f" % kappa)
        print("iou_score is %.4f" % iou_score)


        print('Finish!')
        mylog.close()
        sys.stdout = stdout_backup
        return  f1_score_value,iou_score

    def generate_pseudo_labels(self,pseudo_label_dir,proba_ratio=0.8,use_rgb=False):
        '''
        #===generate pseudo labels, then fine-tune the network using both source labels and target labels
        the value of output label map is: 0 non_building pixls, 1 building pixels, 255  ignore pixels
        :param targetloader:
        :return:
        '''
        targetloader=self.testDataloader
        #from models.Satt_CD import create_model
        predicted_label = np.zeros((len(targetloader), 512, 512),dtype='uint8')
        predicted_prob = np.zeros((len(targetloader), 512, 512))
        image_name = []

        self.net.eval()
        self.net.load_state_dict(torch.load(self.config.pretrained_model_path))  # 通过网络参数形式加载网络

        for index, batch in enumerate(tqdm(self.testDataloader, 0)):
            print("\rprocessing image %d" % index,end='')
            #print("processing cur image %d" % index)
            #if (index )<len(targetloader):
            image, name = batch['img'], batch['name']
            image = image.unsqueeze(0)
            H, W = image.shape[2], image.shape[3]
            with torch.no_grad():
                _, output1 = self.net(Variable(image).cuda())  # [1,1,H,W]
                output2 = torch.zeros(H, W, 2)
                output2[:, :, 0] = 1 - output1[0, 0, :, :]
                output2[:, :, 1] = output1[0, 0, :, :]
                output = output2.cpu().numpy()
                # output = output.transpose(1, 2, 0)  # [H,W,C]
                label, prob = np.argmax(output, axis=2), np.max(output, axis=2)#np.unique(label)
                predicted_label[index] = label.copy()
                predicted_prob[index] = prob.copy()
                image_name.append(name)

        thres = []
        for i in range(2):  # compute a thresh for each class by using medium value of predicted_prob
            x = predicted_prob[predicted_label == i]
            if len(x) == 0:
                thres.append(0)
                continue
            x = np.sort(x)
            thres.append(x[np.int(np.round(len(x) * proba_ratio))])

        thres = np.array(thres)
        print(thres)
        thres[thres > 0.9] = 0.9  # label soft?

        for index in range(len(targetloader)):
            name = image_name[index]
            label = predicted_label[index]
            prob = predicted_prob[index]



            if use_rgb:
                label0=predicted_prob[index]
                label1=1-predicted_prob[index]
                label0[label==0]=label1[label==0]
                label0=label0*255
                label=label0.astype('uint8')
            else:
                for i in range(2):
                    label[(prob < thres[i]) * (label == i)] = 255  # label[0,0]

            print("\rwriting %d image" % index,end='')

            #output.save('%s/%s%s' % (self.config.pseudo_label_dir, name,'.jpg'))
            file_name=pseudo_label_dir+'/'+name+'.png'#cannot be jpg file, or else , label value is [0 ,1,255], while saved_img is [0,1,2,3,...255]
            cv2.imwrite(file_name,label)

    def PR_score_whole(self, y_true, y_pred):
        '''

        :param y_true:
        :param y_pred: probability output
        :return:
        '''
        smooth=1.0
        # y_true = y_true[:, :]#still 2-dim vector
        # y_pred = y_pred[:, :]


        y_true=np.ravel(y_true)#1-dim vector [1825,256,256]==>[119603200]
        y_pred=np.ravel(y_pred)#1-dim vector [1825,256,256]==>[119603200]

        # inter = 0
        # ps = 0
        # ts = 0
        # for t, p in zip(truth, preds):
        #     tr = np.ravel(t)
        #     pr = np.ravel(p)
        #     inter += np.sum(tr * pr)
        #     ps += np.sum(pr)
        #     ts += np.sum(tr)
        # f1 = (2 * inter + smooth) / (ps + ts + smooth)
        # return f1


        c1 = np.sum(np.round(np.clip(y_true * y_pred, 0, 1)))#the clip opetration make sure y_pred y_true and y_true * y_pred are [0,1]
        c2 = np.sum(np.round(np.clip(y_pred, 0, 1)))
        c3 = np.sum(np.round(np.clip(y_true, 0, 1)))
        #=======================================================

        # How many selected items are relevant?
        precision = c1*1.0/ c2
        # How many relevant items are selected?
        recall = c1*1.0/ c3
        # Calculate f1_score
        f1_score = (2 * precision * recall) / (precision + recall)#
        iou_score=(c1)/(c2+c3-c1)


        y_true=np.array(y_true)
        y_pred = np.array(y_pred).astype('float')#must be float
        cond1=(y_true == 1)#119603200 [True,False,...]
        cond2=(y_pred>0.5)
        cond3=(y_true == 0)
        cond4=(y_pred<0.5)
        idx_TP = np.where(cond1&cond2)[0]#not np.where( a and b) np.where(cond1&cond2)==>tuple
        idx_FP = np.where(cond3&cond2)[0]
        idx_FN=np.where(cond1&cond4)[0]
        idx_TN=np.where(cond3&cond4)[0]
        #pix_number = (y_pred.shape[0] * y_pred.shape[1])
        pix_number=y_pred.shape[0]
        acc=(len(idx_TP)+len(idx_TN))*1.0/pix_number

        nTPNum=len(idx_TP)
        nFPNum=len(idx_FP)
        nFNNum=len(idx_FN)
        nTNNum=len(idx_TN)
        temp1 = ((nTPNum + nFPNum) / 1e5) * ((nTPNum + nFNNum) / 1e5)
        temp2 = ((nFNNum + nTNNum) / 1e5) * ((nFPNum + nTNNum) / 1e5)
        temp3 = (pix_number / 1e5) * (pix_number / 1e5)
        dPre = (temp1 + temp2) * 1.0 / temp3
        kappa = (acc - dPre) / (1 - dPre)






        return  precision, recall, f1_score,acc,kappa,iou_score

    def PR_score(self, y_true, y_pred):

        y_true = y_true[:, :]
        y_pred = y_pred[:, :]

        c1 = np.sum(np.round(np.clip(y_true * y_pred, 0, 1)))#the clip opetration make sure y_pred y_true and y_true * y_pred are [0,1]
        c2 = np.sum(np.round(np.clip(y_pred, 0, 1)))
        c3 = np.sum(np.round(np.clip(y_true, 0, 1)))
        image_black=False
        # If there are no true samples, fix the F1 score at 0.
        if c3==0 or c2==0:
            image_black=True
            return 0,0,0,image_black,0
        if c1==0:
            return 0,0,0,image_black,0
        #===============to make the result more reasonable, c2==0 and c3!=0, image_black=False
        # if c3 == 0:
        #     image_black = True
        #     return 0, 0, 0, image_black, 0
        # else:
        #     if c1 == 0 or c2 == 0:
        #         image_black = False
        #         return 0, 0, 0, image_black, 0

        # How many selected items are relevant?
        precision = c1*1.0/ c2
        # How many relevant items are selected?
        recall = c1*1.0/ c3
        # Calculate f1_score
        f1_score = 2 * (precision * recall) / (precision + recall)

        y_true=np.array(y_true)
        y_pred = np.array(y_pred).astype('float')#must be float
        cond1=(y_true == 1)
        cond2=(y_pred>0.5)
        cond3=(y_true == 0)
        cond4=(y_pred<0.5)
        idx_TP = np.where(cond1&cond2)[0]#not np.where( a and b)
        #idx_FP = np.where(cond3&cond2)[0]
        #idx_FN=np.where(cond1&cond4)[0]
        idx_TN=np.where(cond3&cond4)[0]
        pix_number=(y_pred.shape[0]*y_pred.shape[1])
        acc=(len(idx_TP)+len(idx_TN))*1.0/pix_number
        # temp1=(len(idx_TP)+len(idx_FP))*1.0/(len(idx_TP)+len(idx_FN))
        # temp2 = (len(idx_TN) + len(idx_FN)) * 1.0 / (len(idx_FP) + len(idx_TN))
        # temp3=(len(idx_TP)+len(idx_TN)+len(idx_FN)+len(idx_FP))*(len(idx_TP)+len(idx_TN)+len(idx_FN)+len(idx_FP))
        # dPre = (temp1 + temp2) * 1.0 / temp3
        # dKappa = (acc - dPre) / (1 - dPre)
        return  precision, recall, f1_score,image_black,acc

    def inferSenBR(self,multi_outputs=False,use_ave=False,use_CRF=False,use_TTA=True,use_scaleATT=False,mode='_final_iter'):
        '''
        output y_true,y_pred, y_pred_p(of probability not 0,1)
        :return:
        '''

        self.net.eval()
        mode_snap = mode
        model_path = self.config.model_dir + '/' + self.config.pred_name + mode_snap + '.pth'
        self.net.load_state_dict(torch.load(model_path))  # 通过网络参数形式加载网络
        #self.net.load_state_dict(torch.load(self.model_path))  # 通过网络参数形式加载网络
        image_size=self.config.patch_size
        target_test=[]
        pred_test=[]
        pred_test_p=[]

        precision = 0.0
        recall = 0.0
        acc = 0
        f1_score= 0.0
        #pred_batch = 16
        pred_batch=51
        test_num = len(self.testDataloader)/pred_batch
        test_mod=len(self.testDataloader)%pred_batch
        test_batch=pred_batch*test_num

        for i, sample in enumerate(tqdm(self.testDataloader, 0)):

            imgs=sample['img']
            img_name=sample['name']#img_name[0]
            label_test=sample['label']

            label_test=label_test.squeeze(0).data.numpy()
            if self.cuda:
                imgs= imgs.cuda(self.gpuID)
                imgs=imgs.unsqueeze(0)#for batch_size=1

            with torch.no_grad():

                   #===============for multi-output====================
                 if self.config["network_G"]["out_nc"]>1:
                     outputs0, outputs1 = self.net(imgs)
                     pred_prob = F.softmax(outputs1, dim=1)
                     masks_pred = pred_prob[:, 1].unsqueeze(1)
                 if self.config["network_G"]["which_model_G"]=='UNet_2D_PreTrain512_MS':

                     if use_TTA:
                         masks_pred = self.predict_TTA(self.net,imgs,multi_outputs=multi_outputs)
                     else:
                        _,_,_,masks_pred=self.net(imgs)
                 else:
                     if use_TTA:
                         masks_pred = self.predict_TTA(self.net,imgs,multi_outputs=multi_outputs)
                     else:
                        _,masks_pred = self.net(imgs)


                 masks_pred = masks_pred[0].data.cpu().numpy()


                 print("processing image {}, size is {}".format(i,masks_pred.shape))
                #masks_pred=self.net(imgs)#输入必须是训练集的整数倍
                 predict_img=self.grayTrans_numpy(masks_pred)
                 predict_img.save('%s/%s.png'% (self.pred_dir,img_name))
                #pred_test_p+=[masks_pred[0]]
                 predict_img = np.array(predict_img).astype('uint8')
                 if use_CRF:
                    binary_img=np.argmax(masks_out,axis=0)
                    binary_img=(binary_img*255).astype('uint8')
                 else:
                    _, binary_img = cv2.threshold(predict_img, 127, 255, cv2.THRESH_BINARY)#set thresh=90 to fill the fake boundary caused by inaccurate labeling

                 cv2.imwrite('%s/%s%s.png'% (self.pred_dir,'/Binary/',img_name), binary_img)
                # =======remove hole===============
                 save_dir = self.pred_dir + '/ReA'
                 mkdir_if_not_exist(save_dir)
                 res_img = post_proc(binary_img)
                 cv2.imwrite(save_dir + '/' + img_name + '.png', res_img)

                 res_img = np.array(res_img).astype('float')
                 res_img /= 255#=====================for acc evaluation, y_pred must be [0,1]
                #pred_test+=[res_img]#=============for accuracy evaluation

                 binary_img = np.array(binary_img).astype('float')  # for raw binary
                 binary_img /= 255
                 if use_ave == False:
                   pred_test += [binary_img]
                   pred_test_p += [masks_pred[0]]
                   target_test += [label_test]
                 else:
                   pred_test += [binary_img]
                   target_test += [label_test]
                   if i%pred_batch==0:
                       _precision, _recall, _f1_score, _acc = self.PR_score_whole(target_test, pred_test)
                       precision += _precision
                       recall += _recall
                       f1_score += _f1_score
                       acc += _acc
                       pred_test=[]
                       target_test=[]
                   if i==len(self.testDataloader) and test_mod>0:
                       _precision, _recall, _f1_score, _acc = self.PR_score_whole(target_test, pred_test)
                       precision += _precision
                       recall += _recall
                       f1_score += _f1_score
                       acc += _acc
                       test_num+=1








        if use_ave:
            return precision*1.0/test_num,recall*1.0/test_num,f1_score*1.0/test_num,acc*1.0/test_num

        return  np.array(target_test),np.array(pred_test),np.array(pred_test_p)









