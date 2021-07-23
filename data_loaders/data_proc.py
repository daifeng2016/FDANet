import os
import numpy as np
import pandas as pd
from PIL import Image
import skimage.io as io
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from utils.utils import mkdir_if_not_exist
from data_loaders.RSCD_dl import RSCD_DL
import random
import cv2

import albumentations as A  # using open-source library for img aug



class RandomCrop(object):
    def __init__(self, ph, pw,scale=1):
        self.ph = ph
        self.pw = pw
        self.scale=scale#denote to the upsampling scale
    def __call__(self, sample):
        img, label = sample['img'], sample['label']
        h,w=img.shape[:2]#numpy array
        th,tw=int(self.ph*self.scale),int(self.pw*self.scale)
        ix = random.randint(0, w - self.pw)#not using w - self.pw+1
        iy = random.randint(0, h - self.ph)
        tx,ty=int(ix*self.scale),int(iy*self.scale)
        img=img[iy:iy+self.ph, ix:ix+self.pw,:]
        label=label[ty:ty+th, tx:tx+tw]

        return {'img':img,'label':label,'name':sample['name']}#not work when return img_LR[iy:iy+self.ph, ix:ix+self.pw,:]














class ReSize(object):
    def __init__(self, ph, pw,scale=1):
        self.ph = ph
        self.pw = pw
        self.scale=scale#denote to the upsampling scale
    def __call__(self, sample):
        img, label = sample['img'], sample['label']
        img_new=cv2.resize(img,(self.ph,self.pw),interpolation=cv2.INTER_CUBIC)
        label_new=cv2.resize(label,(self.ph,self.pw),interpolation=cv2.INTER_CUBIC)

        return {'img':img_new,'label':label_new,'name':sample['name']}


class RandomFlip(object):
    def __call__(self, sample):
        img,label = sample['img'],sample['label']
        hflip=random.random() < 0.5
        vflip=random.random() < 0.5
        #dfilp=random.random() < 0.5

        if vflip:
            img= np.flipud(img).copy()
            label=np.flipud(label).copy()
        if hflip:
            img= np.fliplr(img).copy()
            label = np.fliplr(label).copy()
        # if dfilp:
        #     img=cv2.flip(img,-1)
        #     label = cv2.flip(label, -1)

        return {'img':img,'label':label,'name':sample['name']}
class RandomRotate(object):
    def __call__(self, sample):
        img, label = sample['img'],sample['label']
        rot90 = random.random() < 0.5
        rot180 = random.random() < 0.5
        rot270 = random.random() < 0.5

        if rot90:
            img = np.rot90(img,1).copy()
            label=np.rot90(label,1).copy()
        if rot180:
            img = np.rot90(img,2).copy()
            label = np.rot90(label,2).copy()
        if rot270:
            img=np.rot90(img,3).copy()
            label = np.rot90(label,3).copy()

        return {'img':img,'label':label,'name':sample['name']}



class ToTensor(object):
    """Convert ndarrays in sample to Tensors.
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
    """
    def __call__(self, sample):
        img_T1, img_T2, label = sample['imgT1'], sample['imgT2'], sample['label']
        label=np.expand_dims(label,axis=-1)
        img_T1_tensor=torch.from_numpy(img_T1.transpose((2,0,1)))
        img_T2_tensor= torch.from_numpy(img_T2.transpose((2, 0, 1)))
        label_tensor = torch.from_numpy(label.transpose((2, 0, 1)))
        img_T1_tensor= img_T1_tensor.float().div(255)
        img_T2_tensor= img_T2_tensor.float().div(255)
        label_tensor=label_tensor.float().div(255)
        return {'imgT1':img_T1_tensor,'imgT2':img_T2_tensor,'label':label_tensor}


class ToTensor_BR(object):
    """Convert ndarrays in sample to Tensors.
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
    """
    def __call__(self, sample):
        img= sample['img']
        timg=torch.from_numpy(img.transpose((2,0,1)))#[512,512,3]==>[3,512,512]
        timg= timg.float().div(255)

        label = sample['label']#[512,512]
        tlabel = torch.from_numpy(label).unsqueeze(0)#F.to_tensor(pic)
        tlabel = tlabel.float().div(255)
        #====https://github.com/zlkanata/DeepGlobe-Road-Extraction-Challenge/issues/8
        #tlabel=tlabel * 3.2 - 1.6#Unet(weight initialized by pytorch default initializer) without batchnorm do have better result using [-1.6,1.6] normalization

        return {'img':timg,'label':tlabel,'name':sample['name']}

class ToTensor_BR2(object):
    """Convert ndarrays in sample to Tensors.
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
    """
    def __call__(self, sample):
        img= sample['img']
        timg=torch.from_numpy(img.transpose((2,0,1)))#[512,512,3]==>[3,512,512]
        timg= timg.float().div(255)

        label = sample['label']#[512,512]
        tlabel = torch.from_numpy(label).unsqueeze(0).float()#F.to_tensor(pic)
        #tlabel = tlabel.float().div(255)
        #====https://github.com/zlkanata/DeepGlobe-Road-Extraction-Challenge/issues/8
        #tlabel=tlabel * 3.2 - 1.6#Unet(weight initialized by pytorch default initializer) without batchnorm do have better result using [-1.6,1.6] normalization

        return {'img':timg,'label':tlabel,'name':sample['name']}

class Normalize_BR(object):
    """Convert ndarrays in sample to Tensors.
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
    """
    def __init__(self, mean, std):
        # self.mean = torch.from_numpy(np.array(mean))
        # self.std =torch.from_numpy(np.array(std))
        self.mean=mean
        self.std=std

    def __call__(self, sample):
        img= sample['img']
        # img-=self.mean
        # img/=self.std
        self.mean = torch.Tensor(self.mean).view(3, 1, 1)#must have .view(3, 1, 1), broadcat can work only the two tensors have same ndims
        self.std=torch.Tensor(self.std).view(3, 1, 1)
        img=img.sub_(self.mean).div_(self.std)


        return {'img':img,'label':sample['label'],'name':sample['name']}



class ToTensor_BR_MStd(object):
    """Convert ndarrays in sample to Tensors.
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
    """
    def __call__(self, sample):
        img= sample['img']/255.0

        img = (img - np.array(mean)) / np.array(std)
        timg=torch.from_numpy(img.transpose((2,0,1)))#[512,512,3]==>[3,512,512]
        timg = timg.float()
        #timg= timg.float().div(255)


        label = sample['label']#[512,512]
        tlabel = torch.from_numpy(label).unsqueeze(0)#F.to_tensor(pic)
        tlabel = tlabel.float().div(255)
        #====https://github.com/zlkanata/DeepGlobe-Road-Extraction-Challenge/issues/8
        #tlabel=tlabel * 3.2 - 1.6#Unet(weight initialized by pytorch default initializer) without batchnorm do have better result using [-1.6,1.6] normalization

        return {'img':timg,'label':tlabel,'name':sample['name']}



#####################################################################
#========================for data loader============================#
#####################################################################






#===============for dataloader speeding=================
class data_prefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        # self.mean = torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255]).cuda().view(1,3,1,1)
        # self.std = torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255]).cuda().view(1,3,1,1)
        # With Amp, it isn't necessary to manually convert data to half.
        # if args.fp16:
        #     self.mean = self.mean.half()
        #     self.std = self.std.half()
        self.preload()

    def preload(self):
        try:
            self.next_input, self.next_target = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True)
            self.next_target = self.next_target.cuda(non_blocking=True)
            # With Amp, it isn't necessary to manually convert data to half.
            # if args.fp16:
            #     self.next_input = self.next_input.half()
            # else:
            self.next_input = self.next_input.float()
            #self.next_input = self.next_input.sub_(self.mean).div_(self.std)
    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        if input is not None:
            input.record_stream(torch.cuda.current_stream())
        if target is not None:
            target.record_stream(torch.cuda.current_stream())
        self.preload()
        return input, target


class TestDatasetBR(Dataset):
    def __init__(self, config, transform=None):
        #self.rootDir = rootDir
        self.transform = transform

        self.test_data=[]
        dl = RSCD_DL(config=config)
        self.test_dir=dl.config.test_dir
        for pic in os.listdir(self.test_dir + "/img"):
            self.test_data.append(pic)

    def __len__(self):

        return len(self.test_data)

    def __getitem__(self, idx):
        #===========================for test dataset=======================
        use_GF=False
        if use_GF:
            img_T1_path = self.test_dir + '/T1/' + self.test_data[idx]
            # (filepath, filename) = os.path.split(img_T1_path)
            img_T2_path = self.test_dir + '/T2/' + self.test2_data[idx]
            substr1 = '_'
            substr2 = '.'
            cur_img = self.test_data[idx]
            index1 = cur_img.rfind(substr1, 0, len(cur_img))
            temp_img=cur_img[:index1]
            index2 = temp_img.rfind(substr1, 0, len(temp_img))
            cur_img_index = temp_img[index2+1:]

            img_name = 'mask_2017_2018_960_960_' + cur_img_index

        else:
            img_T1_path = self.test_dir + '/img/' + self.test_data[idx]
            (filepath, filename) = os.path.split(img_T1_path)

            substr = '.'
            index0 = filename.rfind(substr, 0, len(filename))
            img_name = filename[0:index0]


        inputImage=np.asarray(Image.open(img_T1_path).convert('RGB'))#[H,W,C]
        sample = {'image': inputImage, 'name': str(img_name)}
        if self.transform is not None:

            sample = self.transform(sample)  # [H,W,2C]==>[2C,H,W]

        return sample



class ImagesetDatasetBR(Dataset):
    """ Derived Dataset class for loading many imagesets from a list of directories."""

    def __init__(self, imset_list, config, mode='Train',seed=None,transform=None,visit_tgt=False,use_SSL=False,plabel_dir='plabel'):

        super().__init__()
        #self.dl = RSCD_DL(config=config)
        self.imset_list = imset_list
        self.config=config
        #self.imset_dir=config.data_dir+'/train/aug7'
        if visit_tgt:
            self.imset_dir = config.data_dir_tgt + '/train'
            self.test_dir = config.data_dir_tgt + '/test'
        else:
            self.imset_dir = config.data_dir + '/train'
            self.test_dir = config.data_dir + '/test'

        self.seed = seed  # seed for random patches
        self.mode=mode
        self.transform=transform
        self.out_class=config["network_G"]["out_nc"]
        self.use_SSL=use_SSL
        self.plabel_dir=plabel_dir
        #dl=RSCD_DL(config=config)

    def __len__(self):
        if self.mode=="Train":
            repeat=1
            if self.config.iter_per_epoch>(len(self.imset_list)//self.config.batch_size):
               repeat = self.config.batch_size * self.config.iter_per_epoch /len(self.imset_list)
        else:
            repeat = 1
        return int(len(self.imset_list)*repeat)#int(len(self.imset_list)*1.54)
        #return len(self.imset_list)

    def __getitem__(self, index):

        if self.mode=='Train' or self.mode=='Val':
            index=(index % len(self.imset_list))
            cur_dir=self.imset_dir
        else:
            cur_dir=self.test_dir


        img_path = cur_dir + '/img/' + self.imset_list[index]
        index0 = self.imset_list[index].rfind('.', 0, len(self.imset_list[index]))
        img_name = self.imset_list[index][0:index0]

        if self.use_SSL:
            #label_file='/plabel/'
            label_file='/'+self.plabel_dir+'/'
        else:
            label_file='/label/'


        label_path0=cur_dir + label_file + img_name+'.png'
        label_path = cur_dir + label_file + img_name+'.tif'

        img_path0=cur_dir + '/img/' + img_name+'.jpg'
        img_path1 = cur_dir + '/img/' + img_name + '.tif'
        img_path2= cur_dir + '/img/' + img_name + '.png'

        img = cv2.imread(img_path0, cv2.IMREAD_UNCHANGED)
        if not isinstance(img,np.ndarray):
            img = cv2.imread(img_path1, cv2.IMREAD_UNCHANGED)
            if not isinstance(img, np.ndarray):
                img = cv2.imread(img_path2, cv2.IMREAD_UNCHANGED)
        # else:
        #     #print("image file {} not exist".format(img_path2))
        #     exit(0)

        label = cv2.imread(label_path0, 0)
        if not isinstance(label,np.ndarray):
            label = cv2.imread(label_path, 0)


        sample = {'img': img, 'label': label, 'name': img_name}
        if self.transform is not None:
            sample_train = self.transform(sample)  # must convert to torch tensor
        # if self.out_class >1:
        #     sample_train['label']=sample_train['label'].long().squeeze(0)


        return sample_train




class RandomScaleBR(object):

    def __call__(self, sample):
        img, label = sample['img'], sample['label']

        aug = A.RandomSizedCrop(min_max_height=(40, 100), height=128, width=128, interpolation=2, p=0.5)#for patch128*128
        aug_f = aug(image=img, mask=label)
        img_aug = aug_f['image']
        label_aug = aug_f['mask']



        return {'img': img_aug, 'label': label_aug, 'name': sample['name']}











