from configs.config_utils import process_config, get_train_args
import numpy as np
from data_loaders.data_proc import ReSize,ToTensor_BR2,RandomCrop,RandomFlip,RandomRotate
from trainers.trainer_optim_DA import TrainerOptimDA
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch
import os
from sklearn.model_selection import train_test_split
from data_loaders.data_proc import ImagesetDatasetBR,ToTensor_BR
# fix random seed
rng = np.random.RandomState(37148)

import argparse
from configs.config_utils import get_config_from_json
from data_loaders.RSCD_dl import RSCD_DL


def parse_option(cur_root,up_root):
    parser_file = argparse.ArgumentParser('argument for training')
    # specify folder
    parser_file.add_argument('--input_path', type=str, default=up_root+'/val', help='path to input data')
    parser_file.add_argument('--output_path', type=str, default=up_root+'/output', help='path to output data')
    parser_file.add_argument('--config_path', type=str, default=cur_root+'/configs/config.json', help='path to json')

    args_file = parser_file.parse_args()

    config, _ = get_config_from_json(args_file.config_path)
    return args_file,config

def main_train():


    print('[INFO] loading config...')
    parser = None
    config = None
    # config=args_file.config
    # cur_root=os.path.abspath('__file__')#(__file__) get main_test_CD.py not cur root
    cur_root = os.path.dirname(os.path.abspath(__file__))
    up_root = os.path.abspath(os.path.join(cur_root, ".."))
    args_file, config = parse_option(cur_root, up_root)

    print('[INFO] loading data...')


    torch.backends.cudnn.enabled=True
    torch.backends.cudnn.benchmark=True#使用benchmark以启动CUDNN_FIND自动寻找最快的操作，当计算图不会改变的时候（每次输入形状相同，模型不改变）的情况下可以提高性能，反之则降低性能
    # ==========must construct dl bofore dataloader processing============================
    dl = RSCD_DL(config)
    # ====================================================================================

    #==========================using new dataloader for building and road extraction====================================
    data_dir_src = config.data_dir
    train_set_src = [pic for pic in os.listdir(os.path.join(data_dir_src, 'train', 'img'))]  # for BR segmentation img.jpg
    data_dir_tgt = config.data_dir_tgt
    train_set_tgt = [pic for pic in os.listdir(os.path.join(data_dir_tgt, 'train', 'img'))]
    val_proportion = config.val_proportion
    batch_size = config.batch_size
    num_worker = config.num_worker

    #===================================================================
    train_list_src, val_list_src = train_test_split(train_set_src,
                                            test_size=val_proportion,
                                            random_state=1, shuffle=True)
    train_list_tgt, val_list_tgt = train_test_split(train_set_tgt,
                                                    test_size=val_proportion,
                                                    random_state=1, shuffle=True)


    message1="the number of train and val data for source dataset is {:6d} and {:6d}".format(len(train_list_src),len(val_list_src))
    message2="the number of train and val data for target dataset is {:6d} and {:6d}".format(len(train_list_tgt), len(val_list_tgt))

    print(message1)
    print(message2)



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
    if config.src_name == 'Massachusett_Building':
        src_transforms = train_transforms_LR_128
        tgt_transforms = train_transforms_HR_128
        tgt_transforms2 = train_transforms_LR2_128
    elif config.tgt_name == 'Massachusett_Building':
        src_transforms = train_transforms_HR_128
        tgt_transforms = train_transforms_LR_128
        tgt_transforms2 = train_transforms_LR2_128
    else:
        src_transforms = train_transforms_HR
        tgt_transforms = train_transforms_HR
        tgt_transforms2 = train_transforms_LR2
    # ===========for different resolution test=============
    if config.use_resample == False:
        src_transforms = train_transforms_HR
        tgt_transforms = train_transforms_HR
        tgt_transforms2 = train_transforms_LR2







    train_dataset_src = ImagesetDatasetBR(imset_list=train_list_src, config=config, mode='Train', transform=src_transforms
                                    )
    train_dataloader_src = DataLoader(train_dataset_src, batch_size=batch_size,
                                  shuffle=True, num_workers=0,
                                  # collate_fn=collateFunction(),  # 如何取样本的，我们可以定义自己的函数来准确地实现想要的功能
                                  pin_memory=True,drop_last=True)  # len(train_dataloader)

    val_dataset_src = ImagesetDatasetBR(imset_list=val_list_src, config=config, mode='Val', transform=src_transforms
                                  )
    val_dataloader_src = DataLoader(val_dataset_src, batch_size=batch_size*2,# can be set a large value due to torch.no_grad
                                shuffle=False, num_workers=0,
                                pin_memory=True,drop_last=True)  # len(val_dataloader)

    train_dataset_tgt = ImagesetDatasetBR(imset_list=train_list_tgt, config=config, mode='Train',
                                          transform=tgt_transforms2,visit_tgt=True,use_SSL=config["train"]["use_SSL"]
                                          )
    train_dataloader_tgt = DataLoader(train_dataset_tgt, batch_size=batch_size,
                                      shuffle=True, num_workers=0,
                                      # collate_fn=collateFunction(),  # 如何取样本的，我们可以定义自己的函数来准确地实现想要的功能
                                      pin_memory=True,drop_last=True)  # len(train_dataloader)

    val_dataset_tgt = ImagesetDatasetBR(imset_list=val_list_tgt, config=config, mode='Val', transform=tgt_transforms,visit_tgt=True
                                        )
    val_dataloader_tgt = DataLoader(val_dataset_tgt, batch_size=batch_size*2,  # can be set a large value due to torch.no_grad
                                    shuffle=False, num_workers=0,
                                    pin_memory=True,drop_last=True)  # len(val_dataloader)




    print('[INFO] training net...')

    #==========================for traning using optim of each iteration==========================
    trainer = TrainerOptimDA(config, train_dataloader_src, val_dataloader_src,train_dataloader_tgt, val_dataloader_tgt,train_list_tgt)
    if config["train"]["use_SSL"]:#"ssl_gamma":2.0,
        trainer.train_optim_UDA_DT(use_tmp=False)#

    else:
        trainer.train_optim_UDA()#"ssl_gamma":0.0, define different optim_function for the same netG, otherwise define differnt model.py for different task, such as CD,SR
        #trainer.parameters_tune()


    print('[INFO] training finished...')

if __name__ == '__main__':
    main_train()