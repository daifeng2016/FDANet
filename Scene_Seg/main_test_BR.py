from configs.config_utils import process_config, get_train_args
import  os
import numpy as np
from data_loaders.data_proc import TestDataset,ImagesetDatasetBR,ImagesetDatasetBR2,ImageDataset,TestDatasetBR,TrainDatasetAdvBR,ReSize,RandomCrop
from infers.infers import Infer
from torch.utils.data import Dataset, DataLoader
from data_loaders.data_proc import ToTensor_Test,ToTensor_BR
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from models.utils import print_model_parm_nums
from ptflops import get_model_complexity_info
import  torch
import models.Satt_CD.networks as networks
from data_loaders.RSCD_dl import RSCD_DL
# fix random seed
rng = np.random.RandomState(37148)
import argparse
from configs.config_utils import get_config_from_json

def print_model_para(save_path,net):
    import sys
    mylog = open(save_path, 'w')
    stdout_backup = sys.stdout
    sys.stdout = mylog


    macs, params = get_model_complexity_info(net, (3, 512, 512),
                                             as_strings=True,
                                             print_per_layer_stat=True,
                                             ost=mylog)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    mylog.close()
    sys.stdout = stdout_backup

def parse_option(cur_root,up_root):
    parser_file = argparse.ArgumentParser('argument for training')
    # specify folder
    parser_file.add_argument('--input_path', type=str, default=up_root+'/val', help='path to input data')
    parser_file.add_argument('--output_path', type=str, default=up_root+'/output', help='path to output data')
    parser_file.add_argument('--config_path', type=str, default=cur_root+'/configs/config.json', help='path to json')

    args_file = parser_file.parse_args()

    config, _ = get_config_from_json(args_file.config_path)
    return args_file,config

def main_test():


    print('[INFO] loading config...')
    parser = None
    config = None
    # config=args_file.config
    # cur_root=os.path.abspath('__file__')#(__file__) get main_test_CD.py not cur root
    cur_root = os.path.dirname(os.path.abspath(__file__))
    up_root = os.path.abspath(os.path.join(cur_root, ".."))
    args_file, config = parse_option(cur_root, up_root)

    print('[INFO] loading data...')

    config.mode='Test'

    #===========================for SR test===================================
    test_transforms = transforms.Compose([

        ToTensor_BR()
    ])
    #================================for building and road extraction===============================
    # testDataset = TestDatasetBR(config,transform=test_transforms)
    # testDataloader = DataLoader(testDataset, batch_size=1, num_workers=0, shuffle=False)
    #=============================use imagedataset==============================
    config.use_CRF=False#can not improve the performance?
    # ==========must construct dl bofore dataloader processing============================
    dl = RSCD_DL(config)
    # ====================================================================================

    if config["train"]["visit_tgt"]:#for tgt prediction
        data_dir = config.data_dir_tgt
    else:
        data_dir = config.data_dir



    test_set = [pic for pic in os.listdir(os.path.join(data_dir, 'test', 'img'))]  # img.jpg
    test_dataset = ImagesetDatasetBR(imset_list=test_set, config=config, mode='Test', transform=test_transforms,visit_tgt=config["train"]["visit_tgt"])

    # test_set = [pic for pic in os.listdir(os.path.join(data_dir, 'train', 'img'))]# for generate plabel
    # test_dataset = ImagesetDatasetBR2(imset_list=test_set, config=config, mode='train', transform=test_transforms,visit_tgt=config["train"]["visit_tgt"])  # for generate plabel



    test_dataloader = DataLoader(test_dataset, batch_size=1,
                                  shuffle=False, num_workers=config.num_worker,
                                pin_memory=True
                                  )

    print('[INFO] constructing net...')
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    config.mode = 'Test'

    model = networks.define_G(config).to(device)



    model.cuda(0)
    print('[INFO] predicting data...')
    infer=Infer(config,model,test_dataset,batchsize=1)
    f1_score, iou_score=0,0
    f1_score,iou_score =infer.compute_pred_evaluation(use_TTA=True,use_CRF=config.use_CRF,mode='_best_acc')
    #====================================================================================



    print('[INFO] prediction finishsed...')
    print("f1_score is %.6f" % f1_score)
    print("iou_score is %.6f" % iou_score)

if __name__ == '__main__':
    main_test()
