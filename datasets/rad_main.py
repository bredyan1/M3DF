import logging
import os.path as osp
from glob import glob
from os import makedirs
from sys import stdout
import argparse

import numpy as np

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from torchvision import models
from tqdm import tqdm
#from models_rad.resnet_custom import resnet50_baseline
# from models_rad.RESNET3d import resnet50
# from models_rad.MedLAM import MedLAM
from SAMMed3D.segment_anything.build_sam3D import *
# from model import generate_model
from  datasets_rad import * 
# from generate_model import main
import paddle




def setup_custom_logger(name):
    formatter = logging.Formatter(fmt='%(asctime)s %(levelname)-8s %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')
    handler = logging.FileHandler('log.txt', mode='w')
    handler.setFormatter(formatter)
    screen_handler = logging.StreamHandler(stream=stdout)
    screen_handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.addHandler(screen_handler)
    return logger


logger = setup_custom_logger(__name__)


def get_network(device):

    
    #net = build_sam3D_vit_b(checkpoint='/home/yanyiqun/MCAT/sam_med3d_brain.pth')
    net = build_sam3D_vit_b(checkpoint='/home/yanyiqun/MCAT/sam_med3d_turbo.pth')

    return net.to(device)



def choose_device(use_cuda=True):
    cuda = use_cuda and torch.cuda.is_available()
    device = torch.device("cuda:2" if cuda else "cpu")
    logger.info('Using: ' + str(device))
    return device



def get_features(mydata, net, device, names):
    features = []
    for i,data in tqdm(enumerate(mydata)):  # 使用 enumerate 获取索引和数据
        input_data = data[0].to(device)
        #print(input_data)
    
        name=names[i]
        feature = net(input_data)
        print(feature)
        torch.save(feature, os.path.join(args.out_path, name + '.pt'))
    #return features






def main(args):


    with torch.no_grad():
        device = choose_device()
        # print('loading model checkpoint')
        # model = resnet50()
        # model = model.to(device)
        # print(model)
        model=get_network(device)
        #network = get_network(device)
        mydata , names = build_dataset(False,args)

        get_features(mydata, model, device,names)
       # save_features(features, args.out_path,names=names)

parser = argparse.ArgumentParser(description='Configurations for Survival Analysis on TCGA Data.')
#parser.add_argument('--pretrain_path',   type=str, default='/home/yanyiqun/MCAT/pretrain/resnet_50.pth', help='')
parser.add_argument('--n_seg_classes',   type=int, default='4', help='')
parser.add_argument('--model',   type=str, default='RESNET3d', help='')
parser.add_argument('--model_depth',   type=int, default='152', help='')
parser.add_argument('--resnet_shortcut',   type=str, default='B', help='')
parser.add_argument('--no_cuda',   type=str, default='True', help='')
parser.add_argument('--data_path',   type=str, default='/mnt/sdb/yanyiqun/TCGA_ROOT_DIR/tcga_brca/mri', help='')
parser.add_argument('--out_path',   type=str, default='/mnt/sdb/yanyiqun/TCGA_ROOT_DIR/tcga_brca/tcga_brca_mri_features', help='Data directory to MRI features')
parser.add_argument('--input_size',   type=int, default='256', help='mri image size')

args = parser.parse_args()
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    main(args)