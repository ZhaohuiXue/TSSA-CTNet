import argparse
import logging
import os
import pprint

import torch
from torch import nn
import torch.backends.cudnn as cudnn
from torch.optim import SGD
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml

from dataset.semicd import SemiCDDataset
from model.semseg.TDsemi2UKsim import TDsemi2UKsim
from util.utils import count_params, AverageMeter, acc, init_log,  data_split
import numpy as np
from osgeo import gdal, ogr
import time as time
from PIL import Image

from tqdm import tqdm
import glob


parser = argparse.ArgumentParser(description='Revisiting Weak-to-Strong Consistency in Semi-Supervised Semantic Segmentation')
parser.add_argument('--config', type=str, required=True)
parser.add_argument('--labeled-id-path', type=str, required=True)
parser.add_argument('--unlabeled-id-path', type=str, default=None)
parser.add_argument('--save-path', type=str, required=True)
parser.add_argument('--local_rank', default=0, type=int)
parser.add_argument('--port', default=None, type=int)


def evaluate(model, loader, criterion_s, criterion_c, cfg):
    model.eval()

    total_loss = AverageMeter()
  
    preds_all = []
    labels_all = []

    preds_A_all = []
    labels_A_all = []

    preds_B_all = []
    labels_B_all = []

    with torch.no_grad():
        for i, (imgs_A, imgs_B, labels, labels_A, labels_B) in enumerate(loader):
            
            imgs_A, imgs_B, labels, labels_A, labels_B = imgs_A.cuda().float(), imgs_B.cuda().float(), labels.cuda(), labels_A.cuda(), labels_B.cuda()

            preds, preds_A, preds_B, loss_band, outA_f, outB_f, sim_all_loss, seg_center, seglabelA, seglabelB = model(imgs_A, imgs_B, None, None)

            #loss_s = (
            #        criterion_s(preds_A, labels_A) * 0.5
            #        + criterion_s(preds_B, labels_B) * 0.5
            #    )
            loss_c = criterion_c(preds, labels)
            loss = loss_c #loss_s + 

            total_loss.update(loss.item())

            preds = preds.argmax(dim=1).detach().cpu().numpy().tolist()
            preds_A = preds_A.argmax(dim=1).detach().cpu().numpy().tolist()
            preds_B = preds_B.argmax(dim=1).detach().cpu().numpy().tolist()
            
            preds_all += preds
            labels_all += labels.detach().cpu().numpy().tolist()
            
            preds_A_all += preds_A
            labels_A_all += labels_A.detach().cpu().numpy().tolist()
            preds_B_all += preds_B
            labels_B_all += labels_B.detach().cpu().numpy().tolist()

    kappa_c, OA_c, producer_accuracy_c = acc(labels_all, preds_all)
    labels_A_all += labels_B_all
    preds_A_all += preds_B_all
    kappa_s, OA_s, producer_accuracy_s = acc(labels_A_all, preds_A_all)
    
    accuarcy_c_lis = [kappa_c, OA_c]
    accuarcy_c_lis += producer_accuracy_c

    accuarcy_s_lis = [kappa_s, OA_s]
    accuarcy_s_lis += producer_accuracy_s

    return accuarcy_c_lis, accuarcy_s_lis, total_loss.avg

def test(model, loader, criterion_s, criterion_c, args, cfg):
    model.eval()

    checkpoint_path = os.path.join(args.save_path, args.doc,args.time, args.seed) + "/" + 'best.pth'

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model'])

    total_loss = AverageMeter()
  
    preds_all = []
    labels_all = []

    preds_A_all = []
    labels_A_all = []

    preds_B_all = []
    labels_B_all = []

    with torch.no_grad():
        for i, (imgs_A, imgs_B, labels, labels_A, labels_B) in enumerate(loader):
            
            imgs_A, imgs_B, labels, labels_A, labels_B = imgs_A.cuda().float(), imgs_B.cuda().float(), labels.cuda(), labels_A.cuda(), labels_B.cuda()

            preds, preds_A, preds_B, _, outA_f, outB_f, sim_all_loss, seg_center, seglabelA, seglabelB = model(imgs_A, imgs_B, None, None)

            loss_c = criterion_c(preds, labels)
            loss = loss_c

            total_loss.update(loss.item())

            preds = preds.argmax(dim=1).detach().cpu().numpy().tolist()
            preds_A = preds_A.argmax(dim=1).detach().cpu().numpy().tolist()
            preds_B = preds_B.argmax(dim=1).detach().cpu().numpy().tolist()
            
            preds_all += preds
            labels_all += labels.detach().cpu().numpy().tolist()
            
            preds_A_all += preds_A
            labels_A_all += labels_A.detach().cpu().numpy().tolist()
            preds_B_all += preds_B
            labels_B_all += labels_B.detach().cpu().numpy().tolist()

    kappa_c, OA_c, producer_accuracy_c = acc(labels_all, preds_all)    
    accuarcy_c_lis = [kappa_c, OA_c]
    accuarcy_c_lis += producer_accuracy_c

    return accuarcy_c_lis

def interdata(input):
    if True in np.isnan(input):
        t, f, h, w = input.shape
        input_t = input.transpose(1,2,3,0).reshape(-1, t)
        input_inter = []
        for i in range(f * h * w):
            input_t_i = input_t[i]
            if np.isnan(input_t_i).all():
                input_inter.append(input_t_i)
                continue
            if np.isnan(input_t_i).any():
                missing_indexes = np.isnan(input_t_i)
                new_indexes = np.arange(len(input_t_i))
                input_t_i = np.interp(new_indexes, new_indexes[~missing_indexes], input_t_i[~missing_indexes])
            input_inter.append(input_t_i)
        input_inter = np.array(input_inter).reshape(f, h, w, t).transpose(3, 0, 1, 2)
        return input_inter
    else:
        return input

def normalize8(data):
    max_val = 65535
    min_val = 4894
    data = (data - min_val) / (max_val - min_val)
    return data

def normalize7(data):
    max_val = 255
    min_val = 1
    data = (data - min_val) / (max_val - min_val)
    return data

def main(model, checkpoint_path, loader1, loader2):
    model.eval()

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model'])

    preds_all = []

    t, f, h, w  = loader1.shape
 
    imgs_A = torch.tensor([]).cuda()
    imgs_B = torch.tensor([]).cuda()
    b = 0
    number = (h-4) * (w-4)//128
    with torch.no_grad():
        for i in tqdm(range(2, h-2)):
            
            for j in range(2, w-2):
                
                #time1 = time.time()
                imgs_A_p = torch.tensor(loader1[:, :, i-2:i+2+1, j-2:j+2+1]).cuda().float().unsqueeze(0)
                imgs_B_p = torch.tensor(loader2[:, :, i-2:i+2+1, j-2:j+2+1]).cuda().float().unsqueeze(0)

                imgs_A = torch.cat((imgs_A, imgs_A_p), axis = 0)
                imgs_B = torch.cat((imgs_B, imgs_B_p), axis = 0)

                if b < number and imgs_A.shape[0] == 128:
                    preds, preds_A, preds_B, _, outA_f, outB_f, sim_all_loss, seg_center, seglabelA, seglabelB = model(imgs_A, imgs_B, None, None)
                    preds = preds.argmax(dim=1).detach().cpu().numpy().tolist()
                    preds_all += preds
                    imgs_A = torch.tensor([]).cuda()
                    imgs_B = torch.tensor([]).cuda()
                    b+=1
                
                    #time2 = time.time()
                    #print(time2-time1)
                elif b == number and imgs_A.shape[0] == (h-4) * (w-4) - 128 * number:
                    preds, preds_A, preds_B, _, outA_f, outB_f, sim_all_loss, seg_center, seglabelA, seglabelB = model(imgs_A, imgs_B, None, None)
                    preds = preds.argmax(dim=1).detach().cpu().numpy().tolist()
                    preds_all += preds
                    imgs_A = torch.tensor([]).cuda()
                    imgs_B = torch.tensor([]).cuda()
                    
                    #time2 = time.time()
                    #print(time2-time1)

    preds_all = np.array([preds_all]).reshape(h-4, w-4)

    #np.save('result.npy', preds_all)
        
    return preds_all

def read_tiff(tiff_path):
    dataset = gdal.Open(tiff_path)
    height, width = dataset.RasterYSize, dataset.RasterXSize
    geotransform = dataset.GetGeoTransform()
    proj = dataset.GetProjection()
    image = dataset.ReadAsArray()
    del dataset
    return geotransform, proj, image, width, height

def write_tiff(image, save_path, geotrans, project, width, height):
    datatype = gdal.GDT_Byte
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(save_path, width, height, 1, datatype)
    dataset.SetGeoTransform(geotrans)
    dataset.SetProjection(project)
    dataset.GetRasterBand(1).WriteArray(image)
    del dataset

if __name__ == '__main__':
    cfg = yaml.load(open('configs/second.yaml', "r"), Loader=yaml.Loader)
    
    cfg = yaml.load(open('configs/second.yaml', "r"), Loader=yaml.Loader)

    model_zoo = {'TDsemi2UKsim': TDsemi2UKsim}
    #assert cfg['model'] in model_zoo.keys()
    model = model_zoo[cfg['model']](cfg)
    model.cuda()
    
    checkpoint_path = 'D:/GBA/code/Goose/TSSANet/exp/1221/noignore/6/best.pth'#'D:/GBA/code/Goose/TSSANet/exp/60%/30last/0/best.pth' #'D:/GBA/code/Goose/TSSANet/exp/1021/l2/0/best.pth' #'D:/GBA/code/Goose/TSSANet/exp/80%/1029/8/best.pth' #'D:/GBA/code/Goose/TSSANet/exp/1021/l2/3/best.pth' #D:\GBA\code\Goose\TSSANet\exp\60%\30last\0\best.pth

    input_folder = "D:/GBA/ROIre/JTM/tiff/" #'D:/GBA/Goose/datapre/test/' #
 
    input_rasters_t1 = sorted(glob.glob(input_folder + "0*.tif"))
    geotransform, proj, _, width, height = read_tiff(input_rasters_t1[0])
    for i in range(len(input_rasters_t1)):
        input_rasters_t1[i] = np.expand_dims(gdal.Open(input_rasters_t1[i]).ReadAsArray(), axis=0)
        
        if i == 0:
            t1 = input_rasters_t1[i]
        else:
            t1 = np.concatenate((t1, input_rasters_t1[i]), axis=0)
            
    t1 = interdata(t1)    
    t1 = np.where(np.isnan(t1), 0, t1)
    t1 = normalize7(t1)
    
    input_rasters_t2 = sorted(glob.glob(input_folder + "1*.tif"))
    for i in range(len(input_rasters_t2)):
        input_rasters_t2[i] = np.expand_dims(gdal.Open(input_rasters_t2[i]).ReadAsArray(), axis=0)
        
        if i == 0:
            t2 = input_rasters_t2[i]
        else:
            t2 = np.concatenate((t2, input_rasters_t2[i]), axis=0)
            
    t2 = interdata(t2)    
    t2 = np.where(np.isnan(t2), 0, t2)
    t2 = normalize8(t2)
 

    preds = main(model, checkpoint_path, t1, t2)
    out_tiff_path = r'D:/GBA/ROIre/JTM/predict/JTM_6.tif'

    write_tiff(preds, out_tiff_path, geotransform, proj, width, height)
    
    
    

