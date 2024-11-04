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
import shutil
import numpy as np
import random

from dataset.semicd import SemiCDDataset
from model.semseg.TDsemi2UKsim import TDsemi2UKsim
from supervised import evaluate, test
from util.utils import count_params, init_log, AverageMeter,  data_split
from util.lossim import LabelSmoothCEsim



parser = argparse.ArgumentParser(description='Revisiting Weak-to-Strong Consistency in Semi-Supervised Semantic Segmentation')
parser.add_argument('--config', type=str, required=True)

parser.add_argument('--save-path', type=str, required=True)
parser.add_argument('--local_rank', default=0, type=int)
parser.add_argument('--port', default=None, type=int)
parser.add_argument("--doc",type=str, required=True, help="A string for documentation purpose. " "Will be the name of the log folder.",
)
parser.add_argument("--time",type=str, required=True)                  
#parser.add_argument("--extra_band",type=int, default=4)    


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True          
      

def main(seed):
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)        
    #setup_seed(seed)
    args = parser.parse_args()

    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)
    #cfg["extra_band"] = args.extra_band
    #logger = logging.getLogger()
    #logger = init_log('global', logging.INFO)
    logger = init_log(str(seed), logging.INFO)
    logger.propagate = 0

    setup_seed(seed)

    args.seed = str(seed)
    
    #tb_path = os.path.join(args.save_path, "stdout.log")
    save_dir_path = os.path.join(args.save_path, args.doc)
    if not os.path.exists(save_dir_path):
        os.makedirs(save_dir_path)
    save_doc_path = os.path.join(save_dir_path, args.time, args.seed)
    
    if os.path.exists(save_doc_path):
        overwrite = False
        # response = input("Folder already exists. Overwrite? (Y/N)")
        # if response.upper() == "Y":
        overwrite = True
        if overwrite:
            shutil.rmtree(save_doc_path)
            #shutil.rmtree(tb_path)
            os.makedirs(save_doc_path)
    else:
        os.makedirs(save_doc_path)
   

    handler = logging.FileHandler(os.path.join(save_doc_path, "stdout.log"))
    #formatter = logging.Formatter('[%(asctime)s %(filename)s %(funcName)s] [line:%(lineno)d] %(levelname)s %(message)s')
    #handler.setFormatter(formatter)
    logger.addHandler(handler)

    world_size = 1

    all_args = {**cfg, **vars(args), 'ngpus': world_size}
    logger.info('{}\n'.format(pprint.pformat(all_args)))
    
    writer = SummaryWriter(save_doc_path)
    
    os.makedirs(save_doc_path, exist_ok=True)
    
    cudnn.enabled = True
    cudnn.benchmark = True

    model_zoo = {'TDsemi2UKsim': TDsemi2UKsim}
    assert cfg['model'] in model_zoo.keys()
    model = model_zoo[cfg['model']](cfg)

    logger.info('Total params: {:.1f}M\n'.format(count_params(model)))
    
    
    optimizer = SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg['lr'],
            weight_decay=1e-4,
            momentum=0.9,
            nesterov=True,
        )

    
    #torch.optim.Adam(D_A.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    model.cuda()

    criterion_c = nn.CrossEntropyLoss().cuda()
    criterion_s = LabelSmoothCEsim(ignore_index=0).cuda()   #nn.CrossEntropyLoss(ignore_index=0).cuda()
    
    #criterion_t = nn.CrossEntropyLoss().cuda()
  
    train_sample_number = cfg["train_sample_number"]
    val_sample_number = cfg["val_sample_number"]
    
    
    patch_size = cfg["patch_size"]
    
    train_lis, val_lis, test_lis = data_split(cfg['data_root'], 7 ,train_sample_number, val_sample_number, seed)

    temporal = cfg["temporal"]
    trainset = SemiCDDataset(cfg['data_root'], 'train', train_lis, temporal, patch_size)
    valset = SemiCDDataset(cfg['data_root'], 'val', val_lis, temporal, patch_size)
    testset = SemiCDDataset(cfg['data_root'], 'test', test_lis, temporal, patch_size)
    
    

    trainloader = DataLoader(trainset, batch_size=cfg['batch_size'],
                               pin_memory=True, num_workers=1, drop_last=False, sampler=None, shuffle=True)

    testloader = DataLoader(testset, batch_size=cfg['batch_size'], pin_memory=True, num_workers=1,
                           drop_last=False, sampler=None, shuffle=True)
    
    valloader = DataLoader(valset, batch_size=cfg['batch_size'], pin_memory=True, num_workers=1,
                           drop_last=False, sampler=None, shuffle=True)

    total_iters = len(trainloader) * cfg['epochs']
    previous_best_loss = 10
    previous_best_OA = 0
    epoch = -1
    c = True
    
    if os.path.exists(os.path.join(save_doc_path, 'latest.pth')):
        checkpoint = torch.load(os.path.join(save_doc_path, 'latest.pth'))
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint['epoch']
        #previous_best_loss = checkpoint['previous_best_loss']
        previous_best_OA = checkpoint['previous_best_OA']
        
        
        logger.info('************ Load from checkpoint at epoch %i\n' % epoch)
    
    for epoch in range(epoch + 1, cfg['epochs']):
        model.train()

       
        logger.info('===========> Epoch: {:}, LR: {:.5f}, Previous best loss: {:.2f}, Previous best OA: {:.2f}'.format(
            epoch, optimizer.param_groups[0]['lr'], previous_best_loss, previous_best_OA))

        total_loss = AverageMeter()
        total_loss_s = AverageMeter()
        total_loss_c = AverageMeter()
        total_loss_band = AverageMeter()
        total_loss_sim = AverageMeter()


        for i, (imgs_A, imgs_B, labels, labels_A, labels_B) in enumerate(trainloader):
            
            imgs_A, imgs_B, labels, labels_A, labels_B = imgs_A.cuda().float(), imgs_B.cuda().float(), labels.cuda(), labels_A.cuda(), labels_B.cuda()
            bs_iter = labels_A.size()[0]
            
            preds, preds_A, preds_B, loss_band, outA_f, outB_f, sim_all_loss, seg_center, seglabelA, seglabelB = model(imgs_A, imgs_B, labels_A, labels_B, begin_train = c)

            if not torch.all(labels_A == 0):
                loss_s_A = criterion_s(preds_A, labels_A, outA_f, seg_center)
            else: 
                loss_s_A = 0

            if not torch.all(labels_B == 0):
                loss_s_B = criterion_s(preds_B, labels_B, outB_f, seg_center)
            else: 
                loss_s_B = 0
                
            loss_s = (
                    loss_s_A * 0.5
                    + loss_s_B * 0.5
                )
            
            loss_c = criterion_c(preds, labels)
            #loss = cfg["weight_loss"][0] * loss_c + cfg["weight_loss"][1] * loss_band + cfg["weight_loss"][2] * loss_s  + cfg["weight_loss"][3] * sim_all_loss
            loss = cfg["weight_loss"][0] * loss_c + cfg["weight_loss"][2] * loss_s  + cfg["weight_loss"][3] * sim_all_loss


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss.update(loss.item())
            total_loss_s.update(loss_s.item())
            total_loss_c.update(loss_c.item())
            total_loss_band.update(loss_band.item())
            total_loss_sim.update(sim_all_loss.item())

            iters = epoch * len(trainloader) + i
            lr = cfg['lr'] * (1 - iters / total_iters) ** 0.9

            optimizer.param_groups[0]["lr"] = lr
            
            writer.add_scalar('train/loss_all', loss.item(), iters)
            writer.add_scalar('train/loss_seg', loss_s.item(), iters)
            writer.add_scalar('train/loss_bn', loss_c.item(), iters)
            writer.add_scalar('train/loss_band', loss_band.item(), iters)
            writer.add_scalar('train/loss_sim', sim_all_loss.item(), iters)
            
            if ((i % 10) == 0):
            #if (i % (len(trainloader) // 2) == 0):
                logger.info('Iters: {:}, total_loss: {:.3f}, total_loss_band: {:.3f}, total_loss_bn: {:.3f}, total_loss_sim: {:.3f}'
                            .format(i, total_loss.avg, total_loss_band.avg, total_loss_c.avg, total_loss_sim.avg))
        
        accuarcy_c_lis, accuarcy_s_lis, loss_eval = evaluate(model, valloader, criterion_s, criterion_c, cfg)

        
        logger.info('***** Evaluation ***** >>>> Loss_eval: {:.2f}'.format(loss_eval))
        logger.info('***** Evaluation ***** >>>> Kappa_cha: {:.2f}'.format(accuarcy_c_lis[0]*100))
        logger.info('***** Evaluation ***** >>>> OA_cha: {:.2f}\n'.format(accuarcy_c_lis[1]*100))
        
    
        writer.add_scalar('eval/Loss', loss_eval, epoch)
        writer.add_scalar('eval/Kappa_cha', accuarcy_c_lis[0]*100, epoch)
        writer.add_scalar('eval/OA_cha', accuarcy_c_lis[1]*100, epoch)
            

        is_best = loss_eval < previous_best_loss
        previous_best_loss = min(loss_eval, previous_best_loss)
        if is_best:
            previous_best_OA = accuarcy_c_lis[1]*100
    
        checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'previous_best_OA': previous_best_OA,
        }
        #torch.save(checkpoint, os.path.join(save_doc_path, 'latest.pth'))
        if is_best:
            torch.save(checkpoint, os.path.join(save_doc_path, 'best.pth'))

    accuarcy_c_lis = test(model, testloader, criterion_s, criterion_c, args, cfg)
    logger.info(accuarcy_c_lis)
    logger = logging.shutdown()
    return accuarcy_c_lis

if __name__ == '__main__':
    args = parser.parse_args()
    lis = []
    for seed in range(9,-1, -1):
        setup_seed(seed)
        accuarcy_c_lis = main(seed)
        lis.append(accuarcy_c_lis)
    
    lis = np.squeeze(np.array([lis]))
    mean = np.mean(lis, axis=0, keepdims=True)
    std = np.std(lis, axis=0,  keepdims=True)
    print(mean, std)
    acc_out = np.concatenate((lis, mean, std), axis = 0) *100
    np.savetxt('{}/{}/{}/acc.txt'.format(args.save_path, args.doc, args.time), acc_out, delimiter=' ', fmt='%.2f')