# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 16:00:00 2020

@author: CITI
"""

import os
import numpy as  np
from models.DrumAwareBeatTracker2 import DrumAwareBeatTracker as RNNmodel
import drumaware_dataset as mmdataset
from torch.utils.data import DataLoader
import tqdm
import torch
import torch.nn as nn
from pathlib import Path
import json
#import utils
import da_utils as utils
import time
from lookahead_pytorch import Lookahead
import sys


def train(model, device, train_loader, optimizer):
    model.train()
    train_loss = 0
    ## separate the loss history of each head
    ou_loss = 0
    ou_drum_loss = 0
    fuser_loss = 0
    mix_loss = 0
    nodrum_loss = 0
    drum_loss= 0
    
    pbar = tqdm.tqdm(train_loader, disable = False)
    for x, y, x_nodrum, x_drum in pbar:
        # break
        pbar.set_description("Training batch")
        
        x, y, x_nodrum, x_drum = x.to(device), y.to(device), x_nodrum.to(device), x_drum.to(device)
        
        optimizer.zero_grad()
        
        beat_fused, beat_mix, beat_nodrum, beat_drum, x_nodrum_hat, x_drum_hat = model(x)
        
        beat_fused = beat_fused.reshape((-1, 3))
        beat_mix = beat_mix.reshape((-1, 3))
        beat_nodrum = beat_nodrum.reshape((-1, 3))
        beat_drum = beat_drum.reshape((-1, 3))

        y = y.reshape((-1)).to(dtype = torch.long) # required type of loss function

        weights = [1, 200, 67] # nonbeat, beat, downbeat
        class_weights = torch.FloatTensor(weights).to(device)
        CE = nn.CrossEntropyLoss(weight = class_weights)
        loss_SourceSep = torch.nn.functional.mse_loss(x_nodrum_hat, x_nodrum)
        loss_DrumSourceSep = torch.nn.functional.mse_loss(x_drum_hat, x_drum) 
        loss_fused = CE(beat_fused, y)
        loss_mix = CE(beat_mix, y)
        loss_nodrum = CE(beat_nodrum, y)
        loss_drum = CE(beat_drum, y)
        
        loss = loss_fused + loss_mix + loss_nodrum + 50*loss_SourceSep + loss_drum + 50*loss_DrumSourceSep
        loss.backward()

        ou_loss += loss_SourceSep.item()
        ou_drum_loss += loss_DrumSourceSep.item()
        fuser_loss += loss_fused.item()
        mix_loss += loss_mix.item()
        nodrum_loss += loss_nodrum.item()
        drum_loss += loss_drum.item()
                
        train_loss += loss
        optimizer.step()

    return train_loss/len(train_loader.dataset), [ou_loss/len(train_loader.dataset), 
                         ou_drum_loss/len(train_loader.dataset),
                         fuser_loss/len(train_loader.dataset), 
                         mix_loss/len(train_loader.dataset), 
                         nodrum_loss/len(train_loader.dataset), 
                         drum_loss/len(train_loader.dataset)]

def valid(model, device, valid_loader ):
    model.eval()
    valid_loss = 0
    ## separate the loss history of each head
    ou_loss = 0
    ou_drum_loss = 0
    fuser_loss = 0
    mix_loss = 0
    nodrum_loss = 0
    drum_loss= 0
    with torch.no_grad():
        for x, y, x_nodrum, x_drum in valid_loader:
            x, y, x_nodrum, x_drum = x.to(device), y.to(device), x_nodrum.to(device), x_drum.to(device)
            
            beat_fused, beat_mix, beat_nodrum, beat_drum, x_nodrum_hat, x_drum_hat= model(x) # beat activations (batch, timestep, 3) ==> nonbeat(0), donwbeat(1), beat(2)
        
            beat_fused = beat_fused.reshape((-1, 3))
            beat_mix = beat_mix.reshape((-1, 3))
            beat_nodrum = beat_nodrum.reshape((-1, 3))
            beat_drum = beat_drum.reshape((-1, 3))
    
            y = y.reshape((-1)).to(dtype = torch.long) # required type of loss function
            

            weights = [1, 200, 67] # nonbeat, beat, downbeat
            class_weights = torch.FloatTensor(weights).to(device)
            CE = nn.CrossEntropyLoss(weight = class_weights)
            loss_SourceSep = torch.nn.functional.mse_loss(x_nodrum_hat, x_nodrum)
            loss_DrumSourceSep = torch.nn.functional.mse_loss(x_drum_hat, x_drum) 
            loss_fused = CE(beat_fused, y)
            loss_mix = CE(beat_mix, y)
            loss_nodrum = CE(beat_nodrum, y)
            loss_drum = CE(beat_drum, y)
            loss = loss_fused + loss_mix + loss_nodrum + 50*loss_SourceSep + loss_drum + 50*loss_DrumSourceSep
            
            ou_loss += loss_SourceSep.item()
            ou_drum_loss += loss_DrumSourceSep.item()
            fuser_loss += loss_fused.item()
            mix_loss += loss_mix.item()
            nodrum_loss += loss_nodrum.item()
            drum_loss += loss_drum.item()

            valid_loss += loss
    return valid_loss/len(valid_loader.dataset), [ou_loss/len(valid_loader.dataset), 
                         ou_drum_loss/len(valid_loader.dataset),
                         fuser_loss/len(valid_loader.dataset), 
                         mix_loss/len(valid_loader.dataset), 
                         nodrum_loss/len(valid_loader.dataset), 
                         drum_loss/len(valid_loader.dataset)]




def main():
    ### you may modify the experiment params/info here:
    cuda_num = 0 # int(sys.argv[1])
    cuda_str = 'cuda:'+str(cuda_num)
    device = torch.device(cuda_str if torch.cuda.is_available() else 'cpu')
    train_epochs = 2000
    # must assign
    date = '0615'
    exp_num = 1 #sys.argv[2]
    exp_name = 'RNNBeat_DA2_V'+ str(exp_num)+'_'+date
    exp_dir = os.path.join('./experiments', exp_name)
    target_jsonpath = exp_dir
    lr = 1e-2
    patience = 20
    
    ### model settings, no need to modify
    model_type = 'DA2'
    model_simpname = 'DA2'
    model_dir = exp_dir
    model_setting = dict(
            OU_chkpnt = None, 
            DrumOU_chkpnt = None, 
            DrumBeat_chkpnt = None, 
            NDrumBeat_chkpnt = None, 
            MixBeat_chkpnt = None, 
            FuserBeat_chkpnt = None,
            fixed_DrumOU = False,
            fixed_drum = False, 
            fixed_OU = False,
            fixed_mix = False, 
            fixed_nodrum = False,
            fixed_fuser = False,
            mix_2stage_fsize = 25, 
            mix_out_features = 3,
            nodrum_2stage_fsize = 25,
            nodrum_out_features = 3, 
            drum_2stage_fsize = 25,
            drum_out_features = 3, 
            fuser_2stage_fsize = 0,
            fuser_out_features = 3,)
    model_info = dict(model_type = model_type,
                      model_simpname = model_simpname,
                      model_dir = model_dir, 
                      model_setting = model_setting)
    
    if not os.path.exists(exp_dir):
        Path(exp_dir).mkdir(parents = True, exist_ok = True)
        

    
    mix_main_dir = './datasets/original'
    
    mix_dataset_dirs = os.listdir(mix_main_dir)
    mixtrainset = utils.getMixset(mix_dataset_dirs, folderName = 'features', abname = 'train_dataset.ab' )
    mixvalidset = utils.getMixset(mix_dataset_dirs, folderName= 'features', abname = 'valid_dataset.ab' )
    
    trainset =  mixtrainset
    validset = mixvalidset
    
    train_loader = DataLoader( trainset, batch_size=4, shuffle=True)
    valid_loader = DataLoader( validset, batch_size = 2, shuffle = True)
    

    
    model = RNNmodel( 
            **model_setting)

    model.cuda(cuda_num)
    optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay= 0.00001
        )

    optimizer = Lookahead(optimizer)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=0.3, 
            patience=80,
            cooldown=10
        )

    es = utils.EarlyStopping(patience= patience)
    
    t = tqdm.trange(1, train_epochs +1, disable = False)
    train_losses = []
    valid_losses = []
    sep_trainloss_hist = []
    sep_validloss_hist = []
    
    train_times = []
    lr_change_epoch = []
    best_epoch = 0
    stop_t = 0
    for epoch in t:

        t.set_description("Training Epoch")
        end = time.time()
        train_loss, sep_trainloss = train(model, device, train_loader, optimizer)
        valid_loss, sep_validloss = valid(model, device, valid_loader)
        
        scheduler.step(valid_loss)
        train_losses.append(train_loss.item())
        valid_losses.append(valid_loss.item())
        sep_trainloss_hist.append(sep_trainloss)
        sep_validloss_hist.append(sep_validloss)

        t.set_postfix(
        train_loss=train_loss.item(), val_loss=valid_loss.item()
        )

        stop = es.step(valid_loss.item())

        if valid_loss.item() == es.best:
            best_epoch = epoch
            
        utils.save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'best_loss': es.best,
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict()
                },
                is_best=valid_loss.item() == es.best,
                path=exp_dir,
                target='RNNBeatProc'
            )

            # save params
        params = {
                'epochs_trained': epoch,
                'best_loss': es.best,
                'best_epoch': best_epoch,
                'train_loss_history': train_losses,
                'valid_loss_history': valid_losses,
                'train_time_history': train_times,
                'sep_trainloss_hist': sep_trainloss_hist,
                'sep_validloss_hist': sep_validloss_hist,
                'num_bad_epochs': es.num_bad_epochs,
                'lr_change_epoch': lr_change_epoch,
                'stop_t': stop_t,
                'model_info': model_info,
            }

        with open(os.path.join(target_jsonpath,  'RNNbeat' + '.json'), 'w') as outfile:
            outfile.write(json.dumps(params, indent=4, sort_keys=True))

        train_times.append(time.time() - end)
        
        if stop:
                print("Apply Early Stopping and retrain")

                stop_t +=1
                if stop_t >=5:
                    break
                lr = lr*0.2
                lr_change_epoch.append(epoch)
                optimizer = torch.optim.Adam(
                    model.parameters(),
                    lr=lr,
                    weight_decay= 0.00001
                )
                optimizer = Lookahead(optimizer)
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                        optimizer,
                        factor=0.3, 
                        patience=80,
                        cooldown=10
                    )

                es = utils.EarlyStopping(patience= patience, best_loss = es.best)
                
            
if __name__ == "__main__":
    main()