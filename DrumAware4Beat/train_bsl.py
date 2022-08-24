# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 16:00:00 2020

@author: CITI
"""

import os
import numpy as  np
from models.BaselineBLSTM import RNNDownBeatProc as RNNmodel
import drumaware_dataset as mmdataset
from torch.utils.data import DataLoader
import tqdm
import torch
import torch.nn as nn
from pathlib import Path
import json
import utils
import time
import sys
from lookahead_pytorch import Lookahead

def train(model, device, train_loader, optimizer):
    model.train()
    train_loss = 0
    pbar = tqdm.tqdm(train_loader, disable = False)
    for x, y in pbar:
        # break
        pbar.set_description("Training batch")
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        
        y_hat = model(x) # beat activations (batch, timestep, 3) ==> nonbeat(0), donwbeat(1), beat(2)
        y_hat = y_hat.reshape((-1, 3))
        y = y.reshape((-1)).to(dtype = torch.long) # required type of loss function
        

        weights = [1, 200, 67] # nonbeat, beat, downbeat
        class_weights = torch.FloatTensor(weights).to(device)
        CE = nn.CrossEntropyLoss(weight = class_weights)
        loss = CE(y_hat, y)
        loss.backward()
        train_loss += loss
        optimizer.step()

    return train_loss/len(train_loader.dataset)

def valid(model, device, valid_loader ):
    model.eval()
    valid_loss = 0
    with torch.no_grad():
        for x, y in valid_loader:
            x, y = x.to(device), y.to(device)
            
            y_hat = model(x) # beat activations (batch, timestep, 3) ==> nonbeat(0), donwbeat(1), beat(2)
            y_hat = y_hat.reshape((-1, 3))
            y = y.reshape((-1)).to(dtype = torch.long) # required type of loss function
            
            weights = [1, 200, 67] # nonbeat, beat, downbeat
            class_weights = torch.FloatTensor(weights).to(device)
            CE = nn.CrossEntropyLoss(weight = class_weights)
            loss = CE(y_hat, y)
            valid_loss += loss
    return valid_loss/len(valid_loader.dataset)



def main():

    cuda_num = 0#int(sys.argv[1])
    cuda_str = 'cuda:'+str(cuda_num)
    device = torch.device(cuda_str if torch.cuda.is_available() else 'cpu')
    train_epochs = 2000
    # must assign
    date = '0615'
    exp_num = 1#sys.argv[2] ## added for repeat experiments
    exp_name = 'RNNBeat_bsl_V'+str(exp_num) + '_'+date
    exp_dir = os.path.join('./experiments', exp_name)
    target_jsonpath = exp_dir
    lr = 1e-2
    patience = 20
    
    ### extra information to save in json
    model_type = 'bsl_blstm'
    model_simpname = 'baseline_v'+str(exp_num)
    model_dir = exp_dir
    model_info = dict(model_type = model_type,
                      model_simpname = model_simpname,
                      model_dir = model_dir, 
                      )
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
    
    model = RNNmodel()
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
    train_times = []
    lr_change_epoch = []
    best_epoch = 0
    stop_t = 0
    for epoch in t:
#        break
        t.set_description("Training Epoch")
        end = time.time()
        train_loss = train(model, device, train_loader, optimizer)
        valid_loss = valid(model, device, valid_loader)
        
        scheduler.step(valid_loss)
        train_losses.append(train_loss.item())
        valid_losses.append(valid_loss.item())

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
                is_best=valid_loss == es.best,
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