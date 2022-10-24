import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import argparse
import pickle
from jukebox.make_models import MODELS
from torchvision.utils import make_grid
from model.LanguageModel import JukeTransformer
from model.vqvae import VQVAE, Sampler
from dataset import *

from jukebox.train import get_optimizer
from hparams import OPT, MODEL_LIST, setup_lm_hparams



def get_dataset(hps):
    with open(os.path.join(hps.path, 'dataset.pkl'), 'rb') as f:
        dataset = pickle.load(f)
    
    tr_ids = dataset[0]
    va_ids = dataset[1]
    print(f'number of training data: {len(tr_ids)}')
    print(f'number of validation data: {len(va_ids)}')
    tr_dataset = BeatInfoPairedDataset(tr_ids, hps)
    va_dataset = BeatInfoPairedDataset(va_ids, hps)

    tr_dataloader = DataLoader(
        dataset=tr_dataset,
        batch_size=hps.batch_size,
        num_workers=4,
        shuffle=True,
        drop_last=True,
        pin_memory=True
    )

    va_dataloader = DataLoader(
        dataset=va_dataset,
        batch_size=hps.batch_size,
        num_workers=1,
        shuffle=True,
        drop_last=True,
        pin_memory=True
    )

    return tr_dataloader, va_dataloader

class Solver():
    def __init__(self, model, vqvae, binfo_type, device):
        
        self.device = device
        self.model = model
        self.vqvae = vqvae
        self.binfo_type = binfo_type
        self.criterion = nn.CrossEntropyLoss()
        self.opt, self.shd, scalar = get_optimizer(self.model, OPT)
     
    def run(self, data, summary, training=True, make_sample=False):
        self.opt.zero_grad()
        if training:
            self.model.train()
        else:
            self.model.eval()

        tgz = data[0].long().to(self.device)
        otz = data[1].long().to(self.device)
        ot_binfo = data[2].float().to(self.device)

        bs, l = tgz.size(0), tgz.size(1)

        loss, pred = self.model(tgz, otz, ot_binfo)
        summary['train loss' if training else 'valid loss'] = loss.item()
        # summary['train prime loss' if training else 'valid prime loss'] = metric[1].item()
        if training:
            loss.backward()
            self.opt.step()
            self.shd.step()

        if make_sample:
            with torch.no_grad():
                lr = self.opt.param_groups[0]['lr']
                pred_mel = vqvae.decode(torch.argmax(pred, dim=-1))
                pred_mel = make_grid(pred_mel[:4].unsqueeze(1), nrow=1).detach().cpu().numpy()
                pred_mel = wandb.Image(pred_mel.transpose(1,2,0))
                summary[f'{"train" if training else "valid"} pred mel'] = pred_mel

                real_mel = vqvae.decode(tgz)
                real_mel = make_grid(real_mel[:4].unsqueeze(1), nrow=1).detach().cpu().numpy()
                real_mel = wandb.Image(real_mel.transpose(1,2,0))
                summary[f'{"train" if training else "valid"} real mel'] = real_mel

        return summary

if __name__ == '__main__':
    ## HPARMAS
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', type=int)
    parser.add_argument('--exp_idx', type=int)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--bs', type=int, default=None)
    args = parser.parse_args()
    hps = setup_lm_hparams(MODEL_LIST[args.exp_idx])
    if args.bs:
        hps.batch_size = args.bs
    device = torch.device(f'cuda:{args.cuda}')
    print(f'Cuda: {args.cuda}')
    print(f'Batch Size: {hps.batch_size}')
    print(f'Codebook: {hps.codebook_size}')
    print(f'LM enc layers: {hps.enc_layers}')
    print(f'LM dec layers: {hps.dec_layers}')
    print(f'd model: {hps.d_model}')
    print(f'Beat info type: {hps.binfo_type}')


    ### MODEL 
    model = JukeTransformer(hps).to(device)
    if args.resume:
        ckpt = torch.load(
            os.path.join(hps.ckpt_dir, 'exp'+str(args.exp_idx)+'.pkl'),
            map_location=lambda storage, loc: storage
        )
        model.load_state_dict(ckpt['model'])
        print('resume from previous params...')
    vqvae = VQVAE(
        codebook_size=hps.codebook_size,
        encoder = Sampler(input_dim=80, output_dim=64, z_scale_factors=hps.downsample_ratios),
        decoder = Sampler(input_dim=64, output_dim=80, z_scale_factors=hps.upsample_ratios),
    )
    mean, std = vqvae.restore_from_ckpt(hps, device)
    solver = Solver(model, vqvae, hps.binfo_type, device)

    ### DATA
    tr_dataloader, va_dataloader = get_dataset(hps=hps) 

    ### WANDB
    try:
        import wandb
        is_wandb = True if args.wandb else False
    except ImportError:
        is_wandb = False

    if is_wandb:
        run = wandb.init(
            project='JukeDrummer Language model',
            entity='',
            config=hps,
            dir='./wandb',
            name= 'exp'+str(args.exp_idx),
        )

    ### TRAINING
    for epoch in range(OPT.epochs):
        summary = {}
        # train
        for idx, data in enumerate(tqdm(tr_dataloader)):
            summary = solver.run(
                data, 
                summary,
                make_sample= idx==len(tr_dataloader)-1 and epoch % hps.sample_step == 0 #final one
            )
        # valid
        for idx, data in enumerate(tqdm(va_dataloader)):
            with torch.no_grad():
                summary = solver.run(
                    data, 
                    training=False,
                    summary=summary,
                    make_sample= idx==len(va_dataloader)-1 and epoch % hps.sample_step == 0 #final one
                )
        #save to wandb
        if epoch % hps.sample_step == 0:
            torch.save(
                {
                    'model': model.state_dict(),
                    'hps': dict(hps)
                }, 
                os.path.join(hps.ckpt_dir, 'exp'+str(args.exp_idx)+'.pkl'))
            
        print(f"{str(epoch).zfill(4)} | train loss: {summary['train loss']} | valid loss: {summary['valid loss']}")
        if is_wandb:
            wandb.log(data=summary, step=epoch)


        
