import torch
import torch.nn as nn
import os
from tqdm import tqdm 
from torchvision.utils import make_grid

from dataset import *
from model.vqvae import VQVAE, Sampler
import argparse

from hparams import setup_vq_hparams, OPT
from jukebox.train import get_optimizer

def get_dataset(hps, data_type):
    with open(os.path.join(hps.path, 'dataset.pkl'), 'rb') as f:
        dataset = pickle.load(f)
    
    mean, std, non_nan = compute_mean_std(
        os.path.join(hps['path'], 'mel', args.data_type), dataset
    )

    tr_ids = dataset[0]
    va_ids = dataset[1]

    tr_dataset = MelDataset(tr_ids, hps, data_type)
    va_dataset = MelDataset(va_ids, hps, data_type)

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
        pin_memory=True,
    )
    return tr_dataloader, va_dataloader, mean, std,


### Hyper Parameter Setting
parser = argparse.ArgumentParser()
parser.add_argument('--vq_idx', type=int)
parser.add_argument('--data_type', type=str, choices={'target', 'others'})
parser.add_argument('--cuda', type=int)
parser.add_argument('--wandb', action='store_true')
args = parser.parse_args()
hps = setup_vq_hparams('vq'+str(args.vq_idx))
sequence_length = 4096 // (np.prod(hps['upsample_ratios']))
device = torch.device(f'cuda:{args.cuda}')
if args.data_type == 'others':
    hps['codebook_size'] = 1024

## Model Setting
encoder = Sampler(input_dim=80, output_dim=64, z_scale_factors=hps['downsample_ratios']) 
decoder = Sampler(input_dim=64, output_dim=80, z_scale_factors=hps['upsample_ratios'])
model = VQVAE(
    codebook_size=hps['codebook_size'],
    encoder= encoder,
    decoder= decoder,
    device= device
    ).to(device)

### Data Setting

tr_dataloader, va_dataloader, mean, std = get_dataset(hps, data_type = args.data_type )

### OPT
opt, shd, scalar = get_optimizer(model, OPT)
criterion = nn.MSELoss()

### WANDB
try:
    import wandb
    is_wandb = True if args.wandb else False
except ImportError:
    is_wandb = False

if is_wandb:
    run = wandb.init(
        project='JukeDrummer VQ-VAE',
        entity='',
        dir='./wandb',
        config=hps,
        name=f'{hps.name} {args.data_type}',
    )

### TRAINNING 
mean, std = mean.to(device), std.to(device)
for epoch in range(1000 + 1):
    model.train()
    summary = {}
    for mel in tqdm(tr_dataloader):
        opt.zero_grad()
        mel = mel.to(device) 
        mel = (mel - mean) / std
        r_mel, commit_loss, metric = model(mel)
        reconstruct_loss = criterion(r_mel, mel)
        loss = commit_loss * hps['commit_beta'] + reconstruct_loss

        loss.backward()
        opt.step()
        shd.step()
        summary['train reconstruct loss'] = reconstruct_loss.item()
        summary['train commit loss'] = commit_loss.item()

    model.eval()
    for mel in tqdm(va_dataloader):
        mel = mel.to(device)
        mel = (mel - mean) / std
        with torch.no_grad():
            r_mel, commit_loss, metric = model(mel)
            reconstruct_loss = criterion(r_mel, mel)

        summary['valid reconstruct loss'] = reconstruct_loss.item()
        summary['valid commit loss'] = commit_loss.item()

    if epoch % 50 == 0:
        r_mel_img = make_grid(r_mel[:4].unsqueeze(1), nrow=1).detach().cpu().numpy()
        r_mel_img = wandb.Image(r_mel_img.transpose(1,2,0))

        mel_img = make_grid(mel[:4].unsqueeze(1), nrow=1).detach().cpu().numpy()
        mel_img = wandb.Image(mel_img.transpose(1,2,0))
        if is_wandb:
            wandb.log(data={'real': mel_img, 'reconstruct': r_mel_img})
        print(hps['name'])

    
    model_dict = {
        "model": model.state_dict(),
        "mean": mean.detach().cpu().numpy(),
        "std": std.detach().cpu().numpy(),
        "hps": dict(hps),
    }

    torch.save(
        model_dict, 
        os.path.join(hps.ckpt_dir, f'{hps.name}_{args.data_type}.pkl', ))

    print(summary)
    print(f"{epoch} || usage: {metric['usage'].item()} | used_curr: {metric['used_curr'].item()}")
    if is_wandb:
        wandb.log(data=summary, step=epoch)
