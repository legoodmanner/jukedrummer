
import torch
import os
import numpy as np
from tqdm import tqdm
import argparse

from utils.functions import get_vqvae, mel2token


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', type=int)
    parser.add_argument('--vq_idx', type=int, required=True)
    parser.add_argument('--data_type', type=str, choices=['target', 'others'], required=True)
    parser.add_argument('--ckpt_dir', type=str, required=True)
    parser.add_argument('--mel_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.cuda}' if args.cuda else 'cpu')
    
    vqvae, hps, mean, std = get_vqvae(
        vq_idx=args.vq_idx,
        data_type=args.data_type,
        ckpt_dir=args.ckpt_dir,
        device=device
    )

    os.makedirs(os.path.join(args.output_dir, args.data_type, hps.name), exist_ok=True)
    fl = os.listdir(os.path.join(args.mel_dir, hps.data_type))

    for fname in tqdm(fl):
        if not fname.endswith('.npy'):
            fname += '.npy' 
        dpath = os.path.join(args.mel_dir, hps.data_type, fname)
        try:
            token = mel2token(dpath, vqvae, mean, std, device)
            np.save(os.path.join(args.output_dir, args.data_type, hps.name ,fname), token)
        except:
            pass