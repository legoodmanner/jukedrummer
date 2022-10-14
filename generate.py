import torch
import torch.nn
import os
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from hparams import setup_lm_hparams, MODEL_LIST
import soundfile as sf
import argparse
import time

from model.LanguageModel import JukeTransformer
from data import AllFeatureDataset
from model.vocoder import HiFiVocoder
from hparams import MEL
from utils.functions import get_vqvae
from utils.beats import BeatInfoExtractor
from utils.melspec import Audio2Mel

def get_raw_data(input_dir):
    fns =  os.listdir(input_dir)
    fns = [f for f in fns if f.endswith('.wav')]
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('exp_idx', type=int)
    parser.add_argument('cuda', type=int)
    parser.add_argument('--input_dir', type=str, default='input/')
    parser.add_argument('--output_dir', type=str, default='output/')
    parser.add_argument('--temp', type=float, default=0.96)
    parser.add_argument('--top_p', type=float, default=0.8)
    args = parser.parse_args()

    # Parameters Initialization
    exp_idx = args.exp_idx
    output_dir = os.path.join(args.output_dir, f'exp{exp_idx}')
    hps = setup_lm_hparams(MODEL_LIST[exp_idx])
    hps.cuda = args.cuda
    hps.batch_size = 1
    print(f'setting up exp{exp_idx} parameters')
    print(f'cuda: {hps.cuda}')
    print(f'bs  : {hps.batch_size}')
    print(f'temp: {args.temp}')
    print(f'generate samples from {args.input_dir}')
    print(f'save generated samples to {args.output_dir}')

    os.makedirs(output_dir, exist_ok=True)
    device = torch.device(f'cuda:{hps.cuda}')
    
    # Model Initialization 
    target_vqvae, _, target_mean, target_std = get_vqvae(
        vq_idx=hps.vq_name.strip('vq'),
        data_type='target', 
        ckpt_dir=hps.ckpt_dir, 
        device = device
    )

    others_vqvae, _, others_mean, others_std = get_vqvae(
        vq_idx=hps.vq_name.strip('vq'),
        data_type='others', 
        ckpt_dir=hps.ckpt_dir, 
        device = device
    )

    lm = JukeTransformer(hps).to(device)
    lm_ckpt = torch.load(os.path.join(hps.ckpt_dir, f'exp{exp_idx}.pkl'), map_location=lambda storage, loc: storage)
    lm.load_state_dict(lm_ckpt['model'])

    
    vocoder = HiFiVocoder(
        ckpt_path=os.path.join(hps.ckpt_dir, 'hifigan/generator'), 
        output_dir=output_dir, 
        device=device
    )
    
    beat_extractor = BeatInfoExtractor(info_type=hps.bact_type, device=device)
    mel_extractor = Audio2Mel(MEL)
    
    dataset = AllFeatureDataset(
        args.input_dir, 
        others_vqvae, 
        beat_extractor, 
        mel_extractor, 
        others_mean, 
        others_std, 
        device
    )
    for j in range(20):
        for i in tqdm(range(len(dataset))):

            otz, bact, fn = dataset[i]
            with torch.no_grad():
                lm.eval()
                gen_mel = lm.sample(n_samples=hps.batch_size, otz=otz, bact=bact, vqvae=target_vqvae, temp=args.temp, top_p=args.top_p)
                gen_mel = gen_mel * target_std + target_mean 
                gen_wavs = vocoder(gen_mel)
            orig, _ = sf.read(os.path.join(args.input_dir, fn),always_2d=True )
            sf.write(os.path.join(output_dir, f'{j}_'+fn.replace('.npy', '.wav')), gen_wavs[:,None], 44100)

    # real_wavs = real_wav.detach().cpu().numpy()
