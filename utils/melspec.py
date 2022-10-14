import pickle
import os
import librosa
from tqdm import tqdm
import numpy as np
from multiprocessing import Pool
from librosa.filters import mel as librosa_mel_fn
import torch
from torch import nn
from torch.nn import functional as F
import pickle
import argparse

'''
Modified from
https://github.com/descriptinc/melgan-neurips/blob/master/mel2wav/modules.py#L26
'''



class Audio2Mel(nn.Module):
    def __init__(self, hps):
        super().__init__()
        window = torch.hann_window(hps.win_length).float()
        mel_basis = librosa_mel_fn(
            hps.sampling_rate, hps.n_fft, hps.n_mel_channels, hps.mel_fmin, hps.mel_fmax
        )
        mel_basis = torch.from_numpy(mel_basis).float()
        self.register_buffer("mel_basis", mel_basis)
        self.register_buffer("window", window)
        self.hop_length = hps.hop_length
        self.n_fft = hps.n_fft     
        self.win_length = hps.win_length
        self.window = window
    def forward(self, audio):
        p = (self.n_fft - self.hop_length) // 2
        audio = F.pad(audio, (p, p), "reflect").squeeze(1)
        fft = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=False,
            return_complex=False
        )
        real_part, imag_part = fft.unbind(-1)
        magnitude = torch.sqrt(real_part ** 2 + imag_part ** 2)
        mel_output = torch.matmul(self.mel_basis, magnitude)
        log_mel_spec = torch.log10(torch.clamp(mel_output, min=1e-5))
        return log_mel_spec


def process_audios(fn, out_dir, wav_dir, sample_rate, extract_func):
    y, _ = librosa.load(os.path.join(wav_dir, fn), sr=sample_rate)
    peak = np.abs(y).max()
    if peak > 1.0:
        y /= peak
    y = torch.from_numpy(y)
    try:
        mel = extract_func(y[None, None])
        mel = mel.numpy()[0].astype(np.float32)
        if np.any(np.isnan(mel)) or np.sum(np.mean(mel, axis=0, keepdims=False) < -8) > mel.shape[1] // 2:
            return id, 0
        np.save(os.path.join(out_dir, fn.replace('wav', 'npy')), mel, allow_pickle=False)
    except:
        print('error occur')
        return id, 0
    return fn, mel.shape[-1]


if __name__ == "__main__":
    import sys
    sys.path.append(os.getcwd())
    from hparams import MEL as MEL_HPARAMS
    from functools import partial

    parser = argparse.ArgumentParser()
    parser.add_argument('--wav_dir', type=str, help='path of input wave audio', required=True)
    parser.add_argument('--out_dir', type=str, help='path of output mel', required=True)
    parser.add_argument('--process_num', type=int, help='number of processor used to run for multitask pool', default=20)
    args = parser.parse_args()

    # gitignore
    # out_dir = f'/home/lego/NAS189/home/codify/data/drums/feature/mel/hop/others'
    # wav_dir =  f'/home/lego/NAS189/home/codify/data/drums/hop_audio_24s/others/' 
    
    os.makedirs(args.out_dir, exist_ok=True)
    hps = MEL_HPARAMS
    extract_func = Audio2Mel(hps)
    sr = hps.sampling_rate

    # Get list of files
    audio_fns = [fn for fn in os.listdir(args.wav_dir) if fn.endswith(hps.extension)]
    audio_fns = sorted(list(audio_fns))

    # Initiate a pool
    pool = Pool(processes=args.process_num)
    dataset = []

    # Process
    for i, (fn, length) in enumerate(tqdm(pool.imap(
        partial(process_audios, 
                out_dir=args.out_dir,
                wav_dir=args.wav_dir,
                sample_rate=sr,
                extract_func=extract_func,
                ), audio_fns)),1):

