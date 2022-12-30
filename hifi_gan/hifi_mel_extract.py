# import glob
from copyreg import pickle
import os
import soundfile as sf
import librosa
from librosa.util import normalize
import numpy as np
# from utils.display import *
# from utils.dsp import *
# import hparams as hp
# from multiprocessing import Pool, cpu_count
from multiprocessing import Pool
from librosa.filters import mel as librosa_mel_fn
import torch
from torch import nn
from torch.nn import functional as F
import pickle
import argparse

from meldataset import mel_spectrogram, MAX_WAV_VALUE

'''
Modified from
https://github.com/descriptinc/melgan-neurips/blob/master/mel2wav/modules.py#L26
'''



def convert_file(path):
    y, _ = sf.read(path)
    y = y / MAX_WAV_VALUE
    y = normalize(y) * 0.95
    y = torch.FloatTensor(y).mean(dim=-1)
    mel = mel_spectrogram(
        y.unsqueeze(0),
        n_fft=n_fft, 
        num_mels=n_mel_channels, 
        sampling_rate=sampling_rate,
        hop_size=hop_length,
        win_size=win_length,
        fmax=fmax,
        fmin=fmin,
    )
    mel = mel.squeeze().numpy()
    return mel.astype(np.float32)


def process_audios(path):
    id = path.split('/')[-1][:-4]
    out_fp = os.path.join(base_out_dir, f'{id}.npy')
    if os.path.exists(out_fp):
        print('Done before')
        return id, 0

    m = convert_file(path)
    if np.any(np.isnan(m)):
        return id, 0
    if np.sum(np.mean(m, axis=0, keepdims=False) < -8) > m.shape[1] // 2:
        return id, 0
    np.save(out_fp, m, allow_pickle=False)
    # except Exception:
    #     return id, 0
    return id, m.shape[-1]


if __name__ == "__main__":

    # parser = argparse.ArgumentParser()
    # parser.add_argument('--cuda', type=int, default=0)
    # args = parser.parse_args()

    clip_dir =  f'/home/lego/NAS189/home/codify/data/drums/hop_audio_24s/others/'  #out_dir from step1
    base_out_dir = f'/home/lego/NAS189/home/codify/data/drums/feature/mel/hifi/others/'
    os.makedirs(base_out_dir, exist_ok=True)

    n_fft = 1024
    hop_length = 256 #[241, 482, 964, 1928, 3856]
    win_length = 1024
    sampling_rate = 44100
    n_mel_channels = 80 #[80, 40, 20, 10, 5]
    fmin= 0
    fmax= 8000

    # ### Process ###

    audio_fns = [fn for fn in os.listdir(clip_dir) if fn.endswith('.wav')]
    audio_fns = sorted(list(audio_fns))
    audio_files = [os.path.join(clip_dir, fn) for fn in audio_fns]

    pool = Pool(processes=20)
    dataset = []

    for i, (id, length) in enumerate(pool.imap_unordered(process_audios, audio_files), 1):
        print(id)
        if length == 0:
            continue
        dataset += [id]

    with open(os.path.join(base_out_dir, 'others_dataset.pkl'), 'wb') as f:
        pickle.dump(dataset, f)
