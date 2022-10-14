from distutils.command.config import config
from turtle import forward
import numpy as np
import torch.nn as nn
import torch
import json
import os
from scipy.io.wavfile import write
import argparse
from hifi_gan.models import Generator 
from hifi_gan.meldataset import MAX_WAV_VALUE

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

class HiFiVocoder():
    def __init__(self, ckpt_path, output_dir, device):
        self.ckpt_path = ckpt_path
        self.output_dir = output_dir
        self.device = device
        with open('src/HiFiGan_config.json') as f:
            data = f.read()
        json_config = json.loads(data)
        self.h = AttrDict(json_config)
        self.model_init()
    
    def model_init(self):
        self.generator = Generator(self.h).to(self.device)
        state_dict_g = torch.load(self.ckpt_path, map_location=self.device)
        self.generator.load_state_dict(state_dict_g['generator'])
        self.generator.eval()
        self.generator.remove_weight_norm()
        
    def __call__(self, x, write_fname=None):
        with torch.no_grad():
            if len(x.shape) < 3:
                x = x.unsqueeze(0)
            y_g_hat = self.generator(x)
            audio = y_g_hat.squeeze()
            audio = audio * MAX_WAV_VALUE
            audio = audio.cpu().numpy().astype('int16')
            if write_fname:
                output_file = os.path.join(self.output_dir, str(write_fname) + '_generated.wav')
                write(output_file, h.sampling_rate, audio)
                print(output_file)
            return audio




def main():
    print('Initializing Inference Process..')

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_wavs_dir', default='test_files')
    parser.add_argument('--output_dir', default='generated_files')
    parser.add_argument('--checkpoint_file', required=True)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--cuda', default=0, type=int)
    a = parser.parse_args()

    config_file = os.path.join(os.path.split(a.checkpoint_file)[0], 'config.json')
    with open(config_file) as f:
        data = f.read()

    global h
    json_config = json.loads(data)
    h = AttrDict(json_config)

    torch.manual_seed(h.seed)
    global device
    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)
        device = torch.device(f'cuda:{a.cuda}')
    else:
        device = torch.device('cpu')



if __name__ == '__main__':
    main()
