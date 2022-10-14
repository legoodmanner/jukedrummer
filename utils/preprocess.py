import numpy as np
import soundfile as sf
import os
from tqdm import tqdm
import argparse 
import musdb
import yaml
import pickle
import random

def make_final_filelist(output_dir):
    others_fl = os.listdir(os.path.join(output_dir,'others'))
    others_fl = [f for f in others_fl if f.endswith('npy')]
    target_fl = os.listdir(os.path.join(output_dir, 'target'))
    target_fl = [f for f in target_fl if f.endswith('npy')]

    for f in target_fl:
        if f not in others_fl:
            target_fl.remove(f)
    with open(f'{output_dir}/dataset.pkl') as f:
        pickle.dump(target_fl,f)

def pad_as(orig_arr, target_shape):

    diff = target_shape[0] - orig_arr.shape[0]
    if diff > 0:
        reshape_arr = np.pad(orig_arr, ((0,diff), (0,0)), mode='constant', constant_values=0)
    elif diff < 0:
        reshape_arr = orig_arr[:diff]
    else:
        reshape_arr = orig_arr
    assert reshape_arr.shape == target_shape, f'target shape {target_shape}, orig_arr shape {orig_arr.shape} reshape {reshape_arr.shape}'
    return reshape_arr

class MUSDB():
    def __init__(self, musdb_root, output_dir) -> None:
        self.output_dir = output_dir
        self.mus = musdb.DB(root=musdb_root)

    def write(self):
        tg_option = ['drums', 'bass', 'other', 'vocals']
        for track in tqdm(self.mus):
            assert track.rate == 44100

            # target
            tg_audio = track.targets['drums'].audio
            sf.write(os.path.join(self.output_dir, 'target' ,track.name + '.wav'), tg_audio, track.rate)

            # other
            other_audio = np.zeros_like(tg_audio)
            other_option = [op for op in tg_option if not op == 'drums']
            for op in other_option:
                other_audio += track.targets[op].audio
            sf.write(os.path.join(self.output_dir, 'others' ,track.name + '.wav'), other_audio, track.rate)
    
    def get_fns(self,):
        fns = []
        for track in tqdm(self.mus):
            fns += [track.name]
        return fns

class MeldyDB():
    def __init__(self, mddb_root, output_dir) -> None:
        self.mddb_root = mddb_root
        self.output_dir = output_dir

    def write(self):
        fl = os.listdir(os.path.join(self.mddb_root, 'audio'))
        for file in tqdm(fl):
            meta_path = os.path.join(self.mddb_root, 'metadata', file+'_METADATA.yaml')
            audio_path = os.path.join(self.mddb_root, 'audio', file)
            with open(meta_path, 'r') as f:
                data = yaml.safe_load(f)

            other_audio = None
            drum_exist = False
            for stem in data['stems']:
                if data['stems'][stem]['instrument'] == 'drum set':
                    drum_exist = True
                    y, sr = sf.read(os.path.join(audio_path, file+'_STEMS', data['stems'][stem]['filename']))  
                    assert sr == 44100
                    sf.write(os.path.join(self.output_dir, 'target' ,file + '.wav'), y, sr)
                else:
                    y, sr = sf.read(os.path.join(audio_path, file+'_STEMS', data['stems'][stem]['filename']))
                    assert sr == 44100
                    if other_audio is None:
                        other_audio = y
                    else:
                        other_audio += y

        sf.write(os.path.join(self.output_dir, 'others' ,file + '.wav'), other_audio, sr)
        if not drum_exist:
            sf.write(os.path.join(self.output_dir, 'target' ,file + '.wav'), np.zeros_like(other_audio), sr)
    
    def get_fns(self):
        return os.listdir(os.path.join(self.mddb_root, 'audio'))

class MixingSecret():
    def __init__(self, ms_root, output_dir) -> None:
        self.output_dir = output_dir
        self.ms_root = ms_root
    def write(self):

        fl = os.listdir(os.path.join(ms_root, 'audios'))
        for file in tqdm(fl):
            print(file)
            meta_path = os.path.join('/home/lego/NAS189/home/MixingSecrets/', 'metadata', file+'_METADATA.yaml')
            audio_path = os.path.join(ms_root, 'audios', file)
            with open(meta_path, 'r') as f:
                data = yaml.safe_load(f)

            other_audio = None
            target_audio = None
            drum_exist = False
            
            for stem in data['stems']:
                if data['stems'][stem]['instrument'] == 'drum set':
                    drum_exist = True
                    for raw in data['stems'][stem]['raw']:
                        fn = data['stems'][stem]['raw'][raw]['filename']
                        try:
                            y, sr = sf.read(os.path.join(audio_path, fn), always_2d=True) 
                        except:
                            continue 
                        if y.shape[-1] == 1:
                            y = np.repeat(y, 2, axis=-1)  
                        assert sr == 44100
                        if target_audio is None:
                            target_audio = y
                        else:
                            y = pad_as(y, target_audio.shape)
                            target_audio += y
                    sf.write(os.path.join(self.output_dir, 'target' ,file + '.wav'), target_audio, sr)

                else:
                    for raw in data['stems'][stem]['raw']:
                        fn = data['stems'][stem]['raw'][raw]['filename']
                        try:
                            y, sr = sf.read(os.path.join(audio_path, fn), always_2d=True)
                        except:
                            continue
                        if y.shape[-1] == 1:
                            y = np.repeat(y, 2, axis=-1)  
                        assert sr == 44100

                        if other_audio is None:
                            other_audio = y
                        else:
                            y = pad_as(y, other_audio.shape)
                            other_audio += y
                    sf.write(os.path.join(self.output_dir, 'others' ,file + '.wav'), other_audio, sr)

            if not drum_exist:
                sf.write(os.path.join(self.output_dir, 'target' ,file + '.wav'), np.zeros_like(other_audio), sr)
    def get_fns(self):
        return os.listdir(os.path.join(self.ms_root, 'audios'))


audio_root = '/home/lego/NAS189/home/codify/data/drums/audio/others/'
output_path = '/home/lego/NAS189/home/codify/data/drums/hop_audio_24s/others/'

def seg_clip():
    # dclass = 'target'
    # fdir = f'/home/lego/NAS189/home/codify/data/drums/audio/{dclass}'
    # pkl_output = '/home/lego/data/codify/feature/'
    # propotion = 0.2

    fl = os.listdir(fdir)
    fl = [f.strip('.wav') for f in fl]
    length = len(fl)
    print(f'songs count {length}')
    assert length > 0

    random.seed(1224)
    random.shuffle(fl)
    train_list = fl[round(length*propotion):]
    valid_list = fl[:round(length*propotion)]
    print(f'train data number: {len(train_list)}')
    print(f'valid data number: {len(valid_list)}')
    with open(os.path.join(pkl_output,'train_files.pkl'), 'wb') as f:
        pickle.dump(sorted(train_list), f)
    with open(os.path.join(pkl_output,'valid_files.pkl'), 'wb') as f:
        pickle.dump(sorted(valid_list), f)


length = 8192 * 8 * 4 * 4
fl = os.listdir(audio_root)
hop_length = length // 2

os.makedirs(output_path, exist_ok=True)
for fn in tqdm(fl):
    fpath = os.path.join(audio_root, fn)
    y, sr = sf.read(fpath)
    count = 0
    while(count*hop_length+length < len(y)):
        seg = y[count*hop_length : count*hop_length+length]
        segname = os.path.join(output_path,f'{fn.strip(".wav")}_{count}.wav')
        count += 1
        if os.path.exists(segname):
            print('already done')
            continue
        else:
            sf.write(segname, seg, sr)

def multi_dataset_proc(musdb_root, mddb_root, ms_root, output_path):
    gather_musdb(musdb_root, output_path)
    gather_mddb(mddb_root, output_path)
    gather_ms(ms_root, output_path)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    args = parser.parse_args()
    musdb_root = os.path.join(args.dataset_root, 'musdb')
    mddb_root = os.path.join(args.dataset_root, 'mddb')
    ms_root = os.path.join(args.dataset_root, 'ms')
    multi_dataset_proc(musdb_root, mddb_root, ms_root, args.output_path)