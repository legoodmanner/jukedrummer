import os
import numpy as np
import pickle
import ffmpeg
import librosa
from tqdm import tqdm
import soundfile as sf
from madmom.features.downbeats import DBNDownBeatTrackingProcessor
from madmom.features.downbeats import RNNDownBeatProcessor
import argparse

def pad_to(audio, length):
    assert len(audio.shape) == 1, audio.shape
    if len(audio) == length: 
        return audio
    return np.pad(audio, (0,length - len(audio)), 'constant', constant_values=0)

def get_downbeats(fn, beat_proc, track_proc, root):
    drums, sr = librosa.load(os.path.join(root, 'audio', 'target', fn), 44100)
    others, sr = librosa.load(os.path.join(root, 'audio', 'others', fn), 44100)
    drums = pad_to(drums, max(len(drums), len(others)))
    others = pad_to(others, max(len(drums), len(others)))
    act = beat_proc(others+drums)
    downbeats = [ t[0] for t in track_proc(act) if t[1]==1]
    return downbeats

def segmentation(fn, downbeats, length, root):
    others, sr = librosa.load(os.path.join(root,'audio', 'others', fn), sr=44100)
    drums, sr = librosa.load(os.path.join(root,'audio', 'target', fn), sr=44100)
    if not len(others) == len(drums):
        to_pad = max(len(others), len(drums))
        others = pad_to(others, to_pad)
        drums = pad_to(drums, to_pad)
    if downbeats == None:
        while(count*length+length < len(others)):
            others_s = others[count*length:count*length+length]
            drums_s = drums[count*length:count*length+length]
            sf.write(os.path.join(root, 'segment_audio', 'others',  f'{fn.split(".")[0]}_{count}.wav'), others_s, 44100)
            sf.write(os.path.join(root, 'segment_audio', 'target',  f'{fn.split(".")[0]}_{count}.wav'), drums_s, 44100)
            count += 1
    else:
        start = downbeats.pop(0)
        count = 0 
        while len(downbeats) != 0:
            cur = downbeats.pop(0)
            if cur - start > 24:
                start = round(start * 44100 )
                drums_s = drums[start:start+length]
                others_s = others[start:start+length]
                sf.write(os.path.join(root, 'segment_audio', 'others',  f'{fn.split(".")[0]}_{count}.wav'), others_s, 44100)
                sf.write(os.path.join(root, 'segment_audio', 'target',  f'{fn.split(".")[0]}_{count}.wav'), drums_s, 44100)
                start = cur
                count += 1



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=str, required=True)
    parser.add_argument('seg_by_downbeats', type=bool, default=False)
    args = parser.parse_args()

    root = args.root
    beat_proc = RNNDownBeatProcessor()
    track_proc = DBNDownBeatTrackingProcessor(beats_per_bar=[3, 4], fps=100)
    length = 8192 * 8 * 4 * 4

    with open(os.path.join(args.root ,'dataset.pkl'), 'rb') as f:
        fl = pickle.read(f)

    for fn in tqdm(fl):
        if args.seg_by_downbeats:
            downbeats = get_downbeats(fn, beat_proc, track_proc, root)
        else:
            downbeats = None
        segmentation(fn, downbeats, length, root)