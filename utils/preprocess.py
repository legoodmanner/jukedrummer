import numpy as np
import soundfile as sf
import os
from tqdm import tqdm
import argparse 
import pickle
import random

from segmentation import inference as data_segmentation
from melspec import inference as melspec_extraction
from subset_division import inference as subset_division
from beats import inference as beat_info_extraction


def main(args):
    # segment -> mel extract -> div subset -> beat information extract
    audio_dir = args.audio_dir
    mel_dir = args.mel_dir
    beat_dir= args.beat_dir
    
    

    length = 8192 * 8 * 4 * 4 # This variation is recommended to be fixed
    fns = os.listdir(audio_dir, 'target')
    fns = [f for f in fns if f.endwiths('.wav')]
    # 1. Segmentation by either downbeats or hop window
    data_segmentation(fns, args.segment_by_downbeats, length)
    # 2. Extract Mel spectrograms from segemented audio waves
    melspec_extraction(audio_dir, mel_dir)
    # 3. Divide dataset into train & valid subset
    subset_division(mel_dir, args.dataset_pkl_path)
    # 4. Beat Information Extraction
    beat_info_extraction(fns, 'low', audio_dir, beat_dir, args.cuda)


    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio_dir', type=str, help='directory path of unsegemented audio', default='data/audio')
    parser.add_argument('--segement_audio_dir', type=str, help='directory path of segemented audio', default='data/segment_audio')
    parser.add_argument('--mel_dir', type=str, help='directory path of segemented audio', default='data/mel')
    parser.add_argument('--beat_dir', type=str, help='directory path of beat information', default='data/beats')

    parser.add_argument('--cuda', type=int, help='the id of cuda want to use')
    parser.add_argument('--dataset_pkl_path', type=str, help='the path of final dataset .pkl file', default='data/')
    parser.add_argument('--segment_by_downbeats', type=bool, default=True, help='determine whether the segement would be made according to downbeats or not')
    args = parser.parse_args()
    main(args)