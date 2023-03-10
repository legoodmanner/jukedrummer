#!/bin/bash

# File id
tracker=15GjIBsGbULRyDL3ze4wsR4opzUQN809d
vq1_target=1sblulUyla-R61BU5Ky9Z4QDDISDwG2v5
vq1_others=1YRC0jFzPw1sgQoKD0tsjQ3KUknvZf8_t
exp11=1sMRrOWqE9GvtxeO1jq8iEMsJjLl4lkij
exp1=18uw2gvEXL6yQ2dQk3eHyvgPVhkGBB-Wa
generator=1Un1pm_8NaG5lUIrVTS4l0s0oWbGkllrb

# download
wget -O ckpt/vq1_target.pkl 'https://docs.google.com/uc?export=download&id='$vq1_target
wget -O ckpt/vq1_others.pkl 'https://docs.google.com/uc?export=download&id='$vq1_others
wget -O ckpt/exp1.pkl 'https://docs.google.com/uc?export=download&id='$exp1
wget -O ckpt/exp11.pkl 'https://docs.google.com/uc?export=download&id='$exp11
wget -O ckpt/RNNBeatProc.pth 'https://docs.google.com/uc?export=download&id='$tracker
wget -O ckpt/vocoder/generator 'https://docs.google.com/uc?export=download&id='$generator



