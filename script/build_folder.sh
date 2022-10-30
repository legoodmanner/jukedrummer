#!/bin/bash

# Data directory:
# | data 
# |--- audio
#      |------target
#      |------others
# |--- segment_audio
#      |------target
#      |------others
# |--- beats
#      |------low
#      |------mid
#      |------high
# |--- mel
#      |------target
#      |------others
# |--- token
#      |------target
#             |------vq#
#      |------others
#             |------vq#

# | ckpt
# |--- vocoder
# |--- beat_tracker

make_target_others(){
    mkdir $1/target
    mkdir $1/others
}

mkdir -p data/audio data/segment_audio data/beats data/mel data/token
mkdir -p ckpt/vocoder ckpt/beat_tracker
mkdir data/token/low data/token/mid data/token/high
make_target_others data/audio
make_target_others data/segment_audio
make_target_others data/mel
make_target_others data/token