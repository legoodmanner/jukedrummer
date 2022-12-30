
python3 train.py \
    --config config_v1.json \
    --input_wavs_dir /home/lego/NAS189/home/codify/data/drums/hop_audio_24s/wavs \
    --input_training_file /home/lego/NAS189/home/codify/data/drums/feature/finetune_mel/melgan/train_files.txt \
    --input_validation_file /home/lego/NAS189/home/codify/data/drums/feature/finetune_mel/melgan/test_files.txt \
    --checkpoint_path /home/lego/NAS189/home/codify/ckpt/hifigan/ \
    --cuda 0 \
    --input_mels_dir /home/lego/NAS189/home/codify/data/drums/feature/finetune_mel/melgan \
    --fine_tuning true
    
