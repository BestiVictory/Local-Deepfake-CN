#!/bin/bash
# 进入指定文件夹
cd "/home/ubuntu/mnt_4T/why/AV/wav2lip256"
# 执行 Python 脚本
python inference_torch.py \
  --face_dir "/home/ubuntu/mnt_4T/why/AV/data/temp_1000_temp/video/quarter_1" \
  --audio_dir "/home/ubuntu/mnt_4T/why/AV/data/temp_1000_temp/audio/quarter_1" \
  --outfile_dir "/home/ubuntu/mnt_4T/why/AV/data/temp_1000/all_fake_results/all_fake_results1"