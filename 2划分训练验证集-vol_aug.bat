@echo off
chcp 65001

miniconda3\python.exe preprocess_flist_config.py --speech_encoder vec768l12 --vol_aug

cmd /k
