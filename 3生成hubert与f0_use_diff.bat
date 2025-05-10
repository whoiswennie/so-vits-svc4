@echo off
chcp 65001

miniconda3\python.exe preprocess_hubert_f0.py --f0_predictor dio --use_diff

cmd /k