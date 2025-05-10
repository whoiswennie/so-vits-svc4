@echo off
chcp 65001

miniconda3\python.exe train_diff.py -c configs/diffusion.yaml

cmd /k