@echo off
chcp 65001

miniconda3\python.exe train.py -c configs/config.json -m 44k

cmd /k