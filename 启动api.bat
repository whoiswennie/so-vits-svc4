@echo off
chcp 65001

miniconda3\python.exe flask_api_full_song.py -mn "trained/huan/release.pth" -cn "trained/huan/config.json" 

cmd /k
