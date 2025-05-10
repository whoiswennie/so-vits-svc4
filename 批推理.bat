@echo off
chcp 65001

miniconda3\python.exe batch_inference.py -m "trained/huan/release.pth" -c "trained/huan/config.json" -t 0 -s "huan" -cl 90 -f0p "rmvpe"

cmd /k
