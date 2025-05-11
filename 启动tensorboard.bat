chcp 65001
@echo off

echo 正在启动Tensorboard...
echo 如果看到输出了一条网址（大概率是localhost:6006）就可以访问该网址进入Tensorboard了

REM 使用指定的 miniconda3 的 python 来运行 tensorboard
miniconda3\python.exe -m tensorboard.main --logdir=%~dp0logs/44k

pause