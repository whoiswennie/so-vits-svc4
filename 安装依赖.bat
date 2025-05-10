@echo off
chcp 65001

miniconda3\python.exe -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/

echo 如果都成功了，那没事了，如果有失败的，请手动补装
cmd /k