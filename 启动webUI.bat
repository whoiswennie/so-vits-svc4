@echo off
chcp 65001

set PATH=%cd%;%PATH%

miniconda3\python.exe webUI.py

cmd /k