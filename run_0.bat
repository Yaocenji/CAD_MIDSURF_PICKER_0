@echo off
title 点云渲染一键启动

cd /d "%~dp0"

echo 正在激活 Conda 环境...
call D:\ANACONDA3\Scripts\activate.bat C:\Users\27800\miniconda3\envs\picker

echo 正在运行 ...
python face_highlighter_1.py

echo.
pause