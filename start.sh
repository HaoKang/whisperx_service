#!/bin/bash

# WhisperX 服务启动脚本

# 激活 conda 环境
source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || source /opt/homebrew/Caskroom/miniforge/base/etc/profile.d/conda.sh
conda activate speech_mac

# 进入项目目录
cd "$(dirname "$0")"

# 启动服务
echo "Starting WhisperX service..."
python app.py
