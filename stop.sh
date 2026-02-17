#!/bin/bash

# WhisperX 服务停止脚本

echo "Stopping WhisperX service..."

# 查找并终止 python app.py 进程
pid=$(ps aux | grep "python app.py" | grep -v grep | awk '{print $2}')

if [ -n "$pid" ]; then
    kill $pid
    echo "Service stopped (PID: $pid)"
else
    echo "No running service found"
fi
