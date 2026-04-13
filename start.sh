#!/bin/bash
# 量化交易系统 启动脚本
# Usage: ./start.sh [--port PORT] [--host HOST]

PORT=${1:-8787}
HOST="0.0.0.0"

echo "============================================"
echo "  量化交易系统 启动中..."
echo "  地址: http://${HOST}:${PORT}"
echo "  按 Ctrl+C 停止"
echo "============================================"

cd "$(dirname "$0")"
python3 web/app.py
