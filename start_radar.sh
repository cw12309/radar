#!/usr/bin/env bash
# RADAR 系统看门狗：保证 Flask 主服务与 OHIF 阅片器常驻。
#
# 设计目标：首次启动后两个服务一直活着，崩溃/被杀后 5 秒内自动拉起，
# 不再依赖人工 nohup。OHIF 若已在跑（如之前手工启动）则直接复用不重启。
#
# 用法：
#   ./start_radar.sh          # 前台看门狗（Ctrl+C 退出，服务随之保留）
#   nohup ./start_radar.sh >> radar_watchdog.log 2>&1 &   # 后台常驻
set -u

RADAR_DIR="$(cd "$(dirname "$0")" && pwd)"
PY=/home/rimagai/anaconda3/envs/radar/bin/python
FLASK_LOG="$RADAR_DIR/flask.log"
OHIF_LOG="$RADAR_DIR/logs/ohif.log"
mkdir -p "$RADAR_DIR/logs"

port_busy() { ss -tln 2>/dev/null | awk '{print $4}' | grep -qE "[:.]$1\$"; }

start_flask() {
    cd "$RADAR_DIR/flask-app" || exit 1
    nohup "$PY" app.py >> "$FLASK_LOG" 2>&1 &
    echo "[watchdog] $(date '+%F %T') 拉起 Flask (pid $!)"
}

start_ohif() {
    cd "$RADAR_DIR/ohif-viewer/platform/app" || exit 1
    nohup yarn run dev >> "$OHIF_LOG" 2>&1 &
    echo "[watchdog] $(date '+%F %T') 拉起 OHIF (pid $!)"
}

# 首次启动：缺谁补谁，已存在的不动（避免打断正在用的阅片器）
port_busy 5000 || start_flask
port_busy 3002 || start_ohif

while true; do
    sleep 5
    port_busy 5000 || start_flask
    port_busy 3002 || start_ohif
done
