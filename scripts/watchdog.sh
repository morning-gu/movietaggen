#!/bin/bash
# 04 生成看门狗：崩溃自动重启（断点续跑零损失）+ 卡死检测（jsonl 长时间无增长
# 即杀掉重启——SDK 默认 600s 超时下挂起调用曾实测卡死 10+ 分钟）。
# 04 正常退出（码 0 = 任务矩阵全部完成）后自动停止。
# 服务器用法（项目根目录）：
#   source server-env.sh && nohup bash scripts/watchdog.sh configs/pilot.yaml \
#     results/pilot/stage-gen/responses.jsonl 900 >> /tmp/pilot.log 2>&1 < /dev/null &
set -u
CONFIG="${1:?用法: watchdog.sh <config路径> <jsonl路径> [卡死阈值秒=900]}"
JSONL="${2:?用法: watchdog.sh <config路径> <jsonl路径> [卡死阈值秒=900]}"
STALL="${3:-900}"
MAX_RESTARTS=30
cd "$(dirname "$0")/.."
# 服务器部署用 conda env：server-env.sh export MTG_PYTHON=<conda>/bin/python；
# 未设置时回退本地 .venv（开发机路径）
PY="${MTG_PYTHON:-.venv/bin/python}"

restarts=0
while true; do
  $PY scripts/04_run_generation.py --config "$CONFIG" &
  PID=$!
  echo "[watchdog] $(date '+%F %T') 启动 pid=$PID（第 $((restarts + 1)) 次）"
  killed=0
  last_lines=-1
  last_growth=$(date +%s)
  while kill -0 "$PID" 2>/dev/null; do
    sleep 60
    n=$(wc -l < "$JSONL" 2>/dev/null || echo 0)
    now=$(date +%s)
    if [ "$n" -gt "$last_lines" ]; then
      last_lines=$n
      last_growth=$now
    fi
    if [ $((now - last_growth)) -gt "$STALL" ]; then
      echo "[watchdog] $(date '+%F %T') 卡死：${STALL}s 无 jsonl 增长（当前 $n 行），重启"
      killed=1
      break
    fi
  done
  if [ "$killed" -eq 1 ]; then
    kill "$PID" 2>/dev/null
    sleep 10
    kill -9 "$PID" 2>/dev/null
    wait "$PID" 2>/dev/null
  else
    if wait "$PID"; then
      echo "[watchdog] $(date '+%F %T') 04 正常退出——全部任务完成"
      exit 0
    fi
    echo "[watchdog] $(date '+%F %T') 04 异常退出（码 $?），重启续跑"
  fi
  restarts=$((restarts + 1))
  if [ "$restarts" -ge "$MAX_RESTARTS" ]; then
    echo "[watchdog] 重启次数达上限 $MAX_RESTARTS，放弃（jsonl 保留，可手动续跑）"
    exit 1
  fi
  sleep 30
done
