#!/bin/bash
# ece115 hardware monitor v4 — shared-server aware.
# Facts: GPU PIDs visible in OUR namespace = ours; invisible = other
# containers (e.g. other lab users). Alerts: zombie signature, GPU BECOMES
# FREE (so queued Ada experiments can claim it), very-high disk, hourly HB.
CSV=/root/invdx/runs/hwmon.csv
z0=0; z1=0; a0=0; a1=0; e0=0; e1=0; hb=0; adisk=0
f0=0; f1=0; free0=0; free1=0
now() { date +%s; }
classify() {  # pids "p1:p2:" -> "ours:N ext:M"
  local ours=0 ext=0 p
  for p in $(echo "$1" | tr ':' ' '); do
    [ -d "/proc/$p" ] && ours=$((ours+1)) || ext=$((ext+1))
  done
  echo "ours:$ours ext:$ext"
}
while true; do
  ts=$(now)
  g=$(nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
  g0=$(echo "$g" | sed -n 1p); g1=$(echo "$g" | sed -n 2p)
  u0=${g0%%,*}; r0=${g0#*,}; m0=${r0%%,*}; t0=${r0##*,}
  u1=${g1%%,*}; r1=${g1#*,}; m1=${r1%%,*}; t1=${r1##*,}
  load=$(cut -d' ' -f1 /proc/loadavg)
  ram=$(free | awk '/Mem:/{printf "%.0f", $3/$2*100}')
  disk=$(df --output=pcent / | tail -1 | tr -d ' %')
  procs=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sort | tr '\n' ':')
  echo "$ts,$u0,$m0,$t0,$u1,$m1,$t1,$load,$ram,$disk,$procs" >> "$CSV"

  # zombie: big memory + sustained idle
  if [ "${m0:-0}" -ge 8192 ] 2>/dev/null && [ "${u0:-100}" -le 5 ] 2>/dev/null; then
    z0=$((z0+1)); else z0=0; [ $a0 = 1 ] && echo "[HW-OK] GPU0 zombie signature cleared"; a0=0; fi
  if [ "${m1:-0}" -ge 8192 ] 2>/dev/null && [ "${u1:-100}" -le 5 ] 2>/dev/null; then
    z1=$((z1+1)); else z1=0; [ $a1 = 1 ] && echo "[HW-OK] GPU1 zombie signature cleared"; a1=0; fi
  t=$(now)
  if [ $z0 -ge 30 ] && { [ $a0 = 0 ] || [ $((t-e0)) -ge 1800 ]; }; then
    echo "[HW-ZOMBIE] GPU0: ${m0}MiB held, util ${u0}%, >=5min idle ($(classify "$procs"))"
    a0=1; e0=$t
  fi
  if [ $z1 -ge 30 ] && { [ $a1 = 0 ] || [ $((t-e1)) -ge 1800 ]; }; then
    echo "[HW-ZOMBIE] GPU1: ${m1}MiB held, util ${u1}%, >=5min idle ($(classify "$procs"))"
    a1=1; e1=$t
  fi

  # GPU became free: <5% util AND <1GiB for 12 samples (2 min)
  if [ "${u0:-100}" -le 5 ] 2>/dev/null && [ "${m0:-9999}" -le 1024 ] 2>/dev/null; then
    f0=$((f0+1)); else f0=0; free0=0; fi
  if [ $f0 -ge 12 ] && [ $free0 = 0 ]; then
    echo "[HW-FREE] GPU0 空閒 2 分鐘(外部工作結束?)— 排隊的 Ada 實驗可以上"
    free0=1
  fi
  if [ "${u1:-100}" -le 5 ] 2>/dev/null && [ "${m1:-9999}" -le 1024 ] 2>/dev/null; then
    f1=$((f1+1)); else f1=0; free1=0; fi
  if [ $f1 -ge 12 ] && [ $free1 = 0 ]; then
    echo "[HW-FREE] GPU1 空閒 2 分鐘 — 排隊的 Ada 實驗可以上"
    free1=1
  fi

  if [ "${disk:-0}" -ge 95 ] && [ $adisk = 0 ]; then
    echo "[HW-ALERT] DISK ${disk}%"; adisk=1
  elif [ "${disk:-0}" -lt 93 ]; then adisk=0; fi

  hb=$((hb+1))
  if [ $hb -ge 360 ]; then
    echo "[HW-HB] gpu0 ${u0}%/${m0}MiB($(classify "$procs")) gpu1 ${u1}%/${m1}MiB | load $load ram ${ram}% disk ${disk}%"
    hb=0
  fi
  sleep 10
done
