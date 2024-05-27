#!/bin/sh

set -e

BINARY=$1

GPU_ID=0
WPB=4

echo "timestamp,power" > power.out

# avoid coldstart for ROCm
HIP_VISIBLE_DEVICES=$GPU_ID        $BINARY 1 $WPB> /dev/null
HIP_VISIBLE_DEVICES=$(($GPU_ID+1)) $BINARY 1 $WPB > /dev/null

bin/./power_sampler 100 &
pid_mes=$!

list_n="1 10 20 30 40 50 60 70 80 90 100 110 111 120 130 140 150 160 170 180 190 200 210 220"

sleep 2

for BLOCKS in $list_n; do
    echo blocks=$BLOCKS

    # launch the benchmark
    HIP_VISIBLE_DEVICES=$GPU_ID $BINARY $BLOCKS $WPB | grep kernel &
    pid0=$!
    HIP_VISIBLE_DEVICES=$(($GPU_ID + 1)) $BINARY $BLOCKS $WPB | grep kernel

    wait $pid0 
    sleep 3
done

sleep 1
kill $pid_mes
