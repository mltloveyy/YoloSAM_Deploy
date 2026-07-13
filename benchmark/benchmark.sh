#!/bin/bash

CONFIG_FILE="config.yaml"
MODEL_DIR="models"
IMAGE_PATH="test.jpg"
EXEC_DIR="."
LOOP="5"
WARMUP="0"

export LD_LIBRARY_PATH=${EXEC_DIR}:$LD_LIBRARY_PATH

EXEC="DETECT SEGMENT"
MODEL_FILE="yolo26x_best sam2.1_b sam2.1_s sam2.1_t"
QUANTIZE="fp32 fp16 int8"
FORWARD_TYPE="0"  # forward_type(0:CPU, 3:OPENCL, 5:NN, 7:VULKAN)
PRECISION="0 2"  # precision_mode(0:NORMAL, 2:LOW)
NUM_THREADS="1 2 4"  # CPU:1 2 4, OPENCL:1 4 64 128 68 132 576 580, VULKAN:1 4 516

show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo "Options:"
    echo "  -e, --exec       executable"
    echo "  -m, --model      model type"
    echo "  -q, --quantize   quantize type"
    echo "  -b, --backend    bachend (0:CPU, 3:OPENCL, 5:NN, 7:VULKAN)"
    echo "  -p, --precision  precision mode (0:NORMAL, 2:LOW)"
    echo "  -n, --numthread  thread number"
    echo "  --monitor        enable memory monitor"
}

while [[ $# -gt 0 ]]; do
    case $1 in
        -e|--exec)
            EXEC="$2"
            shift 2
            ;;
        -m|--model)
            MODEL_FILE="$2"
            shift 2
            ;;
        -q|--quantize)
            QUANTIZE="$2"
            shift 2
            ;;
        -b|--backend)
            FORWARD_TYPE="$2"
            shift 2
            ;;
        -p|--precision)
            PRECISION="$2"
            shift 2
            ;;
        -n|--numthread)
            NUM_THREADS="$2"
            shift 2
            ;;
        --monitor)
            MONITOR=true
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# memery monitor
MEM_MONITOR_PID=""
mem_monitor_start() {
    TARGET_PID="$1"
    OUTPUT_CSV="$2"
    echo "timestamp_ms,rss_kb,pss_kb" > "$OUTPUT_CSV"
    (
        trap 'exit 0' TERM INT HUP   
        while true; do
            kill -0 "$TARGET_PID" 2>/dev/null || exit 0
            
            if [ -r "/proc/$TARGET_PID/status" ]; then
                TS=$(date +%s%3N 2>/dev/null || date +%s)
                RSS=$(awk '/^VmRSS:/{print $2}' "/proc/$TARGET_PID/status" 2>/dev/null || echo 0)
                PSS=$(awk '/^Pss:/{print $2}' "/proc/$TARGET_PID/smaps_rollup" 2>/dev/null || echo 0)
                echo "${TS},${RSS},${PSS}" >> "$OUTPUT_CSV"
            fi
            sleep 1
        done
    ) &
    MEM_MONITOR_PID=$!
}

mem_monitor_stop() {
    if [ -n "$MEM_MONITOR_PID" ]; then
        kill "$MEM_MONITOR_PID" 2>/dev/null || true
        MEM_MONITOR_PID=""
        sleep 1
    fi
}

mkdir -p benchmark

for exe in $EXEC; do
    for model in $MODEL_FILE; do
        case "$exe:$model" in
            DETECT:*sam*|SEGMENT:*yolo*) continue ;;
        esac
        for qat in $QUANTIZE; do
            for type in $FORWARD_TYPE; do
                for prc in $PRECISION; do
                    for num in $NUM_THREADS; do
                        PREFIX="${exe}_${model}_${qat}_t${type}_p${prc}_n${num}"
                        LOG_FILE="benchmark/${PREFIX}.log"
                        MEM_LOG="benchmark/${PREFIX}_mem.csv"
                        MODEL_STEM="${MODEL_DIR}/${qat}/${model}"

                        if [[ "$exe" == "DETECT" ]]; then
                            CMD="${EXEC_DIR}/$exe ${MODEL_STEM}.mnn $IMAGE_PATH $CONFIG_FILE 0.2 $LOOP $WARMUP $type $num $prc"
                        elif [[ "$exe" == "SEGMENT" ]]; then
                            CMD="${EXEC_DIR}/$exe ${MODEL_STEM}_enc.mnn ${MODEL_STEM}_dec.mnn $IMAGE_PATH 500,720,560,1070 1,1 $LOOP $WARMUP $type $num $prc 0"
                        fi
                        
                        echo "Command: $CMD"
                        echo "Log: $LOG_FILE"
                        
                        if [ "$MONITOR" = true ]; then
                            MEM_LOG="benchmark/${PREFIX}_mem.csv"
                            echo "MemLog: $MEM_LOG"
                            
                            $CMD > "$LOG_FILE" 2>&1 &
                            TARGET_PID=$!
                            
                            mem_monitor_start "$TARGET_PID" "$MEM_LOG"
                            wait "$TARGET_PID" 2>/dev/null
                            EXIT_CODE=$?
                            mem_monitor_stop
                        else
                            $CMD > "$LOG_FILE" 2>&1
                            EXIT_CODE=$?
                        fi
                        
                        if [ $EXIT_CODE -ne 0 ]; then
                            echo "⚠️  Exited with code $EXIT_CODE" | tee -a "$LOG_FILE"
                        fi
                        
                        echo "Finished. Cooling down 5s..."
                        sleep 5
                    done
                done
            done
        done
    done
done
