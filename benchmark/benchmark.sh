#!/bin/bash

CONFIG_FILE="data/config.yaml"
MODEL_DIR="models"

IMAGE_DIR="data/test"
IMAGE_PATH="data/test.jpg"

EXEC_DIR="sdk/bin/x86"
EXEC=("${EXEC_DIR}/DETECT" "${EXEC_DIR}/SEGMENT")
export LD_LIBRARY_PATH=${EXEC_DIR}:$LD_LIBRARY_PATH

MODEL_FILE="yolo26x_best sam2.1_b sam2.1_s sam2.1_t"

QUANTIZE="fp32 fp16 int8"

FORWARD_TYPE="0 3 4 5 7"  # forward_type(0:CPU, 3:OPENCL, 4:AUTO, 5:NN, 7:VULKAN)

PRECISION="0 2"  # precision_mode(0:NORMAL, 2:LOW)

NUM_THREADS="4 2 1"

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
    case "$exe" in
        *DETECT*)  TASK="DET" ;;
        *SEGMENT*) TASK="SEG" ;;
        *)         echo "⚠️ Unknown exe: $exe"; continue ;;
    esac

    for model in $MODEL_FILE; do
        case "$TASK:$model" in
            DET:*sam*|SEG:*yolo*) continue ;;
        esac

        case "$model" in
            *yolo*) MODEL="yolo" ;;
            *)      MODEL="${model}" ;;
        esac

        model="${MODEL_DIR}/${model}"

        for qat in $QUANTIZE; do
            for type in $FORWARD_TYPE; do
                for prc in $PRECISION; do
                    for num in $NUM_THREADS; do
                        PREFIX="${TASK}_${MODEL}_${qat}_t${type}_p${prc}_n${num}"
                        LOG_FILE="benchmark/${PREFIX}.log"
                        MEM_LOG="benchmark/${PREFIX}_mem.csv"

                        if [[ "$TASK" == "DET" ]]; then
                            CMD="$exe ${model}_${qat}.mnn $IMAGE_DIR $CONFIG_FILE 0.2 $type $prc $num"
                        elif [[ "$TASK" == "SEG" ]]; then
                            CMD="$exe ${model}_enc_${qat}.mnn ${model}_dec_${qat}.mnn $IMAGE_PATH 500,720,560,1070 1,1 $type $prc $num"
                        fi
                        
                        echo "Command: $CMD"
                        echo "Log: $LOG_FILE | MemLog: $MEM_LOG" 
                        $CMD > "$LOG_FILE" 2>&1 &
                        TARGET_PID=$!

                        mem_monitor_start "$TARGET_PID" "$MEM_LOG"
                        wait "$TARGET_PID" 2>/dev/null || true
                        EXIT_CODE=$?
                        mem_monitor_stop
                        
                        if [ $EXIT_CODE -ne 0 ]; then
                            echo "⚠️ Exited with code $EXIT_CODE" | tee -a "$LOG_FILE"
                        fi
                        echo "Finished. Waiting 5s..."
                        sleep 5
                    done
                done
            done
        done
    done
done
