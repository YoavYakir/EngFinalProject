#!/bin/bash
export PYTHONPATH=$(pwd)

# Parse arguments
while [ "$#" -gt 0 ]; do
    case "$1" in
        --type)
            TEST="$2"
            shift
            ;;
        --test)
            TYPE="$2"
            shift
            ;;
        *)
            echo "Unknown parameter: $1"
            exit 1
            ;;
    esac
    shift
done

# Ensure required arguments are provided
if [[ -z "$TYPE" || -z "$TEST" ]]; then
    echo "Usage: $0 --type <type> --test <test>"
    exit 1
fi

# Construct module path
MODULE_PATH="tests.${TYPE}.run_${TEST}_tests"
JSON_PATH="/results/gtx2080/${TEST}/${TYPE}.json"

# Check if file exists and remove it
if [[ -f "$JSON_PATH" ]]; then
    echo "Removing existing file: $JSON_PATH"
    rm "$JSON_PATH"
fi

# Create a new empty file
touch "./$JSON_PATH"
echo "Created empty file: $JSON_PATH"

if [[ "$TEST" == "cifar" || "$TEST" == "mnist" ]]; then
    if [[ "$TEST" == "mnist" ]]; then
        # Define loop variables
        WORKERS=(1 3 6 10)
        BATCH=(256 512 1024)
        EPOCH=(1 3 6 10)
        LEARNING=(0.001 0.01 0.1)
        MODEL=(small medium huge)
        
        for w in "${WORKERS[@]}"; do
            for b in "${BATCH[@]}"; do
                for e in "${EPOCH[@]}"; do
                    for l in "${LEARNING[@]}"; do
                        for m in "${MODEL[@]}"; do
                            echo "Calling python script with workers=$w, batch=$b, epoch=$e, learning=$l, model=$m"
                            python -m "$MODULE_PATH" "$w" "$b" "$e" "$l" "$m"
                        done
                    done
                done
            done
        done
    else
        # Define loop variables
        WORKERS=(1 3 6 10)
        BATCH=(256 512 1024)
        EPOCH=(1 5 10 15)
        LEARNING=(0.001 0.01 0.1)
        MODEL=(small medium huge)
        
        for w in "${WORKERS[@]}"; do
            for b in "${BATCH[@]}"; do
                for e in "${EPOCH[@]}"; do
                    for l in "${LEARNING[@]}"; do
                        for m in "${MODEL[@]}"; do
                            echo "Calling python script with workers=$w, batch=$b, epoch=$e, learning=$l, model=$m"
                            python -m "$MODULE_PATH" "$w" "$b" "$e" "$l" "$m"
                        done
                    done
                done
            done
        done
    fi
    
else
    echo "Running test without nested loops: $MODULE_PATH"
    python -m "$MODULE_PATH"
fi

# python3 <<EOF
# from utilities.ResourceMonitor import ResourceMonitor
# import os
# full_json_path = os.getcwd() + "${JSON_PATH}"
# ResourceMonitor.fix_json_file(full_json_path)
# EOF
