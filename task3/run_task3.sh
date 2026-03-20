#!/bin/bash

# Check if at least 3 arguments are provided
if [ "$#" -lt 3 ]; then
    echo "Usage: $0 <master_ip> <master_port> <rank> [task_name] [glue_dir]"
    exit 1
fi

# Assign arguments for clarity
MASTER_IP=$1
MASTER_PORT=$2
RANK=$3
export GLUE_DIR=$HOME/cos568_pa2/glue_data
export TASK_NAME=RTE
export GLOO_SOCKET_IFNAME=enp1s0f1np1

# Execute the script passing the new arguments as flags
python3 task3.py \
  --master_ip "$MASTER_IP" \
  --master_port "$MASTER_PORT" \
  --local_rank "$RANK" \
  --model_type bert \
  --model_name_or_path bert-base-cased \
  --task_name "$TASK_NAME" \
  --do_train \
  --do_eval \
  --data_dir "$GLUE_DIR/$TASK_NAME" \
  --max_seq_length 128 \
  --per_device_train_batch_size 64 \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --output_dir "/tmp/$TASK_NAME/" \
  --overwrite_output_dir