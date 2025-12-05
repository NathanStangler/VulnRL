#!/bin/bash

#SBATCH --job-name=VulnRL
#SBATCH -p msigpu
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=20G
#SBATCH --time=1:00:00
#SBATCH --output=./run_logs/%j.log

ARTIFACT=${ARTIFACT:-""}
MODEL_NAME=${MODEL_NAME:-"Qwen/Qwen2.5-Coder-1.5B-Instruct"}
OUTPUT_DIR=${OUTPUT_DIR:-"./finetuned_model"}
LOG_DIR=${LOG_DIR:-"./logs"}
EPOCHS=${EPOCHS:-1}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-4}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-4}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-8}
LR=${LR:-2e-5}
USE_LORA=${USE_LORA:-"true"}
LOAD_IN_4BIT=${LOAD_IN_4BIT:-"true"}
REPORT_TO=${REPORT_TO:-"wandb"}

module load python3/3.10.9_anaconda2023.03_libmamba cuda/12.0 clang/7.0.1 llvm/7.0.1
cd ~/VulnRL
if [ ! -d .venv ]; then
    python -m venv .venv
fi
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if [ -z "$RUN_MODE" ] || [ "$RUN_MODE" = "finetune" ]; then
  python finetune.py \
    --model_name "$MODEL_NAME" \
    --output_dir "$OUTPUT_DIR" \
    --log_dir "$LOG_DIR" \
    --epochs "$EPOCHS" \
    --train_batch_size "$TRAIN_BATCH_SIZE" \
    --eval_batch_size "$EVAL_BATCH_SIZE" \
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
    --lr "$LR" \
    $( [ "$USE_LORA" = "true" ] && echo --use_lora ) \
    $( [ "$LOAD_IN_4BIT" = "true" ] && echo --load_in_4bit ) \
    --report_to "$REPORT_TO"
fi

if [ "$RUN_MODE" = "rl" ]; then
  python rl_train.py \
    --artifact "$ARTIFACT" \
    --model_name "$MODEL_NAME" \
    --output_dir "$OUTPUT_DIR" \
    --log_dir "$LOG_DIR" \
    --epochs "$EPOCHS" \
    --train_batch_size "$TRAIN_BATCH_SIZE" \
    --max_new_tokens "${MAX_NEW_TOKENS:-32}" \
    $( [ "$USE_LORA" = "true" ] && echo --use_lora ) \
    $( [ "$LOAD_IN_4BIT" = "true" ] && echo --load_in_4bit )
fi

if [ "$RUN_MODE" = "evaluate" ]; then
  python evaluate.py \
    --model_dir "$OUTPUT_DIR" \
    --log_dir "$LOG_DIR" \
    --eval_batch_size "$EVAL_BATCH_SIZE" \
    --report_to "none"
fi

if [ "$RUN_MODE" = "performance" ]; then
  python performance.py \
    --artifact "$ARTIFACT" \
    --model_dir "$OUTPUT_DIR" \
    --log_dir "$LOG_DIR" \
    --max_new_tokens 1024 \
    --chunk_max_tokens 1024 \
    --chunk_overlap 128 \
    $( [ "$LOAD_IN_4BIT" = "true" ] && echo --load_in_4bit )
fi