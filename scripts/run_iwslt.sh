#!/bin/bash
# scripts/run_iwslt.sh
set -e

echo "Starting IWSLT2017 (en→de) Transformer training..."

OUTPUT_DIR="results/run_experiments/run_base"
mkdir -p ${OUTPUT_DIR}

python -m src.train_mt \
    --batch_size 64 \
    --d_model 256 \
    --num_layers 4 \
    --num_heads 8 \
    --d_ff 1024 \
    --epochs 20 \
    --output_dir ${OUTPUT_DIR} \
    --device cuda

echo "Evaluating BLEU..."
python -m src.eval_bleu --ckpt ${OUTPUT_DIR}/ckpt_epoch20.pt --split validation --device cuda
