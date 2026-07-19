#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src"
mkdir -p outputs/demo

echo "[1/4] Train demo Stage-1 VAE"
python -u -m fastde.training.train_stage1_vae   --config configs/stage1_vae_pretrain.yaml   --input data/demo/demo_pretrain_sequences.csv   --out outputs/demo/stage1

echo "[2/4] Compute FoldX composite pseudo-label score"
python -u -m fastde.scoring.foldx_composite_score   --input data/demo/demo_foldx_pseudolabels.csv   --config configs/foldx_pseudolabel_score.yaml   --out outputs/demo/demo_foldx_scored.csv

echo "[3/4] Run Stage-2 active-training benchmarks"
python -u -m fastde.training.train_stage2_active   --config configs/stage2_active_training.yaml   --vae-checkpoint outputs/demo/stage1/stage1_vae.pt   --labels outputs/demo/demo_foldx_scored.csv   --out outputs/demo/stage2

echo "[4/4] Summarize active rounds"
python -u -m fastde.benchmark.summarize_active_rounds   --input outputs/demo/stage2/active_round_metrics.csv   --out outputs/demo/stage2/active_round_summary.csv

echo "Demo completed. See outputs/demo/"
