#!/bin/zsh
# Cross-dataset zero-shot: HOT3D-only egit -> DexYCB (hic gorulmemis) test.
# split_hot3d_only.json: train=HOT3D, val=HOT3D P0018, test=TUM DexYCB.
cd "$(dirname "$0")/.."
source .venv/bin/activate
cd src
AURAXR_SPLIT=split_hot3d_only.json python -u train.py --epochs 25 --no_prev --obj_size \
    --lambda_fk 1.0 --arch lstm --tag cross_h2d
echo "--- DexYCB zero-shot kirilim ---"
AURAXR_SPLIT=split_hot3d_only.json python -u eval_metrics.py --tag cross_h2d --split test
