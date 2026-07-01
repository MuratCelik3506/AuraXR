#!/bin/bash
# Overnight training + eval pipeline — checkpoint recovery destekli.
#
# Her adım tamamlandığında .done dosyası bırakır.
# Yeniden çalıştırılırsa tamamlanmış adımları atlar.
#
# Kullanım:
#   bash run_overnight.sh          # baştan veya kaldığı yerden devam
#   bash run_overnight.sh --reset  # .done dosyalarını sil, baştan başla
#
# Log: logs/overnight_YYYYMMDD_HHMMSS.log

set -euo pipefail

REPO="$(cd "$(dirname "$0")" && pwd)"
PYTHONPATH="$REPO/src"
export PYTHONPATH

LOGS="$REPO/logs"
mkdir -p "$LOGS"
TS=$(date +%Y%m%d_%H%M%S)
LOGFILE="$LOGS/overnight_${TS}.log"
DONE_DIR="$REPO/.pipeline_done"
mkdir -p "$DONE_DIR"

# --reset flag
if [[ "${1:-}" == "--reset" ]]; then
    echo "Reset: .done dosyaları siliniyor..."
    rm -rf "$DONE_DIR"
    mkdir -p "$DONE_DIR"
fi

# Tee: hem terminale hem log dosyasına yaz
exec > >(tee -a "$LOGFILE") 2>&1

echo "========================================"
echo "AuraXR Overnight Pipeline — $TS"
echo "Log: $LOGFILE"
echo "========================================"

# ── Yardımcı fonksiyon ───────────────────────────────────────────────────────
step() {
    local name="$1"; shift
    local done_file="$DONE_DIR/${name}.done"
    if [[ -f "$done_file" ]]; then
        echo ""
        echo "── [SKIP] $name (zaten tamamlandı) ──"
        return 0
    fi
    echo ""
    echo "══════════════════════════════════════"
    echo "  BAŞLIYOR: $name"
    echo "  $(date '+%Y-%m-%d %H:%M:%S')"
    echo "══════════════════════════════════════"
    "$@"
    local status=$?
    if [[ $status -ne 0 ]]; then
        echo ""
        echo "HATA: $name başarısız (exit=$status)"
        echo "Pipeline durdu. 'bash run_overnight.sh' ile kaldığı yerden devam edilebilir."
        exit $status
    fi
    touch "$done_file"
    echo "── [TAMAM] $name — $(date '+%H:%M:%S') ──"
}

PY() { python3 "$@"; }

# ── ADIM 1: Phase 1 — OakInk static pre-training ────────────────────────────
step "phase1_train" PY src/training/train_grasp.py \
    --phase 1 \
    --epochs 50 \
    --batch 64 \
    --lr 3e-4 \
    --kl_warmup_epochs 10 \
    --kl_weight 0.01 \
    --limit_weight 1.0 \
    --contact_weight 0.3 \
    --penetration_weight 0.1 \
    --quality_weight 0.1 \
    --tip_weight 0.5 \
    --lr_patience 5 \
    --early_stopping 15 \
    --checkpoint-prefix "aura" \
    --workers 0

# ── ADIM 2: Phase 1 Eval ─────────────────────────────────────────────────────
step "phase1_eval_val" PY src/evaluation/evaluate.py \
    --checkpoint checkpoints/aura_phase1_best.pt \
    --source oakink \
    --split val \
    --phase 1 \
    --batch 128 \
    --k 1

step "phase1_eval_test" PY src/evaluation/evaluate.py \
    --checkpoint checkpoints/aura_phase1_best.pt \
    --source oakink \
    --split test \
    --phase 1 \
    --batch 128 \
    --k 1

# ── ADIM 3: Phase 2 — HOT3D temporal + OakInk replay ────────────────────────
step "phase2_train" PY src/training/train_grasp.py \
    --phase 2 \
    --epochs 30 \
    --batch 64 \
    --lr 1e-4 \
    --kl_warmup_epochs 5 \
    --kl_weight 0.01 \
    --limit_weight 1.0 \
    --contact_weight 0.3 \
    --penetration_weight 0.1 \
    --quality_weight 0.1 \
    --vel_weight 0.1 \
    --acc_weight 0.05 \
    --tip_weight 0.5 \
    --oakink_replay_ratio 0.3 \
    --lr_patience 5 \
    --early_stopping 10 \
    --checkpoint-prefix "aura" \
    --checkpoint checkpoints/aura_phase1_best.pt \
    --workers 0

# ── ADIM 4: Phase 2 Eval — HOT3D val + test ──────────────────────────────────
step "phase2_eval_hot3d_val" PY src/evaluation/evaluate.py \
    --checkpoint checkpoints/aura_phase2_best.pt \
    --source hot3d \
    --split val \
    --phase 2 \
    --batch 64 \
    --k 1

step "phase2_eval_hot3d_test" PY src/evaluation/evaluate.py \
    --checkpoint checkpoints/aura_phase2_best.pt \
    --source hot3d \
    --split test \
    --phase 2 \
    --batch 64 \
    --k 1

step "phase2_eval_oakink_val" PY src/evaluation/evaluate.py \
    --checkpoint checkpoints/aura_phase2_best.pt \
    --source oakink \
    --split val \
    --phase 2 \
    --batch 128 \
    --k 1

# ── ADIM 5: K=3 ve K=5 CVAE diversity eval (phase2 best) ────────────────────
step "phase2_eval_k3" PY src/evaluation/evaluate.py \
    --checkpoint checkpoints/aura_phase2_best.pt \
    --source hot3d \
    --split val \
    --phase 2 \
    --batch 64 \
    --k 3

step "phase2_eval_k5" PY src/evaluation/evaluate.py \
    --checkpoint checkpoints/aura_phase2_best.pt \
    --source hot3d \
    --split val \
    --phase 2 \
    --batch 64 \
    --k 5

# ── BİTTİ ────────────────────────────────────────────────────────────────────
echo ""
echo "========================================"
echo "Pipeline tamamlandı — $(date '+%Y-%m-%d %H:%M:%S')"
echo "Sonuçlar: results/ dizininde"
echo "Log: $LOGFILE"
echo "========================================"
