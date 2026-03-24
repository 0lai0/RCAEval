#!/bin/bash
# =============================================================
# retrain_all.sh — 重新訓練所有 graph_fastshap 預訓練模型
# 使用修復後的 pretrain.py（含 gradient clipping + LR scheduler）
# =============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs/pretrain"
mkdir -p "$LOG_DIR"

# 啟用虛擬環境
source "${SCRIPT_DIR}/venv310/bin/activate"

# 所有需要訓練的 dataset（對應 cache/datasets/ 中的快取檔案）
DATASETS=(
    "re1-ob"
    "re1-ss"
    "re1-tt"
    "re2-ob"
    "re2-ss"
    "re2-tt"
)

SURR_EPOCHS=100
EXPL_EPOCHS=150
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "============================================================"
echo "🔄 Graph-FastSHAP 批次重新訓練"
echo "   時間: $(date)"
echo "   Surrogate Epochs: ${SURR_EPOCHS}"
echo "   Explainer Epochs: ${EXPL_EPOCHS}"
echo "   Log 目錄: ${LOG_DIR}"
echo "============================================================"
echo ""

TOTAL=${#DATASETS[@]}
CURRENT=0
FAILED=()

for DATASET in "${DATASETS[@]}"; do
    CURRENT=$((CURRENT + 1))
    LOG_FILE="${LOG_DIR}/${DATASET}_${TIMESTAMP}.log"
    
    echo "──────────────────────────────────────────────────────────"
    echo "  [${CURRENT}/${TOTAL}] 訓練 ${DATASET}"
    echo "  Log: ${LOG_FILE}"
    echo "──────────────────────────────────────────────────────────"
    
    START_TIME=$(date +%s)
    
    if python -u "${SCRIPT_DIR}/pretrain.py" \
        --dataset "$DATASET" \
        --surr-epochs "$SURR_EPOCHS" \
        --expl-epochs "$EXPL_EPOCHS" \
        2>&1 | tee "$LOG_FILE"; then
        
        END_TIME=$(date +%s)
        DURATION=$((END_TIME - START_TIME))
        echo "  ✅ ${DATASET} 完成 (耗時 ${DURATION}s)"
    else
        END_TIME=$(date +%s)
        DURATION=$((END_TIME - START_TIME))
        echo "  ❌ ${DATASET} 失敗 (耗時 ${DURATION}s)"
        FAILED+=("$DATASET")
    fi
    echo ""
done

echo "============================================================"
echo "🏁 全部訓練結束 — $(date)"
echo "   成功: $((TOTAL - ${#FAILED[@]}))/${TOTAL}"
if [ ${#FAILED[@]} -gt 0 ]; then
    echo "   ❌ 失敗: ${FAILED[*]}"
fi
echo "   Logs 存放在: ${LOG_DIR}"
echo "============================================================"
