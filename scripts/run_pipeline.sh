#!/usr/bin/env bash
# End-to-end pipeline. Expect 1-3 days on a single GPU workstation.
# Stages are independent once their inputs exist; comment out what you
# already have. See docs/REPRODUCE.md for runtimes and hardware notes.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export HSI_ROOT="${HSI_ROOT:-$ROOT}"
CKPT="$HSI_ROOT/results/ssl/masked_autoencoder_contiguous_b64.pt"
TRAITS_ALL="cab cw cm LAI cp cbc car anth"
TRAITS_OOD="cab cw cm LAI cp cbc car"
SEEDS="0 1 2"

cd "$ROOT"
echo "HSI_ROOT = $HSI_ROOT"

echo "== 0/6  data check =="
python src/inspect_dataset.py

echo "== 1/6  trait-specific wavelength selection =="
# Already committed under results/beam/ - uncomment to regenerate (slow).
# for t in $TRAITS_ALL; do
#     python src/beam_band_selector.py --target "$t" --beam-width 5 --top-pool 220
# done
python src/train_beam_compact_regressors.py --budget 30
python src/beam_supervised_modern_heads.py \
    --models mlp residual dcn polynomial kan nam \
    --seeds $SEEDS --budget 30 --run-name all8_3seeds

echo "== 2/6  masked self-supervised pretraining =="
if [ ! -f "$CKPT" ]; then
    python src/ssl_pretrain_mae.py \
        --mask-mode contiguous --block-size 64 --mask-ratio 0.30 \
        --latent-channels 128 --epochs 80 --batch-size 256 \
        --out masked_autoencoder_contiguous_b64.pt
else
    echo "   checkpoint exists, skipping"
fi

echo "== 3/6  hybrid fusion and label efficiency =="
python src/hybrid_ssl_fusion_ablation.py --checkpoint "$CKPT" \
    --seeds $SEEDS --budget 30 --run-name fusion_ablation_all8_3seeds
python src/label_efficiency.py --seeds $SEEDS --budget 30
python src/hybrid_ssl_label_efficiency.py --seeds $SEEDS --budget 30

echo "== 4/6  leave-source-out transfer =="
python src/ood_source_validation.py --checkpoint "$CKPT" \
    --traits $TRAITS_OOD --seeds $SEEDS --budget 30 \
    --max-sources 5 --min-test-labels 30

echo "== 5/6  few-sample affine recalibration =="
python src/few_sample_source_recalibration.py --checkpoint "$CKPT" \
    --fractions 0.0 0.05 0.10 0.20 0.30 0.40 0.50 \
    --seeds $SEEDS --budget 30 --max-sources 5

echo "== 6/6  tables and figures =="
python src/generate_paper_artifacts.py
python src/annotate_selected_wavelengths.py
python src/statistical_model_comparison.py
python src/band_stability_analysis.py

echo
echo "Done. Compare results/paper_tables/ against the committed copies."
