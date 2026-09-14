# Reproduction guide

The pipeline runs in five stages. Each stage writes CSVs into `results/`; the
committed copies of those CSVs are what the manuscript reports, so you can stop
after any stage and compare.

## 0. Environment

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

or with conda:

```bash
conda env create -f environment.yml
conda activate greenhyperspectra
```

**GPU.** `requirements.txt` installs the default PyTorch wheel. For CUDA:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

Stages 1, 4 and 5 run acceptably on CPU. Stages 2 and 3 want a GPU.

**Project root.** All scripts resolve paths through `src/hsi_paths.py`, which
defaults to the repository checkout. Override with `HSI_ROOT` if data and
results live on a different volume:

```bash
export HSI_ROOT=/scratch/hsi        # Windows: set HSI_ROOT=D:\hsi
```

## 1. Trait-specific wavelength selection

Beam search over the 1,721-band grid, run once per trait.

```bash
for t in cab cw cm LAI cp cbc car anth; do
    python src/beam_band_selector.py --target $t --beam-width 5 --top-pool 220
done
```

Writes `results/beam/<trait>_beam_selected_wavelengths.csv` — the 30 bands for
that trait. **These files are already committed**, so stages 2–5 can be run
without repeating the search.

Then fit the compact heads on the selected bands:

```bash
python src/train_beam_compact_regressors.py --budget 30
python src/beam_supervised_modern_heads.py \
    --models mlp residual dcn polynomial kan nam \
    --seeds 0 1 2 --budget 30 --run-name all8_3seeds
```

→ `table4_compact_model_comparison.csv` (mean R² 0.624 vs 0.538)

Runtime: beam search ≈ 40–90 min per trait on CPU; compact heads ≈ 25 min total.

## 2. Self-supervised pretraining

```bash
python src/ssl_pretrain_mae.py \
    --mask-mode contiguous --block-size 64 --mask-ratio 0.30 \
    --latent-channels 128 --epochs 80 --batch-size 256 \
    --out masked_autoencoder_contiguous_b64.pt
```

→ `results/ssl/masked_autoencoder_contiguous_b64.pt`

Runtime: ≈ 3–5 h on one modern GPU over the 139,295 unlabelled spectra.
The checkpoint is **not** committed (size); every downstream script takes
`--checkpoint`.

## 3. Hybrid fusion and label efficiency

```bash
python src/hybrid_ssl_fusion_ablation.py \
    --checkpoint results/ssl/masked_autoencoder_contiguous_b64.pt \
    --seeds 0 1 2 --budget 30 --run-name fusion_ablation_all8_3seeds

python src/label_efficiency.py --seeds 0 1 2 --budget 30
python src/hybrid_ssl_label_efficiency.py --seeds 0 1 2 --budget 30
```

→ `table37_fusion_ablation_leaderboard.csv` (best hybrid 0.601)
→ `table6_label_efficiency.csv`, `table32_…_mean_r2_by_fraction.csv`

Runtime: ≈ 6–10 h on GPU for the full ablation.

## 4. Leave-source-out transfer

```bash
python src/ood_source_validation.py \
    --checkpoint results/ssl/masked_autoencoder_contiguous_b64.pt \
    --traits cab cw cm LAI cp cbc car \
    --seeds 0 1 2 --budget 30 --max-sources 5 --min-test-labels 30
```

→ `table35_ood_source_validation_model_leaderboard.csv` (mean R² −0.504)

Seven eligible traits × five held-out sources × three seeds = 105 evaluations
per model. `anth` is excluded: its labels come from a single source.

## 5. Few-sample affine recalibration

```bash
python src/few_sample_source_recalibration.py \
    --checkpoint results/ssl/masked_autoencoder_contiguous_b64.pt \
    --fractions 0.0 0.05 0.10 0.20 0.30 0.40 0.50 \
    --seeds 0 1 2 --budget 30 --max-sources 5
```

→ `table43_few_sample_source_recalibration_extended.csv`

Calibration observations are excluded from the evaluation set. Mean R² moves
−0.504 → 0.155 (10 %) → 0.237 (20 %), then plateaus.

## 6. Tables and figures

```bash
python src/generate_paper_artifacts.py
python src/annotate_selected_wavelengths.py
python src/statistical_model_comparison.py
python src/band_stability_analysis.py
```

→ `results/paper_tables/`, `results/paper_figures/`

## Full pipeline

```bash
bash scripts/run_pipeline.sh          # stages 1–6 in order
```

Expect **1–3 days end to end** on a single GPU workstation. Stages are
independent once their inputs exist, so they can be scheduled separately.

## Determinism

All experiments use seeds 0, 1, 2 and report the mean across them. Exact
reproduction of individual values to the last decimal is not guaranteed across
PyTorch versions, CUDA versions and hardware — cuDNN kernel selection and
reduction order vary. Aggregate values and every ranking reported in the
manuscript are stable across the three seeds.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `FileNotFoundError` on a `.parquet` | Data not downloaded — see `docs/DATA.md` |
| `FileNotFoundError` on `masked_autoencoder_*.pt` | Stage 2 not run, or wrong `--checkpoint` path |
| CUDA out of memory | Lower `--batch-size` and `--embedding-batch-size` |
| Results land in the wrong folder | `HSI_ROOT` points somewhere unexpected — `echo $HSI_ROOT` |
| Beam search is very slow | Expected. Use the committed `results/beam/*.csv` and skip stage 1 |
