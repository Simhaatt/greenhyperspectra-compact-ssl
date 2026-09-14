# CPU-Completed Work and GPU Experiment Plan

Date: 2026-06-11

## CPU-completed additions

The following reviewer-facing additions were completed locally on CPU:

1. **OOD/source metadata audit**
   - The labelled parquet files contain a `dataset` source/domain column.
   - This means leave-source-out or cross-source validation is feasible in principle.
   - Output tables:
     - `results/paper_tables/table28_dataset_source_metadata_audit.csv`
     - `results/paper_tables/table29_ood_leave_source_feasibility.csv`

2. **Bootstrap confidence intervals and paired trait-level tests**
   - Bootstrap 95% confidence intervals were computed over trait-level mean R2 values.
   - Paired trait-level comparisons were computed for key model pairs.
   - Output tables:
     - `results/paper_tables/table26_model_mean_r2_bootstrap_ci.csv`
     - `results/paper_tables/table27_paired_trait_significance_tests.csv`

3. **Trait-wise wavelength interpretation**
   - BeamSearch K=30 wavelengths were mapped to visible, red-edge, NIR, water, and SWIR biochemical regions.
   - Output files:
     - `results/paper_tables/table30_traitwise_selected_wavelength_regions_long.csv`
     - `results/paper_tables/table31_traitwise_wavelength_interpretation_summary.csv`
     - `results/traitwise_wavelength_interpretation.md`

4. **Hybrid SSL label-efficiency Kaggle bundle**
   - A new runner and notebook were created for the most important missing journal experiment.
   - Smoke-tested locally with 1 trait, 1 fraction, 1 seed, 2 epochs.
   - Upload bundle:
     - `kaggle/upload_bundles/kaggle_hybrid_ssl_label_efficiency_upload_bundle.zip`
   - Notebook inside bundle:
     - `kaggle_hybrid_ssl_label_efficiency_run_all.ipynb`

## GPU-required or GPU-recommended experiments

Use Kaggle T4 for the following. If GPU is even slightly better, use GPU.

### 1. Hybrid SSL label-efficiency - highest priority

Purpose:
Show whether unlabelled SSL features improve label efficiency under scarce labels.

Run:

```text
kaggle/upload_bundles/kaggle_hybrid_ssl_label_efficiency_upload_bundle.zip
```

Notebook:

```text
kaggle_hybrid_ssl_label_efficiency_run_all.ipynb
```

Compares:

```text
BeamK30 ResidualMLP
Hybrid Cross-Gated NAM
Hybrid Cross-Gated DCN
Hybrid Cross-Gated MLP
```

Fractions:

```text
1%, 5%, 10%, 25%, 50%, 100%
```

Seeds:

```text
0, 1, 2
```

Expected output:

```text
greenhyperspectra_hybrid_ssl_label_efficiency_outputs.zip
```

Estimated time on T4:

```text
3-6 hours
```

This is the single most important remaining experiment.

### 2. Leave-source-out / OOD validation

Use the `dataset` column as the source/domain identifier.

Recommended design:

1. Select sources with enough labels per target.
2. For each target, hold out one source or a group of sources.
3. Train on remaining sources.
4. Test on the held-out source.
5. Compare:
   - BeamK30 ResidualMLP,
   - BeamK30 MLP/Polynomial where they are trait winners,
   - Hybrid Cross-Gated NAM,
   - Hybrid Cross-Gated DCN,
   - Cross-Gated MLP.

GPU recommendation:
Use T4. The repeated neural retraining across sources/traits can become slow on CPU.

Estimated time:

```text
4-10 hours depending on number of held-out sources
```

### 3. Parameter-matched fusion ablation

Purpose:
Prove that cross-gating matters, not just parameter count.

Compare:

```text
selected bands only
SSL embedding only
naive concatenation
residual summation
FiLM conditioning
standard cross-attention
cross-gated fusion
```

Recommended protocol:

Use the same hidden size, dropout, optimizer, seeds, and train/test split across all variants.

GPU recommendation:
Use T4 for full all-trait, 3-seed runs.

### 4. Compact deployment / distillation

Purpose:
Close the gap between compact sensing and hybrid SSL, because the current hybrid model still needs full-spectrum input for its 128-d SSL embedding.

Recommended experiments:

1. **Teacher-student trait distillation**
   - Teacher: best hybrid SSL model.
   - Student: 30 BeamSearch bands only.
   - Student learns true labels plus teacher predictions.

2. **Embedding distillation**
   - Train a 30-band student to predict the 128-d SSL embedding.
   - Then feed predicted embedding plus 30 bands to the hybrid head.

GPU recommendation:
Use T4. CPU pilot is possible, but full all-trait distillation should run on GPU.

### 5. Prior baseline comparison

Purpose:
Compare against HyspectraSSL-style baselines such as MAE, GAN, RTM-AE, and supervised multitask baselines.

Recommendation:
Do this only after the label-efficiency result. It is setup-heavy and may require adapting external code.

GPU recommendation:
Use T4 or better.

## Current practical order

1. Run **hybrid SSL label-efficiency** on Kaggle T4.
2. If label-efficiency shows SSL helps at low labels, update paper around that result.
3. Run OOD/source validation using `dataset` if time allows.
4. Add fusion ablation if reviewers/professor need stronger architecture evidence.
5. Do distillation only if compact deployment becomes the main paper angle.
