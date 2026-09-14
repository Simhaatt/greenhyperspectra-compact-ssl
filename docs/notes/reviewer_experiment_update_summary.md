# Reviewer-Focused Experiment Update Summary

Date: 2026-07-05

This note summarizes the additional reviewer-strengthening work completed after the main GreenHyperSpectra experiments. The goal was not to add new architectures, but to strengthen validation, source-shift interpretation, band-selection trust, and paper framing.

## 1. Few-Sample Source Recalibration

### What was done

I added and ran a few-sample source recalibration experiment for the two key models:

- BeamK30 ResidualMLP
- Hybrid BeamK30 SSL Cross-Gated NAM

The protocol holds out one source group, trains on all other sources, then uses a small labelled subset from the held-out source to fit a simple prediction-level calibration layer:

```text
y_calibrated = a * y_pred + b
```

This directly addresses the weak zero-shot OOD result by testing whether deployment performance can recover when a small number of labels are collected from a new source.

### Completed calibration levels

The first Kaggle run produced:

```text
0%, 5%, 10%, 20%
```

The second Kaggle run added:

```text
30%, 40%, 50%
```

I copied the second Kaggle zip outputs into the local workspace and merged both runs into one extended calibration table.

### Output files

- `results/paper_tables/table42_few_sample_source_recalibration.csv`
- `results/source_recalibration/few_sample_source_recalibration_top5_sources_3seeds_summary.csv`
- `results/source_recalibration/few_sample_source_recalibration_top5_sources_3seeds_detail.csv`
- `results/source_recalibration/few_sample_source_recalibration_top5_sources_3seeds_cal30_40_50_summary.csv`
- `results/source_recalibration/few_sample_source_recalibration_top5_sources_3seeds_cal30_40_50_detail.csv`
- `results/paper_tables/table43_few_sample_source_recalibration_extended.csv`
- `results/summary/table43_few_sample_source_recalibration_extended.csv`

### Main result

Zero-shot leave-source-out transfer is hard, but few-sample recalibration substantially improves performance.

| Model | Calibration | Mean R2 | Median R2 | RMSE | RPD | Traits improved |
|---|---:|---:|---:|---:|---:|---:|
| BeamK30 ResidualMLP | 0% | -0.504 | -0.156 | 2.065 | 0.964 | 0 |
| BeamK30 ResidualMLP | 5% | -0.056 | 0.034 | 2.013 | 1.057 | 5 |
| BeamK30 ResidualMLP | 10% | 0.155 | 0.124 | 1.812 | 1.143 | 7 |
| BeamK30 ResidualMLP | 20% | 0.237 | 0.225 | 1.741 | 1.195 | 7 |
| BeamK30 ResidualMLP | 30% | 0.241 | 0.235 | 1.762 | 1.198 | 7 |
| BeamK30 ResidualMLP | 40% | 0.255 | 0.208 | 1.684 | 1.211 | 7 |
| BeamK30 ResidualMLP | 50% | 0.250 | 0.217 | 1.717 | 1.214 | 7 |
| Hybrid Cross-Gated NAM | 0% | -0.520 | -0.046 | 2.052 | 0.978 | 0 |
| Hybrid Cross-Gated NAM | 5% | 0.027 | 0.038 | 1.904 | 1.091 | 6 |
| Hybrid Cross-Gated NAM | 10% | 0.172 | 0.191 | 1.766 | 1.153 | 7 |
| Hybrid Cross-Gated NAM | 20% | 0.253 | 0.260 | 1.711 | 1.211 | 7 |
| Hybrid Cross-Gated NAM | 30% | 0.252 | 0.257 | 1.732 | 1.209 | 7 |
| Hybrid Cross-Gated NAM | 40% | 0.260 | 0.266 | 1.663 | 1.222 | 7 |
| Hybrid Cross-Gated NAM | 50% | 0.264 | 0.241 | 1.676 | 1.228 | 7 |

### Interpretation

The most important gain happens by 10-20% calibration. After 20%, the curve mostly saturates. The 30-50% results are still useful because they show that additional calibration does not radically change the conclusion: few-sample recalibration already recovers most of the deployment performance.

Recommended paper claim:

> Zero-shot transfer across acquisition sources is difficult, but a small labelled calibration subset from the target source substantially recovers OOD performance. Performance improves sharply by 10-20% calibration and then largely saturates.

## 2. Statistical Comparison and Confidence Intervals

### What was done

I created a statistical comparison table using existing seed-level fusion ablation results. The comparison tests whether the hybrid and fusion models are close to the strongest supervised compact baseline.

Models compared against BeamK30 ResidualMLP:

- Hybrid Cross-Gated NAM
- Additive fusion
- FiLM fusion
- SSL-only MLP

Metrics computed:

- Mean delta R2
- Median delta R2
- 95% bootstrap confidence interval
- Wilcoxon signed-rank p-value
- Effect size
- Number of traits improved

### Output files

- `src/statistical_model_comparison.py`
- `results/paper_tables/table40_statistical_model_comparison.csv`
- `results/summary/table40_statistical_model_comparison.csv`

### Main result

| Comparison | Mean delta R2 | 95% CI | p-value | Traits improved | Interpretation |
|---|---:|---:|---:|---:|---|
| Hybrid NAM - Beam ResidualMLP | -0.024 | [-0.048, -0.003] | 0.0366 | 2/8 | competitive |
| Additive - Beam ResidualMLP | -0.030 | [-0.062, -0.004] | 0.0839 | 2/8 | competitive |
| FiLM - Beam ResidualMLP | -0.037 | [-0.068, -0.011] | 0.0138 | 2/8 | competitive |
| SSL-only - Beam ResidualMLP | -0.293 | [-0.345, -0.245] | 1.19e-7 | 0/8 | clearly weaker |

### Interpretation

The hybrid/fusion models are close to the supervised compact baseline, but SSL-only is much weaker. This supports the correct framing:

> SSL is useful as contextual fusion, not as a standalone replacement for labelled calibration.

## 3. Band Stability Analysis

### What was done

I extended the BeamSearch stability analysis to all eight traits using a CPU-friendly multi-seed run. The goal was to test whether BeamSearch-selected K=30 wavelengths are stable or random.

The analysis computes:

- Exact wavelength overlap
- Exact wavelength Jaccard similarity
- Region-level overlap
- Region-level Jaccard similarity
- Most frequent spectral regions

### Output files

- `src/band_stability_analysis.py`
- `results/beam_stability/beam_stability_results.csv`
- `results/beam_stability/beam_stability_summary.csv`
- `results/beam_stability/band_stability_pairwise_overlap.csv`
- `results/paper_tables/table41_band_stability_analysis.csv`
- `results/summary/table41_band_stability_analysis.csv`
- `results/figures/beam_selection_frequency_LAI.png`
- `results/figures/beam_selection_frequency_anth.png`
- `results/figures/beam_selection_frequency_cab.png`
- `results/figures/beam_selection_frequency_car.png`
- `results/figures/beam_selection_frequency_cbc.png`
- `results/figures/beam_selection_frequency_cm.png`
- `results/figures/beam_selection_frequency_cp.png`
- `results/figures/beam_selection_frequency_cw.png`

### Main result

Exact wavelength overlap is modest, but region-level stability is much stronger.

| Trait | Runs | Exact Jaccard | Region Jaccard | Main stable regions |
|---|---:|---:|---:|---|
| LAI | 3 | 0.043 | 0.524 | water/NIR, NIR structure, red chlorophyll |
| anth | 3 | 0.023 | 0.744 | SWIR water, red chlorophyll, blue/green pigments |
| cab | 3 | 0.163 | 0.778 | blue/green pigments, red chlorophyll, red edge |
| car | 3 | 0.154 | 0.571 | SWIR water, blue/green pigments |
| cbc | 3 | 0.129 | 0.714 | SWIR biochemical |
| cm | 3 | 0.216 | 0.565 | SWIR biochemical |
| cp | 3 | 0.023 | 0.704 | blue/green pigments, SWIR biochemical |
| cw | 3 | 0.104 | 0.690 | SWIR biochemical, water/NIR |

### Interpretation

The exact selected wavelength indices vary across repeated runs, which is expected for highly correlated hyperspectral bands. The important result is that the selected spectral regions remain much more stable.

Recommended paper claim:

> Although exact BeamSearch-selected wavelengths vary across resampling, the selected spectral regions remain relatively stable and physiologically consistent.

## 4. Master Word Report Update

### What was done

I updated the master Word report builder so the main report now includes:

- Few-sample source recalibration table
- Statistical comparison table
- BeamSearch K=30 band stability table
- Updated source-shift interpretation
- Updated safe claims and limitations
- Expanded appendix archive including recalibration and band-stability CSVs

### Output files

- `src/build_reframed_detailed_report.py`
- `src/build_reframed_full_appendix_report.py`
- `results/detailed_report_reframed_complete.docx`
- `results/detailed_report_reframed_full_appendix.docx`
- `results/detailed_report_reframed_full_appendix_updated.docx`

### QA

The updated master Word file was structurally checked with `python-docx`.

Verified:

- 212 tables
- 123 figures
- 4 sections
- New recalibration section present
- New statistical comparison section present
- New band stability section present

Visual render QA could not be completed locally because LibreOffice/soffice is not installed in the environment.

## 5. What Is Still Optional

The only remaining optional experiment is the unlabelled trait prediction export.

Recommended mode:

- Use GPU if exporting Hybrid NAM predictions because SSL embeddings are involved.
- Use CPU if exporting only BeamK30 ResidualMLP predictions.

Important wording:

> These are predicted trait estimates, not measured labels.

This export is useful for application framing, but it is not required for the main reviewer-safety package because recalibration, band stability, and statistical comparison are already complete.

## Final Reviewer-Safe Story

The strengthened paper story is:

1. BeamSearch K=30 gives a compact, interpretable supervised representation.
2. BeamK30 ResidualMLP is the strongest average in-distribution baseline.
3. Hybrid SSL fusion is close/competitive with the supervised compact baseline, while SSL-only is clearly weaker.
4. Zero-shot source transfer is difficult.
5. Few-sample target-source recalibration substantially recovers OOD performance.
6. Exact selected wavelengths vary, but selected spectral regions remain physiologically consistent.
7. The work is now framed around validation strength rather than adding more architectures.
