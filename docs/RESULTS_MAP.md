# Results map

Every numbered table and figure in the manuscript, the committed file that
holds it, and the script that produced it. All paths are relative to the
repository root.

## Tables

| Manuscript label | Content | Committed file | Produced by |
|---|---|---|---|
| `tab:dataset_traits` | Dataset and trait summary | `results/paper_tables/table1_dataset_summary.csv` | `src/generate_paper_artifacts.py` |
| `tab:implementation_settings` | Implementation settings | — (stated in text) | — |
| `tab:full_compact_comparison` | **Best full-spectrum reference vs 30-band residual MLP** (0.538 → 0.624) | `results/paper_tables/table4_compact_model_comparison.csv` | `src/train_beam_compact_regressors.py` → `src/generate_paper_artifacts.py` |
| `tab:compact_heads` | Compact head comparison (MLP, residual, DCN, polynomial, KAN, NAM) | `results/paper_tables/table13_labelled_modern_heads_r2.csv`, `table14_labelled_best_head_per_trait.csv` | `src/beam_supervised_modern_heads.py` |
| `tab:selector_comparison` | Band-selector comparison | `results/paper_tables/table3_band_selection_comparison.csv` | `src/train_cab_band_selection.py`, `src/rl_band_selector_cab.py`, `src/beam_band_selector.py` |
| `tab:band_stability` | Band-selection stability | `results/paper_tables/table41_band_stability_analysis.csv` | `src/band_stability_analysis.py`, `src/beam_stability.py` |
| `tab:selected_wavelengths_full` | The 30 wavelengths per trait | `results/paper_tables/table44_selected_wavelengths_by_trait.csv`, `results/beam/<trait>_beam_selected_wavelengths.csv` | `src/beam_band_selector.py` |
| `tab:ssl_masking` | Random vs contiguous masking ablation | `results/paper_tables/table7_ssl_ablation_results.csv`, `table9_contiguous_ssl_ablation_results.csv` | `src/ssl_pretrain_mae.py` |
| `tab:ssl_finetuning` | Frozen features vs end-to-end fine-tuning (0.219 → 0.497) | `results/paper_tables/table12_hybrid_ssl_beam_all8_results.csv` | `src/ssl_finetune_label_efficiency.py` |
| `tab:fusion_ablation` | **Fusion strategies** (best hybrid 0.601) | `results/paper_tables/table37_fusion_ablation_leaderboard.csv`, `table38_fusion_ablation_trait_summary.csv`, `table39_fusion_ablation_best_per_trait.csv` | `src/hybrid_ssl_fusion_ablation.py` |
| `tab:statistical_tests` | Paired significance tests | `results/paper_tables/table40_statistical_model_comparison.csv`, `table53_statistical_significance.csv`, `table27_paired_trait_significance_tests.csv` | `src/statistical_model_comparison.py` |
| `tab:label_efficiency` | Reduced-label results (25 % → 0.490, 50 % → 0.541) | `results/paper_tables/table6_label_efficiency.csv`, `table6_label_efficiency_compact.csv`, `table32_hybrid_ssl_label_efficiency_mean_r2_by_fraction.csv` | `src/label_efficiency.py`, `src/hybrid_ssl_label_efficiency.py` |
| `tab:source_transfer` | **Zero-shot leave-source-out** (mean R² −0.504) | `results/paper_tables/table35_ood_source_validation_model_leaderboard.csv`, `table36_ood_source_validation_trait_model_summary.csv` | `src/ood_source_validation.py` |
| `tab:recalibration` | **Few-sample affine recalibration** (0 → 50 % calibration) | `results/paper_tables/table43_few_sample_source_recalibration_extended.csv`, `table42_few_sample_source_recalibration.csv` | `src/few_sample_source_recalibration.py` |
| `tab:recent_methods_comparison` | Comparison with recent methods | — (compiled from the literature) | — |

## Figures

| Manuscript label | Content | Committed file | Produced by |
|---|---|---|---|
| `fig:study_workflow` | Experimental workflow (Figure 1) | `figures/study_workflow.png` | drawn schematic — see `docs/AI_USE.md` |
| `fig:model_architecture` | Model architecture | `figures/model_architecture.png` | drawn schematic |
| `fig:selected_wavelengths` | Selected wavelengths and stability | `results/paper_figures/fig1_selected_wavelengths_annotated.png` | `src/annotate_selected_wavelengths.py` |
| `fig:ssl_pretraining` | SSL pretraining and fine-tuning | `figures/ssl_pretraining_and_finetuning.png` | `src/ssl_finetune_label_efficiency.py` |
| `fig:label_efficiency` | Label-efficiency curves | `figures/label_efficiency.png` | `src/label_efficiency.py` |
| `fig:source_recalibration` | Recalibration curve | `figures/source_recalibration.png` | `src/few_sample_source_recalibration.py` |
| Graphical abstract | Study summary | `figures/graphical_abstract.svg` / `.pdf` | vector schematic — see `docs/AI_USE.md` |

## Supplementary tables

The repository carries more tables than the manuscript reports. These support
the reviewer-response experiments and the appendix:

| File | Content |
|---|---|
| `table49_band_budget_study.csv`, `table49b_band_budget_overall.csv` | Accuracy as a function of band budget |
| `table50_noise_band_dropout_robustness.csv` | Noise and band-dropout robustness |
| `table51_shap_top_wavelengths.csv` | SHAP attribution over selected wavelengths |
| `table52_compact_vs_full_paired_results.csv` | Paired compact-vs-full comparison |
| `table54_source_shift_by_source.csv`, `table55_source_shift_by_trait_source.csv` | Source shift broken down per source and trait |
| `table56*_finite_bandpass*.csv`, `table57_finite_bandpass_channel_audit.csv` | Finite-bandpass (real filter) simulation |
| `table58_cw_direct_budget_vs_merged_bandpass.csv` | Direct band budget vs merged bandpass for C<sub>w</sub> |
| `table59*_vegetation_index_baseline*.csv` | Vegetation-index baseline |
| `table60*_calibration_absolute_counts*.csv` | Absolute calibration sample counts behind each fraction |
| `table61_baseline_reconciliation_*` | Reconciliation of baseline numbers across runs |

## Headline numbers at a glance

| Claim in the abstract | File | Column |
|---|---|---|
| Compact model mean R² = 0.624 | `table4_compact_model_comparison.csv` | compact R² |
| Full-spectrum reference = 0.538 | `table4_compact_model_comparison.csv` | full-spectrum R² |
| Higher for 7 of 8 traits | `table4_compact_model_comparison.csv` | per-trait difference |
| 25 % labels → 0.490; 50 % → 0.541 | `table6_label_efficiency.csv` | fraction, mean R² |
| SSL frozen 0.219 → fine-tuned 0.497 | `table12_hybrid_ssl_beam_all8_results.csv` | mode, mean R² |
| Best structured hybrid = 0.601 | `table37_fusion_ablation_leaderboard.csv` | mean R² |
| Zero-shot LOSO mean R² = −0.504 | `table35_ood_source_validation_model_leaderboard.csv` | mean R² |
| Recalibration 10 % → 0.155, 20 % → 0.237 | `table43_few_sample_source_recalibration_extended.csv` | fraction, mean R² |
| 98.3 % fewer spectral variables | 30 / 1,721 = 1.74 % retained | derived |
