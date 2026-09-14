# Band- and Label-Efficient Hyperspectral Plant Trait Prediction

Reproducibility package for the article *Band- and Label-Efficient Hyperspectral
Plant Trait Prediction with Compact Wavelength Selection and Self-Supervised
Pretraining*, submitted to **Spectrochimica Acta Part A: Molecular and
Biomolecular Spectroscopy**.

---

## What this study asks

Hyperspectral trait models usually consume the full spectral range and a large
labelled calibration set. This work tests how far both requirements can be cut,
and whether unlabelled spectra add anything beyond a small selected band set.

| Question | Answer in this repository |
|---|---|
| Can 30 trait-specific bands replace 1,721? | Yes — mean R² **0.624** vs **0.538** for the best full-spectrum reference, higher for 7 of 8 traits (`results/paper_tables/table4_compact_model_comparison.csv`) |
| Does self-supervised pretraining beat that? | No — the strongest structured hybrid reaches **0.601**; SSL is complementary, not superior (`table37_fusion_ablation_leaderboard.csv`) |
| Does the model transfer to a new acquisition source? | Not directly — zero-shot leave-source-out mean R² is **−0.504** (`table35_ood_source_validation_model_leaderboard.csv`) |
| Does a small local calibration set help? | Partly — 10–20 % target-source labels lift mean R² to **0.155 → 0.237**, without recovering in-distribution performance (`table43_few_sample_source_recalibration_extended.csv`) |

## Dataset

`GreenHyperSpectra`: 5,635 labelled and 139,295 unlabelled spectra on a common
1,721-band grid spanning **400–2450 nm**, covering eight biochemical and
structural traits (C<sub>ab</sub>, C<sub>w</sub>, C<sub>m</sub>, LAI,
C<sub>p</sub>, C<sub>bc</sub>, C<sub>ar</sub>, C<sub>anth</sub>).

The data are **not** redistributed here (≈ 3.3 GB, separate licence).
See **[docs/DATA.md](docs/DATA.md)** for how to obtain and place them.

## Quick start

```bash
git clone https://github.com/OWNER/greenhyperspectra-compact-ssl.git
cd greenhyperspectra-compact-ssl

python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python scripts/download_data.py          # fetches GreenHyperSpectra into data/raw/
python src/inspect_dataset.py            # sanity check: shapes, bands, label counts
```

Every script resolves paths through `src/hsi_paths.py`, which defaults to the
repository root. Point the pipeline elsewhere with:

```bash
export HSI_ROOT=/scratch/hsi          # Windows: set HSI_ROOT=D:\hsi
```

Full instructions, runtimes and hardware notes: **[docs/REPRODUCE.md](docs/REPRODUCE.md)**.

## Repository layout

```
src/                     48 analysis and training scripts
  hsi_paths.py           project root resolution (no hard-coded paths)
  beam_band_selector.py  trait-specific beam-search wavelength selection
  ssl_pretrain_mae.py    1D masked spectral autoencoder pretraining
  ood_source_validation.py        leave-source-out transfer
  few_sample_source_recalibration.py  affine recalibration sweep
results/
  paper_tables/          65 CSVs — one per table reported in the manuscript
  paper_figures/         figures generated from those tables
  beam/                  the 30 selected wavelengths per trait
  summary/               cross-experiment aggregations
figures/                 manuscript figures and the graphical abstract
docs/                    reproduction guide, data guide, results map, AI disclosure
scripts/                 data download and end-to-end pipeline driver
```

## Reproducing a specific number

`docs/RESULTS_MAP.md` maps every manuscript table and figure to the CSV that
holds it and the script that produced it. The tables are committed, so a
reviewer can verify any reported value without rerunning the pipeline.

## Key parameters

| Setting | Value |
|---|---|
| Selected bands per trait | 30 (beam search, width 5, top-pool 220) |
| SSL encoder | 1D convolutional masked autoencoder, 128-D latent |
| Masking | contiguous intervals, ratio 0.30, block size 64 |
| Seeds | 0, 1, 2 (three initialisations throughout) |
| Leave-source-out | 7 eligible traits × 5 held-out sources × 3 seeds = 105 evaluations |
| Recalibration fractions | 0, 5, 10, 20, 30, 40, 50 % |

## Citation

See `CITATION.cff`. Please cite both the article and this archived package.

## Licence

CC BY 4.0 — see `LICENSE`. The GreenHyperSpectra dataset is covered by its own
licence and is not redistributed here.

## AI use

Parts of this repository's documentation and one figure were prepared with AI
assistance. See **[docs/AI_USE.md](docs/AI_USE.md)** for the disclosure that
accompanies the manuscript.
