# Obtaining the data

The GreenHyperSpectra collection is roughly **3.3 GB** and is not redistributed
in this repository. It carries its own licence and must be obtained from the
original source.

## Source

GreenHyperSpectra is published on the Hugging Face Hub and derives from an
aggregated trait–spectroscopy collection:

- Cherif et al., *GreenHyperSpectra* (Hugging Face dataset)
- Cherif et al. (2023), the underlying aggregated spectra collection

Cite the dataset separately from this software; the terms of its own licence
apply to any redistribution.

## Automatic download

```bash
python scripts/download_data.py
```

This writes into `data/raw/` (or `$HSI_ROOT/data/raw/` if that variable is set).
Set a cache location first if your home directory is small:

```bash
export HF_HOME=/scratch/hf_cache
```

## Expected layout

After download, `data/raw/` must contain:

```
data/raw/
├── greenhyperspectra_labeled_all.parquet         ~151 MB   5,635 spectra
├── greenhyperspectra_labeled_train.parquet       ~ 73 MB
├── greenhyperspectra_labeled_test.parquet        ~ 19 MB
├── greenhyperspectra_labeled_all_columns.txt              column manifest
└── greenhyperspectra_unlabeled_hf/               ~1.9 GB   139,295 spectra
    ├── dataset_dict.json
    └── train/
        ├── data-00000-of-00004.arrow
        ├── data-00001-of-00004.arrow
        ├── data-00002-of-00004.arrow
        ├── data-00003-of-00004.arrow
        ├── dataset_info.json
        └── state.json
```

Verify with:

```bash
python src/inspect_dataset.py
```

Expected output: 1,721 wavelength columns per file, wavelengths running
400–2450 nm, and per-trait label counts for the eight targets.

## Structure of a labelled file

| Column group | Meaning |
|---|---|
| `400` … `2450` | 1,721 reflectance values, one per wavelength (column names are integers) |
| `cab`, `cw`, `cm`, `LAI`, `cp`, `cbc`, `car`, `anth` | trait targets; missing where a spectrum was not paired with that measurement |
| source / metadata columns | acquisition source identifier used for leave-source-out splits |

Trait availability is uneven — not every spectrum carries all eight
measurements, so the usable sample size differs per target. `anth`
(anthocyanin) comes from a single acquisition source and is therefore excluded
from leave-source-out evaluation, leaving seven eligible traits.

## Disk budget

| Item | Size |
|---|---|
| Raw data | ~3.3 GB |
| Hugging Face cache during download | up to ~2 GB (can be deleted afterwards) |
| SSL checkpoints | ~50 MB |
| Generated result CSVs and figures | ~40 MB |
