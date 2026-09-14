# SSL Run Guide for T4

The self-supervised learning code is implemented and smoke-tested locally. Full SSL training should be run on the T4 machine.

## 1. Pretrain Masked Spectral Autoencoder

Run from `D:/hsi` or the equivalent project directory:

```powershell
.\.venv\Scripts\python.exe src\ssl_pretrain_mae.py `
  --epochs 80 `
  --batch-size 256 `
  --mask-ratio 0.30 `
  --out masked_autoencoder.pt
```

Expected output:

```text
results/ssl/masked_autoencoder.pt
results/ssl/masked_autoencoder_history.csv
results/ssl/masked_autoencoder_meta.json
```

## 2. Fine-Tune SSL Encoder for Label Efficiency

After pretraining finishes:

```powershell
.\.venv\Scripts\python.exe src\ssl_finetune_label_efficiency.py `
  --checkpoint D:\hsi\results\ssl\masked_autoencoder.pt `
  --traits cab cw cm LAI cp cbc car anth `
  --fractions 0.10 0.25 0.50 1.00 `
  --seeds 0 1 2 `
  --epochs 500 `
  --batch-size 128
```

Expected output:

```text
results/ssl_label_efficiency/ssl_label_efficiency_finetuned_results.csv
results/ssl_label_efficiency/ssl_label_efficiency_finetuned_summary.csv
results/summary/ssl_label_efficiency_finetuned_summary.csv
results/figures/ssl_label_efficiency_r2.png
results/figures/ssl_label_efficiency_rmse.png
```

## 3. Compare Against Current Label-Efficiency Baseline

Current supervised compact baseline:

```text
results/summary/label_efficiency_summary.csv
```

SSL result to compare:

```text
results/summary/ssl_label_efficiency_finetuned_summary.csv
```

Primary comparison:

```text
10% labels and 25% labels
```

SSL is most valuable if it improves low-label performance for:

```text
anth
LAI
car
cw
```

## Status

Implemented:

- `src/ssl_pretrain_mae.py`
- `src/ssl_finetune_label_efficiency.py`

Smoke-tested locally:

- MAE pretraining on 1,024 unlabelled spectra for 2 epochs.
- SSL fine-tuning on Cab at 10% labels for 2 epochs.

Full SSL results are not yet available until the T4 run is completed.
