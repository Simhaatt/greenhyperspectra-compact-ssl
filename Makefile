.PHONY: help setup data check selection pretrain fusion transfer recal artifacts all clean

help:
	@echo "GreenHyperSpectra reproducibility package"
	@echo "  make setup       install Python dependencies"
	@echo "  make data        download GreenHyperSpectra into data/raw/"
	@echo "  make check       verify the data layout"
	@echo "  make selection   stage 1  compact models on 30 selected bands"
	@echo "  make pretrain    stage 2  masked spectral pretraining (GPU)"
	@echo "  make fusion      stage 3  hybrid fusion + label efficiency (GPU)"
	@echo "  make transfer    stage 4  leave-source-out validation"
	@echo "  make recal       stage 5  few-sample affine recalibration"
	@echo "  make artifacts   stage 6  regenerate tables and figures"
	@echo "  make all         run every stage in order"

CKPT = results/ssl/masked_autoencoder_contiguous_b64.pt
SEEDS = 0 1 2

setup:
	pip install -r requirements.txt

data:
	python scripts/download_data.py

check:
	python src/inspect_dataset.py

selection:
	python src/train_beam_compact_regressors.py --budget 30
	python src/beam_supervised_modern_heads.py --models mlp residual dcn polynomial kan nam --seeds $(SEEDS) --budget 30 --run-name all8_3seeds

pretrain:
	python src/ssl_pretrain_mae.py --mask-mode contiguous --block-size 64 --mask-ratio 0.30 --latent-channels 128 --epochs 80 --batch-size 256 --out masked_autoencoder_contiguous_b64.pt

fusion:
	python src/hybrid_ssl_fusion_ablation.py --checkpoint $(CKPT) --seeds $(SEEDS) --budget 30 --run-name fusion_ablation_all8_3seeds
	python src/label_efficiency.py --seeds $(SEEDS) --budget 30
	python src/hybrid_ssl_label_efficiency.py --seeds $(SEEDS) --budget 30

transfer:
	python src/ood_source_validation.py --checkpoint $(CKPT) --traits cab cw cm LAI cp cbc car --seeds $(SEEDS) --budget 30 --max-sources 5 --min-test-labels 30

recal:
	python src/few_sample_source_recalibration.py --checkpoint $(CKPT) --fractions 0.0 0.05 0.10 0.20 0.30 0.40 0.50 --seeds $(SEEDS) --budget 30 --max-sources 5

artifacts:
	python src/generate_paper_artifacts.py
	python src/annotate_selected_wavelengths.py
	python src/statistical_model_comparison.py
	python src/band_stability_analysis.py

all:
	bash scripts/run_pipeline.sh

clean:
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
	find . -name '*.pyc' -delete
