# HistoGPT-LH

HistoGPT-LH provides a PyTorch Lightning pipeline for fine-tuning the HistoGPT-L vision-language model on digital pathology slides. It supports offline operation, LoRA-based tuning and full model training.

## Quick start

1. Install the Python requirements:
   ```bash
   pip install -r requirements.txt
   ```
2. Download BioGPT and the pretrained HistoGPT weights as described in `fine_tune_config.py`.
3. (Optional) verify your environment:
   ```bash
   python check_offline_setup.py
   ```

## Training

Launch training with `train_lightning.py` and choose one of the configuration presets in `fine_tune_config.py`:

```bash
# LoRA-only quick run
python train_lightning.py --config quick --batch-size 2

# Full fine‑tuning
python train_lightning.py --config full --batch-size 4
```

Cross-validation can be run with `train_cross_validation.py`:

```bash
python train_cross_validation.py \
    --data-path /path/to/h5dir \
    --config efficient \
    --batch-size 4 \
    --max-steps 2000 \
    --output-dir ./cross_validation_results
```

## Inference

After training, load the Lightning module and call `predict`:

```python
from lightning_trainer import LightningHistoGPT
model = LightningHistoGPT.load_from_checkpoint("path/to/checkpoint.ckpt")

# features and coords are lists of tensors per slide
preds, short_texts = model.predict(features, coords, mode="binary")
preds, reports = model.predict(features, coords, mode="full_report", max_length=200)
```

## Repository layout

```
lightning_trainer.py       # Lightning module and trainer utilities
slide_level_dataset.py     # Dataset loading with slide-level grouping
fine_tune_config.py        # Hyperparameters and presets
train_lightning.py         # Training script
helpers/                   # Text generation helpers
models/                    # Model components (Perceiver, cross attention, etc.)
check_offline_setup.py     # Verifies local files for offline runs
```

## License

This repository includes code from prior HistoGPT work © Manuel Tran / Helmholtz Munich. See individual file headers for details. All other code in this repository is under the project’s original license.

