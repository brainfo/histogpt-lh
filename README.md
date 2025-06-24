# HistoGPT-L Fine-tuning: Slide-Level Diagnosis Prediction

## Overview

This repository contains a PyTorch Lightning-based fine-tuning pipeline for HistoGPT-L, adapted for **slide-level diagnosis prediction** from histopathology images. The system uses Multiple Instance Learning (MIL) to process variable numbers of patches per slide and predict a single diagnosis.

## Key Features

### 🎯 **Slide-Level Training**
- **Input**: 251 whole slide images (H5 files with extracted features)
- **Output**: Single diagnosis per slide (e.g., "basal cell carcinoma")
- **Approach**: Multiple Instance Learning (MIL) with patch aggregation

### ⚡ **PyTorch Lightning Framework**
- Robust training infrastructure with automatic GPU handling
- Built-in checkpointing, logging, and early stopping
- Mixed precision training for memory efficiency
- Comprehensive metrics tracking

### 🧠 **Efficient Fine-tuning**
- **LoRA** (Low-Rank Adaptation) support for parameter-efficient training
- Selective layer freezing (vision encoder, language model, aggregator)
- Gradient checkpointing for large model training
- Multiple training configurations (quick, efficient, full)

## Architecture

```
WSI Slide → Multiple Patches → Feature Aggregation → Diagnosis Text
     ↓              ↓                    ↓               ↓
[H5 Files]   [1000 patches]      [Perceiver]    ["Final diagnosis: ..."]
                  ↓                    ↓
            [1024-dim features]  [Cross-attention]
```

### Core Components
1. **UNI Vision Encoder**: Pre-trained ViT-L/16 (frozen during training)
2. **Perceiver Aggregator**: Combines variable number of patches into fixed representation
3. **BioGPT Language Model**: Generates diagnosis text from aggregated features
4. **Cross-attention Layers**: Enable vision-language interaction

## Installation (Offline Training Ready)

```bash
# Install dependencies (no network required during training)
pip install -r requirements.txt

# Install flash attention (if needed)
pip install flash-attn --no-build-isolation

# Verify offline setup
python check_offline_setup.py
```

### 🌐 Offline Training Prerequisites

**Before training, ensure these are downloaded:**
1. **BioGPT model**: `../microsoft_biogpt-large/` (config.json, pytorch_model.bin, tokenizer.json)
2. **HistoGPT weights**: `../histogpt-l-6k-pruned.pt`
3. **Data**: `../anne_data/512px_uni-vit-l-16_0.5mpp_0xdown_normal/*.h5`

**No network access required during training!**

These files can be found and expected in `../microsoft_biogpt-large/` directory as specified in `fine_tune_config.py:23`.

- Option 1: Hugging Face Hub (easiest)

  ```bash
  git lfs install
  git clone https://huggingface.co/microsoft/biogpt-large ../microsoft_biogpt-large
  ```

- Option 2: Python script


```python
  from transformers import BioGptForCausalLM, BioGptTokenizer

  # This will download and cache the files
  model =
  BioGptForCausalLM.from_pretrained("microsoft/biogpt-large")
  tokenizer =
  BioGptTokenizer.from_pretrained("microsoft/biogpt-large")

  # Save locally
  model.save_pretrained("../microsoft_biogpt-large")
  tokenizer.save_pretrained("../microsoft_biogpt-large")
```

## Quick Start

### 1. Basic Training
```bash
# Quick training (LoRA only, ~42 minutes on A100)
python train_lightning.py --config quick --batch-size 2

# Efficient training (LoRA + unfrozen, ~3.5 hours on A100)  
python train_lightning.py --config efficient --batch-size 4

# Full fine-tuning (~7 hours on A100)
python train_lightning.py --config full --batch-size 4
```

### 2. Custom Configuration
```bash
python train_lightning.py \
  --train-data "../anne_data/512px_uni-vit-l-16_0.5mpp_0xdown_normal" \
  --val-data "../anne_data/512px_uni-vit-l-16_0.5mpp_0xdown_normal" \
  --batch-size 4 \
  --max-steps 5000 \
  --learning-rate 1e-4 \
  --use-wandb \
  --experiment-name "histogpt-diagnosis"
```

### 3. Resume Training
```bash
python train_lightning.py --config efficient --resume checkpoints/last.ckpt
```

## Data Format

### Expected Data Structure
```
../anne_data/512px_uni-vit-l-16_0.5mpp_0xdown_normal/
├── patient_sBBC_001.h5     # Basal cell carcinoma
├── patient_PEK_002.h5      # Squamous cell carcinoma  
├── sample_sBBC_003.h5
└── ...
```

### H5 File Contents
Each H5 file should contain:
- `features`: Array of shape `[num_patches, 1024]` (UNI-ViT features)
- `coordinates`: Array of shape `[num_patches, 3]` (x, y, z positions)

### Diagnosis Extraction
The system extracts diagnosis labels from filenames:
- Files containing `sBBC` → "basal cell carcinoma"  
- Files containing `PEK` → "squamous cell carcinoma"
- Unknown patterns → "unknown_pathology"

## Training Configurations

| Configuration | Description | Training Time (A100) | Memory Usage |
|---------------|-------------|---------------------|--------------|
| **quick** | LoRA only, frozen LM | ~42 minutes | ~8GB |
| **efficient** | LoRA + unfrozen LM | ~3.5 hours | ~12GB |
| **full** | Complete fine-tuning | ~7 hours | ~15GB |

## GPU Requirements

### Recommended Setup
- **Single A100 40GB**: Optimal for all configurations
- **Memory needed**: ~12.5GB (with mixed precision + gradient checkpointing)
- **Batch size**: 4-6 on A100, 2-4 on V100

### Alternative Options
- **V100 32GB**: Supported but slower (~60% speed of A100)
- **Multiple GPUs**: Not necessary for 251 slides

## Key Files

```
histogpt-lh/
├── train_lightning.py          # Main training script
├── lightning_trainer.py        # PyTorch Lightning model wrapper
├── slide_level_dataset.py      # Slide-level dataset (MIL approach)  
├── fine_tune_config.py         # Training configurations
├── models/
│   ├── histogpt.py            # HistoGPT model architecture
│   ├── aggregator.py          # Perceiver-based patch aggregation
│   └── embedder.py            # Positional embeddings
├── helpers/
│   └── inference.py           # Text generation utilities
└── requirements.txt           # Dependencies
```

## Training Output

The system generates:
- **Checkpoints**: `checkpoints/histogpt-{epoch}-{val_loss}.ckpt`
- **Logs**: TensorBoard logs in `logs/`
- **Metrics**: Loss, perplexity, learning rate tracking
- **Generated samples**: Example diagnoses during validation

## Example Training Command

```bash
# Full training run with monitoring
python train_lightning.py \
  --config efficient \
  --batch-size 4 \
  --max-steps 5000 \
  --use-wandb \
  --wandb-project "histogpt-diagnosis" \
  --experiment-name "basal-squamous-classification" \
  --output-dir "./checkpoints" \
  --num-workers 0
```

## Troubleshooting

### Common Issues
1. **CUDA out of memory**: Reduce `--batch-size` or `--max-patches`
2. **H5 file errors**: Ensure files contain `features` and `coordinates` datasets
3. **Slow loading**: Set `--num-workers 0` to avoid multiprocessing issues with H5 files

### Performance Tips
- Use `--batch-size 4-6` on A100 40GB for optimal speed
- Enable `--use-wandb` for comprehensive experiment tracking
- Start with `--config quick` to verify everything works

## Citation

Based on the original HistoGPT work:
```bibtex
@article{histogpt2024,
  title={HistoGPT: Vision-Language Model for Pathology Report Generation},
  author={[Original Authors]},
  journal={[Journal]},
  year={2024}
}
```