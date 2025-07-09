`>>> CLAUDE.md`
# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **HistoGPT-L fine-tuning project** for slide-level histopathology diagnosis prediction. It combines computer vision and natural language processing to analyze whole slide images (WSI) and generate diagnostic text using Multiple Instance Learning (MIL).

**Key Architecture**: WSI Slide → Multiple Patches → Feature Aggregation → Diagnosis Text

## Common Commands

### Setup and Verification

```bash
# Install dependencies
pip install -r requirements.txt

# Verify offline training setup (critical before training)
python check_offline_setup.py

# View training logs
tensorboard --logdir logs/
```

## Architecture and Core Components

### Multi-Modal Vision-Language Model

1. **UNI Vision Encoder**: Pre-trained ViT-L/16 (frozen during training)
2. **Perceiver Aggregator**: Combines variable number of patches (1-1000) into fixed representation
3. **BioGPT Language Model**: Generates diagnosis text from aggregated features  
4. **Cross-attention Layers**: Enable vision-language interaction via flamingo-pytorch

### Key Model Files

- `models/histogpt.py`: Main HistoGPT model implementation
- `models/aggregator.py`: Perceiver-based patch aggregation
- `models/perceiver.py`: Core perceiver attention mechanism
- `lightning_trainer.py`: PyTorch Lightning wrapper with training logic

### Data Pipeline

- **Input**: H5 files containing UNI-ViT extracted features (1024-dim per patch)
- **Processing**: Multiple Instance Learning with variable patch counts per slide
- **Output**: Slide-level diagnosis text (e.g., "basal cell carcinoma")
- **Dataset**: `slide_level_dataset.py` handles MIL data loading


### Unified prediction interface

The model can output only the binary diagnosis or generate a full report:

```{python}
from lightning_trainer import LightningHistoGPT
model = LightningHistoGPT.load_from_checkpoint("path/to/checkpoint.ckpt")

# features and coords are lists of tensors per slide
preds, short_texts = model.predict(features, coords, mode="binary")
preds, reports = model.predict(features, coords, mode="full_report", max_length=200)
```

## Training Configurations

Three preset configurations available in `fine_tune_config.py`:

| Config | Description | Time (A100) | Memory | Use Case |
|--------|-------------|-------------|---------|----------|
| `quick` | LoRA only, frozen LM | ~42 min | ~8GB | Fast iteration |
| `efficient` | LoRA + unfrozen LM | ~3.5 hours | ~12GB | Recommended |
| `full` | Complete fine-tuning | ~7 hours | ~15GB | Full customization |

### Run 5-fold cross validation with:

```bash
  python train_cross_validation.py \
      --data-path ../anne_data/512px_uni-vit-l-16_0.5mpp_0xdown_normal \
      --config efficient \
      --batch-size 4 \
      --max-steps 2000 \
      --output-dir ./cross_validation_results
```

## Data Requirements

### Expected Data Structure

```{bash}
../anne_data/512px_uni-vit-l-16_0.5mpp_0xdown_normal/
├── patient_sBBC_001.h5     # Basal cell carcinoma
├── patient_PEK_002.h5      # Squamous cell carcinoma  
└── ...
```

### H5 File Format

Each file must contain:

- `features`: Array `[num_patches, 1024]` (UNI-ViT features)
- `coordinates`: Array `[num_patches, 3]` (x, y, z positions)

### Diagnosis Labels

Extracted from filenames:

- `sBBC` or `iBBC` → "basal cell carcinoma"
- `PEK` → "squamous cell carcinoma"
- Other → "unknown_pathology"

## Offline Training Setup

**Critical**: This system is designed for offline training with pre-downloaded models.

### Required Pre-downloads

1. **BioGPT model**: `../microsoft_biogpt-large/` (config.json, pytorch_model.bin, tokenizer files)
2. **HistoGPT weights**: `../histogpt-l-6k-pruned.pt`
3. **Training data**: H5 files in expected directory

### Environment Variables for Offline Mode

```bash
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
```

**Always run `python check_offline_setup.py` before training to verify all dependencies are available locally.**

## Hardware Requirements

### GPU Specifications

- **Recommended**: Single A100 40GB (optimal for all configs)
- **Alternative**: V100 32GB (60% speed of A100)
- **Batch sizes**: 4-6 on A100, 2-4 on V100
- **Memory usage**: ~12.5GB with mixed precision + gradient checkpointing

### Training Optimizations

- Mixed precision training (automatic)
- Gradient checkpointing for memory efficiency
- Parameter-efficient fine-tuning with LoRA/PEFT
- Selective layer freezing (vision encoder, language model, aggregator)

## Key Parameters and Flags

### Critical Training Parameters

- `--batch-size`: 2-6 depending on GPU memory
- `--max-steps`: 2000-10000 typical range
- `--learning-rate`: 1e-4 to 5e-5 typical range
- `--num-workers`: Use 0 to avoid H5 multiprocessing issues

### Important Flags

- `--use-wandb`: Enable Weights & Biases logging (requires network)
- `--resume`: Resume from checkpoint
- `--test-only`: Test mode without training
- `--output-dir`: Checkpoint save directory

## Troubleshooting

### Common Issues

1. **CUDA OOM**: Reduce `--batch-size` or `--max-patches`
2. **H5 loading errors**: Ensure files have required datasets, use `--num-workers 0`
3. **Model not found**: Run `check_offline_setup.py` to verify pre-downloads
4. **Slow training**: Check batch size optimization for your GPU

### Performance Tips

- Start with `--config quick` to verify setup
- Use maximum batch size your GPU can handle
- Monitor with TensorBoard: `tensorboard --logdir logs/`
- Enable gradient checkpointing for larger models

## Training Output

The system produces:
- **Checkpoints**: `checkpoints/histogpt-{epoch}-{val_loss}.ckpt`
- **Logs**: TensorBoard logs in `logs/` directory
- **Metrics**: Loss, perplexity, learning rate tracking
- **Generated samples**: Example diagnoses during validation

## File Organization

### Core Training Files

- `train_lightning.py`: Main training entry point with CLI
- `lightning_trainer.py`: PyTorch Lightning model and data module
- `fine_tune_config.py`: Pre-defined training configurations
- `slide_level_dataset.py`: MIL dataset for slide-level training

### Model Architecture

- `models/`: All model components (HistoGPT, aggregator, perceiver, etc.)
- `helpers/`: Utilities for inference and patch processing

### Key Dependencies

- PyTorch Lightning ≥2.0.0 for training infrastructure
- transformers ≥4.37.0 for BioGPT integration  
- PEFT ≥0.4.0 for parameter-efficient fine-tuning
- flamingo-pytorch ≥0.1.2 for cross-attention layers
- h5py ≥3.13.0 for WSI feature file handling
`<<< CLAUDE.md`
