# HistoGPT-L Fine-tuning

This project provides a PyTorch Lightning pipeline to fine-tune the HistoGPT-L vision-language model on slide-level diagnosis data. The code supports offline training, LoRA tuning, and generating either short diagnoses or full pathology reports.

---

## Repository structure

```
histogpt-lh/
├── lightning_trainer.py       # Lightning model and trainer utilities
├── slide_level_dataset.py     # Dataset loading with slide-level grouping
├── fine_tune_config.py        # Hyperparameters and presets
├── train_lightning.py         # Command line training script
├── helpers/
│   └── inference.py           # Text generation helpers
├── models/                    # Model components
│   ├── aggregator.py          # FlashPerceiver-based patch aggregator
│   ├── histogpt.py            # Cross-attention model wrapper
│   ├── perceiver.py           # FlashPerceiver implementation
│   └── embedder.py            # Positional embedding utilities
└── check_offline_setup.py     # Verifies local files for offline runs
```

---

## Key features

### Slide-level dataset with offline mode
The `SlideLevelDataset` groups patches by slide and forces transformers to load from local paths:

```
class SlideLevelDataset(Dataset):
    ...
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    self.tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        local_files_only=offline_mode
    )
```

### FlashPerceiver-based aggregator
Patch features are embedded and aggregated using a FlashAttention Perceiver:

```
class Aggregator(nn.Module):
    def __init__(self, d_input: int = 1024, d_model: int = 1536, num_cls: int = 167):
        ...
        self.model = FlashPerceiver(
            d_input=d_input,
            d_model=d_model,
            n_heads=16,
            n_layers=6,
            n_latents=640,
            attn_drop=0.0,
            concat_latents=True,
        )
```

### Cross-attention model
`HistoGPTModel` inserts gated cross-attention blocks between the Perceiver and BioGPT layers:

```
for i in range(len(biogpt.layers)):
    self.layers.append(
        nn.ModuleList(
            [
                GatedCrossAttentionBlock(
                    dim=self.biogpt_config.hidden_size,
                    dim_head=(
                        self.biogpt_config.hidden_size //
                        self.biogpt_config.num_attention_heads
                    ),
                    heads=self.biogpt_config.num_attention_heads,
                    ff_mult=4,
                    only_attend_immediate_media=True
                ),
                biogpt.layers[i],
            ]
        )
    )
```

### LoRA and fine-tuning controls
The Lightning module can freeze parts of the model and optionally apply LoRA weights:

```
if hasattr(self.config, 'use_lora') and self.config.use_lora:
    self.apply_lora()

def apply_lora(self):
    ...
    lora_config = LoraConfig(
        r=getattr(self.config, 'lora_rank', 8),
        lora_alpha=getattr(self.config, 'lora_alpha', 16),
        target_modules=getattr(self.config, 'lora_target_modules', ["q_proj", "v_proj"]),
        lora_dropout=getattr(self.config, 'lora_dropout', 0.1),
        bias="none",
        task_type="CAUSAL_LM",
    )
    self.model = get_peft_model(self.model, lora_config)
    print(f"Applied LoRA with rank {lora_config.r}")
```

### Balanced binary loss
The training loop uses a token-level loss with validity penalties to keep predictions consistent:

```
def compute_balanced_binary_loss(self, batch):
    ...
    basal_log_probs = torch.zeros(batch_size, device=logits.device)
    squamous_log_probs = torch.zeros(batch_size, device=logits.device)
    ...
    binary_logits = torch.stack([basal_log_probs, squamous_log_probs], dim=1)
    binary_loss = F.cross_entropy(binary_logits, binary_labels)
    ...
    invalid_penalty = (~valid_predictions).float().mean() * 10.0
    total_loss = binary_loss + invalid_penalty
    return total_loss, logits
```

### Unified prediction interface
The model can output only the binary diagnosis or generate a full report:

```
def predict(self, image_features, coordinates=None, mode="binary", **generation_kwargs):
    """
    Unified prediction interface with multiple modes
    ...
    """
    if mode == "binary":
        predictions, predictions_text, _ = self.constrained_predict(image_features, coordinates)
        return predictions, predictions_text
    elif mode == "full_report":
        predictions, _, full_reports = self.generate_full_report(
            image_features, coordinates, **generation_kwargs
        )
        return predictions, full_reports
    else:
        raise ValueError(f"Unknown prediction mode: {mode}. Use 'binary' or 'full_report'")
```

### Config presets
Several ready-to-use configurations are defined for quick, full or efficient runs:

```
QUICK_TUNE_CONFIG = FineTuningConfig(
    learning_rate=5e-5,
    max_steps=2000,
    batch_size=2,
    freeze_language_model=True,
    use_lora=True,
    lora_rank=8
)

FULL_TUNE_CONFIG = FineTuningConfig(
    learning_rate=1e-4,
    max_steps=10000,
    batch_size=4,
    freeze_language_model=False,
    use_lora=False,
    train_cross_attention=True,
    train_aggregator=True
)

EFFICIENT_TUNE_CONFIG = FineTuningConfig(
    learning_rate=2e-4,
    max_steps=5000,
    batch_size=8,
    freeze_language_model=False,
    use_lora=True,
    lora_rank=16,
    gradient_accumulation_steps=2
)
```

### Offline readiness check
Run `check_offline_setup.py` to ensure all data and models are available locally:

```
print("🔍 Checking Offline Training Setup")
...
if all_ready:
    print("🎉 READY FOR OFFLINE TRAINING!")
    print()
    print("Start training with:")
    print("python train_lightning.py --config quick --batch-size 2")
else:
    print("❌ NOT READY - Fix missing components above")
```

---

## Installation

1. Clone the repository and install dependencies:

```bash
pip install -r requirements.txt
```

2. Download BioGPT and pretrained HistoGPT weights to the paths specified in `fine_tune_config.py`.

3. (Optional) run the offline setup checker:

```bash
python check_offline_setup.py
```

---

## Training

Use the `train_lightning.py` script. Choose a preset or specify a custom configuration:

```bash
# Quick LoRA-only tuning
python train_lightning.py --config quick --batch-size 2

# Full fine-tuning
python train_lightning.py --config full --batch-size 4

# Custom path and parameters
python train_lightning.py \
  --config custom \
  --train-data /path/to/h5dir \
  --val-data /path/to/h5dir \
  --batch-size 4 \
  --max-steps 5000
```

Training uses the Lightning module defined in `lightning_trainer.py`, which handles dataset loading, optimizer setup, and evaluation.

---

## Inference example

After training, predictions can be obtained via the `predict` method:

```python
from lightning_trainer import LightningHistoGPT
model = LightningHistoGPT.load_from_checkpoint('path/to/checkpoint.ckpt')

# features and coordinates should be lists of tensors per slide
preds, texts = model.predict(features, coords, mode="binary")
preds, reports = model.predict(features, coords, mode="full_report", max_length=200)
```

---

## License

The repository includes code from prior HistoGPT work © Manuel Tran / Helmholtz Munich. See individual file headers for details. All other code in this repository is under the project’s original license.

---

This new README reflects the current codebase and documents offline training, LoRA support, the balanced binary loss, and the ability to generate full diagnostic reports.

