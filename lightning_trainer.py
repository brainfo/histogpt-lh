"""
PyTorch Lightning Trainer for HistoGPT-L Fine-tuning
Adopts the better paradigm from original HistoGPT with slide-level MIL training
"""

import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
import torchmetrics
from torch.nn import functional as F
from torch.optim.lr_scheduler import _LRScheduler
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from typing import Dict, List, Any, Optional
import numpy as np

from models.histogpt import HistoGPTForCausalLM
from models.aggregator import Aggregator
from fine_tune_config import FineTuningConfig
from helpers.inference import generate


class WarmupCosineAnnealingLR(_LRScheduler):
    """
    Cosine Annealing Learning Rate Scheduler with Linear Warmup
    Adapted from original HistoGPT trainer
    """
    def __init__(
        self, optimizer, warmup_steps, total_steps, min_lr,
        max_lr, eta_min=0, last_step=-1, verbose=False
    ):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.eta_min = eta_min
        super(WarmupCosineAnnealingLR, self).__init__(optimizer, last_step, verbose)

    def get_lr(self):
        if self._step_count <= self.warmup_steps:
            return [
                self.min_lr + (self.max_lr - self.min_lr) *
                (self._step_count) / self.warmup_steps
                for base_lr in self.base_lrs
            ]
        else:
            t = self._step_count - self.warmup_steps
            T = self.total_steps - self.warmup_steps
            return [
                self.eta_min + (self.max_lr - self.eta_min) *
                (1 + torch.cos(torch.tensor((t / T) * torch.pi)).item()) / 2
                for base_lr in self.base_lrs
            ]


class LightningHistoGPT(pl.LightningModule):
    """
    PyTorch Lightning wrapper for HistoGPT-L with slide-level MIL training
    Combines the best of original trainer with current fine-tuning needs
    """
    
    def __init__(self, config: FineTuningConfig):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        
        # Force offline mode for transformers
        if config.offline_mode:
            import os
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            os.environ["HF_DATASETS_OFFLINE"] = "1"
        
        # Initialize tokenizer (local path only)
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.language_model_name,
            local_files_only=config.offline_mode
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Setup model
        self.setup_model()
        
        # Setup metrics for language modeling
        self.train_loss = torchmetrics.MeanMetric()
        self.val_loss = torchmetrics.MeanMetric()
        self.test_loss = torchmetrics.MeanMetric()
        
        # Perplexity metrics
        self.train_perplexity = torchmetrics.MeanMetric()
        self.val_perplexity = torchmetrics.MeanMetric()
        self.test_perplexity = torchmetrics.MeanMetric()
        
        # Track generation quality (optional)
        self.generation_samples = []
        
    def setup_model(self):
        """Initialize HistoGPT model with proper configuration"""
        # Create aggregator
        self.aggregator = Aggregator(
            d_input=1024,  # Assuming UNI-ViT features
            d_model=self.config.aggregator_d_model,
            num_cls=167  # From original config
        )
        
        # Load language model (local only)
        from transformers import BioGptForCausalLM
        self.language_model = BioGptForCausalLM.from_pretrained(
            self.config.language_model_name,
            local_files_only=self.config.offline_mode
        )
        
        # Create HistoGPT model
        self.model = HistoGPTForCausalLM(
            self.aggregator,
            self.language_model,
            checkpoint=self.config.gradient_checkpointing
        )
        
        # Load pretrained weights if available
        if hasattr(self.config, 'pretrained_model_path') and self.config.pretrained_model_path:
            self.load_pretrained_weights()
        
        # Setup fine-tuning (freeze/unfreeze layers)
        self.setup_fine_tuning()
    
    def load_pretrained_weights(self):
        """Load pretrained HistoGPT weights"""
        try:
            state_dict = torch.load(self.config.pretrained_model_path, map_location='cpu')
            self.model.load_state_dict(state_dict, strict=False)
            print(f"Loaded pretrained weights from {self.config.pretrained_model_path}")
        except Exception as e:
            print(f"Warning: Could not load pretrained weights: {e}")
    
    def setup_fine_tuning(self):
        """Configure which parameters to fine-tune"""
        # Freeze language model if specified
        if hasattr(self.config, 'freeze_language_model') and self.config.freeze_language_model:
            for param in self.language_model.parameters():
                param.requires_grad = False
        
        # Freeze aggregator if specified
        if hasattr(self.config, 'freeze_aggregator') and self.config.freeze_aggregator:
            for param in self.aggregator.parameters():
                param.requires_grad = False
        
        # Apply LoRA if specified
        if hasattr(self.config, 'use_lora') and self.config.use_lora:
            self.apply_lora()
    
    def apply_lora(self):
        """Apply LoRA for efficient fine-tuning"""
        try:
            from peft import LoraConfig, get_peft_model
            
            lora_config = LoraConfig(
                r=getattr(self.config, 'lora_rank', 8),
                lora_alpha=getattr(self.config, 'lora_alpha', 16),
                target_modules=getattr(self.config, 'lora_target_modules', ["q_proj", "v_proj"]),
                lora_dropout=getattr(self.config, 'lora_dropout', 0.1),
                bias="none",
                task_type="CAUSAL_LM"
            )
            
            self.model = get_peft_model(self.model, lora_config)
            print(f"Applied LoRA with rank {lora_config.r}")
            
        except ImportError:
            print("PEFT library not available. Skipping LoRA application.")
    
    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler"""
        # Get trainable parameters
        trainable_params = [p for p in self.parameters() if p.requires_grad]
        
        # Setup optimizer
        optimizer = optim.AdamW(
            trainable_params,
            lr=self.config.learning_rate,
            betas=getattr(self.config, 'betas', (0.9, 0.999)),
            weight_decay=self.config.weight_decay,
            eps=1e-8
        )
        
        # Setup scheduler
        if self.config.scheduler == "warmup_cosine":
            scheduler = WarmupCosineAnnealingLR(
                optimizer=optimizer,
                warmup_steps=self.config.warmup_steps,
                total_steps=self.config.max_steps,
                max_lr=self.config.learning_rate,
                min_lr=getattr(self.config, 'min_lr', 1e-6),
                eta_min=getattr(self.config, 'end_lr', 1e-7)
            )
            
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                    "frequency": 1,
                },
            }
        
        elif self.config.scheduler == "linear_warmup":
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.config.warmup_steps,
                num_training_steps=self.config.max_steps
            )
            
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                    "frequency": 1,
                },
            }
        
        return optimizer
    
    def forward(self, batch):
        """Forward pass through the model"""
        # Unpack batch
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        image_features = batch['image_features']  # List of tensors (variable length)
        
        # Process each slide in the batch
        batch_size = len(image_features)
        all_logits = []
        
        for i in range(batch_size):
            # Get features for this slide
            slide_features = image_features[i]  # Shape: [num_patches, feature_dim]
            slide_input_ids = input_ids[i:i+1]  # Shape: [1, seq_len]
            
            # Create dummy positions for patches
            num_patches = slide_features.shape[0]
            positions = torch.arange(num_patches, device=slide_features.device).unsqueeze(0)
            
            # Forward pass for this slide
            outputs = self.model(slide_input_ids, slide_features.unsqueeze(0), positions)
            all_logits.append(outputs.logits)
        
        # Stack logits
        logits = torch.cat(all_logits, dim=0)
        
        return logits
    
    def compute_loss(self, batch):
        """Compute causal language modeling loss"""
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        
        # Forward pass
        logits = self.forward(batch)
        
        # Prepare labels for causal LM
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100  # Ignore padding tokens
        
        # Compute loss
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100
        )
        
        return loss, logits
    
    def training_step(self, batch, batch_idx):
        """Training step"""
        loss, logits = self.compute_loss(batch)
        
        # Calculate perplexity
        perplexity = torch.exp(loss)
        
        # Update metrics
        self.train_loss(loss)
        self.train_perplexity(perplexity)
        
        # Log metrics
        self.log("train/loss", self.train_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/perplexity", self.train_perplexity, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/lr", self.optimizers().param_groups[0]['lr'], on_step=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step"""
        loss, logits = self.compute_loss(batch)
        
        # Calculate perplexity
        perplexity = torch.exp(loss)
        
        # Update metrics
        self.val_loss(loss)
        self.val_perplexity(perplexity)
        
        # Log metrics
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/perplexity", self.val_perplexity, on_step=False, on_epoch=True, prog_bar=True)
        
        # Sample generation for monitoring (occasionally)
        if batch_idx == 0 and len(self.generation_samples) < 5:
            self.sample_generation(batch)
    
    def test_step(self, batch, batch_idx):
        """Test step"""
        loss, logits = self.compute_loss(batch)
        
        # Calculate perplexity
        perplexity = torch.exp(loss)
        
        # Update metrics
        self.test_loss(loss)
        self.test_perplexity(perplexity)
        
        # Log metrics
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/perplexity", self.test_perplexity, on_step=False, on_epoch=True, prog_bar=True)
    
    def sample_generation(self, batch):
        """Generate sample text for monitoring training progress"""
        try:
            with torch.no_grad():
                # Take first sample from batch
                sample_features = batch['image_features'][0:1]  # First slide
                sample_input = batch['input_ids'][0:1, :10]  # First 10 tokens as prompt
                
                # Generate text
                generated = generate(
                    self.model,
                    self.tokenizer,
                    sample_features,
                    sample_input,
                    max_length=50,
                    temperature=0.7
                )
                
                # Store sample
                self.generation_samples.append({
                    'epoch': self.current_epoch,
                    'slide_id': batch['slide_ids'][0],
                    'target': batch['texts'][0],
                    'generated': generated
                })
                
                # Log generation
                print(f"\nGeneration Sample (Epoch {self.current_epoch}):")
                print(f"Slide: {batch['slide_ids'][0]}")
                print(f"Target: {batch['texts'][0]}")
                print(f"Generated: {generated}")
                
        except Exception as e:
            print(f"Generation sampling failed: {e}")
    
    def on_train_epoch_end(self):
        """Called at the end of training epoch"""
        # Log trainable parameters count
        if self.current_epoch == 0:
            trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in self.parameters())
            self.log("model/trainable_params", float(trainable_params))
            self.log("model/total_params", float(total_params))
            self.log("model/trainable_ratio", trainable_params / total_params)
    
    def on_validation_epoch_end(self):
        """Called at the end of validation epoch"""
        # Print generation samples if available
        if self.generation_samples and self.current_epoch % 5 == 0:
            print(f"\n=== Generation Samples (Epoch {self.current_epoch}) ===")
            for sample in self.generation_samples[-2:]:  # Show last 2 samples
                print(f"Target: {sample['target']}")
                print(f"Generated: {sample['generated']}")
                print("-" * 50)


class HistoGPTDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for HistoGPT
    """
    
    def __init__(
        self,
        train_data_path: str,
        val_data_path: str,
        test_data_path: Optional[str] = None,
        batch_size: int = 4,
        num_workers: int = 4,
        tokenizer_name: str = "../microsoft_biogpt-large",
        max_text_length: int = 512,
        max_patches_per_slide: int = 1000,
    ):
        super().__init__()
        self.train_data_path = train_data_path
        self.val_data_path = val_data_path
        self.test_data_path = test_data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.tokenizer_name = tokenizer_name
        self.max_text_length = max_text_length
        self.max_patches_per_slide = max_patches_per_slide
    
    def setup(self, stage: str = None):
        """Setup datasets"""
        from slide_level_dataset import SlideLevelDataset
        
        if stage == "fit" or stage is None:
            self.train_dataset = SlideLevelDataset(
                self.train_data_path,
                tokenizer_name=self.tokenizer_name,
                max_text_length=self.max_text_length,
                max_patches_per_slide=self.max_patches_per_slide
            )
            
            self.val_dataset = SlideLevelDataset(
                self.val_data_path,
                tokenizer_name=self.tokenizer_name,
                max_text_length=self.max_text_length,
                max_patches_per_slide=self.max_patches_per_slide
            )
        
        if stage == "test" or stage is None:
            if self.test_data_path:
                self.test_dataset = SlideLevelDataset(
                    self.test_data_path,
                    tokenizer_name=self.tokenizer_name,
                    max_text_length=self.max_text_length,
                    max_patches_per_slide=self.max_patches_per_slide
                )
    
    def train_dataloader(self):
        from slide_level_dataset import SlideCollator
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=SlideCollator(),
            pin_memory=True
        )
    
    def val_dataloader(self):
        from slide_level_dataset import SlideCollator
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=SlideCollator(),
            pin_memory=True
        )
    
    def test_dataloader(self):
        if hasattr(self, 'test_dataset'):
            from slide_level_dataset import SlideCollator
            return torch.utils.data.DataLoader(
                self.test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                collate_fn=SlideCollator(),
                pin_memory=True
            )
        return None


def create_lightning_trainer(config: FineTuningConfig) -> pl.Trainer:
    """Create PyTorch Lightning trainer with proper configuration"""
    
    # Setup callbacks
    callbacks = []
    
    # Model checkpoint callback
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=config.output_dir,
        filename='histogpt-{epoch:02d}-{val/loss:.2f}',
        monitor='val/loss',
        mode='min',
        save_top_k=3,
        save_last=True
    )
    callbacks.append(checkpoint_callback)
    
    # Early stopping
    if hasattr(config, 'early_stopping_patience'):
        early_stop_callback = pl.callbacks.EarlyStopping(
            monitor='val/loss',
            patience=config.early_stopping_patience,
            mode='min',
            verbose=True
        )
        callbacks.append(early_stop_callback)
    
    # Learning rate monitor
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='step')
    callbacks.append(lr_monitor)
    
    # Setup trainer
    trainer = pl.Trainer(
        max_steps=config.max_steps,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices='auto',
        precision='16-mixed' if config.mixed_precision else 32,
        gradient_clip_val=1.0,
        accumulate_grad_batches=config.gradient_accumulation_steps,
        log_every_n_steps=config.log_steps,
        val_check_interval=config.eval_steps,
        callbacks=callbacks,
        default_root_dir=config.output_dir,
        enable_checkpointing=True,
        enable_progress_bar=True,
        enable_model_summary=True,
    )
    
    return trainer


if __name__ == "__main__":
    # Example usage
    from fine_tune_config import FineTuningConfig
    
    # Create config
    config = FineTuningConfig()
    
    # Create data module
    data_module = HistoGPTDataModule(
        train_data_path="../anne_data/512px_uni-vit-l-16_0.5mpp_0xdown_normal",
        val_data_path="../anne_data/512px_uni-vit-l-16_0.5mpp_0xdown_normal",  # Use same for demo
        batch_size=2,
        num_workers=0
    )
    
    # Create model
    model = LightningHistoGPT(config)
    
    # Create trainer
    trainer = create_lightning_trainer(config)
    
    # Start training
    trainer.fit(model, data_module)