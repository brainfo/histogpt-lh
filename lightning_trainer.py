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
        
        # Get binary classification token sequences for full phrases
        self.basal_tokens = self.tokenizer.encode("basal cell carcinoma", add_special_tokens=False)
        self.squamous_tokens = self.tokenizer.encode("squamous cell carcinoma", add_special_tokens=False)
        print(f"Basal tokens: {self.basal_tokens} ({self.tokenizer.decode(self.basal_tokens)})")
        print(f"Squamous tokens: {self.squamous_tokens} ({self.tokenizer.decode(self.squamous_tokens)})")
        
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
        
        # Classification metrics
        self.train_accuracy = torchmetrics.Accuracy(task='binary')
        self.val_accuracy = torchmetrics.Accuracy(task='binary')
        self.test_accuracy = torchmetrics.Accuracy(task='binary')
        self.train_precision = torchmetrics.Precision(task='binary')
        self.val_precision = torchmetrics.Precision(task='binary')
        self.test_precision = torchmetrics.Precision(task='binary')
        self.train_recall = torchmetrics.Recall(task='binary')
        self.val_recall = torchmetrics.Recall(task='binary')
        self.test_recall = torchmetrics.Recall(task='binary')
        self.train_f1 = torchmetrics.F1Score(task='binary')
        self.val_f1 = torchmetrics.F1Score(task='binary')
        self.test_f1 = torchmetrics.F1Score(task='binary')
        self.train_auc = torchmetrics.AUROC(task='binary')
        self.val_auc = torchmetrics.AUROC(task='binary')
        self.test_auc = torchmetrics.AUROC(task='binary')
        
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
            
            # Use real coordinates when available, otherwise fallback to dummy positions
            if batch.get('coordinates') and len(batch['coordinates']) > i and batch['coordinates'][i] is not None:
                # Use real 3D coordinates from the slide
                slide_coordinates = batch['coordinates'][i]  # Shape: [num_patches, 3]
                positions = slide_coordinates.unsqueeze(0)  # Shape: [1, num_patches, 3]
            else:
                # Fallback to dummy sequential positions (should rarely happen)
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
    
    def compute_balanced_binary_loss(self, batch):
        """Compute balanced binary loss for multi-token sequences"""
        # Forward pass to get logits
        logits = self.forward(batch)  # [batch_size, seq_len, vocab_size]
        
        # Find the position of "Final diagnosis:" in the sequence
        # We want to predict the tokens that come after this prompt
        prompt_text = "Final diagnosis:"
        prompt_tokens = self.tokenizer.encode(prompt_text, add_special_tokens=False)
        
        # Extract binary labels from batch (0=BCC, 1=SCC)
        binary_labels = batch.get('binary_labels')
        if binary_labels is None:
            # Fallback: extract from diagnosis text
            diagnoses = batch.get('diagnoses', [])
            binary_labels = torch.zeros(len(diagnoses), dtype=torch.long, device=logits.device)
            for i, diagnosis in enumerate(diagnoses):
                if 'squamous' in diagnosis.lower():
                    binary_labels[i] = 1
                # basal remains 0
        
        # Compute sequence probabilities for both diagnoses
        batch_size = logits.size(0)
        seq_len = logits.size(1)
        
        # Get the maximum sequence length to predict
        max_seq_len = max(len(self.basal_tokens), len(self.squamous_tokens))
        
        # Compute log probabilities for each diagnosis sequence
        basal_log_probs = torch.zeros(batch_size, device=logits.device)
        squamous_log_probs = torch.zeros(batch_size, device=logits.device)
        
        # For each position in the sequence
        for pos in range(max_seq_len):
            if seq_len > pos:
                token_logits = logits[:, -(max_seq_len - pos), :]  # Get logits for this position
                
                # Add log prob for basal sequence
                if pos < len(self.basal_tokens):
                    basal_token_id = self.basal_tokens[pos]
                    basal_log_probs += F.log_softmax(token_logits, dim=-1)[:, basal_token_id]
                
                # Add log prob for squamous sequence  
                if pos < len(self.squamous_tokens):
                    squamous_token_id = self.squamous_tokens[pos]
                    squamous_log_probs += F.log_softmax(token_logits, dim=-1)[:, squamous_token_id]
        
        # Normalize by sequence length
        basal_log_probs = basal_log_probs / len(self.basal_tokens)
        squamous_log_probs = squamous_log_probs / len(self.squamous_tokens)
        
        # Create binary logits for cross-entropy
        binary_logits = torch.stack([basal_log_probs, squamous_log_probs], dim=1)
        
        # Compute binary cross-entropy loss
        binary_loss = F.cross_entropy(binary_logits, binary_labels)
        
        # Compute validity penalty based on greedy decoding
        greedy_predictions = self._get_greedy_sequence_predictions(logits, max_seq_len)
        valid_predictions = self._check_sequence_validity(greedy_predictions)
        invalid_penalty = (~valid_predictions).float().mean() * 10.0
        
        total_loss = binary_loss + invalid_penalty
        return total_loss, logits
    
    def _get_greedy_sequence_predictions(self, logits, max_seq_len):
        """Get greedy sequence predictions for validity checking"""
        batch_size = logits.size(0)
        seq_len = logits.size(1)
        
        predictions = []
        for i in range(batch_size):
            pred_tokens = []
            for pos in range(max_seq_len):
                if seq_len > pos:
                    token_logits = logits[i, -(max_seq_len - pos), :]
                    pred_token = torch.argmax(token_logits).item()
                    pred_tokens.append(pred_token)
            predictions.append(pred_tokens)
        
        return predictions
    
    def _check_sequence_validity(self, predictions):
        """Check if predicted sequences match either basal or squamous"""
        valid = torch.zeros(len(predictions), dtype=torch.bool)
        
        for i, pred_tokens in enumerate(predictions):
            # Check if prediction matches basal sequence
            if len(pred_tokens) >= len(self.basal_tokens):
                if pred_tokens[:len(self.basal_tokens)] == self.basal_tokens:
                    valid[i] = True
                    continue
            
            # Check if prediction matches squamous sequence
            if len(pred_tokens) >= len(self.squamous_tokens):
                if pred_tokens[:len(self.squamous_tokens)] == self.squamous_tokens:
                    valid[i] = True
                    continue
        
        # Move to the same device as the model parameters
        device = next(self.parameters()).device
        return valid.to(device)
    
    def _get_binary_predictions(self, logits):
        """Extract binary predictions and probabilities from logits"""
        # Get sequence probabilities for both diagnoses like in compute_balanced_binary_loss
        batch_size = logits.size(0)
        max_seq_len = max(len(self.basal_tokens), len(self.squamous_tokens))
        
        # Compute log probabilities for each diagnosis sequence
        basal_log_probs = torch.zeros(batch_size, device=logits.device)
        squamous_log_probs = torch.zeros(batch_size, device=logits.device)
        
        # For each position in the sequence
        for pos in range(max_seq_len):
            if logits.size(1) > pos:
                token_logits = logits[:, -(max_seq_len - pos), :]
                
                # Add log prob for basal sequence
                if pos < len(self.basal_tokens):
                    basal_token_id = self.basal_tokens[pos]
                    basal_log_probs += F.log_softmax(token_logits, dim=-1)[:, basal_token_id]
                
                # Add log prob for squamous sequence  
                if pos < len(self.squamous_tokens):
                    squamous_token_id = self.squamous_tokens[pos]
                    squamous_log_probs += F.log_softmax(token_logits, dim=-1)[:, squamous_token_id]
        
        # Normalize by sequence length
        basal_log_probs = basal_log_probs / len(self.basal_tokens)
        squamous_log_probs = squamous_log_probs / len(self.squamous_tokens)
        
        # Binary predictions (0=basal, 1=squamous)
        binary_preds = (squamous_log_probs > basal_log_probs).long()
        
        # Convert log probabilities to probabilities for AUC
        binary_logits = torch.stack([basal_log_probs, squamous_log_probs], dim=1)
        binary_probs = F.softmax(binary_logits, dim=1)[:, 1]  # Prob of squamous (class 1)
        
        return binary_preds, binary_probs
    
    def _get_binary_targets(self, batch):
        """Extract binary targets from batch"""
        # Try to get binary labels from batch
        binary_targets = batch.get('binary_labels')
        if binary_targets is not None:
            return binary_targets
        
        # Fallback: extract from diagnosis text
        diagnoses = batch.get('diagnoses', [])
        if diagnoses:
            binary_targets = torch.zeros(len(diagnoses), dtype=torch.long, device=next(self.parameters()).device)
            for i, diagnosis in enumerate(diagnoses):
                if 'squamous' in diagnosis.lower():
                    binary_targets[i] = 1
                # basal remains 0
            return binary_targets
        
        # Last resort: extract from text field
        texts = batch.get('texts', [])
        if texts:
            binary_targets = torch.zeros(len(texts), dtype=torch.long, device=next(self.parameters()).device)
            for i, text in enumerate(texts):
                if 'squamous' in text.lower():
                    binary_targets[i] = 1
                # basal remains 0
            return binary_targets
        
        # If we can't determine targets, return dummy targets (should not happen in practice)
        batch_size = batch['input_ids'].size(0)
        return torch.zeros(batch_size, dtype=torch.long, device=next(self.parameters()).device)
    
    def training_step(self, batch, batch_idx):
        """Training step with binary loss"""
        # Use balanced binary loss instead of standard causal LM loss
        loss, logits = self.compute_balanced_binary_loss(batch)
        
        # Calculate perplexity (approximate for binary case)
        perplexity = torch.exp(loss)
        
        # Get binary predictions and targets for classification metrics
        binary_preds, binary_probs = self._get_binary_predictions(logits)
        binary_targets = self._get_binary_targets(batch)
        
        # Update metrics
        self.train_loss(loss)
        self.train_perplexity(perplexity)
        self.train_accuracy(binary_preds, binary_targets)
        self.train_precision(binary_preds, binary_targets)
        self.train_recall(binary_preds, binary_targets)
        self.train_f1(binary_preds, binary_targets)
        self.train_auc(binary_probs, binary_targets)
        
        # Log metrics
        self.log("train/loss", self.train_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/perplexity", self.train_perplexity, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/accuracy", self.train_accuracy, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/precision", self.train_precision, on_step=True, on_epoch=True)
        self.log("train/recall", self.train_recall, on_step=True, on_epoch=True)
        self.log("train/f1", self.train_f1, on_step=True, on_epoch=True)
        self.log("train/auc", self.train_auc, on_step=True, on_epoch=True)
        self.log("train/lr", self.optimizers().param_groups[0]['lr'], on_step=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step with binary loss"""
        # Use balanced binary loss for validation too
        loss, logits = self.compute_balanced_binary_loss(batch)
        
        # Calculate perplexity
        perplexity = torch.exp(loss)
        
        # Get binary predictions and targets for classification metrics
        binary_preds, binary_probs = self._get_binary_predictions(logits)
        binary_targets = self._get_binary_targets(batch)
        
        # Update metrics
        self.val_loss(loss)
        self.val_perplexity(perplexity)
        self.val_accuracy(binary_preds, binary_targets)
        self.val_precision(binary_preds, binary_targets)
        self.val_recall(binary_preds, binary_targets)
        self.val_f1(binary_preds, binary_targets)
        self.val_auc(binary_probs, binary_targets)
        
        # Log metrics
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/perplexity", self.val_perplexity, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/accuracy", self.val_accuracy, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/precision", self.val_precision, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/recall", self.val_recall, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/f1", self.val_f1, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/auc", self.val_auc, on_step=False, on_epoch=True, prog_bar=True)
        
        # Sample generation for monitoring (occasionally)
        if batch_idx == 0 and len(self.generation_samples) < 5:
            self.sample_generation(batch)
    
    def test_step(self, batch, batch_idx):
        """Test step with binary loss"""
        # Use balanced binary loss for testing too
        loss, logits = self.compute_balanced_binary_loss(batch)
        
        # Calculate perplexity
        perplexity = torch.exp(loss)
        
        # Get binary predictions and targets for classification metrics
        binary_preds, binary_probs = self._get_binary_predictions(logits)
        binary_targets = self._get_binary_targets(batch)
        
        # Update metrics
        self.test_loss(loss)
        self.test_perplexity(perplexity)
        self.test_accuracy(binary_preds, binary_targets)
        self.test_precision(binary_preds, binary_targets)
        self.test_recall(binary_preds, binary_targets)
        self.test_f1(binary_preds, binary_targets)
        self.test_auc(binary_probs, binary_targets)
        
        # Log metrics
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/perplexity", self.test_perplexity, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/accuracy", self.test_accuracy, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/precision", self.test_precision, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/recall", self.test_recall, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/f1", self.test_f1, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/auc", self.test_auc, on_step=False, on_epoch=True, prog_bar=True)
    
    def constrained_predict(self, image_features, coordinates=None):
        """Generate constrained sequence predictions for full diagnostic phrases"""
        with torch.no_grad():
            batch_size = len(image_features)
            device = image_features[0].device
            
            # Prepare input for "Final diagnosis:" prompt
            prompt_text = "Final diagnosis:"
            prompt_tokens = self.tokenizer.encode(prompt_text, add_special_tokens=False, return_tensors='pt')
            prompt_tokens = prompt_tokens.to(device)
            
            # Initialize sequences with the prompt
            current_sequences = prompt_tokens.repeat(batch_size, 1)  # [batch_size, prompt_len]
            attention_masks = torch.ones_like(current_sequences)
            
            # Get maximum sequence length to generate
            max_seq_len = max(len(self.basal_tokens), len(self.squamous_tokens))
            
            # Store sequence probabilities for each diagnosis
            basal_log_probs = torch.zeros(batch_size, device=device)
            squamous_log_probs = torch.zeros(batch_size, device=device)
            
            # Generate tokens step by step
            for step in range(max_seq_len):
                # Create batch for forward pass
                mock_batch = {
                    'input_ids': current_sequences,
                    'attention_mask': attention_masks,
                    'image_features': image_features,
                    'coordinates': coordinates
                }
                
                # Forward pass
                logits = self.forward(mock_batch)
                next_token_logits = logits[:, -1, :]  # [batch_size, vocab_size]
                
                # Compute probabilities for valid next tokens
                log_probs = F.log_softmax(next_token_logits, dim=-1)
                
                # Add probabilities for basal sequence
                if step < len(self.basal_tokens):
                    basal_token_id = self.basal_tokens[step]
                    basal_log_probs += log_probs[:, basal_token_id]
                
                # Add probabilities for squamous sequence
                if step < len(self.squamous_tokens):
                    squamous_token_id = self.squamous_tokens[step]
                    squamous_log_probs += log_probs[:, squamous_token_id]
                
                # Determine next token based on current best sequence
                # For simplicity, use greedy selection from valid options
                next_tokens = torch.zeros(batch_size, dtype=torch.long, device=device)
                
                for i in range(batch_size):
                    # Compare current probabilities and choose the better sequence
                    current_basal_prob = basal_log_probs[i] / max(1, step + 1)
                    current_squamous_prob = squamous_log_probs[i] / max(1, step + 1)
                    
                    if current_basal_prob >= current_squamous_prob:
                        # Choose basal sequence
                        if step < len(self.basal_tokens):
                            next_tokens[i] = self.basal_tokens[step]
                        else:
                            next_tokens[i] = self.tokenizer.eos_token_id
                    else:
                        # Choose squamous sequence
                        if step < len(self.squamous_tokens):
                            next_tokens[i] = self.squamous_tokens[step]
                        else:
                            next_tokens[i] = self.tokenizer.eos_token_id
                
                # Append next tokens to sequences
                current_sequences = torch.cat([current_sequences, next_tokens.unsqueeze(1)], dim=1)
                attention_masks = torch.cat([attention_masks, torch.ones(batch_size, 1, device=device)], dim=1)
            
            # Determine final predictions based on total probabilities
            normalized_basal_probs = basal_log_probs / len(self.basal_tokens)
            normalized_squamous_probs = squamous_log_probs / len(self.squamous_tokens)
            
            # Binary predictions (0=basal, 1=squamous)
            predictions = (normalized_squamous_probs > normalized_basal_probs).long()
            
            # Convert to text labels
            predictions_text = []
            for pred in predictions:
                if pred == 0:
                    predictions_text.append("basal cell carcinoma")
                else:
                    predictions_text.append("squamous cell carcinoma")
            
            # Also return the generated sequences for debugging
            generated_sequences = []
            for i in range(batch_size):
                # Extract only the generated part (after prompt)
                generated_tokens = current_sequences[i, prompt_tokens.size(1):].tolist()
                generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                generated_sequences.append(generated_text.strip())
            
            return predictions, predictions_text, generated_sequences
    
    def generate_full_report(self, image_features, coordinates=None, max_length=200, temperature=0.7, do_sample=True):
        """Generate full diagnostic reports during inference"""
        with torch.no_grad():
            batch_size = len(image_features)
            device = image_features[0].device
            
            # First get binary classification to guide the report
            binary_preds, binary_texts, _ = self.constrained_predict(image_features, coordinates)
            
            # Prepare input for full report generation
            prompt_text = "Final diagnosis:"
            prompt_tokens = self.tokenizer.encode(prompt_text, add_special_tokens=False, return_tensors='pt')
            prompt_tokens = prompt_tokens.to(device)
            
            # Initialize sequences with the prompt
            input_ids = prompt_tokens.repeat(batch_size, 1)
            attention_mask = torch.ones_like(input_ids)
            
            full_reports = []
            
            for i in range(batch_size):
                # Get single sample for generation
                single_features = [image_features[i]]
                single_coords = [coordinates[i]] if coordinates else None
                single_input = input_ids[i:i+1]
                single_mask = attention_mask[i:i+1]
                
                # Create batch for this single sample
                single_batch = {
                    'input_ids': single_input,
                    'attention_mask': single_mask,
                    'image_features': single_features,
                    'coordinates': single_coords
                }
                
                # Generate the report
                generated_ids = self._generate_autoregressive(
                    single_batch, 
                    max_length=max_length,
                    temperature=temperature,
                    do_sample=do_sample,
                    binary_guidance=binary_texts[i]
                )
                
                # Decode the full report
                generated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                
                # Clean up the text (remove prompt if it appears)
                if prompt_text in generated_text:
                    report_text = generated_text.split(prompt_text, 1)[1].strip()
                else:
                    report_text = generated_text.strip()
                
                full_reports.append(report_text)
            
            return binary_preds, binary_texts, full_reports
    
    def _generate_autoregressive(self, batch, max_length=200, temperature=0.7, do_sample=True, binary_guidance=None):
        """Autoregressive generation for full reports"""
        input_ids = batch['input_ids'].clone()
        attention_mask = batch['attention_mask'].clone()
        
        # Add binary diagnosis as guidance at the start
        if binary_guidance:
            guidance_tokens = self.tokenizer.encode(f" {binary_guidance}.", add_special_tokens=False, return_tensors='pt')
            guidance_tokens = guidance_tokens.to(input_ids.device)
            input_ids = torch.cat([input_ids, guidance_tokens], dim=1)
            guidance_mask = torch.ones_like(guidance_tokens)
            attention_mask = torch.cat([attention_mask, guidance_mask], dim=1)
        
        # Generate additional tokens
        for _ in range(max_length - input_ids.size(1)):
            # Update batch with current sequence
            current_batch = {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'image_features': batch['image_features'],
                'coordinates': batch['coordinates']
            }
            
            # Forward pass
            logits = self.forward(current_batch)
            next_token_logits = logits[:, -1, :] / temperature
            
            # Sample or take greedy
            if do_sample:
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            # Append token
            input_ids = torch.cat([input_ids, next_token], dim=1)
            new_mask = torch.ones_like(next_token)
            attention_mask = torch.cat([attention_mask, new_mask], dim=1)
            
            # Stop if EOS token
            if next_token.item() == self.tokenizer.eos_token_id:
                break
            
            # Stop if we see sentence endings for reports
            if next_token.item() in [self.tokenizer.encode(".", add_special_tokens=False)[0], 
                                    self.tokenizer.encode("!", add_special_tokens=False)[0]] and input_ids.size(1) > 50:
                # Allow some minimum length before stopping on punctuation
                break
        
        return input_ids
    
    def predict(self, image_features, coordinates=None, mode="binary", **generation_kwargs):
        """
        Unified prediction interface with multiple modes
        
        Args:
            image_features: List of feature tensors for each slide
            coordinates: Optional coordinates for each slide  
            mode: "binary" for classification only, "full_report" for detailed reports
            **generation_kwargs: Additional arguments for text generation
            
        Returns:
            predictions: Binary predictions (0=BCC, 1=SCC)
            text_outputs: Either diagnostic phrases or full reports
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


def create_lightning_trainer(config: FineTuningConfig, loggers=None) -> pl.Trainer:
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
    
    # Calculate validation check interval (ensure it's not larger than training batches)
    val_check_config = {}
    if hasattr(config, 'eval_steps') and config.eval_steps:
        val_check_config['val_check_interval'] = config.eval_steps
    else:
        val_check_config['check_val_every_n_epoch'] = 1
    
    # Setup trainer
    trainer = pl.Trainer(
        max_steps=config.max_steps,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices='auto',
        precision='16-mixed' if config.mixed_precision else 32,
        gradient_clip_val=1.0,
        accumulate_grad_batches=config.gradient_accumulation_steps,
        log_every_n_steps=config.log_steps,
        callbacks=callbacks,
        logger=loggers,
        default_root_dir=config.output_dir,
        enable_checkpointing=True,
        enable_progress_bar=True,
        enable_model_summary=True,
        **val_check_config
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