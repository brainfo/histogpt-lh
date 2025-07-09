#!/usr/bin/env python3
"""
5-Fold Cross Validation Training Script for HistoGPT-L
Implements stratified cross-validation with proper evaluation metrics
"""

import os
import argparse
import json
import numpy as np
import torch
import pytorch_lightning as pl
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor

from lightning_trainer import LightningHistoGPT, HistoGPTDataModule
from slide_level_dataset import SlideLevelDataset
from fine_tune_config import FineTuningConfig


def get_slide_ids_and_labels(data_path):
    """Extract slide IDs and labels from H5 files"""
    slide_ids = []
    labels = []
    
    data_dir = Path(data_path)
    for h5_file in data_dir.glob("*.h5"):
        slide_id = h5_file.stem
        slide_ids.append(slide_id)
        
        # Extract label from filename (sBBC/iBBC = 0, PEK = 1)
        if 'sBBC' in slide_id or 'iBBC' in slide_id:
            labels.append(0)  # Basal cell carcinoma
        elif 'PEK' in slide_id:
            labels.append(1)  # Squamous cell carcinoma
        else:
            labels.append(0)  # Default to basal for unknown
    
    return slide_ids, labels


def create_fold_datasets(data_path, train_indices, val_indices, slide_ids, config):
    """Create training and validation datasets for a specific fold"""

    train_files = [f"{slide_ids[i]}.h5" for i in train_indices]
    val_files = [f"{slide_ids[i]}.h5" for i in val_indices]

    base_train = SlideLevelDataset(
        data_path,
        tokenizer_name=config.tokenizer_name,
        max_text_length=config.max_text_length,
        max_patches_per_slide=config.max_patches_per_slide,
    )

    base_val = SlideLevelDataset(
        data_path,
        tokenizer_name=config.tokenizer_name,
        max_text_length=config.max_text_length,
        max_patches_per_slide=config.max_patches_per_slide,
    )

    base_train.slides = [s for s in base_train.slides if f"{s['slide_id']}.h5" in train_files]
    base_val.slides = [s for s in base_val.slides if f"{s['slide_id']}.h5" in val_files]

    return base_train, base_val


def train_fold(fold_idx, train_dataset, val_dataset, config, output_dir):
    """Train a single fold"""
    print(f"\n=== Training Fold {fold_idx + 1} ===")
    
    # Create fold-specific output directory
    fold_output_dir = os.path.join(output_dir, f"fold_{fold_idx + 1}")
    os.makedirs(fold_output_dir, exist_ok=True)
    
    # Setup logging
    logger = TensorBoardLogger(
        save_dir=fold_output_dir,
        name="histogpt_fold",
        version=f"fold_{fold_idx + 1}"
    )
    
    # Create data module with fold datasets
    class FoldDataModule(pl.LightningDataModule):
        def __init__(self, train_dataset, val_dataset, batch_size, num_workers):
            super().__init__()
            self.train_dataset = train_dataset
            self.val_dataset = val_dataset
            self.batch_size = batch_size
            self.num_workers = num_workers
        
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
    
    data_module = FoldDataModule(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers
    )
    
    # Create model
    model = LightningHistoGPT(config)
    
    # Setup callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=fold_output_dir,
            filename=f'fold_{fold_idx + 1}_best_model',
            monitor='val/f1',
            mode='max',
            save_top_k=1,
            save_last=True
        ),
        EarlyStopping(
            monitor='val/f1',
            patience=config.early_stopping_patience,
            mode='max',
            verbose=True
        ),
        LearningRateMonitor(logging_interval='step')
    ]
    
    # Create trainer
    trainer = pl.Trainer(
        max_steps=config.max_steps,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices='auto',
        precision='16-mixed' if config.mixed_precision else 32,
        gradient_clip_val=1.0,
        accumulate_grad_batches=config.gradient_accumulation_steps,
        log_every_n_steps=config.log_steps,
        check_val_every_n_epoch=1,
        callbacks=callbacks,
        logger=logger,
        default_root_dir=fold_output_dir,
        enable_checkpointing=True,
        enable_progress_bar=True,
        enable_model_summary=True
    )
    
    # Train the model
    trainer.fit(model, data_module)
    
    # Get best model metrics
    best_metrics = {}
    if hasattr(trainer.callback_metrics, 'items'):
        for key, value in trainer.callback_metrics.items():
            if key.startswith('val/'):
                best_metrics[key] = float(value)
    
    return best_metrics, trainer.checkpoint_callback.best_model_path


def evaluate_fold(model_path, test_dataset, config):
    """Evaluate a trained model on test data"""
    print(f"Evaluating model: {model_path}")
    
    # Load the model
    model = LightningHistoGPT.load_from_checkpoint(model_path)
    model.eval()
    
    # Create test dataloader
    from slide_level_dataset import SlideCollator
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=SlideCollator(),
        pin_memory=True
    )
    
    all_predictions = []
    all_probabilities = []
    all_targets = []
    
    with torch.no_grad():
        for batch in test_loader:
            # Move batch to device
            device = next(model.parameters()).device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # Get predictions
            _, logits = model.compute_loss(batch)
            binary_preds, binary_probs = model._get_binary_predictions(logits)
            binary_targets = model._get_binary_targets(batch)
            
            all_predictions.extend(binary_preds.cpu().tolist())
            all_probabilities.extend(binary_probs.cpu().tolist())
            all_targets.extend(binary_targets.cpu().tolist())
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(all_targets, all_predictions),
        'precision': precision_score(all_targets, all_predictions, average='binary'),
        'recall': recall_score(all_targets, all_predictions, average='binary'),
        'f1': f1_score(all_targets, all_predictions, average='binary'),
        'auc': roc_auc_score(all_targets, all_probabilities)
    }
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='5-Fold Cross Validation Training')
    parser.add_argument('--data-path', type=str, required=True,
                        help='Path to H5 files directory')
    parser.add_argument('--config', type=str, default='efficient',
                        choices=['quick', 'efficient', 'full'],
                        help='Training configuration preset')
    parser.add_argument('--output-dir', type=str, default='./cross_validation_results',
                        help='Output directory for results')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Batch size for training')
    parser.add_argument('--num-workers', type=int, default=0,
                        help='Number of workers for data loading')
    parser.add_argument('--max-steps', type=int, default=2000,
                        help='Maximum training steps per fold')
    parser.add_argument('--early-stopping-patience', type=int, default=10,
                        help='Early stopping patience')
    parser.add_argument('--n-folds', type=int, default=5,
                        help='Number of folds for cross validation')
    parser.add_argument('--random-seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Set random seeds
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get slide IDs and labels
    print("Loading slide information...")
    slide_ids, labels = get_slide_ids_and_labels(args.data_path)
    print(f"Found {len(slide_ids)} slides")
    print(f"Label distribution: {np.bincount(labels)}")
    
    # Create stratified K-fold splitter
    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.random_seed)
    
    # Load configuration
    config = FineTuningConfig.from_preset(args.config)
    config.batch_size = args.batch_size
    config.num_workers = args.num_workers
    config.max_steps = args.max_steps
    config.early_stopping_patience = args.early_stopping_patience
    config.output_dir = args.output_dir
    
    # Store cross-validation results
    cv_results = {
        'fold_metrics': [],
        'fold_paths': [],
        'config': config.to_dict()
    }
    
    # Perform cross validation
    for fold_idx, (train_indices, val_indices) in enumerate(skf.split(slide_ids, labels)):
        print(f"\nFold {fold_idx + 1}/{args.n_folds}")
        print(f"Train samples: {len(train_indices)}, Val samples: {len(val_indices)}")
        
        # Create fold datasets
        train_dataset, val_dataset = create_fold_datasets(
            args.data_path, train_indices, val_indices, slide_ids, config
        )
        
        # Train fold
        fold_metrics, best_model_path = train_fold(
            fold_idx, train_dataset, val_dataset, config, args.output_dir
        )
        
        # Evaluate fold
        test_metrics = evaluate_fold(best_model_path, val_dataset, config)
        
        # Store results
        fold_result = {
            'fold': fold_idx + 1,
            'train_indices': train_indices.tolist(),
            'val_indices': val_indices.tolist(),
            'val_metrics': fold_metrics,
            'test_metrics': test_metrics,
            'best_model_path': best_model_path
        }
        
        cv_results['fold_metrics'].append(fold_result)
        cv_results['fold_paths'].append(best_model_path)
        
        print(f"Fold {fold_idx + 1} Results:")
        print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"  Precision: {test_metrics['precision']:.4f}")
        print(f"  Recall: {test_metrics['recall']:.4f}")
        print(f"  F1: {test_metrics['f1']:.4f}")
        print(f"  AUC: {test_metrics['auc']:.4f}")
    
    # Calculate average metrics across folds
    avg_metrics = {}
    for metric_name in ['accuracy', 'precision', 'recall', 'f1', 'auc']:
        values = [fold['test_metrics'][metric_name] for fold in cv_results['fold_metrics']]
        avg_metrics[metric_name] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'values': values
        }
    
    cv_results['average_metrics'] = avg_metrics
    
    # Save results
    results_file = os.path.join(args.output_dir, 'cross_validation_results.json')
    with open(results_file, 'w') as f:
        json.dump(cv_results, f, indent=2)
    
    # Print summary
    print("\n" + "="*50)
    print("CROSS VALIDATION SUMMARY")
    print("="*50)
    for metric_name, stats in avg_metrics.items():
        print(f"{metric_name.upper()}: {stats['mean']:.4f} ± {stats['std']:.4f}")
    
    print(f"\nDetailed results saved to: {results_file}")
    print(f"Individual fold models saved in: {args.output_dir}")


if __name__ == "__main__":
    main()