"""
Lightning Training Script for HistoGPT-L Fine-tuning
Combines the best paradigms from original HistoGPT with current needs
"""

import argparse
import logging
import os
from pathlib import Path

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

from fine_tune_config import FineTuningConfig, QUICK_TUNE_CONFIG, FULL_TUNE_CONFIG, EFFICIENT_TUNE_CONFIG
from lightning_trainer import LightningHistoGPT, HistoGPTDataModule, create_lightning_trainer


def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('training.log')
        ]
    )


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train HistoGPT-L with PyTorch Lightning")
    
    # Configuration
    parser.add_argument(
        "--config", 
        type=str, 
        choices=["quick", "full", "efficient", "custom"],
        default="quick",
        help="Training configuration preset"
    )
    
    # Data paths
    parser.add_argument(
        "--train-data", 
        type=str,
        default="../anne_data/512px_uni-vit-l-16_0.5mpp_0xdown_normal",
        help="Path to training data directory"
    )
    parser.add_argument(
        "--val-data", 
        type=str,
        default="../anne_data/512px_uni-vit-l-16_0.5mpp_0xdown_normal",
        help="Path to validation data directory"
    )
    parser.add_argument(
        "--test-data", 
        type=str,
        help="Path to test data directory (optional)"
    )
    
    # Training parameters
    parser.add_argument("--batch-size", type=int, default=2, help="Training batch size")
    parser.add_argument("--max-steps", type=int, help="Maximum training steps")
    parser.add_argument("--learning-rate", type=float, help="Learning rate")
    parser.add_argument("--num-workers", type=int, default=0, help="Number of data loader workers")
    
    # Model configuration
    parser.add_argument("--max-patches", type=int, default=1000, help="Maximum patches per slide")
    parser.add_argument("--pretrained-path", type=str, help="Path to pretrained model weights")
    
    # Output
    parser.add_argument("--output-dir", type=str, default="./checkpoints", help="Output directory")
    parser.add_argument("--experiment-name", type=str, default="histogpt-finetune", help="Experiment name")
    
    # Logging
    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--use-wandb", action="store_true", help="Use Weights & Biases logging")
    parser.add_argument("--wandb-project", type=str, default="histogpt-lightning", help="W&B project name")
    
    # Resume training
    parser.add_argument("--resume", type=str, help="Path to checkpoint to resume from")
    
    # Testing
    parser.add_argument("--test-only", action="store_true", help="Only run testing")
    parser.add_argument("--test-checkpoint", type=str, help="Checkpoint to test")
    
    return parser.parse_args()


def get_config(args) -> FineTuningConfig:
    """Get configuration based on arguments"""
    
    # Load preset configuration
    if args.config == "quick":
        config = QUICK_TUNE_CONFIG
    elif args.config == "full":
        config = FULL_TUNE_CONFIG
    elif args.config == "efficient":
        config = EFFICIENT_TUNE_CONFIG
    else:
        config = FineTuningConfig()
    
    # Override with command line arguments
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.max_steps:
        config.max_steps = args.max_steps
    if args.learning_rate:
        config.learning_rate = args.learning_rate
    if args.pretrained_path:
        config.pretrained_model_path = args.pretrained_path
    if args.output_dir:
        config.output_dir = args.output_dir
    
    # Set data paths
    config.train_data_path = args.train_data
    config.val_data_path = args.val_data
    if args.test_data:
        config.test_data_path = args.test_data
    
    return config


def setup_logger(args, config):
    """Setup experiment logger"""
    loggers = []
    
    # TensorBoard logger
    tb_logger = TensorBoardLogger(
        save_dir=config.logging_dir,
        name=args.experiment_name,
        version=None
    )
    loggers.append(tb_logger)
    
    # Weights & Biases logger
    if args.use_wandb:
        try:
            wandb_logger = WandbLogger(
                name=args.experiment_name,
                project=args.wandb_project,
                save_dir=config.logging_dir,
                config=config.__dict__
            )
            loggers.append(wandb_logger)
        except ImportError:
            logging.warning("Weights & Biases not available. Skipping W&B logging.")
    
    return loggers


def main():
    """Main training function"""
    args = parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logging.info("Starting HistoGPT-L Lightning training")
    
    # Get configuration
    config = get_config(args)
    logging.info(f"Using configuration: {args.config}")
    logging.info(f"Training steps: {config.max_steps}")
    logging.info(f"Batch size: {config.batch_size}")
    logging.info(f"Learning rate: {config.learning_rate}")
    
    # Create output directories
    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(config.logging_dir, exist_ok=True)
    
    # Setup data module
    logging.info("Setting up data module")
    data_module = HistoGPTDataModule(
        train_data_path=config.train_data_path,
        val_data_path=config.val_data_path,
        test_data_path=config.test_data_path,
        batch_size=config.batch_size,
        num_workers=args.num_workers,
        tokenizer_name=config.language_model_name,
        max_text_length=config.max_sequence_length,
        max_patches_per_slide=args.max_patches
    )
    
    # Setup model
    logging.info("Setting up model")
    model = LightningHistoGPT(config)
    
    # Setup loggers
    loggers = setup_logger(args, config)
    
    # Create trainer
    logging.info("Creating Lightning trainer")
    trainer = create_lightning_trainer(config, loggers)
    
    if args.test_only:
        # Test only mode
        if not args.test_checkpoint:
            logging.error("Test checkpoint must be specified for test-only mode")
            return
        
        logging.info(f"Testing model from checkpoint: {args.test_checkpoint}")
        trainer.test(model, datamodule=data_module, ckpt_path=args.test_checkpoint)
        
    else:
        # Training mode
        logging.info("Starting training")
        
        # Print dataset statistics
        data_module.setup("fit")
        logging.info(f"Training slides: {len(data_module.train_dataset)}")
        logging.info(f"Validation slides: {len(data_module.val_dataset)}")
        
        # Print model statistics
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logging.info(f"Total parameters: {total_params:,}")
        logging.info(f"Trainable parameters: {trainable_params:,}")
        logging.info(f"Trainable ratio: {trainable_params/total_params:.2%}")
        
        # Start training
        if args.resume:
            logging.info(f"Resuming training from: {args.resume}")
            trainer.fit(model, datamodule=data_module, ckpt_path=args.resume)
        else:
            trainer.fit(model, datamodule=data_module)
        
        # Test on best model if test data available
        if config.test_data_path:
            logging.info("Testing best model")
            trainer.test(datamodule=data_module, ckpt_path="best")
    
    logging.info("Training completed!")


if __name__ == "__main__":
    main()