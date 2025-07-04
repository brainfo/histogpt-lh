"""
Fine-tuning Configuration for HistoGPT-L
© Modified for Hyperparameter Tuning
"""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class FineTuningConfig:
    """Configuration for fine-tuning HistoGPT-L model with PyTorch Lightning"""
    
    # Model Architecture Hyperparameters
    aggregator_d_model: int = 1536  # Aggregator hidden dimension
    aggregator_n_heads: int = 16    # Number of attention heads
    aggregator_n_layers: int = 6    # Number of Perceiver layers
    aggregator_n_latents: int = 640 # Number of latent tokens
    aggregator_attn_drop: float = 0.0  # Attention dropout
    
    # Model Paths (local paths for offline training)
    vision_model_name: str = "uni-vit-l-16"
    language_model_name: str = "../microsoft_biogpt-large"  # Local path
    pretrained_model_path: Optional[str] = "../histogpt-l-6k-pruned.pt"
    offline_mode: bool = True  # Force offline mode for transformers
    
    # Positional Embedding Hyperparameters
    pos_embed_max_len: int = 1000   # Maximum sequence length
    pos_embed_d_model: int = 1024   # Position embedding dimension
    
    # Training Hyperparameters (Lightning compatible)
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    betas: tuple = (0.9, 0.999)      # Adam betas from original trainer
    warmup_steps: int = 1000
    max_steps: int = 10000
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    
    # Learning Rate Scheduler
    scheduler: str = "warmup_cosine"  # "warmup_cosine", "linear_warmup", "cosine"
    min_lr: float = 1e-6             # Minimum learning rate
    end_lr: float = 1e-7             # End learning rate for cosine annealing
    
    # Fine-tuning Strategy
    freeze_vision_encoder: bool = True      # Keep UNI frozen
    freeze_language_model: bool = False     # Allow BioGPT fine-tuning
    freeze_aggregator: bool = False         # Allow aggregator training
    train_cross_attention: bool = True      # Train cross-attention layers
    train_aggregator: bool = True           # Train aggregator module
    train_projection: bool = True           # Train vision-language projection
    train_positional_embedding: bool = True # Train positional embeddings
    
    # LoRA Configuration (for efficient fine-tuning)
    use_lora: bool = True
    lora_rank: int = 16
    lora_alpha: float = 32
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = None
    
    # Data Configuration
    train_data_path: str = "../anne_data/512px_uni-vit-l-16_0.5mpp_0xdown_normal"
    val_data_path: str = "../anne_data/512px_uni-vit-l-16_0.5mpp_0xdown_normal"
    test_data_path: Optional[str] = None
    max_sequence_length: int = 512
    max_patches_per_slide: int = 1000
    
    # Optimization
    optimizer: str = "adamw"
    gradient_checkpointing: bool = True
    mixed_precision: bool = True
    
    # Lightning-specific configurations
    early_stopping_patience: Optional[int] = 5  # Early stopping patience
    task: str = "binary"                         # For metrics (binary classification)
    num_classes: int = 2                         # Number of diagnosis classes
    
    # Cross-validation configuration
    n_folds: int = 5                             # Number of folds for cross validation
    random_seed: int = 42                        # Random seed for reproducibility
    
    # Data loading
    num_workers: int = 0                         # Number of workers for data loading
    tokenizer_name: str = "../microsoft_biogpt-large"  # Tokenizer path
    max_text_length: int = 512                   # Maximum text length
    
    # Logging and Checkpointing
    log_steps: int = 100
    eval_steps: int = 500
    save_steps: int = 1000
    save_total_limit: int = 3
    output_dir: str = "./checkpoints"
    logging_dir: str = "./logs"
    
    def __post_init__(self):
        if self.lora_target_modules is None:
            self.lora_target_modules = [
                "q_proj", "v_proj", "k_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]
    
    @classmethod
    def from_preset(cls, preset_name: str):
        """Create configuration from preset name"""
        if preset_name == "quick":
            return QUICK_TUNE_CONFIG
        elif preset_name == "efficient":
            return EFFICIENT_TUNE_CONFIG
        elif preset_name == "full":
            return FULL_TUNE_CONFIG
        else:
            raise ValueError(f"Unknown preset: {preset_name}. Available: quick, efficient, full")
    
    def to_dict(self):
        """Convert configuration to dictionary"""
        return {
            field.name: getattr(self, field.name)
            for field in self.__dataclass_fields__.values()
        }


# Predefined configurations for different fine-tuning scenarios
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