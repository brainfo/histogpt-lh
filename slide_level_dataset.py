"""
Slide-Level Dataset for HistoGPT Fine-tuning
Groups patches by slide for proper MIL training
"""

import h5py
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict, List, Tuple
import logging
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


class SlideLevelDataset(Dataset):
    """Dataset that groups patches by slide for Multiple Instance Learning"""
    
    def __init__(
        self,
        data_path: str,
        tokenizer_name: str = "../microsoft_biogpt-large",
        max_text_length: int = 512,
        max_patches_per_slide: int = 1000,  # Limit patches per slide
        offline_mode: bool = True,  # Force offline mode
    ):
        self.data_path = Path(data_path)
        
        # Force offline mode for transformers
        if offline_mode:
            import os
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            os.environ["HF_DATASETS_OFFLINE"] = "1"
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            local_files_only=offline_mode
        )
        self.max_text_length = max_text_length
        self.max_patches_per_slide = max_patches_per_slide
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.slides = self._load_slides()
        
    def _load_slides(self) -> List[Dict]:
        """Load slides with their patches grouped together"""
        slides = []
        
        # Get all H5 files
        h5_files = list(self.data_path.rglob('*.h5'))
        
        for h5_path in h5_files:
            slide_data = self._load_slide_data(h5_path)
            if slide_data:
                slides.append(slide_data)
        
        logger.info(f"Loaded {len(slides)} slides from {self.data_path}")
        return slides
    
    def _load_slide_data(self, h5_path: Path) -> Dict:
        """Load all patches from a single slide"""
        try:
            with h5py.File(h5_path, 'r') as f:
                if 'features' not in f:
                    return None
                
                features = f['features'][:]
                coordinates = f.get('coordinates', None)
                
                # Limit number of patches per slide
                if len(features) > self.max_patches_per_slide:
                    # Random sampling of patches
                    indices = np.random.choice(
                        len(features), 
                        self.max_patches_per_slide, 
                        replace=False
                    )
                    features = features[indices]
                    if coordinates is not None:
                        coordinates = coordinates[indices]
                
                # Extract diagnosis from filename
                diagnosis = self._extract_diagnosis_from_filename(h5_path.stem)
                target_text = f"Final diagnosis: {diagnosis}"
                
                return {
                    'slide_id': h5_path.stem,
                    'features': torch.tensor(features, dtype=torch.float32),
                    'coordinates': torch.tensor(coordinates, dtype=torch.float32) if coordinates is not None else None,
                    'diagnosis': diagnosis,
                    'text': target_text,
                    'num_patches': len(features)
                }
                
        except Exception as e:
            logger.warning(f"Failed to load {h5_path}: {e}")
            return None
    
    def _extract_diagnosis_from_filename(self, filename: str) -> str:
        """Extract diagnosis from filename - same logic as original"""
        filename = filename.lower()
        
        disease_patterns = {
            'sbbc': 'basal cell carcinoma',
            'ibbc': 'basal cell carcinoma',
            'pek': 'squamous cell carcinoma'
        }
        
        for file_code, diagnosis in disease_patterns.items():
            if file_code.lower() in filename:
                return diagnosis
        
        parts = filename.replace('-', '_').split('_')
        for part in parts:
            for file_code, diagnosis in disease_patterns.items():
                if file_code.lower() in part:
                    return diagnosis
        
        return "unknown_pathology"
    
    def __len__(self) -> int:
        return len(self.slides)
    
    def __getitem__(self, idx: int) -> Dict:
        slide = self.slides[idx]
        
        # Tokenize text
        text_encoding = self.tokenizer(
            slide['text'],
            max_length=self.max_text_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'slide_id': slide['slide_id'],
            'image_features': slide['features'],  # Shape: [num_patches, feature_dim]
            'coordinates': slide['coordinates'],  # Shape: [num_patches, 3] or None
            'input_ids': text_encoding['input_ids'].squeeze(0),
            'attention_mask': text_encoding['attention_mask'].squeeze(0),
            'text': slide['text'],
            'diagnosis': slide['diagnosis'],
            'num_patches': slide['num_patches']
        }


class SlideCollator:
    """Custom collator for variable-length slide data"""
    
    def __init__(self, pad_token_id: int = 0):
        self.pad_token_id = pad_token_id
    
    def __call__(self, batch: List[Dict]) -> Dict:
        """Collate batch of slides with variable number of patches"""
        
        # Stack text data (fixed length)
        input_ids = torch.stack([item['input_ids'] for item in batch])
        attention_mask = torch.stack([item['attention_mask'] for item in batch])
        
        # Handle variable-length image features
        # Pack as list since each slide has different number of patches
        image_features = [item['image_features'] for item in batch]
        coordinates = [item['coordinates'] for item in batch if item['coordinates'] is not None]
        
        # Collect metadata
        slide_ids = [item['slide_id'] for item in batch]
        texts = [item['text'] for item in batch]
        diagnoses = [item['diagnosis'] for item in batch]
        num_patches = [item['num_patches'] for item in batch]
        
        return {
            'slide_ids': slide_ids,
            'image_features': image_features,  # List of tensors
            'coordinates': coordinates if coordinates else None,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'texts': texts,
            'diagnoses': diagnoses,
            'num_patches': num_patches
        }


def create_slide_dataloader(
    data_path: str,
    batch_size: int = 4,
    shuffle: bool = True,
    **kwargs
) -> DataLoader:
    """Create DataLoader for slide-level training"""
    
    dataset = SlideLevelDataset(data_path, **kwargs)
    collator = SlideCollator()
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collator,
        num_workers=0  # Avoid issues with H5 files
    )


if __name__ == "__main__":
    # Test the dataset
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Test slide-level loading
    dataloader = create_slide_dataloader(
        "../anne_data/512px_uni-vit-l-16_0.5mpp_0xdown_normal",
        batch_size=2
    )
    
    print(f"Dataset has {len(dataloader.dataset)} slides")
    
    for batch in dataloader:
        print(f"Batch with {len(batch['slide_ids'])} slides:")
        for i, slide_id in enumerate(batch['slide_ids']):
            print(f"  {slide_id}: {batch['num_patches'][i]} patches, diagnosis: {batch['diagnoses'][i]}")
        break