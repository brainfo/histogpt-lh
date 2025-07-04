from train_cross_validation import get_slide_ids_and_labels
import numpy as np

slide_ids, labels = get_slide_ids_and_labels('../anne_data/512px_uni-vit-l-16_0.5mpp_0xdown_normal')
print(f'Total slides: {len(slide_ids)}')
print(f'Label distribution: {np.bincount(labels)}')
print(f'Class 0 (BCC): {np.sum(np.array(labels) == 0)}')
print(f'Class 1 (SCC): {np.sum(np.array(labels) == 1)}')

  # Check if you have enough samples for 5-fold CV
min_class_count = min(np.bincount(labels))
print(f'Minimum class count: {min_class_count}')
print(f'Can do 5-fold CV: {min_class_count >= 5}')