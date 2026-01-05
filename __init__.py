"""
LPR Baseline - Multi-Frame License Plate Recognition

A modular implementation of CRNN with attention-based temporal fusion
for license plate recognition from multi-frame sequences.
"""

from .config import Config
from .transforms import get_train_transforms, get_val_transforms, get_degradation_transforms
from .dataset import AdvancedMultiFrameDataset
from .utils import seed_everything, decode_predictions
from .models import AttentionFusion, MultiFrameCRNN

__version__ = "1.0.0"
__all__ = [
    'Config',
    'get_train_transforms',
    'get_val_transforms', 
    'get_degradation_transforms',
    'AdvancedMultiFrameDataset',
    'seed_everything',
    'decode_predictions',
    'AttentionFusion',
    'MultiFrameCRNN',
]
