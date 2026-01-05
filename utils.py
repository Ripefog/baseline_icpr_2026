import os
import random

import numpy as np
import torch


def seed_everything(seed=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"🔒 Đã cố định Seed: {seed}")


def decode_predictions(preds, idx2char):
    """
    Decode CTC predictions to text strings.
    
    Args:
        preds: Tensor of shape [B, T] containing predicted character indices
        idx2char: Dictionary mapping indices to characters
    
    Returns:
        List of decoded strings
    """
    result_list = []
    for p in preds:
        pred_str = ""
        last_char = 0
        for char_idx in p:
            c = char_idx.item()
            if c != 0 and c != last_char:
                pred_str += idx2char[c]
            last_char = c
        result_list.append(pred_str)
    return result_list
