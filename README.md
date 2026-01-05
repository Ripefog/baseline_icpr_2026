# LPR Baseline

Multi-Frame License Plate Recognition using CRNN with Attention-based Temporal Fusion.

## Features

- **Multi-frame input**: Uses 5 frames per sample for robust recognition
- **Attention Fusion**: Learnable attention weights for temporal feature fusion
- **Synthetic Degradation**: Augments HR images with blur, noise, and compression
- **CTC Loss**: Handles variable-length license plate text

## Project Structure

```
lpr_baseline/
├── __init__.py         # Package exports
├── config.py           # Configuration
├── transforms.py       # Data augmentation
├── dataset.py          # Dataset class
├── utils.py            # Utilities
├── train.py            # Training script
├── requirements.txt
└── models/
    ├── __init__.py
    ├── fusion.py       # AttentionFusion
    └── crnn.py         # MultiFrameCRNN
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

1. **Configure data path** in `config.py`:
   ```python
   DATA_ROOT = "path/to/your/data"
   ```

2. **Run training**:
   ```bash
   cd lpr_baseline
   python train.py
   ```

## Data Format

Expected directory structure:
```
data/train/
├── Scenario-A/
│   └── track_00001/
│       ├── annotations.json    # {"plate_text": "ABC123"}
│       ├── lr-00.png          # Low-resolution frames
│       ├── lr-01.png
│       └── hr-00.png          # High-resolution frames (optional)
```

## Model Architecture

```
Input [B, 5, 3, 32, 128]
    ↓
CNN Backbone (7 layers) → [B*5, 512, 1, W]
    ↓
AttentionFusion → [B, 512, W]
    ↓
BiLSTM (2 layers) → [B, W, 512]
    ↓
FC + LogSoftmax → [B, W, 38]
```

## License

MIT
