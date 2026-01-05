import albumentations as A
from albumentations.pytorch import ToTensorV2

from .config import Config


def get_train_transforms():
    """Get training data augmentation transforms."""
    return A.Compose([
        A.Resize(height=Config.IMG_HEIGHT, width=Config.IMG_WIDTH),
        
        # Geometric transforms
        A.Affine(
            scale=(0.95, 1.05), 
            translate_percent=(0.05, 0.05), 
            rotate=(-5, 5), 
            p=0.5, 
            fill=128
        ),
        A.Perspective(scale=(0.02, 0.05), p=0.3),
        
        # Color transforms
        A.RandomBrightnessContrast(p=0.5),
        A.HueSaturationValue(
            hue_shift_limit=10, 
            sat_shift_limit=20, 
            val_shift_limit=20, 
            p=0.3
        ),
        
        # Dropout
        A.CoarseDropout(
            num_holes_range=(1, 3), 
            hole_height_range=(4, 8), 
            hole_width_range=(4, 8), 
            p=0.3
        ),
        
        # Normalization
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2()
    ])


def get_degradation_transforms():
    """Get synthetic degradation transforms to simulate LR images from HR."""
    return A.Compose([
        # Blur
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
            A.MotionBlur(blur_limit=(3, 7), p=1.0),
            A.Defocus(radius=(1, 3), alias_blur=(0.1, 0.3), p=1.0),
        ], p=0.8),
        
        # Noise
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0),
            A.MultiplicativeNoise(multiplier=(0.9, 1.1), p=1.0),
        ], p=0.8),
        
        # Compression artifacts
        A.ImageCompression(quality_range=(10, 50), p=0.5),
        A.Downscale(scale_range=(0.25, 0.5), p=0.5),
    ])


def get_val_transforms():
    """Get validation/inference transforms (no augmentation)."""
    return A.Compose([
        A.Resize(height=Config.IMG_HEIGHT, width=Config.IMG_WIDTH),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2()
    ])
