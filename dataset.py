import os
import glob
import json
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

try:
    from .config import Config
    from .transforms import get_train_transforms, get_val_transforms, get_degradation_transforms
except ImportError:
    from config import Config
    from transforms import get_train_transforms, get_val_transforms, get_degradation_transforms


class AdvancedMultiFrameDataset(Dataset):
    """
    Multi-frame license plate dataset.
    
    Loads tracks containing multiple LR/HR frames and their annotations.
    For training, randomly switches between LR images and HR+degradation.
    """
    
    def __init__(self, root_dir, mode='train', split_ratio=0.9):
        """
        Args:
            root_dir: Path to data directory containing track_* folders
            mode: 'train' or 'val'
            split_ratio: Ratio of data to use for training
        """
        self.mode = mode
        self.samples = []
        
        if mode == 'train':
            self.transform = get_train_transforms()
            self.degrade = get_degradation_transforms()
        else:
            self.transform = get_val_transforms()
            self.degrade = None

        print(f"[{mode.upper()}] Scanning: {root_dir}")
        abs_root = os.path.abspath(root_dir)
        search_path = os.path.join(abs_root, "**", "track_*")
        all_tracks = sorted(glob.glob(search_path, recursive=True))
        
        if not all_tracks:
            print("❌ LỖI: Không tìm thấy data.")
            return

        train_tracks, val_tracks = self._split_tracks(all_tracks, split_ratio)
        
        selected_tracks = train_tracks if mode == 'train' else val_tracks
        print(f"[{mode.upper()}] Loaded {len(selected_tracks)} tracks.")
        
        self._load_samples(selected_tracks)
    
    def _split_tracks(self, all_tracks, split_ratio):
        """Split tracks into train/val sets, persisting to JSON for reproducibility."""
        train_tracks = []
        val_tracks = []
        
        if os.path.exists(Config.VAL_SPLIT_FILE):
            print(f"📂 Loading split from '{Config.VAL_SPLIT_FILE}'...")
            try:
                with open(Config.VAL_SPLIT_FILE, 'r') as f:
                    val_ids = set(json.load(f))
            except:
                val_ids = set()
                print("⚠️ Lỗi đọc file split, sẽ tạo lại.")

            for t in all_tracks:
                track_name = os.path.basename(t)
                if track_name in val_ids:
                    val_tracks.append(t)
                else:
                    train_tracks.append(t)
            
            if not val_tracks and len(all_tracks) > 0:
                print("⚠️ File split không khớp với dữ liệu hiện tại. Chia lại...")
                train_tracks, val_tracks = self._create_new_split(all_tracks, split_ratio)
        else:
            print("⚠️ Creating new split...")
            train_tracks, val_tracks = self._create_new_split(all_tracks, split_ratio)
        
        return train_tracks, val_tracks
    
    def _create_new_split(self, all_tracks, split_ratio):
        """Create a new train/val split and save to JSON."""
        random.Random(Config.SEED).shuffle(all_tracks)
        split_idx = int(len(all_tracks) * split_ratio)
        train_tracks = all_tracks[:split_idx]
        val_tracks = all_tracks[split_idx:]
        
        val_ids = [os.path.basename(t) for t in val_tracks]
        with open(Config.VAL_SPLIT_FILE, 'w') as f:
            json.dump(val_ids, f, indent=2)
        
        return train_tracks, val_tracks
    
    def _load_samples(self, tracks):
        """Load sample metadata from track directories."""
        for track_path in tqdm(tracks, desc=f"Indexing {self.mode}"):
            json_path = os.path.join(track_path, "annotations.json")
            if not os.path.exists(json_path):
                continue
            
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
                if isinstance(data, list):
                    data = data[0]
                
                label = data.get('plate_text', data.get('license_plate', data.get('text', '')))
                if not label:
                    continue

                lr_files = sorted(
                    glob.glob(os.path.join(track_path, "lr-*.png")) + 
                    glob.glob(os.path.join(track_path, "lr-*.jpg"))
                )
                hr_files = sorted(
                    glob.glob(os.path.join(track_path, "hr-*.png")) + 
                    glob.glob(os.path.join(track_path, "hr-*.jpg"))
                )
                
                if len(lr_files) > 0:
                    self.samples.append({
                        'lr_paths': lr_files,
                        'hr_paths': hr_files,
                        'label': label
                    })
            except:
                pass

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        label = item['label']
        
        # During training, 50% chance to use HR images with degradation
        use_hr = (self.mode == 'train') and (len(item['hr_paths']) > 0) and (random.random() < 0.5)
        
        if use_hr:
            images_list = self._load_frames(item['hr_paths'], apply_degradation=True)
        else:
            images_list = self._load_frames(item['lr_paths'], apply_degradation=False)

        images_tensor = torch.stack(images_list, dim=0)
        target = [Config.CHAR2IDX[c] for c in label if c in Config.CHAR2IDX]
        if len(target) == 0:
            target = [0]
            
        return images_tensor, torch.tensor(target, dtype=torch.long), len(target), label
    
    def _load_frames(self, paths, apply_degradation=False):
        """Load and process frames, padding to 5 frames if needed."""
        # Ensure exactly 5 frames
        if len(paths) < 5:
            paths = paths + [paths[-1]] * (5 - len(paths))
        else:
            paths = paths[:5]
        
        images_list = []
        for p in paths:
            image = cv2.imread(p)
            if image is None:
                image = np.zeros((Config.IMG_HEIGHT, Config.IMG_WIDTH, 3), dtype=np.uint8)
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            if apply_degradation and self.degrade:
                image = self.degrade(image=image)['image']
            
            image = self.transform(image=image)['image']
            images_list.append(image)
        
        return images_list

    @staticmethod
    def collate_fn(batch):
        """Collate function for DataLoader."""
        images, targets, target_lengths, labels_text = zip(*batch)
        images = torch.stack(images, 0)
        targets = torch.cat(targets)
        target_lengths = torch.tensor(target_lengths, dtype=torch.long)
        return images, targets, target_lengths, labels_text
