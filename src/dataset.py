import os
import pickle
import numpy as np
from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

# ─── Optional OpenCV (for CLAHE) ───────────────────────────────────────────
try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False


def _lime_preprocess(img_np: np.ndarray, sigma: float = 5.0) -> np.ndarray:
    
    from scipy.ndimage import gaussian_filter

    img_f = img_np.astype(np.float32) / 255.0
    illum = img_f.max(axis=2)
    illum_smooth = gaussian_filter(illum, sigma=sigma)
    illum_smooth = np.clip(illum_smooth, 1e-3, 1.0)
    enhanced = img_f / illum_smooth[:, :, np.newaxis]
    enhanced = np.clip(enhanced, 0.0, 1.0)
    return (enhanced * 255).astype(np.uint8)


def _clahe_preprocess(img_np: np.ndarray,
                      clip_limit: float = 2.0,
                      tile_grid: tuple = (8, 8)) -> np.ndarray:
    
    if not _CV2_AVAILABLE:
        raise RuntimeError("cv2 is required for CLAHE preprocessing. "
                           "Install with: pip install opencv-python")
    img_bgr  = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    img_lab  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b  = cv2.split(img_lab)
    clahe    = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    l_eq     = clahe.apply(l)
    img_lab2 = cv2.merge([l_eq, a, b])
    img_bgr2 = cv2.cvtColor(img_lab2, cv2.COLOR_LAB2BGR)
    return cv2.cvtColor(img_bgr2, cv2.COLOR_BGR2RGB)



_SDNET_SPLITS = {
    'D': ('UD', 'CD'),   # Bridge decks
    'P': ('UP', 'CP'),   # Pavement
    'W': ('UW', 'CW'),   # Walls
}


def build_image_index(
    dataset_dir: str,
    surface_types: list = None,
) -> list:
    
    if surface_types is None:
        surface_types = ['D', 'P', 'W']

    records = []
    for surface in surface_types:
        uncracked_dir, cracked_dir = _SDNET_SPLITS[surface]
        for label, subdir in [(0, uncracked_dir), (1, cracked_dir)]:
            folder = os.path.join(dataset_dir, surface, subdir)
            if not os.path.isdir(folder):
                print(f"  Warning: {folder} not found, skipping.")
                continue
            for fname in sorted(os.listdir(folder)):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    records.append({
                        'image_path': os.path.join(folder, fname),
                        'label': label,
                        'surface': surface,
                    })

    return records



_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]

_TO_TENSOR = T.ToTensor()
_NORMALISE = T.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD)
_AUGMENT   = T.Compose([
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
    T.RandomRotation(15),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
])


class CrackDataset(Dataset):
    

    def __init__(
        self,
        records: list,
        preprocessing: str = 'none',
        augment: bool = False,
        normal_only: bool = False,
    ):
        assert preprocessing in ('lime', 'clahe', 'none'), \
            "preprocessing must be 'lime', 'clahe', or 'none'"

        if normal_only:
            records = [r for r in records if r['label'] == 0]

        self.records       = records
        self.preprocessing = preprocessing
        self.augment       = augment

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        rec = self.records[idx]

        with Image.open(rec['image_path']) as img:
            img_np = np.array(img.convert('RGB'))   # (256, 256, 3) uint8

        # Apply preprocessing
        if self.preprocessing == 'lime':
            img_np = _lime_preprocess(img_np)
        elif self.preprocessing == 'clahe':
            img_np = _clahe_preprocess(img_np)

        img_pil = Image.fromarray(img_np)

        # Optional augmentation
        if self.augment:
            img_pil = _AUGMENT(img_pil)

        tensor = _NORMALISE(_TO_TENSOR(img_pil))   # (3, 256, 256)
        label  = torch.tensor(rec['label'], dtype=torch.long)
        return tensor, label


def save_splits(train_records: list, val_records: list, test_records: list,
                splits_dir: str) -> None:
    os.makedirs(splits_dir, exist_ok=True)
    for name, records in [('train', train_records),
                          ('val',   val_records),
                          ('test',  test_records)]:
        path = os.path.join(splits_dir, f'patch_index_{name}.pkl')
        with open(path, 'wb') as f:
            pickle.dump(records, f)
    print(f"Splits saved to {splits_dir}/")
    print(f"  train: {len(train_records):,} images")
    print(f"  val:   {len(val_records):,} images")
    print(f"  test:  {len(test_records):,} images")


def load_splits(splits_dir: str) -> tuple:
    splits = {}
    for name in ('train', 'val', 'test'):
        path = os.path.join(splits_dir, f'patch_index_{name}.pkl')
        with open(path, 'rb') as f:
            splits[name] = pickle.load(f)
    return splits['train'], splits['val'], splits['test']