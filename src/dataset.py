"""
src/dataset.py
==============

Kermany OCT veri seti için PyTorch Dataset sınıfı + augmentation pipeline'ları.

OCT'ye özel augmentation kararları (literatürden):
- HFlip ✓ (retina simetrisi koruyor)
- Vertical flip ✗ (retina katmanlarını ters çeviriyor — anatomik olarak yanlış)
- Rotasyon ±10° ✓ (büyük rotasyon B-scan oryantasyonunu bozar)
- CLAHE ✓ (OCT için kontrast iyileştirme)
- Brightness/Contrast ±15% ✓
- Gaussian noise (hafif) ✓
- Color jitter ✗ (görüntüler grayscale)
- Heavy elastic distortion ✗ (klinik bulguları distorte eder)
- CutMix DRUSEN için tehlikeli (küçük lezyonlar kaybolabilir) — şimdilik kapalı
"""

from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

from src.paths import resolve_image

CLASS_TO_IDX = {"CNV": 0, "DME": 1, "DRUSEN": 2, "NORMAL": 3}
IDX_TO_CLASS = {v: k for k, v in CLASS_TO_IDX.items()}

# ImageNet normalization (ImageNet-pretrained backbone'ler için)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class OCTDataset(Dataset):
    """
    Kermany OCT Dataset.

    Args:
        csv_path: train.csv / val.csv / test.csv yolu
        transform: albumentations Compose objesi
        image_size: hedef görüntü boyutu (default 224)
    """

    def __init__(self, csv_path, transform=None, image_size=224):
        self.df = pd.read_csv(csv_path, dtype={"patient_id": str})
        self.transform = transform
        self.image_size = image_size

        # Sınıf etiketlerini sayısallaştır
        self.df["label"] = self.df["class"].map(CLASS_TO_IDX)

        # Sanity check
        assert self.df["label"].notna().all(), "Bilinmeyen sınıf etiketi var"

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        filepath = row["filepath"]

        # OCT görüntüleri grayscale, ImageNet pretrained model 3-channel bekliyor
        img = Image.open(resolve_image(filepath)).convert("L")  # önce grayscale
        img = np.array(img)  # numpy array (H, W)

        # 3-channel'a kopyala (RGB pretrain ile uyumluluk)
        img = np.stack([img, img, img], axis=-1)  # (H, W, 3)

        if self.transform is not None:
            transformed = self.transform(image=img)
            img = transformed["image"]

        label = int(row["label"])

        return img, label

    def get_class_weights(self):
        """Sınıf dengesizliği için weighted loss'a inverse-frequency ağırlıkları."""
        class_counts = self.df["class"].value_counts()
        total = len(self.df)
        n_classes = len(CLASS_TO_IDX)
        weights = torch.zeros(n_classes)
        for cls, idx in CLASS_TO_IDX.items():
            count = class_counts.get(cls, 1)
            # PyTorch standart formül: weight_i = total / (n_classes * count_i)
            weights[idx] = total / (n_classes * count)
        return weights


def get_train_transform(image_size: int = 224, normalization: str = "imagenet"):
    """
    Eğitim için augmentation pipeline.

    OCT'ye özel: vertical flip yok, rotasyon ±10° max, CLAHE var.
    """
    if normalization == "imagenet":
        mean, std = IMAGENET_MEAN, IMAGENET_STD
    else:  # raw
        mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)

    return A.Compose([
        # 1. Geometrik
        A.LongestMaxSize(max_size=int(image_size * 1.15)),
        A.PadIfNeeded(min_height=int(image_size * 1.15),
                      min_width=int(image_size * 1.15),
                      border_mode=0, fill=0),
        A.RandomResizedCrop(size=(image_size, image_size),
                             scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=10, border_mode=0, p=0.5),

        # 2. Yoğunluk / kontrast
        A.OneOf([
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
            A.RandomBrightnessContrast(brightness_limit=0.15,
                                        contrast_limit=0.15, p=1.0),
        ], p=0.5),

        # 3. Gürültü (hafif)
        A.OneOf([
            A.GaussNoise(std_range=(0.03, 0.08), p=1.0),
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
        ], p=0.2),

        # 4. Normalize + tensor
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])


def get_val_transform(image_size: int = 224, normalization: str = "imagenet"):
    """Validation/test için: sadece resize + normalize, augmentation yok."""
    if normalization == "imagenet":
        mean, std = IMAGENET_MEAN, IMAGENET_STD
    else:
        mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)

    return A.Compose([
        A.LongestMaxSize(max_size=image_size),
        A.PadIfNeeded(min_height=image_size, min_width=image_size,
                      border_mode=0, fill=0),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])


def get_visualization_transform(image_size: int = 224):
    """Augmentation'ı görselleştirmek için (normalize/tensor yok)."""
    return A.Compose([
        A.LongestMaxSize(max_size=int(image_size * 1.15)),
        A.PadIfNeeded(min_height=int(image_size * 1.15),
                      min_width=int(image_size * 1.15),
                      border_mode=0, fill=0),
        A.RandomResizedCrop(size=(image_size, image_size),
                             scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=10, border_mode=0, p=0.5),
        A.OneOf([
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
            A.RandomBrightnessContrast(brightness_limit=0.15,
                                        contrast_limit=0.15, p=1.0),
        ], p=0.5),
    ])
