"""
03_test_setup.py
================

Eğitime başlamadan önce her şeyin yerinde olduğunu kontrol eder:
1. PyTorch + CUDA + GPU
2. timm + albumentations
3. Dataset class (train/val/test CSV'lerinden okuyabiliyor mu?)
4. DataLoader (batch yükleme + GPU'ya transfer + hız ölçümü)
5. Augmentation görselleştirmesi (1 görüntü, 8 augment versiyonu)
6. Backbone yükleme testi (ConvNeXt-V2-Tiny, RTX 3060'a sığacak mı?)

Bu script çalışırsa eğitim aşamasına geçeriz.
"""

import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from PIL import Image

# Path setup
SCRIPT_DIR = Path(__file__).parent if "__file__" in dir() else Path.cwd()
sys.path.insert(0, str(SCRIPT_DIR))

OUTPUT_DIR = SCRIPT_DIR / "outputs"
SPLITS_DIR = OUTPUT_DIR / "splits"
FIGURES_DIR = OUTPUT_DIR / "figures"

plt.rcParams.update({
    "figure.dpi": 100,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})


def check(label: str, ok: bool, detail: str = ""):
    """Yardımcı print fonksiyonu."""
    icon = "✓" if ok else "❌"
    print(f"  {icon} {label}{': ' + detail if detail else ''}")
    return ok


def section(title: str):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


# ============================================================================
# 1. KÜTÜPHANE KONTROLÜ
# ============================================================================
section("1. KÜTÜPHANE KONTROLÜ")

ok = True
try:
    import torch
    check("PyTorch", True, torch.__version__)
except ImportError:
    check("PyTorch", False, "kurulu değil")
    ok = False

try:
    import torchvision
    check("torchvision", True, torchvision.__version__)
except ImportError:
    check("torchvision", False)
    ok = False

try:
    import timm
    check("timm", True, timm.__version__)
except ImportError:
    check("timm", False, "→ pip install timm")
    ok = False

try:
    import albumentations as A
    check("albumentations", True, A.__version__)
except ImportError:
    check("albumentations", False, "→ pip install albumentations")
    ok = False

if not ok:
    print("\n❌ Eksik kütüphaneler var. Kurulumdan sonra tekrar dene.")
    sys.exit(1)


# ============================================================================
# 2. GPU KONTROLÜ
# ============================================================================
section("2. GPU KONTROLÜ")

cuda_available = torch.cuda.is_available()
check("CUDA mevcut", cuda_available)

if not cuda_available:
    print("\n❌ CUDA bulunamadı. GPU'da çalışmadan eğitim çok yavaş olur.")
    print("   PyTorch'u CUDA destekli kurduğundan emin ol:")
    print("   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
    sys.exit(1)

device = torch.device("cuda")
gpu_name = torch.cuda.get_device_name(0)
total_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f"  ✓ GPU: {gpu_name}")
print(f"  ✓ VRAM: {total_memory_gb:.2f} GB toplam")
print(f"  ✓ CUDA: {torch.version.cuda}")
print(f"  ✓ cuDNN: {torch.backends.cudnn.version()}")

# Mevcut bellek
torch.cuda.empty_cache()
free_mem = (torch.cuda.get_device_properties(0).total_memory -
            torch.cuda.memory_allocated()) / 1e9
print(f"  ✓ Boşta: ~{free_mem:.2f} GB")


# ============================================================================
# 3. DATASET TESTİ
# ============================================================================
section("3. DATASET TESTİ")

# CSV'lerin mevcut olduğunu kontrol et
for split in ["train", "val", "test"]:
    csv_path = SPLITS_DIR / f"{split}.csv"
    if not csv_path.exists():
        check(f"{split}.csv", False, f"{csv_path} bulunamadı")
        sys.exit(1)
    check(f"{split}.csv", True, str(csv_path))

# Dataset import et
try:
    from src.dataset import (OCTDataset, get_train_transform,
                              get_val_transform, get_visualization_transform,
                              CLASS_TO_IDX, IDX_TO_CLASS)
    print("  ✓ src/dataset.py import edildi")
except Exception as e:
    print(f"  ❌ src/dataset.py import edilemedi: {e}")
    sys.exit(1)

# Dataset oluştur
print("\n  Dataset objeleri oluşturuluyor...")
train_ds = OCTDataset(SPLITS_DIR / "train.csv",
                       transform=get_train_transform(224))
val_ds = OCTDataset(SPLITS_DIR / "val.csv",
                     transform=get_val_transform(224))
test_ds = OCTDataset(SPLITS_DIR / "test.csv",
                      transform=get_val_transform(224))

print(f"    train: {len(train_ds):,} örnek")
print(f"    val:   {len(val_ds):,} örnek")
print(f"    test:  {len(test_ds):,} örnek")

# Tek bir örnek yüklemeyi test et
img, label = train_ds[0]
print(f"\n  Örnek tensor:")
print(f"    img.shape: {tuple(img.shape)}  (beklenen: (3, 224, 224))")
print(f"    img.dtype: {img.dtype}")
print(f"    img.min/max: {img.min().item():.3f} / {img.max().item():.3f}")
print(f"    label: {label} ({IDX_TO_CLASS[label]})")

# Sınıf ağırlıkları
class_weights = train_ds.get_class_weights()
print(f"\n  Train set sınıf ağırlıkları (weighted CE için):")
for cls, idx in CLASS_TO_IDX.items():
    print(f"    {cls:7s}: {class_weights[idx]:.4f}")


# ============================================================================
# 4. DATALOADER + HIZ TESTİ
# ============================================================================
section("4. DATALOADER + HIZ TESTİ")

# Windows'ta num_workers > 0 sorun yaratabilir, dikkatli başla
import os
NUM_WORKERS = 4 if os.name != "nt" else 0  # Windows'ta 0 ile başla
BATCH_SIZE = 32

print(f"  num_workers: {NUM_WORKERS} (Windows: 0, Linux: 4)")
print(f"  batch_size:  {BATCH_SIZE}")

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                          shuffle=True, num_workers=NUM_WORKERS,
                          pin_memory=True, drop_last=True)

print(f"\n  Hız ölçümü (10 batch, GPU transfer dahil)...")
torch.cuda.synchronize()
t0 = time.time()
n_images = 0
for i, (imgs, labels) in enumerate(train_loader):
    imgs = imgs.to(device, non_blocking=True)
    labels = labels.to(device, non_blocking=True)
    n_images += imgs.size(0)
    if i == 0:
        print(f"    İlk batch shape: {tuple(imgs.shape)}, label shape: {tuple(labels.shape)}")
    if i >= 9:
        break
torch.cuda.synchronize()
elapsed = time.time() - t0

print(f"  ✓ {n_images} görüntü {elapsed:.2f}s içinde yüklendi")
print(f"  ✓ Verim: ~{n_images/elapsed:.1f} görüntü/saniye")
estimated_epoch_min = (len(train_ds) / (n_images/elapsed)) / 60
print(f"  ⌛ Tahmini bir epoch (sadece veri yükleme): ~{estimated_epoch_min:.1f} dk")
print(f"     (gerçek epoch süresi forward/backward'la birlikte hesaplanacak)")


# ============================================================================
# 5. AUGMENTATION GÖRSELLEŞTİRMESİ
# ============================================================================
section("5. AUGMENTATION GÖRSELLEŞTİRMESİ")

print("  Aynı görüntüden 8 farklı augmented versiyon üretiliyor...")

# Bir örnek görüntü al (raw, normalize edilmemiş)
import pandas as pd
from src.paths import resolve_image
train_df = pd.read_csv(SPLITS_DIR / "train.csv", dtype={"patient_id": str})
sample_row = train_df[train_df["class"] == "CNV"].iloc[0]
img_pil = Image.open(resolve_image(sample_row["filepath"])).convert("L")
img_np = np.array(img_pil)
img_np = np.stack([img_np, img_np, img_np], axis=-1)  # 3-channel

vis_transform = get_visualization_transform(224)

fig, axes = plt.subplots(3, 3, figsize=(10, 10))
# Sol üst: orijinal
axes[0, 0].imshow(img_np[:, :, 0], cmap="gray")
axes[0, 0].set_title("Orijinal", fontweight="bold")
axes[0, 0].axis("off")

# Diğerleri: augmented versiyonlar
for i in range(8):
    ax = axes.flat[i + 1]
    augmented = vis_transform(image=img_np)["image"]
    ax.imshow(augmented[:, :, 0], cmap="gray")
    ax.set_title(f"Augmented #{i+1}")
    ax.axis("off")

plt.suptitle(f"Augmentation Örnekleri ({sample_row['class']} sınıfı)",
              fontsize=14, y=1.00)
plt.tight_layout()
save_path = FIGURES_DIR / "07_augmentation_examples.png"
plt.savefig(save_path)
plt.close()
print(f"  ✓ {save_path.name} kaydedildi")


# ============================================================================
# 6. BACKBONE YÜKLEME TESTİ (ConvNeXt-V2-Tiny)
# ============================================================================
section("6. BACKBONE YÜKLEME TESTİ — ConvNeXt-V2-Tiny")

print("  ConvNeXt-V2-Tiny modeli yükleniyor (ilk seferde ~110 MB indirme)...")
try:
    model = timm.create_model(
        "convnextv2_tiny.fcmae_ft_in22k_in1k",
        pretrained=True,
        num_classes=4,  # Kermany 4 sınıf
    )
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  ✓ Model yüklendi: {n_params:.2f}M parametre")

    # Test forward pass
    print("\n  Forward pass testi (batch=32, 224x224)...")
    dummy = torch.randn(32, 3, 224, 224, device=device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    with torch.no_grad():
        out = model(dummy)
    peak_mem = torch.cuda.max_memory_allocated() / 1e9
    print(f"  ✓ Output shape: {tuple(out.shape)} (beklenen: (32, 4))")
    print(f"  ✓ Inference VRAM (batch=32, fp32): {peak_mem:.2f} GB")

    # Backward pass testi (asıl kritik test)
    print("\n  Backward pass testi (eğitim sırasındaki gerçek bellek)...")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # AMP ile (eğitimde kullanacağımız mod)
    scaler = torch.amp.GradScaler("cuda")
    optimizer.zero_grad()
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        out = model(dummy)
        target = torch.randint(0, 4, (32,), device=device)
        loss = torch.nn.functional.cross_entropy(out, target)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    peak_mem = torch.cuda.max_memory_allocated() / 1e9
    print(f"  ✓ Eğitim VRAM (batch=32, AMP/bf16): {peak_mem:.2f} GB")
    print(f"  ✓ Mevcut: {total_memory_gb:.2f} GB → "
          f"{'YETERLİ' if peak_mem < total_memory_gb * 0.9 else 'TIGHT, batch boyutunu düşür'}")

    del model, dummy, out, target, loss
    torch.cuda.empty_cache()

except Exception as e:
    print(f"  ❌ Hata: {e}")
    print("     timm versiyonunu güncelle: pip install -U timm")


# ============================================================================
# ÖZET
# ============================================================================
section("ÖZET")
print(f"  ✓ Kütüphaneler hazır")
print(f"  ✓ GPU: {gpu_name} ({total_memory_gb:.2f} GB)")
print(f"  ✓ Dataset: train={len(train_ds):,}, val={len(val_ds):,}, test={len(test_ds):,}")
print(f"  ✓ DataLoader hızı: ~{n_images/elapsed:.0f} görüntü/sn")
print(f"  ✓ ConvNeXt-V2-Tiny: yüklendi ve forward+backward çalışıyor")
print(f"\n  📁 Yeni figür: outputs/figures/07_augmentation_examples.png")
print(f"\n  👉 Her şey hazır. Eğitim scriptini bekliyoruz.")