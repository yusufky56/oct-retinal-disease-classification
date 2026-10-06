"""
04_train_convnext.py
====================

ConvNeXt-V2-Tiny ile Kermany OCT 4-sınıf sınıflandırma eğitimi.

Q1 yayını için tasarım kararları:
- ImageNet-22k → ImageNet-1k pretrained (FCMAE+supervised)
- Weighted Cross Entropy + label smoothing 0.1
- AdamW + cosine annealing + 5 epoch linear warmup
- AMP bf16 (RTX 3060 Ampere'de native)
- Val macro-F1'e göre best model checkpoint
- Patient-disjoint splitler üzerinde (Tampu 2022'ye uygun)
- 30 epoch (early stopping yok, ablation sonrası ekleyebiliriz)

Output:
- outputs/runs/convnext_<timestamp>/
  ├── checkpoint_best.pt          (val macro-F1 best)
  ├── checkpoint_last.pt          (son epoch)
  ├── training_log.csv            (her epoch metrikleri)
  ├── config.json                 (kullanılan hiperparametreler)
  ├── test_results.json           (final test metrikleri)
  ├── confusion_matrix.png        (test set için)
  └── training_curves.png         (loss, acc, F1 eğrileri)
"""

import os
import sys
import json
import time
import random
import argparse
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import timm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, f1_score,
                              precision_recall_fscore_support,
                              confusion_matrix, classification_report,
                              cohen_kappa_score)

# Path setup
SCRIPT_DIR = Path(__file__).parent if "__file__" in dir() else Path.cwd()
sys.path.insert(0, str(SCRIPT_DIR))

from src.dataset import (OCTDataset, get_train_transform, get_val_transform,
                          CLASS_TO_IDX, IDX_TO_CLASS)


# ============================================================================
# CONFIG
# ============================================================================

@dataclass
class Config:
    # Model
    backbone: str = "convnextv2_tiny.fcmae_ft_in22k_in1k"
    pretrained: bool = True
    num_classes: int = 4
    image_size: int = 224

    # Training
    epochs: int = 30
    batch_size: int = 32
    num_workers: int = 2  # Windows'ta 2-4, Linux'ta 8
    learning_rate: float = 1e-4
    weight_decay: float = 0.05
    warmup_epochs: int = 5
    grad_clip: float = 1.0

    # Loss
    loss_type: str = "weighted_ce"  # "weighted_ce" veya "ce" (ablation için)
    label_smoothing: float = 0.1

    # AMP
    amp_enabled: bool = True
    amp_dtype: str = "bfloat16"  # bf16 RTX 3060+ için ideal

    # Logging / Checkpoints
    save_dir: str = "outputs/runs"
    log_every_n_batches: int = 50

    # Reproducibility
    seed: int = 42

    # Paths
    splits_dir: str = "outputs/splits"


# ============================================================================
# UTILS
# ============================================================================

def set_seed(seed: int):
    """Reproducibility için tüm seed'leri sabitle."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Deterministik davranış (biraz yavaşlatır ama tekrarlanabilirlik için kritik)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_time(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}h {m}m {s}s"
    return f"{m}m {s}s"


# ============================================================================
# WARMUP + COSINE LR SCHEDULER
# ============================================================================

class WarmupCosineLR:
    """5 epoch linear warmup + sonrasında cosine annealing."""

    def __init__(self, optimizer, warmup_epochs, total_epochs,
                 base_lr, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lr = base_lr
        self.min_lr = min_lr

    def step(self, epoch):
        if epoch < self.warmup_epochs:
            # Linear warmup: 0 → base_lr
            lr = self.base_lr * (epoch + 1) / self.warmup_epochs
        else:
            # Cosine annealing: base_lr → min_lr
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * \
                 (1 + np.cos(np.pi * progress))

        for pg in self.optimizer.param_groups:
            pg["lr"] = lr
        return lr


# ============================================================================
# TRAIN / VALIDATE
# ============================================================================

def train_one_epoch(model, loader, optimizer, criterion, scaler,
                     device, amp_dtype, grad_clip, log_every_n_batches):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    n_batches = len(loader)
    t_start = time.time()

    for batch_idx, (imgs, labels) in enumerate(loader):
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=scaler is not None or amp_dtype != torch.float32):
            outputs = model(imgs)
            loss = criterion(outputs, labels)

        if scaler is not None:
            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += imgs.size(0)

        if (batch_idx + 1) % log_every_n_batches == 0 or batch_idx == n_batches - 1:
            elapsed = time.time() - t_start
            speed = total / elapsed
            print(f"    [{batch_idx+1:4d}/{n_batches}] "
                  f"loss={total_loss/total:.4f} "
                  f"acc={100*correct/total:.2f}% "
                  f"speed={speed:.1f} img/s "
                  f"lr={optimizer.param_groups[0]['lr']:.2e}")

    avg_loss = total_loss / total
    avg_acc = correct / total
    return avg_loss, avg_acc


@torch.no_grad()
def validate(model, loader, criterion, device, amp_dtype):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []

    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", dtype=amp_dtype):
            outputs = model(imgs)
            loss = criterion(outputs, labels)

        total_loss += loss.item() * imgs.size(0)
        probs = F.softmax(outputs.float(), dim=1)
        _, preds = outputs.max(1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.append(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.concatenate(all_probs, axis=0)

    avg_loss = total_loss / len(all_labels)
    accuracy = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    weighted_f1 = f1_score(all_labels, all_preds, average="weighted")
    kappa = cohen_kappa_score(all_labels, all_preds)
    per_class_f1 = f1_score(all_labels, all_preds, average=None,
                              labels=list(range(len(CLASS_TO_IDX))))

    metrics = {
        "loss": avg_loss,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "cohen_kappa": kappa,
    }
    for i, cls in enumerate(["CNV", "DME", "DRUSEN", "NORMAL"]):
        metrics[f"{cls.lower()}_f1"] = per_class_f1[i]

    return metrics, all_preds, all_labels, all_probs


# ============================================================================
# PLOTTING
# ============================================================================

def plot_training_curves(log_df, save_path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Loss
    axes[0].plot(log_df["epoch"], log_df["train_loss"], label="Train", linewidth=2)
    axes[0].plot(log_df["epoch"], log_df["val_loss"], label="Val", linewidth=2)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss Eğrisi")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Accuracy
    axes[1].plot(log_df["epoch"], log_df["train_acc"] * 100, label="Train", linewidth=2)
    axes[1].plot(log_df["epoch"], log_df["val_acc"] * 100, label="Val", linewidth=2)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_title("Accuracy Eğrisi")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    # F1 (per-class on val)
    for cls in ["cnv", "dme", "drusen", "normal"]:
        col = f"val_{cls}_f1"
        if col in log_df.columns:
            axes[2].plot(log_df["epoch"], log_df[col], label=cls.upper(), linewidth=2)
    if "val_macro_f1" in log_df.columns:
        axes[2].plot(log_df["epoch"], log_df["val_macro_f1"],
                      label="Macro-F1", linewidth=2.5, linestyle="--", color="black")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("F1 Score")
    axes[2].set_title("Sınıf-Bazlı Val F1")
    axes[2].legend()
    axes[2].grid(alpha=0.3)
    axes[2].set_ylim(0, 1.0)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_confusion_matrix(y_true, y_pred, save_path):
    cm = confusion_matrix(y_true, y_pred,
                            labels=list(range(len(CLASS_TO_IDX))))
    cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
    classes = list(CLASS_TO_IDX.keys())

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Sayısal
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes, ax=axes[0],
                cbar_kws={"label": "Sayı"}, square=True,
                annot_kws={"size": 13, "weight": "bold"})
    axes[0].set_title("Confusion Matrix (sayı)")
    axes[0].set_xlabel("Tahmin")
    axes[0].set_ylabel("Gerçek")

    # Yüzdelik (sınıf-bazlı normalize)
    sns.heatmap(cm_norm * 100, annot=True, fmt=".1f", cmap="Blues",
                xticklabels=classes, yticklabels=classes, ax=axes[1],
                cbar_kws={"label": "%"}, square=True,
                annot_kws={"size": 13, "weight": "bold"},
                vmin=0, vmax=100)
    axes[1].set_title("Confusion Matrix (% — sınıf normalize)")
    axes[1].set_xlabel("Tahmin")
    axes[1].set_ylabel("Gerçek")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


# ============================================================================
# MAIN
# ============================================================================

def main():
    cfg = Config()

    # ----- Setup -----
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"convnext_{timestamp}"
    save_dir = SCRIPT_DIR / cfg.save_dir / run_name
    save_dir.mkdir(parents=True, exist_ok=True)

    # AMP dtype
    amp_dtype = torch.bfloat16 if cfg.amp_dtype == "bfloat16" else torch.float16
    use_scaler = (amp_dtype == torch.float16)  # bf16 scaler gerektirmez

    print(f"\n{'#'*70}")
    print(f"# EĞİTİM BAŞLIYOR: {run_name}")
    print(f"# Cihaz: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")
    print(f"# Tasarruf yolu: {save_dir}")
    print(f"{'#'*70}")

    # Config'i kaydet
    with open(save_dir / "config.json", "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    # ----- Datasets & Loaders -----
    print(f"\n[1/5] Dataset'ler hazırlanıyor...")
    splits_dir = SCRIPT_DIR / cfg.splits_dir

    train_ds = OCTDataset(splits_dir / "train.csv",
                           transform=get_train_transform(cfg.image_size))
    val_ds = OCTDataset(splits_dir / "val.csv",
                         transform=get_val_transform(cfg.image_size))
    test_ds = OCTDataset(splits_dir / "test.csv",
                          transform=get_val_transform(cfg.image_size))

    print(f"  train: {len(train_ds):,} | val: {len(val_ds):,} | test: {len(test_ds):,}")

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size,
                              shuffle=True, num_workers=cfg.num_workers,
                              pin_memory=True, drop_last=True,
                              persistent_workers=cfg.num_workers > 0)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size,
                             shuffle=False, num_workers=cfg.num_workers,
                             pin_memory=True,
                             persistent_workers=cfg.num_workers > 0)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size,
                              shuffle=False, num_workers=cfg.num_workers,
                              pin_memory=True,
                              persistent_workers=cfg.num_workers > 0)

    # Class weights (weighted CE için)
    class_weights = train_ds.get_class_weights().to(device)
    print(f"  Class weights: {class_weights.cpu().numpy().round(3)}")

    # ----- Model -----
    print(f"\n[2/5] Model yükleniyor: {cfg.backbone}")
    model = timm.create_model(
        cfg.backbone,
        pretrained=cfg.pretrained,
        num_classes=cfg.num_classes,
    ).to(device)
    n_params = count_parameters(model) / 1e6
    print(f"  Trainable params: {n_params:.2f}M")

    # ----- Loss -----
    if cfg.loss_type == "weighted_ce":
        criterion = nn.CrossEntropyLoss(weight=class_weights,
                                          label_smoothing=cfg.label_smoothing)
        print(f"  Loss: Weighted CE + label smoothing {cfg.label_smoothing}")
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
        print(f"  Loss: CE + label smoothing {cfg.label_smoothing}")

    # ----- Optimizer & Scheduler -----
    optimizer = torch.optim.AdamW(model.parameters(),
                                    lr=cfg.learning_rate,
                                    weight_decay=cfg.weight_decay,
                                    betas=(0.9, 0.999))
    scheduler = WarmupCosineLR(optimizer,
                                  warmup_epochs=cfg.warmup_epochs,
                                  total_epochs=cfg.epochs,
                                  base_lr=cfg.learning_rate)

    scaler = torch.amp.GradScaler("cuda") if use_scaler else None
    print(f"  Optimizer: AdamW lr={cfg.learning_rate} wd={cfg.weight_decay}")
    print(f"  Scheduler: Linear warmup ({cfg.warmup_epochs} epoch) + Cosine annealing")
    print(f"  AMP: {cfg.amp_dtype}")

    # ----- Training Loop -----
    print(f"\n[3/5] Eğitim başlıyor: {cfg.epochs} epoch\n")
    print(f"{'='*70}")

    log_records = []
    best_val_macro_f1 = 0.0
    best_epoch = -1
    t_train_start = time.time()

    for epoch in range(cfg.epochs):
        lr = scheduler.step(epoch)
        print(f"\n  Epoch {epoch+1}/{cfg.epochs} (lr={lr:.2e})")
        print(f"  {'-'*60}")

        t_epoch = time.time()

        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion,
            scaler, device, amp_dtype, cfg.grad_clip,
            cfg.log_every_n_batches
        )

        # Validate
        val_metrics, _, _, _ = validate(
            model, val_loader, criterion, device, amp_dtype
        )

        epoch_time = time.time() - t_epoch
        elapsed_total = time.time() - t_train_start
        eta = elapsed_total / (epoch + 1) * (cfg.epochs - epoch - 1)

        print(f"\n  Epoch {epoch+1} özeti:")
        print(f"    Train: loss={train_loss:.4f}  acc={train_acc*100:.2f}%")
        print(f"    Val:   loss={val_metrics['loss']:.4f}  "
              f"acc={val_metrics['accuracy']*100:.2f}%  "
              f"macro-F1={val_metrics['macro_f1']:.4f}  κ={val_metrics['cohen_kappa']:.4f}")
        print(f"    Sınıf F1: CNV={val_metrics['cnv_f1']:.3f}  "
              f"DME={val_metrics['dme_f1']:.3f}  "
              f"DRUSEN={val_metrics['drusen_f1']:.3f}  "
              f"NORMAL={val_metrics['normal_f1']:.3f}")
        print(f"    Süre: {format_time(epoch_time)}  |  Geçen: {format_time(elapsed_total)}  |  Tahmini kalan: {format_time(eta)}")

        # Log
        record = {
            "epoch": epoch + 1,
            "lr": lr,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_weighted_f1": val_metrics["weighted_f1"],
            "val_kappa": val_metrics["cohen_kappa"],
            "val_cnv_f1": val_metrics["cnv_f1"],
            "val_dme_f1": val_metrics["dme_f1"],
            "val_drusen_f1": val_metrics["drusen_f1"],
            "val_normal_f1": val_metrics["normal_f1"],
            "epoch_time_sec": epoch_time,
        }
        log_records.append(record)
        pd.DataFrame(log_records).to_csv(save_dir / "training_log.csv", index=False)

        # Best checkpoint (val macro-F1'e göre)
        if val_metrics["macro_f1"] > best_val_macro_f1:
            best_val_macro_f1 = val_metrics["macro_f1"]
            best_epoch = epoch + 1
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_macro_f1": best_val_macro_f1,
                "val_metrics": val_metrics,
                "config": asdict(cfg),
            }, save_dir / "checkpoint_best.pt")
            print(f"    🎯 Yeni en iyi model! (macro-F1={best_val_macro_f1:.4f})")

    total_train_time = time.time() - t_train_start
    print(f"\n{'='*70}")
    print(f"  Eğitim tamamlandı: {format_time(total_train_time)}")
    print(f"  En iyi epoch: {best_epoch} (val macro-F1={best_val_macro_f1:.4f})")
    print(f"{'='*70}")

    # Last checkpoint
    torch.save({
        "epoch": cfg.epochs,
        "model_state_dict": model.state_dict(),
        "config": asdict(cfg),
    }, save_dir / "checkpoint_last.pt")

    # Training curves
    log_df = pd.DataFrame(log_records)
    plot_training_curves(log_df, save_dir / "training_curves.png")

    # ----- Final Test Evaluation -----
    print(f"\n[4/5] Test seti üzerinde final değerlendirme...")
    print(f"  En iyi checkpoint yükleniyor (epoch {best_epoch})...")

    checkpoint = torch.load(save_dir / "checkpoint_best.pt", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_metrics, test_preds, test_labels, test_probs = validate(
        model, test_loader, criterion, device, amp_dtype
    )

    print(f"\n  📊 TEST SONUÇLARI:")
    print(f"  {'-'*50}")
    print(f"  Accuracy:     {test_metrics['accuracy']*100:.2f}%")
    print(f"  Macro-F1:     {test_metrics['macro_f1']:.4f}")
    print(f"  Weighted-F1:  {test_metrics['weighted_f1']:.4f}")
    print(f"  Cohen κ:      {test_metrics['cohen_kappa']:.4f}")
    print(f"\n  Sınıf-Bazlı F1:")
    for cls in ["CNV", "DME", "DRUSEN", "NORMAL"]:
        print(f"    {cls:7s}: {test_metrics[f'{cls.lower()}_f1']:.4f}")

    # Classification report
    report_str = classification_report(
        test_labels, test_preds,
        target_names=list(CLASS_TO_IDX.keys()),
        digits=4
    )
    print(f"\n{report_str}")

    # Save test results
    test_results = {
        "best_epoch": best_epoch,
        "best_val_macro_f1": float(best_val_macro_f1),
        "test_metrics": {k: float(v) for k, v in test_metrics.items()},
        "test_classification_report": classification_report(
            test_labels, test_preds,
            target_names=list(CLASS_TO_IDX.keys()),
            digits=4,
            output_dict=True
        ),
        "test_confusion_matrix": confusion_matrix(
            test_labels, test_preds,
            labels=list(range(len(CLASS_TO_IDX)))
        ).tolist(),
        "config": asdict(cfg),
        "total_train_time_sec": total_train_time,
    }
    with open(save_dir / "test_results.json", "w") as f:
        json.dump(test_results, f, indent=2)

    # Confusion matrix
    plot_confusion_matrix(test_labels, test_preds,
                            save_dir / "confusion_matrix.png")

    # Tahminleri CSV olarak kaydet (later analysis için)
    test_df = pd.read_csv(SCRIPT_DIR / cfg.splits_dir / "test.csv")
    test_df["true_label"] = test_labels
    test_df["pred_label"] = test_preds
    test_df["correct"] = test_labels == test_preds
    for i, cls in enumerate(["CNV", "DME", "DRUSEN", "NORMAL"]):
        test_df[f"prob_{cls}"] = test_probs[:, i]
    test_df.to_csv(save_dir / "test_predictions.csv", index=False)

    print(f"\n[5/5] Tüm çıktılar kaydedildi: {save_dir}")
    print(f"\n{'#'*70}")
    print(f"# TAMAMLANDI ✓")
    print(f"{'#'*70}")
    print(f"\nÖnemli dosyalar:")
    print(f"  📄 test_results.json       — final metrikler")
    print(f"  📊 confusion_matrix.png    — test confusion matrix")
    print(f"  📈 training_curves.png     — loss, acc, F1 eğrileri")
    print(f"  💾 checkpoint_best.pt      — RETFound karşılaştırması için lazım")


if __name__ == "__main__":
    main()