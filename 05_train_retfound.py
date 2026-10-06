"""
05_train_retfound.py
====================

RETFound (ViT-Large/16, MAE-pretrained, Zhou et al. Nature 2023)
+ LoRA (rank=16, q/v projeksiyonları) ile Kermany OCT 4-sınıf sınıflandırma.

ConvNeXt-V2-Tiny (04_train_convnext.py) için paralel mimari, ancak şu farklarla:
- RETFound checkpoint yükleme (HF Hub veya local)
- LoRA adapter'ları (sadece ~1.2M parametre eğitilebilir)
- Daha küçük learning rate (foundation model fine-tune için)
- Layer-wise learning rate decay (orijinal RETFound önerisi)

ÖNEMLİ ÖNCESİ:
1. peft, huggingface_hub kurulu olmalı:
     pip install peft huggingface_hub
2. RETFound HF erişim izni almak için:
     https://huggingface.co/YukunZhou/RETFound_mae_natureOCT
   adresinde "request access" tıkla (anlık onaylanır)
3. HF token'ı ayarla (terminal):
     $env:HF_TOKEN = "hf_xxxxx"
4. İlk çalıştırmada checkpoint indirme ~1.2 GB olacak

Output: outputs/runs/retfound_<timestamp>/  (ConvNeXt ile aynı yapı)
"""

import os
import sys
import json
import time
import random
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, f1_score,
                              confusion_matrix, classification_report,
                              cohen_kappa_score)

SCRIPT_DIR = Path(__file__).parent if "__file__" in dir() else Path.cwd()
sys.path.insert(0, str(SCRIPT_DIR))

from src.dataset import (OCTDataset, get_train_transform, get_val_transform,
                          CLASS_TO_IDX, IDX_TO_CLASS)
from src.retfound import (build_retfound_for_finetune,
                            download_retfound_checkpoint,
                            RETFOUND_LOCAL_FILENAME)


# ============================================================================
# CONFIG
# ============================================================================

@dataclass
class Config:
    # Model
    num_classes: int = 4
    image_size: int = 224
    use_lora: bool = True
    lora_rank: int = 16
    lora_alpha: int = 32
    drop_path_rate: float = 0.2

    # Checkpoint
    checkpoint_dir: str = "outputs/checkpoints"
    auto_download: bool = True

    # Training
    epochs: int = 30
    batch_size: int = 16  # ViT-L için 32 fazla, 16 güvenli
    num_workers: int = 2
    learning_rate: float = 5e-4  # RETFound paper'da "blr=5e-3" → effective 5e-4
    weight_decay: float = 0.05
    warmup_epochs: int = 5
    grad_clip: float = 1.0
    layer_decay: float = 0.65  # RETFound layer-wise LR decay

    # Loss
    loss_type: str = "weighted_ce"
    label_smoothing: float = 0.1

    # AMP + memory
    amp_enabled: bool = True
    amp_dtype: str = "bfloat16"
    gradient_checkpointing: bool = True  # ViT-L için bellek tasarrufu

    # Logging
    save_dir: str = "outputs/runs"
    log_every_n_batches: int = 50
    seed: int = 42
    splits_dir: str = "outputs/splits"


# ============================================================================
# UTILS (ConvNeXt scripti ile ortak — gerçek projede src/utils.py'a taşınacak)
# ============================================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def format_time(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}h {m}m {s}s"
    return f"{m}m {s}s"


class WarmupCosineLR:
    def __init__(self, optimizer, warmup_epochs, total_epochs, base_lr, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lr = base_lr
        self.min_lr = min_lr

    def step(self, epoch):
        if epoch < self.warmup_epochs:
            scale = (epoch + 1) / self.warmup_epochs
        else:
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            scale = (self.min_lr + 0.5 * (self.base_lr - self.min_lr) *
                     (1 + np.cos(np.pi * progress))) / self.base_lr

        for pg in self.optimizer.param_groups:
            # Her parametre grubu kendi base_lr'sine sahip olabilir (layer-wise decay)
            pg["lr"] = pg.get("base_lr", self.base_lr) * scale
        return scale * self.base_lr  # nominal lr


def get_param_groups_with_layer_decay(model, base_lr: float, weight_decay: float,
                                        layer_decay: float = 0.65, num_layers: int = 24):
    """
    RETFound layer-wise learning rate decay.

    Erken katmanlar (0, 1, ...) küçük LR ile, geç katmanlar daha büyük LR ile öğrenir.
    LoRA modunda çalışırken sadece eğitilebilir parametreleri gruplar.
    """
    no_decay_keywords = ["bias", "norm", "LayerNorm"]

    # Katman kimliklerini çıkarmak için
    def get_layer_id(name: str) -> int:
        # 'blocks.0.attn.qkv...' → katman 0
        # 'patch_embed' veya 'cls_token' → katman -1 (en eski)
        # 'norm' / 'head' → katman num_layers (en yeni)
        if "patch_embed" in name or "cls_token" in name or "pos_embed" in name:
            return 0
        if "blocks." in name:
            try:
                return int(name.split("blocks.")[1].split(".")[0]) + 1
            except (IndexError, ValueError):
                return num_layers + 1
        return num_layers + 1  # head, norm, vs.

    param_groups = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        layer_id = get_layer_id(name)
        # Decay'siz mi (norm, bias)?
        wd = 0.0 if any(k in name for k in no_decay_keywords) else weight_decay

        # LR scale: layer_decay^(num_layers - layer_id)
        lr_scale = layer_decay ** (num_layers + 1 - layer_id)
        actual_lr = base_lr * lr_scale

        group_key = f"layer_{layer_id}_wd_{wd}"
        if group_key not in param_groups:
            param_groups[group_key] = {
                "params": [],
                "weight_decay": wd,
                "lr": actual_lr,
                "base_lr": actual_lr,  # scheduler için referans
                "layer_id": layer_id,
            }
        param_groups[group_key]["params"].append(param)

    return list(param_groups.values())


# ============================================================================
# TRAIN / VALIDATE (ConvNeXt scripti ile aynı)
# ============================================================================

def train_one_epoch(model, loader, optimizer, criterion, scaler,
                     device, amp_dtype, grad_clip, log_every_n_batches):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    n_batches = len(loader)
    t_start = time.time()

    for batch_idx, (imgs, labels) in enumerate(loader):
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", dtype=amp_dtype):
            outputs = model(imgs)
            loss = criterion(outputs, labels)

        if scaler is not None:
            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], grad_clip
                )
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], grad_clip
                )
            optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += imgs.size(0)

        if (batch_idx + 1) % log_every_n_batches == 0 or batch_idx == n_batches - 1:
            elapsed = time.time() - t_start
            speed = total / elapsed
            print(f"    [{batch_idx+1:4d}/{n_batches}] "
                  f"loss={total_loss/total:.4f} acc={100*correct/total:.2f}% "
                  f"speed={speed:.1f} img/s")

    return total_loss / total, correct / total


@torch.no_grad()
def validate(model, loader, criterion, device, amp_dtype):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels, all_probs = [], [], []

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

    metrics = {
        "loss": total_loss / len(all_labels),
        "accuracy": accuracy_score(all_labels, all_preds),
        "macro_f1": f1_score(all_labels, all_preds, average="macro"),
        "weighted_f1": f1_score(all_labels, all_preds, average="weighted"),
        "cohen_kappa": cohen_kappa_score(all_labels, all_preds),
    }
    per_class_f1 = f1_score(all_labels, all_preds, average=None,
                              labels=list(range(len(CLASS_TO_IDX))))
    for i, cls in enumerate(["CNV", "DME", "DRUSEN", "NORMAL"]):
        metrics[f"{cls.lower()}_f1"] = per_class_f1[i]
    return metrics, all_preds, all_labels, all_probs


def plot_training_curves(log_df, save_path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(log_df["epoch"], log_df["train_loss"], label="Train", linewidth=2)
    axes[0].plot(log_df["epoch"], log_df["val_loss"], label="Val", linewidth=2)
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss Eğrisi"); axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(log_df["epoch"], log_df["train_acc"]*100, label="Train", linewidth=2)
    axes[1].plot(log_df["epoch"], log_df["val_acc"]*100, label="Val", linewidth=2)
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_title("Accuracy"); axes[1].legend(); axes[1].grid(alpha=0.3)

    for cls in ["cnv", "dme", "drusen", "normal"]:
        col = f"val_{cls}_f1"
        if col in log_df.columns:
            axes[2].plot(log_df["epoch"], log_df[col], label=cls.upper(), linewidth=2)
    if "val_macro_f1" in log_df.columns:
        axes[2].plot(log_df["epoch"], log_df["val_macro_f1"],
                      label="Macro-F1", linewidth=2.5, linestyle="--", color="black")
    axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("F1 Score")
    axes[2].set_title("Sınıf Bazlı Val F1"); axes[2].legend()
    axes[2].grid(alpha=0.3); axes[2].set_ylim(0, 1.0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_confusion_matrix(y_true, y_pred, save_path):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASS_TO_IDX))))
    cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
    classes = list(CLASS_TO_IDX.keys())
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes, ax=axes[0],
                cbar_kws={"label": "Sayı"}, square=True,
                annot_kws={"size": 13, "weight": "bold"})
    axes[0].set_title("Confusion Matrix (sayı)")
    axes[0].set_xlabel("Tahmin"); axes[0].set_ylabel("Gerçek")
    sns.heatmap(cm_norm * 100, annot=True, fmt=".1f", cmap="Blues",
                xticklabels=classes, yticklabels=classes, ax=axes[1],
                cbar_kws={"label": "%"}, square=True,
                annot_kws={"size": 13, "weight": "bold"}, vmin=0, vmax=100)
    axes[1].set_title("Confusion Matrix (% sınıf normalize)")
    axes[1].set_xlabel("Tahmin"); axes[1].set_ylabel("Gerçek")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


# ============================================================================
# MAIN
# ============================================================================

def main():
    cfg = Config()
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"retfound_lora_{timestamp}"
    save_dir = SCRIPT_DIR / cfg.save_dir / run_name
    save_dir.mkdir(parents=True, exist_ok=True)

    amp_dtype = torch.bfloat16 if cfg.amp_dtype == "bfloat16" else torch.float16
    use_scaler = (amp_dtype == torch.float16)

    print(f"\n{'#'*70}")
    print(f"# EĞİTİM BAŞLIYOR: {run_name}")
    print(f"# Cihaz: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")
    print(f"# Tasarruf yolu: {save_dir}")
    print(f"{'#'*70}")

    with open(save_dir / "config.json", "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    # ----- 1. RETFound checkpoint -----
    print(f"\n[1/6] RETFound checkpoint...")
    ckpt_dir = SCRIPT_DIR / cfg.checkpoint_dir
    ckpt_path = ckpt_dir / RETFOUND_LOCAL_FILENAME

    if not ckpt_path.exists() and cfg.auto_download:
        try:
            ckpt_path = download_retfound_checkpoint(ckpt_dir)
        except Exception as e:
            print(f"\n  ❌ İndirme başarısız: {e}")
            print(f"  Manuel indirme talimatları yukarıda. Devam etmek için checkpoint gerekli.")
            return

    if not ckpt_path.exists():
        print(f"  ❌ Checkpoint bulunamadı: {ckpt_path}")
        return

    # ----- 2. Datasets -----
    print(f"\n[2/6] Dataset'ler...")
    splits_dir = SCRIPT_DIR / cfg.splits_dir
    train_ds = OCTDataset(splits_dir / "train.csv", get_train_transform(cfg.image_size))
    val_ds = OCTDataset(splits_dir / "val.csv", get_val_transform(cfg.image_size))
    test_ds = OCTDataset(splits_dir / "test.csv", get_val_transform(cfg.image_size))
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
    class_weights = train_ds.get_class_weights().to(device)

    # ----- 3. Model -----
    print(f"\n[3/6] RETFound + LoRA modeli kuruluyor...")
    model, model_info = build_retfound_for_finetune(
        num_classes=cfg.num_classes,
        checkpoint_path=ckpt_path,
        use_lora=cfg.use_lora,
        lora_rank=cfg.lora_rank,
        lora_alpha=cfg.lora_alpha,
    )
    model = model.to(device)

    if cfg.gradient_checkpointing:
        try:
            # PEFT modeli için orijinal'a erişip gradient_checkpointing'i aç
            base = model.base_model.model if hasattr(model, "base_model") else model
            if hasattr(base, "set_grad_checkpointing"):
                base.set_grad_checkpointing(True)
                print(f"  ✓ Gradient checkpointing açıldı (bellek tasarrufu)")
        except Exception as e:
            print(f"  ⚠️  Gradient checkpointing açılamadı: {e}")

    # ----- 4. Loss + Optimizer -----
    if cfg.loss_type == "weighted_ce":
        criterion = nn.CrossEntropyLoss(weight=class_weights,
                                          label_smoothing=cfg.label_smoothing)
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)

    # Layer-wise LR decay (RETFound önerisi)
    param_groups = get_param_groups_with_layer_decay(
        model, base_lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        layer_decay=cfg.layer_decay
    )
    optimizer = torch.optim.AdamW(param_groups, lr=cfg.learning_rate,
                                    betas=(0.9, 0.999))
    scheduler = WarmupCosineLR(optimizer, cfg.warmup_epochs, cfg.epochs,
                                  cfg.learning_rate)
    scaler = torch.amp.GradScaler("cuda") if use_scaler else None

    print(f"  Loss: {'Weighted ' if cfg.loss_type == 'weighted_ce' else ''}CE + smoothing {cfg.label_smoothing}")
    print(f"  Optimizer: AdamW lr={cfg.learning_rate} wd={cfg.weight_decay} "
          f"layer_decay={cfg.layer_decay}")
    print(f"  AMP: {cfg.amp_dtype} | Grad checkpointing: {cfg.gradient_checkpointing}")

    # ----- 5. Training Loop -----
    print(f"\n[4/6] Eğitim başlıyor: {cfg.epochs} epoch")
    print(f"{'='*70}")

    log_records = []
    best_val_macro_f1 = 0.0
    best_epoch = -1
    t_start = time.time()

    for epoch in range(cfg.epochs):
        lr = scheduler.step(epoch)
        print(f"\n  Epoch {epoch+1}/{cfg.epochs} (lr={lr:.2e})")
        print(f"  {'-'*60}")
        t_epoch = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, scaler,
            device, amp_dtype, cfg.grad_clip, cfg.log_every_n_batches
        )
        val_metrics, _, _, _ = validate(model, val_loader, criterion,
                                          device, amp_dtype)

        epoch_time = time.time() - t_epoch
        elapsed = time.time() - t_start
        eta = elapsed / (epoch + 1) * (cfg.epochs - epoch - 1)

        print(f"\n  Epoch {epoch+1} özeti:")
        print(f"    Train: loss={train_loss:.4f}  acc={train_acc*100:.2f}%")
        print(f"    Val:   loss={val_metrics['loss']:.4f}  "
              f"acc={val_metrics['accuracy']*100:.2f}%  "
              f"macro-F1={val_metrics['macro_f1']:.4f}  κ={val_metrics['cohen_kappa']:.4f}")
        print(f"    Sınıf F1: CNV={val_metrics['cnv_f1']:.3f}  "
              f"DME={val_metrics['dme_f1']:.3f}  "
              f"DRUSEN={val_metrics['drusen_f1']:.3f}  "
              f"NORMAL={val_metrics['normal_f1']:.3f}")
        print(f"    Süre: {format_time(epoch_time)} | "
              f"Geçen: {format_time(elapsed)} | Kalan: {format_time(eta)}")

        record = {"epoch": epoch + 1, "lr": lr,
                  "train_loss": train_loss, "train_acc": train_acc,
                  "val_loss": val_metrics["loss"],
                  "val_acc": val_metrics["accuracy"],
                  "val_macro_f1": val_metrics["macro_f1"],
                  "val_weighted_f1": val_metrics["weighted_f1"],
                  "val_kappa": val_metrics["cohen_kappa"],
                  "val_cnv_f1": val_metrics["cnv_f1"],
                  "val_dme_f1": val_metrics["dme_f1"],
                  "val_drusen_f1": val_metrics["drusen_f1"],
                  "val_normal_f1": val_metrics["normal_f1"],
                  "epoch_time_sec": epoch_time}
        log_records.append(record)
        pd.DataFrame(log_records).to_csv(save_dir / "training_log.csv", index=False)

        if val_metrics["macro_f1"] > best_val_macro_f1:
            best_val_macro_f1 = val_metrics["macro_f1"]
            best_epoch = epoch + 1
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "val_macro_f1": best_val_macro_f1,
                "val_metrics": val_metrics,
                "config": asdict(cfg),
            }, save_dir / "checkpoint_best.pt")
            print(f"    🎯 Yeni en iyi model! (macro-F1={best_val_macro_f1:.4f})")

    total_time = time.time() - t_start
    print(f"\n{'='*70}")
    print(f"  Eğitim tamamlandı: {format_time(total_time)}")
    print(f"  En iyi epoch: {best_epoch} (val macro-F1={best_val_macro_f1:.4f})")
    print(f"{'='*70}")

    # ----- 6. Test Evaluation -----
    print(f"\n[5/6] Test set evaluation...")
    checkpoint = torch.load(save_dir / "checkpoint_best.pt", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics, test_preds, test_labels, test_probs = validate(
        model, test_loader, criterion, device, amp_dtype
    )

    print(f"\n  📊 RETFound + LoRA TEST SONUÇLARI:")
    print(f"  {'-'*50}")
    print(f"  Accuracy:     {test_metrics['accuracy']*100:.2f}%")
    print(f"  Macro-F1:     {test_metrics['macro_f1']:.4f}")
    print(f"  Weighted-F1:  {test_metrics['weighted_f1']:.4f}")
    print(f"  Cohen κ:      {test_metrics['cohen_kappa']:.4f}")
    print(f"\n  Sınıf-Bazlı F1:")
    for cls in ["CNV", "DME", "DRUSEN", "NORMAL"]:
        print(f"    {cls:7s}: {test_metrics[f'{cls.lower()}_f1']:.4f}")

    print(f"\n{classification_report(test_labels, test_preds, target_names=list(CLASS_TO_IDX.keys()), digits=4)}")

    test_results = {
        "best_epoch": best_epoch,
        "best_val_macro_f1": float(best_val_macro_f1),
        "test_metrics": {k: float(v) for k, v in test_metrics.items()},
        "test_classification_report": classification_report(
            test_labels, test_preds, target_names=list(CLASS_TO_IDX.keys()),
            digits=4, output_dict=True
        ),
        "test_confusion_matrix": confusion_matrix(
            test_labels, test_preds, labels=list(range(len(CLASS_TO_IDX)))
        ).tolist(),
        "model_info": {
            "total_params": int(model_info["total_params"]),
            "trainable_params": int(model_info.get("after_lora_trainable",
                                                       model_info["before_lora_trainable"])),
        },
        "config": asdict(cfg),
        "total_train_time_sec": total_time,
    }
    with open(save_dir / "test_results.json", "w") as f:
        json.dump(test_results, f, indent=2)

    plot_training_curves(pd.DataFrame(log_records), save_dir / "training_curves.png")
    plot_confusion_matrix(test_labels, test_preds, save_dir / "confusion_matrix.png")

    test_df = pd.read_csv(SCRIPT_DIR / cfg.splits_dir / "test.csv")
    test_df["true_label"] = test_labels
    test_df["pred_label"] = test_preds
    test_df["correct"] = test_labels == test_preds
    for i, cls in enumerate(["CNV", "DME", "DRUSEN", "NORMAL"]):
        test_df[f"prob_{cls}"] = test_probs[:, i]
    test_df.to_csv(save_dir / "test_predictions.csv", index=False)

    print(f"\n[6/6] Tüm çıktılar: {save_dir}")
    print(f"\n{'#'*70}")
    print(f"# RETFOUND + LoRA TAMAMLANDI ✓")
    print(f"{'#'*70}")


if __name__ == "__main__":
    main()