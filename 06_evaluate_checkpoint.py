"""
06_evaluate_checkpoint.py
==========================

Eğitim crash ettiğinde (veya bittikten sonra) kaydedilmiş bir checkpoint'ten
test setinde değerlendirme yapar.

Bu script:
1. En son `outputs/runs/convnext_*` klasöründeki checkpoint_best.pt'yi bulur
2. Modeli yükler
3. Test setinde tahmin yapar
4. Tüm metrikleri hesaplar (accuracy, macro-F1, per-class F1, kappa, AUROC)
5. Confusion matrix figürü üretir
6. Bootstrap 95% güven aralıkları hesaplar (yayın için)
7. test_results.json olarak kaydeder
"""

import os
import sys
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import timm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, f1_score, precision_recall_fscore_support,
                              confusion_matrix, classification_report,
                              cohen_kappa_score, roc_auc_score)

SCRIPT_DIR = Path(__file__).parent if "__file__" in dir() else Path.cwd()
sys.path.insert(0, str(SCRIPT_DIR))

from src.dataset import (OCTDataset, get_val_transform, CLASS_TO_IDX, IDX_TO_CLASS)


# ============================================================================
# CONFIG - en son convnext_* run'ını bulacak
# ============================================================================

def find_latest_run(prefix: str = "convnext"):
    """outputs/runs/convnext_* klasörlerinden en son olanı bul."""
    runs_dir = SCRIPT_DIR / "outputs" / "runs"
    if not runs_dir.exists():
        return None
    candidates = sorted([d for d in runs_dir.iterdir()
                         if d.is_dir() and d.name.startswith(prefix)])
    return candidates[-1] if candidates else None


# ============================================================================
# BOOTSTRAP CI (yayın için kritik)
# ============================================================================

def bootstrap_metric(y_true, y_pred, metric_func, n_bootstrap=1000, seed=42, **metric_kwargs):
    """
    Bir metrik için bootstrap %95 güven aralığı hesapla.
    """
    np.random.seed(seed)
    n = len(y_true)
    scores = []
    for _ in range(n_bootstrap):
        indices = np.random.choice(n, n, replace=True)
        try:
            score = metric_func(y_true[indices], y_pred[indices], **metric_kwargs)
            scores.append(score)
        except Exception:
            continue
    scores = np.array(scores)
    return {
        "mean": float(scores.mean()),
        "ci_low": float(np.percentile(scores, 2.5)),
        "ci_high": float(np.percentile(scores, 97.5)),
    }


def bootstrap_auroc(y_true, y_probs, n_bootstrap=1000, seed=42):
    """Bootstrap AUROC for multiclass (one-vs-rest)."""
    np.random.seed(seed)
    n = len(y_true)
    scores = []
    for _ in range(n_bootstrap):
        indices = np.random.choice(n, n, replace=True)
        try:
            score = roc_auc_score(y_true[indices], y_probs[indices],
                                    multi_class="ovr", average="macro")
            scores.append(score)
        except Exception:
            continue
    scores = np.array(scores)
    return {
        "mean": float(scores.mean()),
        "ci_low": float(np.percentile(scores, 2.5)),
        "ci_high": float(np.percentile(scores, 97.5)),
    }


# ============================================================================
# EVALUATION
# ============================================================================

@torch.no_grad()
def evaluate(model, loader, device, amp_dtype=torch.bfloat16):
    """Tüm tahminleri ve olasılıkları topla."""
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=amp_dtype):
            outputs = model(imgs)
        probs = F.softmax(outputs.float(), dim=1)
        _, preds = outputs.max(1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.append(probs.cpu().numpy())

    return (np.array(all_preds),
            np.array(all_labels),
            np.concatenate(all_probs, axis=0))


# ============================================================================
# PLOTS
# ============================================================================

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
    axes[1].set_title("Confusion Matrix (% — sınıf normalize)")
    axes[1].set_xlabel("Tahmin"); axes[1].set_ylabel("Gerçek")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {save_path.name}")


def plot_training_curves_from_log(log_csv_path, save_path):
    """Mevcut training_log.csv'den eğri figürlerini üret."""
    if not log_csv_path.exists():
        print(f"  ⚠️  {log_csv_path} bulunamadı, eğri figürü atlanıyor")
        return

    log_df = pd.read_csv(log_csv_path)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(log_df["epoch"], log_df["train_loss"], label="Train", linewidth=2)
    axes[0].plot(log_df["epoch"], log_df["val_loss"], label="Val", linewidth=2)
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss Eğrisi"); axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(log_df["epoch"], log_df["train_acc"]*100, label="Train", linewidth=2)
    axes[1].plot(log_df["epoch"], log_df["val_acc"]*100, label="Val", linewidth=2)
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_title("Accuracy Eğrisi"); axes[1].legend(); axes[1].grid(alpha=0.3)

    for cls in ["cnv", "dme", "drusen", "normal"]:
        col = f"val_{cls}_f1"
        if col in log_df.columns:
            axes[2].plot(log_df["epoch"], log_df[col], label=cls.upper(), linewidth=2)
    if "val_macro_f1" in log_df.columns:
        axes[2].plot(log_df["epoch"], log_df["val_macro_f1"],
                      label="Macro-F1", linewidth=2.5, linestyle="--", color="black")
    axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("F1 Score")
    axes[2].set_title("Sınıf-Bazlı Val F1"); axes[2].legend()
    axes[2].grid(alpha=0.3); axes[2].set_ylim(0, 1.0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {save_path.name}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print(f"\n{'#'*70}")
    print(f"# CHECKPOINT EVALUATION (crash sonrası kurtarma)")
    print(f"{'#'*70}")

    # 1. En son run'ı bul
    run_dir = find_latest_run("convnext")
    if run_dir is None:
        print("❌ Hiç convnext run bulunamadı. outputs/runs/convnext_* var mı?")
        return

    print(f"\n  Bulunan run: {run_dir.name}")
    ckpt_path = run_dir / "checkpoint_best.pt"
    if not ckpt_path.exists():
        print(f"❌ {ckpt_path} bulunamadı.")
        return
    print(f"  Checkpoint: {ckpt_path}")

    # 2. Checkpoint yükle
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Cihaz: {device}")

    print(f"\n[1/5] Checkpoint yükleniyor...")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = checkpoint.get("config", {})
    backbone = cfg.get("backbone", "convnextv2_tiny.fcmae_ft_in22k_in1k")
    num_classes = cfg.get("num_classes", 4)
    image_size = cfg.get("image_size", 224)
    best_epoch = checkpoint.get("epoch", "?")
    best_val_f1 = checkpoint.get("val_macro_f1", "?")

    print(f"  Backbone: {backbone}")
    print(f"  En iyi epoch: {best_epoch}")
    print(f"  En iyi val macro-F1: {best_val_f1:.4f}" if isinstance(best_val_f1, float) else f"  Val F1: {best_val_f1}")

    # 3. Modeli kur ve ağırlıkları yükle
    print(f"\n[2/5] Model kuruluyor...")
    model = timm.create_model(backbone, pretrained=False, num_classes=num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    print(f"  ✓ Model yüklendi ve eval moduna alındı")

    # 4. Test loader
    print(f"\n[3/5] Test set yükleniyor...")
    test_csv = SCRIPT_DIR / "outputs" / "splits" / "test.csv"
    test_ds = OCTDataset(test_csv, transform=get_val_transform(image_size))
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False,
                              num_workers=0, pin_memory=True)
    print(f"  ✓ Test seti: {len(test_ds):,} görüntü")

    # 5. Evaluation
    print(f"\n[4/5] Test seti üzerinde tahmin...")
    t_start = time.time()
    test_preds, test_labels, test_probs = evaluate(model, test_loader, device)
    t_elapsed = time.time() - t_start
    print(f"  ✓ {len(test_labels)} tahmin {t_elapsed:.1f}s içinde")

    # 6. Metrikleri hesapla
    print(f"\n[5/5] Metrikler ve istatistikler...")

    accuracy = accuracy_score(test_labels, test_preds)
    macro_f1 = f1_score(test_labels, test_preds, average="macro")
    weighted_f1 = f1_score(test_labels, test_preds, average="weighted")
    kappa = cohen_kappa_score(test_labels, test_preds)
    per_class_f1 = f1_score(test_labels, test_preds, average=None,
                              labels=list(range(len(CLASS_TO_IDX))))

    # AUROC (multi-class one-vs-rest)
    try:
        macro_auroc = roc_auc_score(test_labels, test_probs,
                                      multi_class="ovr", average="macro")
        per_class_auroc = roc_auc_score(test_labels, test_probs,
                                          multi_class="ovr", average=None)
    except Exception as e:
        print(f"  ⚠️  AUROC hesaplanamadı: {e}")
        macro_auroc = None
        per_class_auroc = None

    print(f"\n  📊 TEMEL METRİKLER:")
    print(f"    Accuracy:       {accuracy*100:.2f}%")
    print(f"    Macro-F1:       {macro_f1:.4f}")
    print(f"    Weighted-F1:    {weighted_f1:.4f}")
    print(f"    Cohen κ:        {kappa:.4f}")
    if macro_auroc:
        print(f"    Macro-AUROC:    {macro_auroc:.4f}")

    print(f"\n  📊 SINIF-BAZLI F1:")
    for i, cls in enumerate(["CNV", "DME", "DRUSEN", "NORMAL"]):
        auroc_str = f"  AUROC={per_class_auroc[i]:.4f}" if per_class_auroc is not None else ""
        print(f"    {cls:7s}: F1={per_class_f1[i]:.4f}{auroc_str}")

    # Bootstrap 95% güven aralıkları (yayın için)
    print(f"\n  📊 BOOTSTRAP %95 GÜVEN ARALIKLARI (n=1000):")
    print(f"  (Bu birkaç saniye sürer...)")

    acc_ci = bootstrap_metric(test_labels, test_preds, accuracy_score, n_bootstrap=1000)
    macro_f1_ci = bootstrap_metric(test_labels, test_preds, f1_score,
                                       n_bootstrap=1000, average="macro")
    kappa_ci = bootstrap_metric(test_labels, test_preds, cohen_kappa_score, n_bootstrap=1000)

    print(f"    Accuracy:    {acc_ci['mean']:.4f} [{acc_ci['ci_low']:.4f}, {acc_ci['ci_high']:.4f}]")
    print(f"    Macro-F1:    {macro_f1_ci['mean']:.4f} [{macro_f1_ci['ci_low']:.4f}, {macro_f1_ci['ci_high']:.4f}]")
    print(f"    Cohen κ:     {kappa_ci['mean']:.4f} [{kappa_ci['ci_low']:.4f}, {kappa_ci['ci_high']:.4f}]")

    if macro_auroc is not None:
        auroc_ci = bootstrap_auroc(test_labels, test_probs, n_bootstrap=1000)
        print(f"    Macro-AUROC: {auroc_ci['mean']:.4f} [{auroc_ci['ci_low']:.4f}, {auroc_ci['ci_high']:.4f}]")
    else:
        auroc_ci = None

    # Classification report
    print(f"\n  📋 KLASIFIKASYON RAPORU:")
    report_str = classification_report(test_labels, test_preds,
                                          target_names=list(CLASS_TO_IDX.keys()),
                                          digits=4)
    print(report_str)

    # Sonuçları kaydet
    test_results = {
        "best_epoch": int(best_epoch) if isinstance(best_epoch, (int, float)) else best_epoch,
        "best_val_macro_f1": float(best_val_f1) if isinstance(best_val_f1, (int, float)) else None,
        "test_metrics": {
            "accuracy": float(accuracy),
            "macro_f1": float(macro_f1),
            "weighted_f1": float(weighted_f1),
            "cohen_kappa": float(kappa),
            "macro_auroc": float(macro_auroc) if macro_auroc else None,
            "per_class_f1": {cls: float(per_class_f1[i])
                              for i, cls in enumerate(["CNV", "DME", "DRUSEN", "NORMAL"])},
            "per_class_auroc": ({cls: float(per_class_auroc[i])
                                  for i, cls in enumerate(["CNV", "DME", "DRUSEN", "NORMAL"])}
                                 if per_class_auroc is not None else None),
        },
        "bootstrap_95_ci": {
            "accuracy": acc_ci,
            "macro_f1": macro_f1_ci,
            "cohen_kappa": kappa_ci,
            "macro_auroc": auroc_ci,
        },
        "classification_report": classification_report(
            test_labels, test_preds,
            target_names=list(CLASS_TO_IDX.keys()),
            digits=4, output_dict=True
        ),
        "confusion_matrix": confusion_matrix(
            test_labels, test_preds, labels=list(range(len(CLASS_TO_IDX)))
        ).tolist(),
        "n_test_samples": len(test_labels),
    }

    results_path = run_dir / "test_results.json"
    with open(results_path, "w") as f:
        json.dump(test_results, f, indent=2)
    print(f"\n  ✓ {results_path.name} kaydedildi")

    # Confusion matrix figürü
    plot_confusion_matrix(test_labels, test_preds,
                            run_dir / "confusion_matrix.png")

    # Training curves (training_log.csv'den)
    plot_training_curves_from_log(run_dir / "training_log.csv",
                                     run_dir / "training_curves.png")

    # Tahminleri CSV olarak kaydet
    test_df = pd.read_csv(SCRIPT_DIR / "outputs" / "splits" / "test.csv")
    test_df["true_label"] = test_labels
    test_df["pred_label"] = test_preds
    test_df["correct"] = test_labels == test_preds
    for i, cls in enumerate(["CNV", "DME", "DRUSEN", "NORMAL"]):
        test_df[f"prob_{cls}"] = test_probs[:, i]
    test_df.to_csv(run_dir / "test_predictions.csv", index=False)
    print(f"  ✓ test_predictions.csv kaydedildi")

    print(f"\n{'#'*70}")
    print(f"# TAMAMLANDI ✓")
    print(f"{'#'*70}")
    print(f"\n📁 Tüm çıktılar: {run_dir}")
    print(f"\nÖne çıkan rakamlar (rapor için):")
    print(f"  • Test Accuracy:  {accuracy*100:.2f}% [%{acc_ci['ci_low']*100:.2f}, %{acc_ci['ci_high']*100:.2f}]")
    print(f"  • Test Macro-F1:  {macro_f1:.4f} [{macro_f1_ci['ci_low']:.4f}, {macro_f1_ci['ci_high']:.4f}]")
    print(f"  • Test Cohen κ:   {kappa:.4f} [{kappa_ci['ci_low']:.4f}, {kappa_ci['ci_high']:.4f}]")


if __name__ == "__main__":
    main()