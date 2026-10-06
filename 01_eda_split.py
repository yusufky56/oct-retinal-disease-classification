"""
Kermany OCT Dataset - Keşifsel Veri Analizi (EDA) + Hasta-Seviyeli Split
========================================================================

Bu script:
1. Veri setini tarar ve dosya adlarından hasta ID'lerini çıkarır
2. Sınıf dağılımı, hasta dağılımı, görüntü boyutu istatistiklerini hesaplar
3. Tampu et al. (Sci Data 2022) data leakage analizini yapar
   (orijinal Kermany v2 split'inde hastaların splitler arası taşıp taşmadığını kontrol eder)
4. Hasta-seviyeli yeni train/val/test split'i oluşturur (Q1 yayını için zorunlu)
5. Rapor için yayın kalitesinde figürler üretir
6. Her şeyi outputs/ klasörüne kaydeder

ÖNEMLİ: Kermany filename formatı: CLASS-PATIENTID-IMGNUM.jpeg
        Örnek: CNV-1234-5.jpeg → sınıf=CNV, hasta=1234, görüntü no=5
"""

import os
import re
import sys
import json
from pathlib import Path
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# ============================================================================
# YAPILANDIRMA
# ============================================================================

# DİKKAT: Bu yolu kendi makinenize göre değiştirin (raw string r"..." kullanın)
from src.paths import DATA_ROOT, resolve_image  # set OCT_DATA_ROOT to your dataset folder

# Çıktılar bu scriptin çalıştığı dizinde oluşturulacak
SCRIPT_DIR = Path(__file__).parent if "__file__" in dir() else Path.cwd()
OUTPUT_DIR = SCRIPT_DIR / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
SPLITS_DIR = OUTPUT_DIR / "splits"
REPORTS_DIR = OUTPUT_DIR / "reports"

for d in [OUTPUT_DIR, FIGURES_DIR, SPLITS_DIR, REPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

CLASSES = ["CNV", "DME", "DRUSEN", "NORMAL"]
SPLITS = ["train", "val", "test"]
CLASS_COLORS = {"CNV": "#e63946", "DME": "#f4a261", "DRUSEN": "#e9c46a", "NORMAL": "#2a9d8f"}

# Kermany filename pattern: CLASS-PATIENTID-IMGNUM.jpeg
FILENAME_PATTERN = re.compile(r"([A-Z]+)-(\d+)-(\d+)\.jpe?g", re.IGNORECASE)

# Görsel ayarları (yayın kalitesi)
plt.rcParams.update({
    "figure.dpi": 100,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


# ============================================================================
# YARDIMCI FONKSİYONLAR
# ============================================================================

def parse_filename(filename: str):
    """Kermany dosya adını parse et: CLASS-PATIENTID-IMGNUM.jpeg"""
    match = FILENAME_PATTERN.match(filename)
    if match:
        return {
            "class_label": match.group(1).upper(),
            "patient_id": match.group(2),
            "image_num": int(match.group(3)),
        }
    return None


def build_dataframe(data_root: Path) -> pd.DataFrame:
    """Tüm görüntüleri tara ve dataframe oluştur."""
    print(f"\n{'='*70}")
    print("VERİ SETİ TARANIYOR")
    print(f"{'='*70}")
    print(f"Kök dizin: {data_root}")

    if not data_root.exists():
        print(f"\n❌ HATA: {data_root} dizini bulunamadı!")
        print("   Lütfen DATA_ROOT değişkenini kendi yolunuzla güncelleyin.")
        sys.exit(1)

    records = []
    parse_failures = []

    for split in SPLITS:
        split_dir = data_root / split
        if not split_dir.exists():
            print(f"  UYARI: {split_dir} bulunamadı, atlanıyor")
            continue

        for cls in CLASSES:
            cls_dir = split_dir / cls
            if not cls_dir.exists():
                print(f"  UYARI: {cls_dir} bulunamadı, atlanıyor")
                continue

            files = [f for f in cls_dir.iterdir()
                     if f.suffix.lower() in [".jpg", ".jpeg", ".png"]]

            for img_path in files:
                parsed = parse_filename(img_path.name)
                if parsed is None:
                    parse_failures.append(img_path.name)
                    continue

                records.append({
                    "filepath": img_path.relative_to(data_root).as_posix(),
                    "filename": img_path.name,
                    "class": cls,
                    "patient_id": parsed["patient_id"],
                    "image_num": parsed["image_num"],
                    "original_split": split,
                })

    df = pd.DataFrame(records)

    print(f"\n✓ Toplam {len(df):,} görüntü bulundu")
    print(f"✓ {df['patient_id'].nunique():,} farklı hasta")
    if parse_failures:
        print(f"⚠️  {len(parse_failures)} dosya adı parse edilemedi (atlandı)")
        print(f"   Örnekler: {parse_failures[:3]}")

    return df


def print_basic_stats(df: pd.DataFrame):
    """Temel istatistikleri yazdır."""
    print(f"\n{'='*70}")
    print("TEMEL İSTATİSTİKLER")
    print(f"{'='*70}")

    print("\n📊 Orijinal Kermany split dağılımı:")
    pivot = df.pivot_table(index="original_split", columns="class",
                            values="filename", aggfunc="count", fill_value=0)
    pivot = pivot.reindex(SPLITS)
    pivot["TOPLAM"] = pivot.sum(axis=1)
    print(pivot.to_string())

    print("\n👥 Hasta sayıları:")
    for split in SPLITS:
        sub = df[df["original_split"] == split]
        if len(sub) > 0:
            n_patients = sub["patient_id"].nunique()
            n_images = len(sub)
            avg_per_patient = n_images / n_patients if n_patients > 0 else 0
            print(f"  {split:5s}: {n_patients:5,} hasta, {n_images:6,} görüntü "
                  f"(hasta başı ort. {avg_per_patient:.1f})")

    print(f"\n📈 Sınıf dengesizliği (toplam):")
    class_counts = df["class"].value_counts()
    max_class = class_counts.max()
    for cls, count in class_counts.items():
        ratio = max_class / count
        print(f"  {cls:7s}: {count:6,} ({100*count/len(df):.1f}%) — "
              f"en kalabalığa oran: 1:{ratio:.2f}")

    print(f"\n⚖️  Dengesizlik oranı (en büyük/en küçük): "
          f"{max_class / class_counts.min():.2f}x")


def check_patient_leakage(df: pd.DataFrame) -> dict:
    """Tampu et al. 2022 - hastaların splitler arası taşması kontrolü."""
    print(f"\n{'='*70}")
    print("DATA LEAKAGE KONTROLÜ (Tampu et al., Sci Data 2022)")
    print(f"{'='*70}")

    leakage_report = {}
    splits_present = [s for s in SPLITS if (df["original_split"] == s).any()]

    for i, s1 in enumerate(splits_present):
        for s2 in splits_present[i+1:]:
            patients_s1 = set(df[df["original_split"] == s1]["patient_id"])
            patients_s2 = set(df[df["original_split"] == s2]["patient_id"])
            overlap = patients_s1 & patients_s2

            key = f"{s1}_vs_{s2}"
            leakage_report[key] = {
                "patients_in_first": len(patients_s1),
                "patients_in_second": len(patients_s2),
                "overlapping_patients": len(overlap),
                "overlapping_ids": sorted(list(overlap))[:20],  # ilk 20
            }

            status = "✓ TEMİZ" if len(overlap) == 0 else "⚠️ LEAKAGE VAR"
            print(f"\n  {s1.upper()} vs {s2.upper()}: {status}")
            print(f"    {s1}: {len(patients_s1)} hasta, {s2}: {len(patients_s2)} hasta")
            print(f"    Ortak hasta sayısı: {len(overlap)}")
            if len(overlap) > 0:
                print(f"    Örnek ortak ID'ler: {sorted(list(overlap))[:5]}")

    return leakage_report


def create_patient_level_split(df: pd.DataFrame, val_size: float = 0.15,
                                seed: int = RANDOM_SEED) -> pd.DataFrame:
    """
    Q1 yayını için hasta-seviyeli split oluştur.

    Strateji:
    - Orijinal TEST setini OLDUĞU GİBİ KORU (önceki literatürle karşılaştırılabilirlik için)
    - Orijinal TRAIN+VAL setini hasta seviyesinde train/val olarak yeniden böl
      (Kermany'nin orijinal val seti sadece 32 görüntü, kullanılamaz)
    - Sınıf-stratifiye edilmiş hasta bazlı bölme uygula
    """
    print(f"\n{'='*70}")
    print("HASTA-SEVİYELİ SPLİT OLUŞTURULUYOR")
    print(f"{'='*70}")

    df = df.copy()
    df["new_split"] = ""

    # Test setini olduğu gibi koru
    test_mask = df["original_split"] == "test"
    df.loc[test_mask, "new_split"] = "test"
    print(f"  ✓ Orijinal test seti korundu: {test_mask.sum():,} görüntü, "
          f"{df[test_mask]['patient_id'].nunique():,} hasta")

    # Train + val (eğer val varsa) → hasta bazlı yeniden böl
    train_pool = df[df["original_split"].isin(["train", "val"])].copy()

    # Her sınıf için, o sınıfın hastalarını ayır
    new_train_patients = set()
    new_val_patients = set()

    for cls in CLASSES:
        cls_patients = train_pool[train_pool["class"] == cls]["patient_id"].unique()
        if len(cls_patients) < 2:
            print(f"  UYARI: {cls} sınıfında çok az hasta ({len(cls_patients)})")
            continue

        train_pat, val_pat = train_test_split(
            cls_patients, test_size=val_size, random_state=seed
        )
        new_train_patients.update(train_pat)
        new_val_patients.update(val_pat)

    # Hasta ID'lerine göre yeni split atamasını yap
    for idx, row in train_pool.iterrows():
        pid = row["patient_id"]
        if pid in new_val_patients:
            df.loc[idx, "new_split"] = "val"
        elif pid in new_train_patients:
            df.loc[idx, "new_split"] = "train"

    # Doğrulama: hasta taşması kontrolü (yeni splitler arası)
    print(f"\n  YENİ SPLİT İSTATİSTİKLERİ:")
    print(f"  {'-'*60}")
    pivot_new = df[df["new_split"] != ""].pivot_table(
        index="new_split", columns="class", values="filename",
        aggfunc="count", fill_value=0
    ).reindex(["train", "val", "test"])
    pivot_new["TOPLAM"] = pivot_new.sum(axis=1)
    print(pivot_new.to_string())

    # Yeni splitler arası leakage kontrolü
    print(f"\n  YENİ SPLİT LEAKAGE KONTROLÜ:")
    new_splits = ["train", "val", "test"]
    for i, s1 in enumerate(new_splits):
        for s2 in new_splits[i+1:]:
            p1 = set(df[df["new_split"] == s1]["patient_id"])
            p2 = set(df[df["new_split"] == s2]["patient_id"])
            overlap = p1 & p2
            status = "✓ TEMİZ" if len(overlap) == 0 else "❌ HATA"
            print(f"    {s1} ∩ {s2}: {len(overlap)} hasta {status}")

    return df


def sample_image_sizes(df: pd.DataFrame, samples_per_class: int = 200) -> pd.DataFrame:
    """Hızlı görüntü boyutu istatistikleri için örnekleme yap."""
    print(f"\n{'='*70}")
    print(f"GÖRÜNTÜ BOYUTU ANALİZİ (her sınıftan {samples_per_class} örnek)")
    print(f"{'='*70}")

    size_records = []
    for cls in CLASSES:
        cls_df = df[df["class"] == cls]
        sample = cls_df.sample(min(samples_per_class, len(cls_df)),
                                random_state=RANDOM_SEED)
        for _, row in tqdm(sample.iterrows(), total=len(sample),
                            desc=f"  {cls}"):
            try:
                with Image.open(resolve_image(row["filepath"])) as img:
                    w, h = img.size
                    size_records.append({
                        "class": cls,
                        "width": w,
                        "height": h,
                        "aspect_ratio": w / h,
                    })
            except Exception as e:
                continue

    sizes_df = pd.DataFrame(size_records)
    print(f"\n  Boyut istatistikleri:")
    print(sizes_df.groupby("class")[["width", "height"]].describe().round(1))
    return sizes_df


# ============================================================================
# GÖRSELLEŞTİRMELER
# ============================================================================

def plot_class_distribution(df: pd.DataFrame, save_path: Path):
    """Sınıf dağılımı grafiği (orijinal splitler bazında)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Sol: orijinal split bazında
    pivot = df.pivot_table(index="class", columns="original_split",
                            values="filename", aggfunc="count", fill_value=0)
    pivot = pivot.reindex(CLASSES)
    if "test" in pivot.columns:
        pivot = pivot[["train", "val", "test"][:len(pivot.columns)]]

    pivot.plot(kind="bar", ax=axes[0], color=["#264653", "#e76f51", "#2a9d8f"])
    axes[0].set_title("Orijinal Kermany Split Bazında Sınıf Dağılımı", pad=10)
    axes[0].set_xlabel("Sınıf")
    axes[0].set_ylabel("Görüntü Sayısı")
    axes[0].set_xticklabels(CLASSES, rotation=0)
    axes[0].legend(title="Split")
    axes[0].grid(axis="y", alpha=0.3)

    # Sağ: toplam dağılım pasta + bar
    class_counts = df["class"].value_counts().reindex(CLASSES)
    bars = axes[1].bar(class_counts.index, class_counts.values,
                        color=[CLASS_COLORS[c] for c in CLASSES])
    axes[1].set_title("Toplam Sınıf Dağılımı", pad=10)
    axes[1].set_xlabel("Sınıf")
    axes[1].set_ylabel("Görüntü Sayısı")
    axes[1].grid(axis="y", alpha=0.3)

    # Bar üstüne değer ve yüzde yaz
    total = class_counts.sum()
    for bar, val in zip(bars, class_counts.values):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + total*0.01,
                     f"{val:,}\n({100*val/total:.1f}%)",
                     ha="center", va="bottom", fontsize=10)

    axes[1].set_ylim(0, class_counts.max() * 1.15)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  ✓ {save_path.name}")


def plot_patient_distribution(df: pd.DataFrame, save_path: Path):
    """Hasta sayısı ve hasta başına görüntü dağılımı."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Sol: hasta sayısı per class
    patients_per_class = df.groupby("class")["patient_id"].nunique().reindex(CLASSES)
    bars = axes[0].bar(patients_per_class.index, patients_per_class.values,
                        color=[CLASS_COLORS[c] for c in CLASSES])
    axes[0].set_title("Sınıf Başına Hasta Sayısı", pad=10)
    axes[0].set_xlabel("Sınıf")
    axes[0].set_ylabel("Benzersiz Hasta Sayısı")
    axes[0].grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, patients_per_class.values):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                     f"{val:,}", ha="center", va="bottom", fontsize=10)

    # Sağ: hasta başına görüntü dağılımı (histogram)
    images_per_patient = df.groupby(["class", "patient_id"]).size().reset_index(name="count")
    for cls in CLASSES:
        data = images_per_patient[images_per_patient["class"] == cls]["count"]
        axes[1].hist(data, bins=30, alpha=0.5, label=cls, color=CLASS_COLORS[cls])

    axes[1].set_title("Hasta Başına Görüntü Sayısı Dağılımı", pad=10)
    axes[1].set_xlabel("Hasta Başına Görüntü Sayısı")
    axes[1].set_ylabel("Hasta Sayısı (frekans)")
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.3)
    axes[1].set_yscale("log")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  ✓ {save_path.name}")


def plot_sample_images(df: pd.DataFrame, save_path: Path, n_per_class: int = 4):
    """Her sınıftan örnek görüntüler."""
    fig, axes = plt.subplots(len(CLASSES), n_per_class,
                              figsize=(n_per_class * 3, len(CLASSES) * 3))

    for i, cls in enumerate(CLASSES):
        cls_df = df[df["class"] == cls].sample(n_per_class, random_state=RANDOM_SEED)
        for j, (_, row) in enumerate(cls_df.iterrows()):
            try:
                img = Image.open(resolve_image(row["filepath"])).convert("L")
                axes[i, j].imshow(img, cmap="gray")
            except Exception as e:
                axes[i, j].text(0.5, 0.5, "Yüklenemedi", ha="center", va="center")

            axes[i, j].axis("off")
            if j == 0:
                axes[i, j].set_ylabel(cls, fontsize=14, fontweight="bold",
                                       color=CLASS_COLORS[cls], rotation=0,
                                       labelpad=40, va="center")

    plt.suptitle("Her Sınıftan Örnek OCT Görüntüleri", fontsize=14, y=1.00)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  ✓ {save_path.name}")


def plot_image_size_distribution(sizes_df: pd.DataFrame, save_path: Path):
    """Görüntü boyutu dağılımı."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Genişlik
    for cls in CLASSES:
        data = sizes_df[sizes_df["class"] == cls]["width"]
        axes[0].hist(data, bins=30, alpha=0.5, label=cls, color=CLASS_COLORS[cls])
    axes[0].set_title("Genişlik Dağılımı")
    axes[0].set_xlabel("Genişlik (piksel)")
    axes[0].set_ylabel("Frekans")
    axes[0].legend()

    # Yükseklik
    for cls in CLASSES:
        data = sizes_df[sizes_df["class"] == cls]["height"]
        axes[1].hist(data, bins=30, alpha=0.5, label=cls, color=CLASS_COLORS[cls])
    axes[1].set_title("Yükseklik Dağılımı")
    axes[1].set_xlabel("Yükseklik (piksel)")
    axes[1].legend()

    # En-boy oranı
    for cls in CLASSES:
        data = sizes_df[sizes_df["class"] == cls]["aspect_ratio"]
        axes[2].hist(data, bins=30, alpha=0.5, label=cls, color=CLASS_COLORS[cls])
    axes[2].set_title("En/Boy Oranı")
    axes[2].set_xlabel("Genişlik / Yükseklik")
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  ✓ {save_path.name}")


def plot_new_split_distribution(df: pd.DataFrame, save_path: Path):
    """Yeni hasta-seviyeli split dağılımı."""
    df_split = df[df["new_split"] != ""]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    pivot = df_split.pivot_table(index="class", columns="new_split",
                                  values="filename", aggfunc="count", fill_value=0)
    pivot = pivot.reindex(CLASSES)
    if all(s in pivot.columns for s in ["train", "val", "test"]):
        pivot = pivot[["train", "val", "test"]]

    pivot.plot(kind="bar", ax=axes[0], color=["#264653", "#e76f51", "#2a9d8f"])
    axes[0].set_title("Yeni Hasta-Seviyeli Split — Görüntü Sayıları", pad=10)
    axes[0].set_xlabel("Sınıf")
    axes[0].set_ylabel("Görüntü Sayısı")
    axes[0].set_xticklabels(CLASSES, rotation=0)
    axes[0].legend(title="Split")
    axes[0].grid(axis="y", alpha=0.3)

    # Hasta sayıları
    patients_pivot = df_split.groupby(["class", "new_split"])["patient_id"].nunique().unstack(fill_value=0)
    patients_pivot = patients_pivot.reindex(CLASSES)
    if all(s in patients_pivot.columns for s in ["train", "val", "test"]):
        patients_pivot = patients_pivot[["train", "val", "test"]]

    patients_pivot.plot(kind="bar", ax=axes[1], color=["#264653", "#e76f51", "#2a9d8f"])
    axes[1].set_title("Yeni Hasta-Seviyeli Split — Hasta Sayıları", pad=10)
    axes[1].set_xlabel("Sınıf")
    axes[1].set_ylabel("Hasta Sayısı")
    axes[1].set_xticklabels(CLASSES, rotation=0)
    axes[1].legend(title="Split")
    axes[1].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  ✓ {save_path.name}")


# ============================================================================
# ANA AKIŞ
# ============================================================================

def main():
    print(f"\n{'#'*70}")
    print(f"# KERMANY OCT VERİ SETİ — EDA + HASTA-SEVİYELİ SPLİT")
    print(f"# Çalıştırma zamanı: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*70}")

    # 1. Veri setini tara
    df = build_dataframe(DATA_ROOT)

    # 2. Temel istatistikler
    print_basic_stats(df)

    # 3. Data leakage kontrolü (orijinal splitler)
    leakage_report = check_patient_leakage(df)

    # 4. Hasta-seviyeli yeni split oluştur
    df = create_patient_level_split(df, val_size=0.15)

    # 5. Görüntü boyutu örneklemesi
    sizes_df = sample_image_sizes(df, samples_per_class=200)

    # 6. Figürler üret
    print(f"\n{'='*70}")
    print("FİGÜRLER OLUŞTURULUYOR")
    print(f"{'='*70}")

    plot_class_distribution(df, FIGURES_DIR / "01_class_distribution.png")
    plot_patient_distribution(df, FIGURES_DIR / "02_patient_distribution.png")
    plot_sample_images(df, FIGURES_DIR / "03_sample_images.png", n_per_class=4)
    plot_image_size_distribution(sizes_df, FIGURES_DIR / "04_image_sizes.png")
    plot_new_split_distribution(df, FIGURES_DIR / "05_new_split_distribution.png")

    # 7. Split CSV'leri kaydet
    print(f"\n{'='*70}")
    print("SPLIT DOSYALARI KAYDEDİLİYOR")
    print(f"{'='*70}")
    for split_name in ["train", "val", "test"]:
        split_df = df[df["new_split"] == split_name][
            ["filepath", "filename", "class", "patient_id", "image_num"]
        ]
        save_path = SPLITS_DIR / f"{split_name}.csv"
        split_df.to_csv(save_path, index=False)
        print(f"  ✓ {save_path.name}: {len(split_df):,} görüntü")

    # Tam dataframe'i de kaydet
    df.to_csv(OUTPUT_DIR / "full_dataset_index.csv", index=False)
    print(f"  ✓ full_dataset_index.csv: {len(df):,} görüntü (tam index)")

    # 8. JSON rapor
    report = {
        "timestamp": datetime.now().isoformat(),
        "data_root": str(DATA_ROOT),
        "total_images": int(len(df)),
        "total_patients": int(df["patient_id"].nunique()),
        "classes": CLASSES,
        "original_split_counts": df.groupby(["original_split", "class"]).size().unstack(fill_value=0).to_dict(),
        "new_split_counts": df[df["new_split"] != ""].groupby(["new_split", "class"]).size().unstack(fill_value=0).to_dict(),
        "patient_counts_per_class": df.groupby("class")["patient_id"].nunique().to_dict(),
        "leakage_check_original_splits": {
            k: {kk: vv for kk, vv in v.items() if kk != "overlapping_ids"}
            for k, v in leakage_report.items()
        },
        "image_size_stats": sizes_df.groupby("class")[["width", "height"]].describe().round(2).to_dict(),
        "class_imbalance_ratio": float(df["class"].value_counts().max() / df["class"].value_counts().min()),
    }

    with open(REPORTS_DIR / "eda_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n  ✓ JSON raporu: {REPORTS_DIR / 'eda_report.json'}")

    # ÖZET
    print(f"\n{'#'*70}")
    print(f"# TAMAMLANDI ✓")
    print(f"{'#'*70}")
    print(f"\nÇıktılar:")
    print(f"  📁 Figürler:    {FIGURES_DIR}")
    print(f"  📁 Splitler:    {SPLITS_DIR}")
    print(f"  📁 Raporlar:    {REPORTS_DIR}")
    print(f"\n👉 Sonraki adım: Bu çıktıları benimle paylaş, model eğitimine geçelim.")


if __name__ == "__main__":
    main()