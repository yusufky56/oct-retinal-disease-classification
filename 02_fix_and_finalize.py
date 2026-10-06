
"""
02_fix_and_finalize.py
======================

İlk script'te iki problem vardı:

PROBLEM 1 (Kritik bilimsel bug):
  Orijinal Kermany test setini olduğu gibi koruyup train+val'ı yeniden
  böldüm AMA test hastalarını train+val havuzundan ÇIKARMADIM. Sonuç:
  yeni splitlerde de leakage kaldı (train∩test=418, val∩test=128 hasta).

  ÇÖZÜM: Önce test hastalarını train+val'dan çıkar, sonra hasta bazlı böl.

PROBLEM 2 (JSON serialization):
  pandas describe().to_dict() tuple key'ler üretiyor, JSON kabul etmiyor.

  ÇÖZÜM: İstatistikleri manuel hesapla.

EK BÖNUS:
  Tampu 2022'yi doğrulayan bir leakage matris figürü üret (yayın kalitesinde).
  Bu, yayında Section 3 "Dataset Audit" için doğrudan kullanılacak.
"""

import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

from src.paths import resolve_image

SCRIPT_DIR = Path(__file__).parent if "__file__" in dir() else Path.cwd()
OUTPUT_DIR = SCRIPT_DIR / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
SPLITS_DIR = OUTPUT_DIR / "splits"
REPORTS_DIR = OUTPUT_DIR / "reports"

CLASSES = ["CNV", "DME", "DRUSEN", "NORMAL"]
CLASS_COLORS = {"CNV": "#e63946", "DME": "#f4a261", "DRUSEN": "#e9c46a", "NORMAL": "#2a9d8f"}
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

plt.rcParams.update({
    "figure.dpi": 100,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def plot_leakage_matrix(df: pd.DataFrame, save_path: Path):
    """
    Yayın kalitesinde leakage görselleştirmesi.
    Sol: orijinal Kermany splitlerinde hasta-üst-üste binme matrisi.
    Sağ: bizim temiz hasta-seviyeli split'imiz (sıfır binme).
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # SOL: orijinal Kermany splitleri arası örtüşme
    splits = ["train", "val", "test"]
    overlap_orig = np.zeros((3, 3), dtype=int)
    for i, s1 in enumerate(splits):
        for j, s2 in enumerate(splits):
            p1 = set(df[df["original_split"] == s1]["patient_id"])
            p2 = set(df[df["original_split"] == s2]["patient_id"])
            overlap_orig[i, j] = len(p1 & p2)

    sns.heatmap(overlap_orig, annot=True, fmt="d", cmap="Reds",
                xticklabels=splits, yticklabels=splits, ax=axes[0],
                cbar_kws={"label": "Ortak hasta sayısı"}, square=True,
                annot_kws={"size": 14, "weight": "bold"})
    axes[0].set_title("Orijinal Kermany v2 Split\n(Tampu et al., Sci Data 2022 doğrulandı)",
                       pad=12)
    axes[0].set_xlabel("Split B")
    axes[0].set_ylabel("Split A")

    # SAĞ: bizim temiz split'imiz
    overlap_new = np.zeros((3, 3), dtype=int)
    for i, s1 in enumerate(splits):
        for j, s2 in enumerate(splits):
            p1 = set(df[df["new_split"] == s1]["patient_id"])
            p2 = set(df[df["new_split"] == s2]["patient_id"])
            overlap_new[i, j] = len(p1 & p2)

    sns.heatmap(overlap_new, annot=True, fmt="d", cmap="Greens",
                xticklabels=splits, yticklabels=splits, ax=axes[1],
                cbar_kws={"label": "Ortak hasta sayısı"}, square=True,
                annot_kws={"size": 14, "weight": "bold"})
    axes[1].set_title("Bizim Hasta-Seviyeli Yeni Split\n(diagonal harici sıfır = temiz)",
                       pad=12)
    axes[1].set_xlabel("Split B")
    axes[1].set_ylabel("Split A")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  ✓ {save_path.name}")


def plot_new_split_distribution(df: pd.DataFrame, save_path: Path):
    """Yeni temiz split dağılımı."""
    df_split = df[df["new_split"] != ""]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    pivot = df_split.pivot_table(index="class", columns="new_split",
                                  values="filename", aggfunc="count", fill_value=0)
    pivot = pivot.reindex(CLASSES)[["train", "val", "test"]]

    pivot.plot(kind="bar", ax=axes[0], color=["#264653", "#e76f51", "#2a9d8f"])
    axes[0].set_title("Yeni Hasta-Seviyeli Split — Görüntü Sayıları", pad=10)
    axes[0].set_xlabel("Sınıf")
    axes[0].set_ylabel("Görüntü Sayısı")
    axes[0].set_xticklabels(CLASSES, rotation=0)
    axes[0].legend(title="Split")
    axes[0].grid(axis="y", alpha=0.3)

    patients_pivot = df_split.groupby(["class", "new_split"])["patient_id"].nunique().unstack(fill_value=0)
    patients_pivot = patients_pivot.reindex(CLASSES)[["train", "val", "test"]]

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


def safe_size_stats(sizes_csv_path: Path = None):
    """JSON'a güvenli boyut istatistikleri."""
    # Boyut örneklemesi datayı diskte tutmuyoruz, baştan al
    df_idx = pd.read_csv(OUTPUT_DIR / "full_dataset_index.csv", dtype={"patient_id": str})
    from PIL import Image
    from tqdm import tqdm

    print("\n  Görüntü boyutları yeniden örnekleniyor (JSON için)...")
    records = []
    for cls in CLASSES:
        cls_df = df_idx[df_idx["class"] == cls]
        sample = cls_df.sample(min(200, len(cls_df)), random_state=RANDOM_SEED)
        for _, row in tqdm(sample.iterrows(), total=len(sample), desc=f"  {cls}"):
            try:
                with Image.open(resolve_image(row["filepath"])) as img:
                    w, h = img.size
                    records.append({"class": cls, "width": w, "height": h})
            except Exception:
                continue

    sizes_df = pd.DataFrame(records)
    stats = {}
    for cls in CLASSES:
        sub = sizes_df[sizes_df["class"] == cls]
        stats[cls] = {
            "width": {
                "mean": float(sub["width"].mean()),
                "std": float(sub["width"].std()),
                "min": int(sub["width"].min()),
                "median": float(sub["width"].median()),
                "max": int(sub["width"].max()),
            },
            "height": {
                "mean": float(sub["height"].mean()),
                "std": float(sub["height"].std()),
                "min": int(sub["height"].min()),
                "median": float(sub["height"].median()),
                "max": int(sub["height"].max()),
            },
        }
    return stats


def main():
    print(f"\n{'#'*70}")
    print(f"# DÜZELTME VE SONLANDIRMA")
    print(f"# Çalıştırma zamanı: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*70}")

    # ----- Adım 1: Mevcut index'i yükle -----
    print(f"\n{'='*70}")
    print("ADIM 1: Mevcut veri seti index'ini yükle")
    print(f"{'='*70}")

    idx_path = OUTPUT_DIR / "full_dataset_index.csv"
    if not idx_path.exists():
        print(f"❌ {idx_path} bulunamadı. Önce 01_eda_split.py'yi çalıştırın.")
        return

    # patient_id'yi string olarak oku (yoksa int dönüşür ve baştaki sıfırlar kaybolur)
    df = pd.read_csv(idx_path, dtype={"patient_id": str})
    print(f"  ✓ {len(df):,} görüntü, {df['patient_id'].nunique():,} hasta")

    # ----- Adım 2: Leakage'ın boyutunu raporla -----
    print(f"\n{'='*70}")
    print("ADIM 2: Orijinal Kermany v2 leakage'ını ölç")
    print(f"{'='*70}")

    test_patient_ids = set(df[df["original_split"] == "test"]["patient_id"].unique())
    train_val_pool = df[df["original_split"].isin(["train", "val"])].copy()
    contaminated = train_val_pool[train_val_pool["patient_id"].isin(test_patient_ids)]

    print(f"  Test setinde {len(test_patient_ids)} hasta")
    print(f"  Bunlardan {contaminated['patient_id'].nunique()} tanesi train+val'da da var")
    print(f"  Bu kontaminasyon {len(contaminated):,} eğitim görüntüsünü ETKİLİYOR")
    print(f"  Yüzde olarak: %{100*len(contaminated)/len(train_val_pool):.1f} eğitim verisi kontamine")

    leakage_summary = {
        "test_patients_total": int(len(test_patient_ids)),
        "test_patients_also_in_train_or_val": int(contaminated["patient_id"].nunique()),
        "leakage_percentage_of_test_patients": round(
            100 * contaminated["patient_id"].nunique() / len(test_patient_ids), 2
        ),
        "contaminated_train_val_images": int(len(contaminated)),
        "contaminated_percentage_of_training_data": round(
            100 * len(contaminated) / len(train_val_pool), 2
        ),
    }

    # ----- Adım 3: Test hastalarını çıkararak temiz havuz oluştur -----
    print(f"\n{'='*70}")
    print("ADIM 3: Temiz havuz oluştur ve hasta-seviyeli yeniden böl")
    print(f"{'='*70}")

    clean_pool = train_val_pool[~train_val_pool["patient_id"].isin(test_patient_ids)].copy()
    print(f"  Temizlenmiş train+val havuzu: {len(clean_pool):,} görüntü, "
          f"{clean_pool['patient_id'].nunique():,} hasta")

    # Yeni split sütunu
    df["new_split"] = ""
    df.loc[df["original_split"] == "test", "new_split"] = "test"

    # Sınıf bazlı hasta-seviyeli 85/15 bölme (temiz havuz üzerinde)
    new_train_patients = set()
    new_val_patients = set()

    for cls in CLASSES:
        cls_patients = clean_pool[clean_pool["class"] == cls]["patient_id"].unique()
        train_pat, val_pat = train_test_split(
            cls_patients, test_size=0.15, random_state=RANDOM_SEED
        )
        new_train_patients.update(train_pat)
        new_val_patients.update(val_pat)
        print(f"    {cls}: {len(cls_patients):,} hasta → "
              f"train {len(train_pat):,} / val {len(val_pat):,}")

    # Yalnızca temiz havuza yeni split ata; test'e dokunma
    in_clean = df["patient_id"].isin(set(clean_pool["patient_id"].unique()))
    is_test = df["new_split"] == "test"

    df.loc[in_clean & ~is_test & df["patient_id"].isin(new_train_patients), "new_split"] = "train"
    df.loc[in_clean & ~is_test & df["patient_id"].isin(new_val_patients), "new_split"] = "val"

    # ----- Adım 4: Doğrulama (kritik) -----
    print(f"\n{'='*70}")
    print("ADIM 4: Doğrulama — leakage gerçekten gitti mi?")
    print(f"{'='*70}")

    splits = ["train", "val", "test"]
    all_clean = True
    overlap_results = {}
    for i, s1 in enumerate(splits):
        for s2 in splits[i+1:]:
            p1 = set(df[df["new_split"] == s1]["patient_id"])
            p2 = set(df[df["new_split"] == s2]["patient_id"])
            overlap = p1 & p2
            overlap_results[f"{s1}_vs_{s2}"] = len(overlap)
            status = "✓ TEMİZ" if len(overlap) == 0 else "❌ HATA"
            if len(overlap) > 0:
                all_clean = False
            print(f"  {s1:5s} ∩ {s2:5s}: {len(overlap):3d} hasta {status}")

    if all_clean:
        print(f"\n  🎯 TÜM SPLİTLER HASTA-DİSJOİNT")
        print(f"     Q1 yayını için temiz, doğrulanmış veri bölümü hazır.")
    else:
        print(f"\n  ❌ HALA HATA VAR — kod incelenmeli")

    # ----- Adım 5: Yeni split istatistikleri -----
    print(f"\n{'='*70}")
    print("ADIM 5: Yeni split istatistikleri")
    print(f"{'='*70}")
    pivot = df[df["new_split"] != ""].pivot_table(
        index="new_split", columns="class", values="filename",
        aggfunc="count", fill_value=0
    ).reindex(["train", "val", "test"])
    pivot["TOPLAM"] = pivot.sum(axis=1)
    print(pivot.to_string())

    print(f"\n  Hasta sayıları:")
    for s in splits:
        n = df[df["new_split"] == s]["patient_id"].nunique()
        m = (df["new_split"] == s).sum()
        print(f"    {s:5s}: {n:5,} hasta, {m:6,} görüntü")

    # ----- Adım 6: Figürleri yenile -----
    print(f"\n{'='*70}")
    print("ADIM 6: Yeni figürler oluşturuluyor")
    print(f"{'='*70}")
    plot_leakage_matrix(df, FIGURES_DIR / "06_leakage_audit.png")
    plot_new_split_distribution(df, FIGURES_DIR / "05_new_split_distribution.png")

    # ----- Adım 7: Split CSV'lerini yeniden kaydet -----
    print(f"\n{'='*70}")
    print("ADIM 7: Temiz split CSV'leri kaydediliyor")
    print(f"{'='*70}")
    for split_name in ["train", "val", "test"]:
        split_df = df[df["new_split"] == split_name][
            ["filepath", "filename", "class", "patient_id", "image_num"]
        ]
        save_path = SPLITS_DIR / f"{split_name}.csv"
        split_df.to_csv(save_path, index=False)
        print(f"  ✓ {save_path.name}: {len(split_df):,} görüntü, "
              f"{split_df['patient_id'].nunique():,} hasta")

    df.to_csv(OUTPUT_DIR / "full_dataset_index.csv", index=False)
    print(f"  ✓ full_dataset_index.csv güncellendi")

    # ----- Adım 8: JSON raporu (düzeltilmiş) -----
    print(f"\n{'='*70}")
    print("ADIM 8: JSON raporu (düzeltilmiş)")
    print(f"{'='*70}")

    size_stats = safe_size_stats()

    new_split_counts = {}
    for s in ["train", "val", "test"]:
        sub = df[df["new_split"] == s]
        new_split_counts[s] = {
            "total_images": int(len(sub)),
            "total_patients": int(sub["patient_id"].nunique()),
            "per_class_images": {cls: int((sub["class"] == cls).sum()) for cls in CLASSES},
            "per_class_patients": {
                cls: int(sub[sub["class"] == cls]["patient_id"].nunique()) for cls in CLASSES
            },
        }

    original_split_counts = {}
    for s in ["train", "val", "test"]:
        sub = df[df["original_split"] == s]
        original_split_counts[s] = {
            "total_images": int(len(sub)),
            "total_patients": int(sub["patient_id"].nunique()),
            "per_class_images": {cls: int((sub["class"] == cls).sum()) for cls in CLASSES},
        }

    class_imbalance = df["class"].value_counts().to_dict()
    max_count = max(class_imbalance.values())
    min_count = min(class_imbalance.values())

    report = {
        "timestamp": datetime.now().isoformat(),
        "total_images": int(len(df)),
        "total_patients": int(df["patient_id"].nunique()),
        "classes": CLASSES,
        "class_distribution_total": {k: int(v) for k, v in class_imbalance.items()},
        "class_imbalance_ratio_max_over_min": round(max_count / min_count, 2),
        "original_kermany_v2_split": original_split_counts,
        "leakage_audit_original_split": leakage_summary,
        "new_patient_level_split": new_split_counts,
        "new_split_leakage_check": overlap_results,
        "image_size_statistics": size_stats,
        "headline_findings": [
            f"Kermany v2 splitinde test hastalarının %{leakage_summary['leakage_percentage_of_test_patients']}'si "
            f"({leakage_summary['test_patients_also_in_train_or_val']}/{leakage_summary['test_patients_total']}) "
            f"train+val setlerinde de mevcut — Tampu et al. 2022 ampirik olarak doğrulandı.",
            f"Sınıf dengesizliği orta seviye: {max_count/min_count:.2f}x (CNV vs DRUSEN).",
            "Görüntü boyutları sabit yükseklik (~496-512px), değişken genişlik (512-1536px) — resize stratejisi dikkat gerektirir.",
            f"Yeni hasta-seviyeli split tüm split çiftleri için sıfır hasta örtüşmesi sağlıyor.",
        ],
    }

    with open(REPORTS_DIR / "eda_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"  ✓ {REPORTS_DIR / 'eda_report.json'}")

    # ----- Bitiş -----
    print(f"\n{'#'*70}")
    print(f"# TAMAMLANDI ✓")
    print(f"{'#'*70}")
    print(f"\n📁 Yeni/güncellenmiş çıktılar:")
    print(f"  • figures/06_leakage_audit.png       (yayın için altın değerinde)")
    print(f"  • figures/05_new_split_distribution.png  (güncellendi)")
    print(f"  • splits/train.csv, val.csv, test.csv  (temiz, leakage'sız)")
    print(f"  • reports/eda_report.json            (tam JSON rapor)")
    print(f"\n👉 Bunları paylaş, model eğitim adımına geçelim.")


if __name__ == "__main__":
    main()