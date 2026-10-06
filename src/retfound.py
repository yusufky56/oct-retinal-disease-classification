"""
src/retfound.py
===============

RETFound (Zhou et al., Nature 2023) modelinin yüklenmesi ve LoRA ile
parametre-verimli ince ayara hazırlanması için yardımcı modül.

DÜZELTME (v2):
1. task_type="FEATURE_EXTRACTION" parametresi kaldırıldı.
   PEFT bu task_type'ta bile NLP-style input_ids geçirmeye çalışıyordu.
   None bırakınca PEFT generic forward kullanır (timm ViT'siyle uyumlu).
2. norm.weight/norm.bias → fc_norm.weight/fc_norm.bias rename eklendi.
   timm vit_large_patch16_224 global_pool='avg' modunda fc_norm kullanır,
   RETFound checkpoint'i ise norm. ile kaydetmiş.
"""

import os
import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import timm

try:
    from peft import LoraConfig, get_peft_model
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False


RETFOUND_HF_REPO = "YukunZhou/RETFound_mae_natureOCT"
RETFOUND_LOCAL_FILENAME = "RETFound_mae_natureOCT.pth"


def build_retfound_vit_large(num_classes: int = 4, drop_path_rate: float = 0.2):
    """RETFound ile uyumlu ViT-L/16 mimarisi."""
    model = timm.create_model(
        "vit_large_patch16_224",
        pretrained=False,
        num_classes=num_classes,
        drop_path_rate=drop_path_rate,
        global_pool="avg",
    )
    return model


def load_retfound_weights(model: nn.Module, checkpoint_path: str,
                            verbose: bool = True) -> nn.Module:
    """RETFound MAE ön-eğitim ağırlıklarını ViT-L modeline yükle."""
    print(f"  RETFound checkpoint yükleniyor: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    if "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    new_state_dict = {}
    skipped = []
    renamed = 0
    for k, v in state_dict.items():
        if k.startswith("decoder") or "decoder_" in k or k.startswith("mask_token"):
            skipped.append(k)
            continue
        if k.startswith("head.") or k == "head.weight" or k == "head.bias":
            skipped.append(k)
            continue
        # norm.weight, norm.bias → fc_norm.weight, fc_norm.bias
        if k == "norm.weight" or k == "norm.bias":
            new_key = "fc_" + k
            new_state_dict[new_key] = v
            renamed += 1
            continue
        new_state_dict[k] = v

    msg = model.load_state_dict(new_state_dict, strict=False)

    if verbose:
        print(f"  ✓ Yüklendi. {len(skipped)} key atlandı (decoder + eski head)")
        if renamed > 0:
            print(f"  ✓ {renamed} key yeniden adlandırıldı (norm.X → fc_norm.X)")
        if msg.missing_keys:
            critical_missing = [k for k in msg.missing_keys
                                  if not (k.startswith("head.") or "fc_norm" in k)]
            if critical_missing:
                print(f"  ⚠️  Eksik anahtarlar: {critical_missing[:3]}...")
        if msg.unexpected_keys:
            print(f"  ⚠️  Beklenmeyen anahtarlar: {msg.unexpected_keys[:3]}")

    return model


def apply_lora(model: nn.Module, rank: int = 16, alpha: int = 32,
                dropout: float = 0.1, target_modules: Optional[list] = None):
    """
    LoRA adapter'ları uygula.

    DÜZELTME: task_type SET EDİLMİYOR (None). task_type="FEATURE_EXTRACTION"
    bile PEFT'in NLP-style input_ids beklemesine yol açıyordu.
    None ise PEFT generic forward kullanır, args/kwargs olduğu gibi geçer.
    """
    if not PEFT_AVAILABLE:
        raise ImportError("peft kütüphanesi gerekli. Kurulum: pip install peft")

    if target_modules is None:
        target_modules = ["qkv"]

    config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=target_modules,
        lora_dropout=dropout,
        bias="none",
        # task_type'ı SET ETMİYORUZ — None bırakıyoruz
        modules_to_save=["head"],
    )

    model = get_peft_model(model, config)
    return model


def download_retfound_checkpoint(target_dir: Path,
                                    force_download: bool = False) -> Path:
    """RETFound checkpoint'ini HuggingFace Hub'dan indir."""
    target_dir.mkdir(parents=True, exist_ok=True)
    target_file = target_dir / RETFOUND_LOCAL_FILENAME

    if target_file.exists() and not force_download:
        size_mb = target_file.stat().st_size / 1e6
        print(f"  ✓ Checkpoint zaten mevcut: {target_file} ({size_mb:.1f} MB)")
        return target_file

    print(f"  RETFound checkpoint HuggingFace'ten indiriliyor...")
    print(f"  Hedef: {target_file}")
    print(f"  (~1.2 GB, internet hızına göre 5-15 dakika sürebilir)")

    try:
        from huggingface_hub import hf_hub_download
        token = os.environ.get("HF_TOKEN", None)
        downloaded_path = hf_hub_download(
            repo_id=RETFOUND_HF_REPO,
            filename="RETFound_mae_natureOCT.pth",
            local_dir=str(target_dir),
            token=token,
        )
        downloaded_path = Path(downloaded_path)
        if downloaded_path != target_file:
            import shutil
            shutil.move(str(downloaded_path), str(target_file))

        size_mb = target_file.stat().st_size / 1e6
        print(f"  ✓ İndirildi: {target_file} ({size_mb:.1f} MB)")
        return target_file
    except Exception as e:
        print(f"\n  ❌ İndirme başarısız: {e}")
        raise


def build_retfound_for_finetune(num_classes: int = 4,
                                   checkpoint_path: Optional[Path] = None,
                                   use_lora: bool = True,
                                   lora_rank: int = 16,
                                   lora_alpha: int = 32):
    """Eğitime hazır RETFound modeli oluştur."""
    print("\n  RETFound modeli kuruluyor...")
    model = build_retfound_vit_large(num_classes=num_classes)

    if checkpoint_path is not None and Path(checkpoint_path).exists():
        model = load_retfound_weights(model, str(checkpoint_path))
    else:
        print("  ⚠️  Checkpoint yok, modelin random init halinde başlayacak")

    info = {
        "total_params": sum(p.numel() for p in model.parameters()),
        "before_lora_trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }

    if use_lora:
        print(f"\n  LoRA uygulanıyor (rank={lora_rank}, alpha={lora_alpha})...")
        model = apply_lora(model, rank=lora_rank, alpha=lora_alpha)
        info["after_lora_trainable"] = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        trainable_pct = 100 * info["after_lora_trainable"] / info["total_params"]
        print(f"  ✓ Toplam parametre: {info['total_params']/1e6:.1f}M")
        print(f"  ✓ Eğitilebilir (LoRA + head): {info['after_lora_trainable']/1e6:.2f}M "
              f"(%{trainable_pct:.2f})")
    else:
        print(f"\n  Full fine-tune modu (LoRA yok)")
        print(f"  ✓ Toplam parametre: {info['total_params']/1e6:.1f}M")

    return model, info


if __name__ == "__main__":
    print("RETFound modülü hızlı testi:")
    print("=" * 50)

    model = build_retfound_vit_large(num_classes=4)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"\n✓ ViT-L/16 mimarisi: {n_params:.1f}M parametre")

    dummy = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out = model(dummy)
    print(f"✓ Forward pass: {tuple(dummy.shape)} → {tuple(out.shape)}")

    if PEFT_AVAILABLE:
        print("✓ PEFT yüklü")
        model_with_lora = apply_lora(model, rank=8)
        with torch.no_grad():
            out2 = model_with_lora(dummy)
        print(f"✓ LoRA forward: {tuple(out2.shape)}")
    else:
        print("⚠️  PEFT kurulu değil")
