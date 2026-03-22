# IntentFormer: Pipeline & Workflow Guide

Bu rehber, **IntentFormer** projesindeki modelleri eğitmek, değerlendirmek ve interaktif arayüzü çalıştırmak için gereken tüm komutları ve süreçleri içerir.

---

## 1. Modeli Eğitme (Training)

Modeli `H2O`, `HOT3D` veya her ikisinin birleşimi (`combined`) üzerinde eğitebilirsiniz.

### H2O Tekli Eğitim (Varsayılan)
```bash
python -m src.train --dataset h2o --epochs 60 --batch_size 64 --lr 3e-4
```

### Combined (H2O + HOT3D) Gelişmiş Eğitim
En yüksek isabet oranı için önerilen "Büyük Model" konfigürasyonu:
```bash
python -m src.train \
    --dataset combined \
    --fusion shared_head \
    --data_root data/h2o \
    --hot3d_root data/hot3d \
    --epochs 100 \
    --d_model 256 \
    --nhead 8 \
    --num_layers 6 \
    --dim_ff 1024 \
    --out_dir checkpoints/combined_best
```

**Önemli Parametreler:**
- `--obs_ratios`: "0.2,0.25,0.3" (Modelin ne kadar erken tahmin yapacağını belirler)
- `--fusion`: `concat` (tüm sınıflar) veya `shared_head` (3 genel sınıf: H2O, Doğru, Yanlış)
- `--device`: `mps` (Apple Silicon), `cuda` (NVIDIA) veya `cpu`

---

## 2. Model Değerlendirme (Evaluation)

Eğitilen bir modelin performans metriklerini (Accuracy, Precision, Recall) ölçmek için:

```bash
python -m src.evaluate \
    --dataset h2o \
    --checkpoint checkpoints/combined_best/best_model.pt
```

---

## 3. İnteraktif Test Arayüzü (Web UI)

Görselleştirme, iskelet eller, 3D obje ve Ghost Hand takibi için web tabanlı arayüzü başlatın.

```bash
python -m src.tests.test_ui_server --checkpoint checkpoints/combined_best/best_model.pt
```

**Kullanım:**
1. Tarayıcıda `http://127.0.0.1:5001` adresine gidin.
2. **Dataset** sekmesinden bir örnek seçin.
3. **Run Inference** butonuna basın.
4. **Play** butonuna basarak 3D alanı inceleyin.

---

## 4. Modeli CoreML'e Dönüştürme (Export)

Apple Vision Pro veya iOS cihazlarda ANE (Apple Neural Engine) üzerinde çalıştırmak için:

```bash
python -m src.export_coreml \
    --checkpoint checkpoints/combined_best/best_model.pt \
    --out_dir exports/
```

---

## 5. Benchmarking (Hız Testi)

Cihazınızın saniyede kaç çıkarım (Inference) yapabildiğini ölçmek için:

```bash
python -m src.benchmark_mps
```

---

> [!TIP]
> **Eğitim İpucu:** Eğer model overfitting (aşırı öğrenme) yapıyorsa, `--dropout 0.2` ve `--weight_decay 1e-3` parametrelerini artırmayı deneyebilirsiniz.
