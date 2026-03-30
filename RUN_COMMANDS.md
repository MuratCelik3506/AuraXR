# Tüm Çalıştırma Komutları

Bu dosya projedeki tüm ana çalıştırma komutlarını içermektedir.

---

## 📦 Kurulum

### Bağımlılıkları Yükle
```bash
pip install -r requirements.txt
```

### MediaPipe Uyumluluk (ARM64 Mac için)
```bash
pip install mediapipe==0.10.14
```

---

## 📊 Veri Hazırlama

### HOT3D Veri Setini İndir
```bash
# HuggingFace'e giriş yap (dataset erişimi gerekli)
huggingface-cli login

# 10 clip indir (test için)
python scripts/download_hot3d.py --max_clips 10 --device Aria

# Tüm Aria cliplerini indir
python scripts/download_hot3d.py --device Aria

# Quest 3 cliplerini indir
python scripts/download_hot3d.py --device Quest3
```

---

## 🏋️ Model Eğitimi

### Temel Eğitim (H2O Dataset)
```bash
python -m src.train \
    --dataset h2o \
    --epochs 100 \
    --batch_size 64 \
    --lr 0.0001 \
    --device mps
```

### Combined Dataset ile Eğitim (Shared Head)
```bash
python -m src.train \
    --dataset combined \
    --fusion shared_head \
    --epochs 100 \
    --batch_size 64 \
    --lr 0.0001 \
    --device mps
```

### Combined Dataset ile Eğitim (Concatenation)
```bash
python -m src.train \
    --dataset combined \
    --fusion concat \
    --epochs 100 \
    --batch_size 64 \
    --lr 0.0001 \
    --device mps
```

### Checkpoint'ten Devam Etme
```bash
python -m src.train \
    --dataset combined \
    --fusion shared_head \
    --resume checkpoints/combined_best/best_model.pt \
    --epochs 150 \
    --device mps
```

---

## 📈 Model Değerlendirme

### Standart Değerlendirme
```bash
python -m src.evaluate \
    --checkpoint checkpoints/combined_best/best_model.pt \
    --dataset combined \
    --fusion shared_head \
    --device mps
```

### Sliding Window Değerlendirme (Observation Ratios)
```bash
python -m src.sliding_window_eval \
    --checkpoint checkpoints/combined_best/best_model.pt \
    --dataset combined \
    --fusion shared_head \
    --device mps
```

---

## 🎥 Gerçek Zamanlı İnferans (Webcam)

### Basit Kullanım (Default Ayarlar)
```bash
python src/realtime_inference.py
```

### Özel Checkpoint ile
```bash
python src/realtime_inference.py \
    --checkpoint checkpoints/combined_best/best_model.pt \
    --device mps \
    --fusion shared_head
```

### Tüm Parametrelerle
```bash
python src/realtime_inference.py \
    --checkpoint checkpoints/combined_best/best_model.pt \
    --device mps \
    --fusion shared_head \
    --camera_id 0
```

**Kontroller:**
- `q` - Uygulamadan çık
- `p` - Duraklat/Devam et
- `r` - Buffer'ı sıfırla

**Not:** macOS'ta kamera izni gereklidir:
1. System Settings → Privacy & Security → Camera
2. Terminal için kamera erişimini etkinleştir
3. Terminal'i yeniden başlat

---

## 🔄 CoreML Export (iOS/Unity Deployment)

### PyTorch'tan CoreML'e Dönüştürme
```bash
python -m src.export_coreml \
    --checkpoint checkpoints/combined_best/best_model.pt \
    --output build/IntentFormer.mlpackage
```

### ANE Optimizasyonu ile
```bash
python -m src.export_coreml \
    --checkpoint checkpoints/combined_best/best_model.pt \
    --output build/IntentFormer.mlpackage \
    --compute_units ALL
```

---

## ⚡ Performans Testi

### MPS Benchmark (Latency Profiling)
```bash
python -m src.benchmark_mps \
    --checkpoint checkpoints/combined_best/best_model.pt \
    --device mps
```

---

## 🧪 Testler

### UI Server Testleri
```bash
python -m pytest src/tests/test_ui_server.py -v
```

### Tüm Testleri Çalıştır
```bash
python -m pytest src/tests/ -v
```

---

## 🔍 Checkpoint İnceleme

### Checkpoint Bilgilerini Görüntüle
```bash
python -c "
import torch
ckpt = torch.load('checkpoints/combined_best/best_model.pt', map_location='cpu')
print('Epoch:', ckpt.get('epoch'))
print('Val Accuracy:', ckpt.get('val_acc'))
print('Num Classes:', ckpt.get('num_classes'))
print('Dataset:', ckpt.get('dataset'))
print('Fusion:', ckpt.get('fusion'))
"
```

---

## 📁 Dizin Yapısı Kontrol

### Veri Setlerini Kontrol Et
```bash
ls -lh data/
```

### Checkpointleri Listele
```bash
ls -lh checkpoints/
```

### Preprocessed Verileri Kontrol Et
```bash
ls -lh data/preprocessed/
```

---

## 🛠️ Debugging

### PYTHONPATH ile Çalıştırma
```bash
PYTHONPATH=/Users/muratcelik/Desktop/Thesis/Workspace/Phase1:$PYTHONPATH python src/realtime_inference.py
```

### Verbose Logging
```bash
python -m src.train --dataset combined --fusion shared_head --device mps --verbose
```

### GPU/MPS Kullanımını Kontrol Et
```bash
python -c "
import torch
print('MPS Available:', torch.backends.mps.is_available())
print('MPS Built:', torch.backends.mps.is_built())
print('CUDA Available:', torch.cuda.is_available())
"
```

---

## 📝 Notlar

### Python Sürümü
- Python 3.10+ gereklidir
- `python3` komutu kullanın (macOS'ta `python` yerine)

### Apple Silicon (M1/M2/M3) Optimizasyonu
- MPS (Metal Performance Shaders) otomatik olarak kullanılır
- `--device mps` parametresi önerilir

### Veri Seti Yolları
- H2O: `data/h2o/`
- HOT3D: `data/hot3d/`
- Preprocessed: `data/preprocessed/`

### Model Checkpoints
- En iyi model: `checkpoints/combined_best/best_model.pt`
- Training checkpoints: `checkpoints/` dizini

---

## 🎯 Hızlı Başlangıç (Quick Start)

Projeyi ilk defa çalıştırıyorsanız:

```bash
# 1. Bağımlılıkları yükle
pip install -r requirements.txt
pip install mediapipe==0.10.14

# 2. Veri setini indir (opsiyonel - test için)
python scripts/download_hot3d.py --max_clips 5 --device Aria

# 3. Mevcut checkpoint ile gerçek zamanlı inferans
python src/realtime_inference.py

# 4. Model değerlendirmesi
python -m src.evaluate --checkpoint checkpoints/combined_best/best_model.pt --dataset combined --fusion shared_head --device mps
```

---

**Son Güncelleme:** 2024-03-25
