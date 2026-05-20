# Train Başlatmadan Önce Yapılacaklar

## Durum
- `11_train.py` düzeltildi (2026-05-20)
- Kırık `delta_q` ve `delta_t` loss'ları kaldırıldı
- Test seti ham verisi mevcut ama henüz işlenmemiş

---

## Adım 1 — Test Setini İşle (opsiyonel ama önerilir)
Eğitimden ÖNCE test setini hazırla ki `12_evaluate.py` çalışabilsin.

```bash
cd /Users/muratcelik/Desktop/Thesis/Workspace/V3/hot3d_exploration

# Test split'i preprocess et (~20-30 dk)
python 08_preprocess_annotations.py

# HDF5'i test seti dahil yeniden oluştur
python 09_build_dataset.py --no_resume
```

Adım 1'i yaparsan HDF5'te artık train / val / test üç split olur.  
Adım 1'i atlarsan HDF5 değişmez, eğitim başlar ama test split olmaz.

---

## Adım 2 — Eğitimi Başlat

```bash
cd /Users/muratcelik/Desktop/Thesis/Workspace/V3/hot3d_exploration

# Sıfırdan başlat (eski checkpoint'i yoksay)
python 11_train.py --no_resume

# Veya subset ile önce smoke test yap (~10 epoch, birkaç dk)
python 11_train.py --no_resume --subset 5000 --epochs 10
```

- Süre: ~7 saat (100 epoch, 1.2M window, MPS/CPU)
- Çıktı: `data/checkpoints/best.pt`, `data/checkpoints/latest.pt`
- Log: `data/logs/intentformer_training_log.jsonl`

---

## Adım 3 — ONNX Export

```bash
python 13_export_onnx.py
```

Çıktı: `data/intentformer.onnx` → Unity Assets'e sürükle

---

## Değişiklik Özeti (neden)

| Kaldırılan | Neden |
|---|---|
| `l_delta_q` (λ=0.4) | HOT3D'de ctrl_q = wrist_q → delta_q = identity her zaman → normaliz. sonrası [0,0,0,0] → loss = π sabit, gradient = 0 |
| `l_delta_t` (λ=0.4) | HOT3D'de ctrl_t = wrist_t → delta_t = 0 her zaman → model 4 epoch'ta trivial şekilde öğreniyor, gerçek el pozu öğrenmesini yavaşlatıyor |

Bu iki loss kaldırılınca training loss hesabı hızlanır ve gerçek sinyaller (MANO pose, wrist_t, wrist_q) daha güçlü gradient alır.
