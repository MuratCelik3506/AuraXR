# AuraXR Pipeline Planı — Python Scriptleri

Bu belge, AuraXR el pozu tahmin modelini inşa etmek, eğitmek, değerlendirmek ve dışa aktarmak için gereken tüm Python scriptlerini açıklar. Scriptler çalıştırma bağımlılık sırasına göre listelenmiştir: her script kendinden öncekini gerektirir.

---

## Yardımcı Dosyalar (İstediğiniz Zaman Çalıştırın)

Bunlar diğer scriptler tarafından import edilen paylaşımlı modüllerdir. Doğrudan çalıştırılmazlar.

### `hot3d_utils.py` *(mevcut — güncelle)*

**Amaç:** Düşük seviyeli HOT3D veri erişim yardımcıları.

**İçerir:**
- `hand_data.zip` okuyucu → `umetrack_hand_pose_trajectory.jsonl` ve `mano_hand_pose_trajectory.jsonl` ayrıştırır
- `ground_truth.zip` okuyucu → `dynamic_objects.csv` (nesne 6DoF pozları) ayrıştırır
- El + nesne verisinde zaman damgası hizalamalı kare iteratörü
- `rotate_vec(q_wxyz, v)` — bir 3B vektörü kuaterniyon ile döndürür
- `quat_conjugate(q_wxyz)` — birim kuaterniyonun eşleniği
- BOP ID → tutuş kategorisi + bbox lookup (`OBJ_BBOX` tablosu, 33 nesne)

**Veri kaynağı:** `data/quest3/{split}/{seq_id}/` ZIP dosyaları (yalnızca Quest3, `06_download_annotations.py` ile indirilmiş)

**Kullananlar:** `build_dataset.py`, `evaluate.py`

---

### `hot3d_dataset.py` *(mevcut — yeniden yaz)*

**Amaç:** `build_dataset.py` tarafından oluşturulan HDF5 dosyasından akan PyTorch Dataset.

**İçerir:**
- `HOT3DDataset(hdf5_path, split, normalise)` — HDF5'ten train/val bölümü yükler
- Saklanan istatistiklerden özellik/hedef normalizasyonu uygular
- Örnek başına `(feature_11, target_22)` tensor döndürür

**Kullananlar:** `train.py`, `evaluate.py`

---

### `grip_categories.py` *(yeni)*

**Amaç:** HOT3D nesne ID'lerini tutuş kategorisi ve fiziksel boyutlara eşler.

**Girdi:** HOT3D `object_library.json`

**Çıktı (bellekte):** object_id → `{grip_category: int, size_xyz: [float, float, float]}` eşleme sözlüğü

**Tutuş kategorileri:**
| Kategori   | İndeks | Örnek Nesneler |
|------------|--------|----------------|
| Power (Güç)| 0      | bardak, şişe, kap |
| Precision  | 1      | kaşık, kalem, spatula |
| Palmar     | 2      | tabak, klavye, telefon |
| Pinch      | 3      | fare, bulmaca, küçük kutu |

**One-hot kodlama:** Tutuş indeksi → `[1,0,0,0]`, `[0,1,0,0]`, `[0,0,1,0]`, `[0,0,0,1]`

**Çalıştırma:** `python grip_categories.py` (self-test yapar, kategori tablosunu yazdırır)

---

## Adım 1 — Veri Seti Oluştur

### `build_dataset.py` *(yeni)*

**Amaç:** HOT3D'den kullanılabilir tüm kareleri çıkarır ve eğitim veri setini diske yazar.

**Adım adım ne yapar:**
1. `data/quest3/train/` ve `data/quest3/test/` içindeki sekans dizinlerini tarar (yalnızca Quest3)
2. Train/val bölümü atar: test katılımcıları (P0004/5/6/8/16/20) hariç; kalan train katılımcılarında 70/15 bölümü
3. Her sekans için `hand_data.zip` ve `ground_truth.zip` açar
4. Geçerli el annotasyonu olan tüm kareleri iterate eder:
   - **El-nesne mesafesi > 40cm olan kareleri atlar**
   - `umetrack_hand_pose_trajectory.jsonl`'dan bilek pozisyonu ve kuaterniyonunu okur
   - `dynamic_objects.csv`'den en yakın nesne centroid'ini (dünya uzayında) okur
   - Bilek frame'inde göreceli konum hesaplar:
     ```python
     delta = nesne_centroid_world - bilek_pos_world
     rel_pos = rotate_vec(quat_conjugate(bilek_q), delta)  # (3,)
     ```
   - Mesafeyi hesaplar: `mesafe = ‖rel_pos‖`
   - BOP nesne ID'sini → tutuş kategorisi one-hot'a (4) dönüştürür (`OBJ_BBOX` tablosu)
   - Bbox half-extents'i `OBJ_BBOX` lookup'tan okur → 3 değer
   - Özellik vektörünü birleştirir: `[rel_pos(3), tutuş_onehot(4), bbox(3), mesafe(1)]` = **11 değer**
   - Aynı dosyadan UmeTrack eklem açılarını okur → **22 değer** (hedef)
   - Kareyi etiketler: `pre_shape` (10–40cm) veya `grip` (<10cm)
5. Normalizasyon istatistiklerini hesaplar (ortalama, std) — yalnızca train bölümü üzerinde
6. Tek bir HDF5 dosyasına train/val gruplarıyla kaydeder

**Girdiler:**
- `data/quest3/` — indirilmiş Quest3 ZIP dosyaları

**Çıktılar:**
- `data/right/dataset.h5` — train/ ve val/ grupları olan HDF5 dosyası:
  - `features` — şekil `(N, 11)`, float32
  - `targets` — şekil `(N, 22)`, float32
  - `labels` — şekil `(N,)`, bytes: `b"pre_shape"` veya `b"grip"`
  - `distances` — şekil `(N,)`, float32
  - `meta` özelliği — norm istatistiklerini içeren JSON string
- `data/left/dataset.h5` — sol el için aynı yapı

**Çalıştırma:**
```bash
python build_dataset.py --data_dir data/quest3/ --output_dir data/right/ --hand right
python build_dataset.py --data_dir data/quest3/ --output_dir data/left/  --hand left
```

**Temel parametreler:**
| Parametre | Varsayılan | Açıklama |
|-----------|-----------|----------|
| `--data_dir` | zorunlu | `data/quest3/` dizinine yol |
| `--output_dir` | zorunlu | dataset.h5'in kaydedileceği yer |
| `--hand` | `right` | Hangi el çıkarılacak: `right` veya `left` |
| `--max_distance` | 0.40 | Maksimum el-nesne mesafesi (metre) — bu değerin üzerindeki kareler atlanır |

**Notlar:**
- **İki kez çalıştırılır:** bir kez `--hand right`, bir kez `--hand left`
- Norm istatistikleri yalnızca train bölümünde hesaplanır, HDF5 `meta` özelliğinde saklanır
- P0021_5d8b0988 hand_data.zip eksik — otomatik olarak atlanır

---

## Adım 2 — Modeli Tanımla

### `model.py` *(yeni)*

**Amaç:** İki dallı MLP mimarisini tanımlar. `train.py`, `evaluate.py`, `export_onnx.py` tarafından import edilir.

**Mimari:**
```
Girdi: 11 değer

Dal A — Spatial Encoder (Uzamsal Kodlayıcı):
  [göreceli_konum(3) + mesafe(1)] = 4 değer
  FC(4 → 64) → ReLU → FC(64 → 32) → ReLU → spatial_emb(32)

Dal B — Object Encoder (Nesne Kodlayıcı):
  [tutuş_onehot(4) + bbox(3)] = 7 değer
  FC(7 → 64) → ReLU → FC(64 → 32) → ReLU → obj_emb(32)

Tahmin Kafası:
  Birleştir [pose_emb(32), obj_emb(32)] = 64 değer
  FC(64 → 64) → ReLU → FC(64 → 22) → Tanh

Çıktı: 22 eklem açısı (normalleştirilmiş, [-1, 1] aralığında)
```

**Sınıf:**
```python
class AuraXRModel(nn.Module):
    def __init__(self):
        # Pose encoder: 8 → 64 → 32
        # Spatial encoder: 4 → 64 → 32
        # Object encoder:  7 → 64 → 32
        # Head: 64 → 64 → 22

    def forward(self, spatial_input, object_input):
        # spatial_input: (B, 4) — [göreceli_konum(3), mesafe(1)]
        # object_input:  (B, 7) — [tutuş_onehot(4), bbox(3)]
        # döndürür: (B, 22) — normalleştirilmiş eklem açıları
```

**Çalıştırma:** Doğrudan çalıştırılmaz. `from model import AuraXRModel` ile import edilir.

---

## Adım 3 — Eğit

### `train.py` *(yeni)*

**Amaç:** Modeli eğitir, kontrol noktaları ve meta veri kaydeder.

**Ne yapar:**
1. `data/train.h5` ve `data/val.h5` yükler
2. `data/norm_stats.json` yükler — özellik ve hedefleri normalleştirir
3. Özellik vektörünü spatial_input (4) ve object_input (7) olarak böler
4. Mesafeye göre örnek başına kayıp ağırlığı hesaplar:
   - mesafe < 10cm → `ağırlık = 3.0` (tutuş kareleri, az ama kritik)
   - mesafe 10–40cm → `ağırlık = 1.0` (pre-shape kareleri)
5. Ağırlıklı MSE kaybıyla eğitir
6. Her epoch: val bölümünde değerlendirir, val kaybı iyileşirse kontrol noktası kaydeder
7. Final modeli + eğitim meta verisini kaydeder

**Girdiler:**
- `data/train.h5`, `data/val.h5`
- `data/norm_stats.json`

**Çıktılar:**
- `checkpoints/best_model.pt` — en iyi val kayıplı modelin PyTorch state dict'i
- `checkpoints/training_log.json` — epoch başına kayıp (train + val)
- `checkpoints/model_meta.json` — mimari konfigürasyonu + norm istatistikleri (ONNX export ve Unity için gerekli)

**Çalıştırma:**
```bash
python train.py --data_dir data/right/ --output_dir checkpoints/right/
python train.py --data_dir data/left/  --output_dir checkpoints/left/
```

**Temel parametreler:**
| Parametre | Varsayılan | Açıklama |
|-----------|-----------|----------|
| `--data_dir` | zorunlu | train.h5, val.h5, norm_stats.json içeren dizin |
| `--output_dir` | `checkpoints/` | Model ve logların kaydedileceği yer |
| `--epochs` | 100 | Eğitim epoch sayısı |
| `--batch_size` | 64 | Batch boyutu |
| `--lr` | 1e-3 | Adam öğrenme hızı |
| `--weight_decay` | 1e-5 | Adam ağırlık düşürme |
| `--grip_weight` | 3.0 | <10cm kareler için kayıp ağırlık çarpanı |
| `--seed` | 42 | Tekrar üretilebilirlik için rastgele tohum |

**Eğitim döngüsü:**
```python
loss = weighted_mse(pred, target, weights)
# weights[i] = grip_weight if distance[i] < 0.10 else 1.0
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

---

## Adım 4 — Değerlendir

### `evaluate.py` *(yeni)*

**Amaç:** Eğitilmiş modeli val bölümü üzerinde çalıştırır ve sayısal metrikler raporlar.

**Ne yapar:**
1. `checkpoints/best_model.pt` ve `model_meta.json` yükler
2. `data/val.h5` yükler
3. Tüm val karelerinde ileri geçiş yapar (düzeltme yok — ham model performansı ölçülür)
4. Hesaplar:
   - **Eklem Açısı MAE** eklem başına (derece, denormalizasyon sonrası)
   - **Genel MAE** (22 eklem üzerinde ortalama)
   - **MPJPE** (mm cinsinden ortalama eklem başına konum hatası) — açılardan 3B eklem konumları hesaplamak için UmeTrack FK gerektirir
   - **Nesne başına MAE** — her tutuş kategorisi için ayrı döküm
   - **Faz başına MAE** — pre-shape (10–40cm) ve grip (<10cm) kareleri için ayrı döküm

**Girdiler:**
- `checkpoints/right/best_model.pt`
- `checkpoints/right/model_meta.json`
- `data/right/val.h5`

**Çıktılar (konsola yazdırılır + kaydedilir):**
- `results/eval_right.json` — tüm metrikler
- Konsol tablosu: eklem başına MAE, genel MAE, MPJPE, kategori başına hata

**Çalıştırma:**
```bash
python evaluate.py --checkpoint checkpoints/right/ --data_dir data/right/ --output_dir results/
python evaluate.py --checkpoint checkpoints/left/  --data_dir data/left/  --output_dir results/
```

**Hedef metrikler:**
| Metrik | Hedef |
|--------|-------|
| Eklem Açısı MAE | < 5° |
| MPJPE | < 20 mm |
| Pre-shape MAE | < 6° |
| Grip MAE | < 4° (daha sıkı — bu karelerin ağırlığı daha yüksek) |

**Temel parametreler:**
| Parametre | Varsayılan | Açıklama |
|-----------|-----------|----------|
| `--checkpoint` | zorunlu | best_model.pt ve model_meta.json içeren dizin |
| `--data_dir` | zorunlu | val.h5 içeren dizin |
| `--output_dir` | `results/` | Değerlendirme JSON'unun kaydedileceği yer |

---

## Adım 5 — Simüle Et

### `simulate.py` *(yeni)*

**Amaç:** Sentetik bir yaklaşma yörüngesi oluşturur ve her mesafe adımında tahmin edilen eklem açılarını görselleştirir. Unity testinden önce kötü geçişleri yakalar.

**Ne yapar:**
1. Eğitilmiş modeli yükler
2. Sentetik girdi dizisi oluşturur:
   - Mesafe adımları: `[0.40, 0.35, 0.30, 0.25, 0.20, 0.15, 0.10, 0.07, 0.05, 0.02]` metre
   - Sabit nesne: örn. Şişe (tutuş=Power, boyut=[0.07, 0.22, 0.07])
   - Sabit bilek dönüşümü: kimlik (el önden yaklaşıyor)
   - Nesne Z ekseni boyunca 40cm → 2cm öne hareket eder
3. Her mesafe adımında modeli çalıştırır → 22 eklem açısı tahmini
4. Mesafe vs eklem açıları çizgi grafiği çizer (parmak eklemi başına bir çizgi)
5. Adımlar arasındaki büyük sıçramaları kontrol eder (delta > eşik = uyarı)

**Girdiler:**
- `checkpoints/right/best_model.pt`
- `checkpoints/right/model_meta.json`

**Çıktılar:**
- `results/simulation_right_bottle.png` — eklem açısı yörüngesi grafiği
- `results/simulation_right_cup.png` — bardak için aynısı
- Herhangi bir eklem açısı deltası adımlar arası 10°'yi aşarsa konsol uyarısı

**Çalıştırma:**
```bash
python simulate.py --checkpoint checkpoints/right/ --object bottle
python simulate.py --checkpoint checkpoints/right/ --object cup
```

**Temel parametreler:**
| Parametre | Varsayılan | Açıklama |
|-----------|-----------|----------|
| `--checkpoint` | zorunlu | best_model.pt ve model_meta.json içeren dizin |
| `--object` | `bottle` | Simüle edilecek HOT3D nesnesi (`bottle`, `cup`, `pen`, vb.) |
| `--distance_steps` | 10 | 40cm'den 2cm'ye mesafe adımı sayısı |
| `--output_dir` | `results/` | Grafiklerin kaydedileceği yer |

**Beklenen davranış:**
- 40cm'de: eklem açıları ≈ FIXED_DEFAULT_POSE değerleri
- 20cm'de: parmaklar tutuş şekline doğru açılmaya/kapanmaya başlar
- 5cm'de: net tutuş şekli görünür
- 2cm'de: nesne tipine göre tam kavrama pozu
- Adımlar arasında ani sıçrama yok

---

## Adım 6 — ONNX'e Aktar

### `export_onnx.py` *(yeni)*

**Amaç:** Eğitilmiş PyTorch modelini Unity Sentis için ONNX formatına aktarır.

**Ne yapar:**
1. `best_model.pt` yükler
2. Sahte girdi tensörleri oluşturur (spatial_input: `(1,4)`, object_input: `(1,7)`)
3. Dinamik batch boyutu ile `torch.onnx.export()` çalıştırır
4. Aynı sahte girdilerle ONNX Runtime çalıştırarak export'u doğrular — çıktı şeklinin `(1,22)` olduğunu kontrol eder
5. `model_meta.json`'u (norm istatistikleri + mimari konfigürasyonu) çıktı dizinine kopyalar

**Girdiler:**
- `checkpoints/right/best_model.pt`
- `checkpoints/right/model_meta.json`

**Çıktılar:**
- `onnx/auraxr_right.onnx` — Unity için ONNX model dosyası
- `onnx/model_meta_right.json` — Unity'nin çıktıyı denormalize etmek için ihtiyaç duyduğu norm istatistikleri
- Konsol: ONNX doğrulama sonucu (girdi/çıktı şekilleri)

**Çalıştırma:**
```bash
python export_onnx.py --checkpoint checkpoints/right/ --output_dir onnx/
python export_onnx.py --checkpoint checkpoints/left/  --output_dir onnx/
```

**Temel parametreler:**
| Parametre | Varsayılan | Açıklama |
|-----------|-----------|----------|
| `--checkpoint` | zorunlu | best_model.pt ve model_meta.json içeren dizin |
| `--output_dir` | `onnx/` | .onnx ve meta JSON'un kaydedileceği yer |
| `--opset` | 14 | ONNX opset versiyonu (Unity Sentis 14+ destekler) |

**ONNX girdi/çıktı spesifikasyonu (Unity için):**
```
Girdi  "spatial_input": şekil [1, 4]  — [göreceli_konum(3), mesafe(1)]
Girdi  "object_input":  şekil [1, 7]  — [tutuş_onehot(4), bbox(3)]
Çıktı  "joint_angles":  şekil [1, 22] — normalleştirilmiş, kullanmadan önce denorm uygula
```

**Unity'de denormalizasyon:**
```csharp
float angle_i = (model_output[i] * target_std[i]) + target_mean[i];
```

---

## Çalıştırma Sırası Özeti

```
1. python build_dataset.py --data_dir ../data/quest3/ --output_dir ../data/right/ --hand right
2. python build_dataset.py --data_dir ../data/quest3/ --output_dir ../data/left/  --hand left
3. python train.py --data_dir ../data/right/ --output_dir ../checkpoints/right/
4. python train.py --data_dir ../data/left/  --output_dir ../checkpoints/left/
5. python evaluate.py --checkpoint ../checkpoints/right/ --data_dir ../data/right/ --output_dir ../results/
6. python evaluate.py --checkpoint ../checkpoints/left/  --data_dir ../data/left/  --output_dir ../results/
7. python simulate.py --checkpoint ../checkpoints/right/ --object bottle --output_dir ../results/
8. python simulate.py --checkpoint ../checkpoints/right/ --object cup    --output_dir ../results/
9. python export_onnx.py --checkpoint ../checkpoints/right/ --output_dir ../onnx/
10. python export_onnx.py --checkpoint ../checkpoints/left/  --output_dir ../onnx/
```

10. adımdan sonra: `onnx/auraxr_right.onnx`, `onnx/auraxr_left.onnx` ve her iki `model_meta_*.json` dosyasını Unity Assets klasörüne kopyala.

---

## Dosya ve Dizin Yapısı

```
hot3d_exploration/
├── grip_categories.py       (yeni — yardımcı, tutuş haritası)
├── hot3d_utils.py           (mevcut — güncelle)
├── hot3d_dataset.py         (mevcut — güncelle)
├── build_dataset.py         (yeni — adım 1)
├── model.py                 (yeni — adım 2)
├── train.py                 (yeni — adım 3)
├── evaluate.py              (yeni — adım 4)
├── simulate.py              (yeni — adım 5)
└── export_onnx.py           (yeni — adım 6)

data/
├── right/
│   ├── train.h5
│   ├── val.h5
│   └── norm_stats.json
└── left/
    ├── train.h5
    ├── val.h5
    └── norm_stats.json

checkpoints/
├── right/
│   ├── best_model.pt
│   ├── training_log.json
│   └── model_meta.json
└── left/
    └── (aynısı)

onnx/
├── auraxr_right.onnx
├── auraxr_left.onnx
├── model_meta_right.json
└── model_meta_left.json

results/
├── eval_right.json
├── eval_left.json
├── simulation_right_bottle.png
└── simulation_right_cup.png
```

---

## Temel Paylaşımlı Veri Formatı

### Özellik Vektörü (11 değer)

| İndeks | Değerler | Kaynak |
|--------|----------|--------|
| 0–2    | Nesne göreceli konumu (bilek çerçevesinde x, y, z) | Hesaplanır: `R_bilek^T × (nesne_world - bilek_world)` |
| 3–6    | Tutuş kategorisi one-hot | BOP ID → `OBJ_BBOX` tablosu ile 4 sınıf |
| 7–9    | Nesne bbox half-extents (x, y, z metre) | `OBJ_BBOX[bop_id]` lookup |
| 10     | El-nesne mesafesi (metre) | `‖göreceli_konum‖` |

### Hedef Vektör (22 değer)

UmeTrack eklem açıları — 22 parmak eklemi, sol veya sağ el. Bilek bu vektörde **yoktur** (bilek Unity'de controllere sabitlenir, model tarafından tahmin edilmez).

### model_meta.json Yapısı

```json
{
  "feature_mean": [11 float],
  "feature_std":  [11 float],
  "target_mean":  [22 float],
  "target_std":   [22 float],
  "architecture": {
    "pose_input_dim": 8,
    "spatial_input_dim": 4,
    "object_input_dim": 7,
    "output_dim": 22,
    "hidden_dim": 64,
    "embedding_dim": 32
  }
}
```

---

## Bağımlılıklar

```
pip install torch torchvision
pip install onnx onnxruntime
pip install h5py
pip install numpy matplotlib
```

HOT3D verisi doğrudan indirilen ZIP dosyalarından okunur — ek SDK gerekmez.

Sabitlenmiş versiyonlar için `requirements.txt`'e bakın.
