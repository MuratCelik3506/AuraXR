# B. Temporal Geometry-Conditioned Grasp Model

Bu tezde tek ana AI model vardır: **Temporal Geometry-Conditioned Grasp Model**. Model, el objeye temas ettiğinde veya temas eşiğine geldiğinde devreye girer. Görevi, objenin 3B geometrisine ve son frame'lerdeki el hareketine göre parmak eklem rotasyonlarını (3×15), heuristik kalite skorunu (`quality_score`) ve Unity'den öğrenilen başarı olasılığını (`success_prob`) üretmektir.

Ayrı bir Approach Model kullanılmaz; kullanıcının bilek/controller hareketi korunur ve model yalnızca objeye yakın bölgede parmak kapanışını düzeltir.

---

## B0. Model Mimarisi

```
Object Point Cloud
        │
   Mini PointNet
        │
 Object Feature (256)
        │
 frame_feat (B,T,13) ──► GRU ──► Temporal Feature
                                        │
                          ┌─────────────┤
                          │             │
                   Previous Pose   Object Feature
                    (B,45)          (256)
                          │             │
                          └──── Fusion ─┘
                                  │
                      Self-Attention (15 eklem)
                                  │
                             CVAE Decoder
                                  │
                    ┌─────────────┼─────────────┐
              Finger Pose (45)  Quality       Success
                              Score (1)      Prob (1)
```

**Birleşik girdi formatı:** Her iki veri seti aynı `frame_feat (B, T, 13)` formatını kullanır.
- OakInk statik: `T=1`, `rel_vel=0`, `dist` hesaplanır
- HOT3D temporal: `T>1`, sliding window

| Bileşen | Rol | Durum |
|---|---|---|
| Mini PointNet | Point cloud'dan obje feature (256-dim) | Uygulandı |
| FiLM conditioning | Obje feature'ını temporal feature'a enjekte eder | Uygulandı |
| GRU temporal encoder | Son T frame bilek hareketini işler | Hedeflenen |
| Self-Attention (15 eklem) | Parmaklar arası koordinasyonu öğrenir | Hedeflenen |
| CVAE decoder | Çok-modlu parmak pozu üretir | Uygulandı |
| Quality score head | Heuristik kalite skoru üretir (MSE, hemen eğitilebilir) | Uygulandı (label eksik) |
| Success prob head | Unity binary başarı olasılığı üretir (BCE, Aşama 3) | Uygulandı (label eksik) |
| Approach controller | Mesafe/temas tabanlı faz geçişi | Kural tabanlı, AI değil |
| Unity evaluator | Fizik başarısını ölçer, confidence label üretir | Hedeflenen |

### Mevcut Uygulama Durumu

Repo'daki mevcut `src/grasp_model.py` şu statik baseline'ı uygular:

| Bileşen | Mevcut Durum |
|---|---|
| Object encoder | Mini PointNet, `(x,y,z)` point cloud |
| Conditioning | FiLM ile obje embedding → wrist feature |
| Generative yapı | CVAE |
| Input | `wrist_feat(6) + obj_pts(1024,3)` |
| Output | `finger_aa45(45) + quality_score(1) + success_prob(1)` |
| Training | OakInk statik grasp pre-training |

Hedeflenen genişletmeler:

| Genişletme | Amaç |
|---|---|
| `frame_feat (B,T,13)` birleşik arayüzü | `wrist_feat(6)` yerine, OakInk T=1 / HOT3D T>1 |
| GRU temporal encoder | Son T frame bilek hareketini işler |
| Self-Attention (15 eklem) | Parmaklar arası bağımlılık |
| K adaylı CVAE inference | Aynı obje için birden fazla grasp adayı |
| Unity retarget/eval | `finger_aa45` çıktısını XR avatar ve fizik eval'e bağlamak |

---

## B1. Veri Rejimleri, Koordinat Sözleşmesi ve Eğitim

### Koordinat Sözleşmesi

Tüm bilek bilgisi **objeye göre relatif** (object frame) olarak ifade edilir. Her iki veri seti için aynı dönüşüm uygulanır:

```
T_wrist_in_object = T_object_in_world⁻¹ @ T_wrist_in_world

rel_pos   = T_wrist_in_object[:3, 3]               # (3,)  metre
rel_rot6d = rot_matrix_to_6d(T_wrist_in_object[:3,:3])  # (6,)  6D sürekli temsil
rel_vel   = (rel_pos[t] - rel_pos[t-1]) / Δt       # (3,)  m/s  — OakInk için sıfır
dist      = signed_distance_to_mesh_surface(wrist)  # (1,)  metre, yüzeye göre
```

OakInk'te `T_object_in_world` her annotation'da mevcuttur. HOT3D'de per-frame obje pozu annotation'da gelir. Bu dönüşüm `build_oakink_canonical.py` ve `build_hot3d_canonical.py`'ye eklenerek her iki dataset aynı semantik anlamı taşıyan feature üretir.

### Birleşik Girdi Formatı

Her iki veri seti `frame_feat (B, T, 13)` formatında modele verilir:

```
frame_feat = [rel_pos(3), rel_rot6d(6), rel_vel(3), dist(1)]  # 13 boyut
```

| Veri Seti | T | rel_vel | dist |
|---|---|---|---|
| OakInk | 1 | sıfır | bilek–yüzey mesafesi |
| HOT3D | >1 (sliding window) | hesaplanır | bilek–yüzey mesafesi |
| Unity runtime | >1 (ring buffer) | her frame hesaplanır | collider mesafesi |

Unity runtime HOT3D formatıyla özdeştir; bu yüzden tüm format HOT3D-style object-relative.

### Model Girdi/Çıktı Sözleşmesi

**Eğitim girdisi:**

| Alan | Boyut | Açıklama |
|---|---:|---|
| `frame_feat` | `(B, T, 13)` | Object-relative bilek hareketi |
| `contact_flag` | `(B, T, 1)` | Temas/transition sinyali — `frame_feat` ile concat → GRU'ya 14-dim |
| `finger_hist` | `(B, T, 45)` | Önceki parmak pozları — self-attention token kimliği için |
| `history_mask` | `(B,)` | OakInk: 0, HOT3D/Unity: 1 |
| `obj_pts` | `(B, 1024, 3)` | Obje point cloud (canonical frame) |
| `target_pose` | `(B, 45)` | Hedef parmak pozu |

**Çıktı (inference, K aday):**

| Alan | Boyut | Açıklama |
|---|---:|---|
| `candidate_poses` | `(B, K, 45)` | K farklı latent sample → K kavrama adayı |
| `quality_scores` | `(B, K)` | Heuristic kalite skoru (contact ratio + penetration penalty + finger-distance penalty) |
| `success_probs` | `(B, K)` | Unity physics başarı olasılığı |
| `selected_pose` | `(B, 45)` | `argmax(success_prob)` ile seçilen aday |

### Eklem Sırası

Modelin 45 boyutlu çıktısı MANO parmak axis-angle sırasını kullanır:

| Aralık | Parmak / Eklem |
|---|---|
| `0..8` | Index MCP, PIP, DIP |
| `9..17` | Middle MCP, PIP, DIP |
| `18..26` | Ring MCP, PIP, DIP |
| `27..35` | Pinky MCP, PIP, DIP |
| `36..44` | Thumb CMC, MCP, IP |

Unity tarafında bu sıra XR Hands kemik sırasına açık mapping tablosuyla retarget edilmelidir.

### Eğitim Planı (3 Aşama)

**Aşama 1 — OakInk Statik Pre-training**

OakInk öğretir: "Bu obje geometrisine göre geçerli final kavrama pozu nedir?"

```
Input:   frame_feat (B, 1, 13)   # T=1
Loss:    L_recon + β*L_KL + λ_c*L_contact + λ_p*L_penetration + λ_q*L_quality
         L_temporal_smooth YOK
β:       0 → 1, 50 epoch warm-up
Optimizer: Adam, lr=1e-3, batch=64
Durdurma: val L_recon 10 epoch iyileşmezse
```

**Aşama 2 — HOT3D Temporal Fine-tuning**

HOT3D öğretir: "El bu poza zaman içinde nasıl doğal ve stabil kapanır?"

```
Input:   frame_feat (B, T, 13)   # T=8 başlangıç; T=4,16 ablation
Batch:   %70 HOT3D + %30 OakInk  # OakInk'i unutmasın
Loss:    Aşama 1 loss
         + λ_vel * ||pose_t - pose_{t-1}||
         + λ_acc * ||pose_t - 2*pose_{t-1} + pose_{t-2}||
Optimizer: AdamW, lr=1e-4, batch=32
Grad clip: max_norm=1.0           # GRU için kritik
PointNet:  frozen değil, düşük LR ile fine-tune
```

**Aşama 3 — Confidence Kalibrasyonu**

```
quality_score head: heuristic label (contact ratio + penetration penalty + finger-distance penalty), MSE — hemen eğitilebilir
success_prob head:  Unity binary label, BCE — Unity eval tamamlanınca
Backbone:           frozen
```

#### Eklem Sırası

Modelin 45 boyutlu çıktısı MANO parmak axis-angle sırasını kullanır:

| Aralık | Parmak / Eklem |
|---|---|
| `0..8` | Index MCP, PIP, DIP |
| `9..17` | Middle MCP, PIP, DIP |
| `18..26` | Ring MCP, PIP, DIP |
| `27..35` | Pinky MCP, PIP, DIP |
| `36..44` | Thumb CMC, MCP, IP |

Unity tarafında bu sıra doğrudan kullanılmamalıdır; XR Hands kemik sırasına retarget edilirken açık mapping tablosu uygulanmalıdır.

---

## B2. PointNet Obje Encoder

Objenin geometrisi modelin en kritik girdisidir. Basit bir bounding box veya boyut vektörü, ince geometrik farkları (bardak kulpu, makas kolu, kalem ucu) yakalayamaz.

### Obje Temsili

Her obje mesh'i → **N=1024 nokta** örnekleme.

- Mevcut uygulama: her nokta `(x, y, z)` → 3 boyut
- E3 ablation varyantı: `(x, y, z) + yüzey normali (nx, ny, nz)` → 6 boyut
- Obje canonical frame'e dönüştürülür (obje rotasyonu çıkarılır; model orientasyon-invariant girdi alır, orientasyon ayrı input olarak verilir)

### Mimari

**Mini PointNet (mevcut baseline):**
```
1024 × 3
    │
MLP per point: 3 → 64 → 128 → 256
    │
mean pooling + max pooling
    │
Global Feature: 256-dim
```

### Çıktı

- `obj_global_feat`: 256-dim — objenin genel şekli

---

## B3. GRU Temporal Encoder

`frame_feat` ve `contact_flag` birleştirilerek GRU'ya verilir:

```
gru_input = concat(frame_feat(13), contact_flag(1))  # 14 boyut
        │
   GRU (hidden: 256)
        │
 Temporal Feature (B, 256)   ← son hidden state
```

- **Girdi:** `[rel_pos(3), rel_rot6d(6), rel_vel(3), dist(1), contact_flag(1)]` — 14 boyut
- **`contact_flag`:** Model temas/geçiş fazını girdi olarak bilir; sadece loss maskesi değil
- **Başlangıç penceresi:** T=8 (30 FPS'te ~270ms bağlam)
- **T=1 durumu (OakInk):** GRU tek adım çalışır; `contact_flag=0`, `rel_vel=0`
- **Causal:** Sadece geçmiş frame'lere bakar, gelecek bilgisi kullanılmaz

Temporal smoothness loss yalnızca T>1 batch'leri için hesaplanır.

---

## B4. Self-Attention — 15 Parmak Eklemi

GRU ve obje feature'ı fusion'dan sonra 15 parmak eklemi üzerinde self-attention uygulanır. Amaç: parmakların birbirinden bağımsız değil, koordineli hareket etmesini öğretmek (örn. güç kavramasında tüm parmaklar eş zamanlı kapanır; pinch'te sadece işaret ve başparmak).

Her token'a per-joint kimlik bilgisi eklenmeden aynı vektörü 15 kez kopyalamak self-attention'ı anlamsız kılar — tüm tokenlar özdeş başlar ve simetrik çıktı üretilir. Kimlik bilgisi iki kaynaktan gelir:

```python
# fusion_out: (B, 128) — GRU + object + previous pose fusion sonucu
# learned_joint_emb: (15, 128) — her ekleme öğrenilebilir kimlik, trainable param
# finger_hist: (B, T, 45) — önceki parmak pozları

# Önceki pozdan per-joint embedding:
prev_pose_emb = Linear(3, 128)(finger_hist[:, -1, :].view(B, 15, 3))  # (B, 15, 128)

joint_tokens = (fusion_out[:, None, :].expand(-1, 15, -1)  # paylaşılan bağlam
              + learned_joint_emb[None, :, :]               # eklem kimliği
              + prev_pose_emb)                              # önceki poz durumu

attn_out = SelfAttention(joint_tokens)                      # (B, 15, 128)
```

- **Kafa sayısı:** 4 head, eklem başına 128-dim
- **Katman sayısı:** 1 — 15 eklem küçük sequence, tek katman yeterli
- **`learned_joint_emb`:** Index MCP ile Thumb CMC farklı kimlik taşır; model hangi eklemin hangi parmağa ait olduğunu öğrenir
- **`prev_pose_emb`:** Önceki frame'de eklemin bulunduğu açı, token'a kinematik geçmiş katar
- **Neden tam Transformer değil:** Temporal bağımlılık GRU'da; self-attention yalnızca spatial (parmaklar arası) koordinasyon için. İki görevi ayrı tutmak debug'ı kolaylaştırır.

---

## B5. FiLM Conditioning (Obje Şekli)

Obje global feature'ı, ana ağa **FiLM (Feature-wise Linear Modulation)** ile enjekte edilir.

Basitçe FiLM, obje vektörünü ana ağın aktivasyonlarını "ayarlamak" için kullanır. Concatenation'da obje bilgisi sadece girişe eklenir; FiLM'de obje bilgisi katmanların içindeki feature'ları ölçekler ve kaydırır. Örneğin ince-uzun bir obje için parmak kapanış feature'ları farklı, geniş bir kupa için farklı modüle edilir.

### Mekanizma

```python
# obj_global_feat: (B, 256)
gamma = Linear(256, hidden_dim)(obj_global_feat)  # ölçek
beta  = Linear(256, hidden_dim)(obj_global_feat)  # kaydırma

# Ana gövde aktivasyonu:
h_out = gamma * h + beta
```

FiLM, obje bilgisini "ne öğrenileceğini şekillendir" şeklinde kullanır. Mevcut `src/grasp_model.py` implementasyonunda conditioning mekanizması olarak FiLM kullanılmaktadır.

---

## B6. CVAE Yapısı

Aynı objeyi farklı şekillerde kavramanın birden fazla geçerli yolu vardır (üstten güç kavraması, yandan kıstırma kavraması). CVAE bu çok-modluluğu modellemek için kullanılır.

```
Girdiler: bilek pose + obje bilgisi + obj_global_feat
      │
Encoder (eğitimde):  ground-truth parmak pozu → μ, logσ
Decoder (inference): z ~ N(0,1) → parmak pozu (15 × 3 eklem açısı)
      │
Çıktı: parmak konfigürasyonu
```

**Eğitim loss:**
```
L_cvae = L_recon + β * KL(q(z|x) || N(0,1))
```
β-VAE warm-up: β 0→1 arası 50 epoch içinde kademeli artış.

---

## B7. Multi-Task Output Head

Model üç şey üretir: parmak açıları, heuristic kalite skoru ve fizik başarı olasılığı.

### Parmak Açısı Başlığı

```
Self-Attention çıktısı (B, 15, 128)
      │
Linear: 128 → 3   (per eklem)
      │
Çıktı: (B, 15, 3) → flatten → (B, 45)
```

XR Hands eklem başına 1–2 anlamlı serbestlik derecesi var; 3 çıktı vererek gereksiz eksenleri sıfıra doğru regularize etmek daha esnektir.

### Confidence: İki Ayrı Head

Heuristic "kalite" ile Unity "başarı" farklı kavramlardır ve ayrı head'lerle modellenir:

**quality_score** — sürekli, heuristic label, hemen eğitilebilir:
```
Global pooling (B, 15, 128) → (B, 128)
      │
Linear 128 → 64 → 1  +  Sigmoid
      │
quality_score ∈ [0, 1]

Label: clip(w1*contact_ratio - w2*clip(pen/max_pen,0,1) - w3*clip(dist/max_dist,0,1), 0.0, 1.0)
       Başlangıç: w1=1.0, w2=0.3, w3=0.2  # w1<1 max skoru 1.0'ın altına kilitler
Loss:  MSE(quality_score, label)
```

**success_prob** — binary, Unity label, Aşama 3'te eğitilir:
```
concat(global_pool, candidate_pose_embedding)  # adaya özgü
      │
Linear → 64 → 1  +  Sigmoid
      │
success_prob ∈ [0, 1]

Label: Unity physics success/fail
Loss:  BCE(success_prob, unity_label)
```

K aday arasından seçim `argmax(success_prob)` ile yapılır. Her aday için ayrı `success_prob` hesaplanmalıdır; paylaşılan `global_pool`'a sadece `candidate_pose_embedding` eklenerek adaylar birbirinden ayrışır.

### Loss Özeti (Eğitim Aşamalarına Göre)

| Aşama | Aktif Losslar |
|---|---|
| 1 — OakInk | `L_recon + β*L_KL + L_contact + L_penetration + L_quality` |
| 2 — HOT3D | Aşama 1 + `L_vel + L_acc` |
| 3 — Confidence | Sadece `L_success` (backbone frozen) |

---

## B8. Temporal Refinement ve Jitter Önleme

Sürekli inference'da parmak açıları frame'den frame'e zıplayabilir.

### Temporal Grasp Refinement İçin Gerekenler

Temporal çalışmak için model tek frame'den karar vermek yerine kısa bir geçmiş pencere kullanır. Bu pencere parmak kapanışının hızını, temas öncesi hazırlığı ve objeye göre yaklaşma dinamiğini taşır.

`frame_feat (B, T, 13)` ve `contact_flag (B, T, 1)` concat edilerek `gru_input (B, T, 14)` oluşturulur ve GRU'ya verilir (B3). Başlangıç penceresi T=8 (30 FPS'te ~270ms). Ablation: T=4, T=8, T=16.

**Temporal loss — ground-truth hız ve ivme farkı:**

Yalnızca `||p̂_t - p̂_{t-1}||` minimize etmek modelin "hareket etmemeyi" öğrenmesine yol açabilir. Doğru tanım tahmin edilen dinamiklerin GT dinamikleriyle eşleşmesini ister:

```
L_vel = ||(p̂_t - p̂_{t-1}) - (p_t - p_{t-1})||
L_acc = ||(p̂_t - 2p̂_{t-1} + p̂_{t-2}) - (p_t - 2p_{t-1} + p_{t-2})||
```

Burada `p̂` model tahmini, `p` ground-truth (`finger_hist` veya `target_pose`). Ek smoothing istenirse düşük ağırlıklı `λ_smooth * ||p̂_t - p̂_{t-1}||` regularizer olarak eklenebilir.

Temporal katkı şu karşılaştırmayla savunulur:

> Single-frame model benzer pose hatası üretse bile temporal model daha düşük jitter, daha stabil temas ve daha yüksek Unity success sağlamalıdır.

### Yöntem 1 — Önceki Frame Girdisi

Bir önceki frame'in tahmin edilen parmak pozu self-attention token'larına `prev_pose_emb` olarak eklenir (B4'te tanımlanmıştır). Ağ "smooth geçiş" yapmayı öğrenir.

### Yöntem 2 — Exponential Moving Average (EMA) — Runtime

```python
output_smoothed = α * output_new + (1 - α) * output_prev
```
α = 0.7 — modelin kendi pürüzsüzlüğü yeterliyse α=1.0 (filtre yok).

İkisi birlikte kullanılabilir; model içi jitter önleme eğitimde, EMA ise runtime'da son güvenlik katmanı olarak.

---

## B9. Çoklu Kavrama Üretimi

Aynı nesne için birden fazla geçerli kavrama üretme özelliği korunmalıdır. Örneğin bir kupa kulptan, gövdeden veya üstten kavranabilir. Ancak bunu sağlamak için diffusion modelini ana sisteme almak şart değildir.

**Aday** nedir: Aynı bilek pozu + aynı obje için CVAE'den K farklı `z ~ N(0,1)` örneklenir. Her `z` farklı eklem konfigürasyonu üretebilir. PointNet global pooling kullandığı için lokal yüzey bölgeleri (kulp vs gövde) açıkça ayrıştırılamayabilir; adaylar "farklı kavrama stratejileri" değil "farklı eklem konfigürasyonları" olarak ifade edilmelidir.

Encoder bir kez çalışır; K aday paralel decoder batch'iyle üretilir:

```python
fusion_feat = encoder(frame_feat, obj_pts)        # (B, H) — bir kez hesaplanır
z = torch.randn(B, K, latent_dim)                 # (B, K, Z)
fusion_k = fusion_feat[:, None, :].expand(B, K, -1)  # (B, K, H)
candidate_poses = decoder(fusion_k, z)             # (B, K, 45) — tek batched call

# Per-candidate confidence (adaya özgü)
pose_emb = Linear(45, H)(candidate_poses)          # (B, K, H)
conf_input = fusion_k + pose_emb                   # (B, K, H)
success_probs = sigmoid(Linear(H, 1)(conf_input))  # (B, K, 1)

best_pose = candidate_poses[argmax(success_probs)] # (B, 45)
```

K=3 → tek encoder geçişi + paralel decoder batch < 10ms hedefi.

Tez ana sisteminde çoklu kavrama için CVAE kullanılır.

---

## B10. Offline Değerlendirme Metrikleri

### 1. Geodesic Rotation Error (Ana Metrik)

Axis-angle vektörleri üzerinde doğrudan MAE, SO(3) manifold yapısını görmezden gelir. Ana metrik geodesic error:

```
d(R_pred, R_gt) = arccos( (trace(R_pred^T R_gt) - 1) / 2 )   [radyan veya derece]
```

Her eklem için hesaplanır, ortalama raporlanır. Axis-angle MAE yalnızca debug/yardımcı metrik olarak tutulur.

### 2. MPJPE ve Fingertip Position Error

```
mpjpe         = mean over joints of ||FK(pred) - FK(gt)||      [mm]
fingertip_err = mean over 5 fingertips of ||pos_pred - pos_gt|| [mm]
```

Sezgisel olarak anlamlı: kaç mm kadar hatalı?

### 3. Joint-Limit Violation Rate

Anatomik sınırları aşan eklem yüzdesi. Modelin fiziksel olarak imkânsız pozlar üretip üretmediğini gösterir.

### 4. Contact Ratio

```
contact_ratio = (FK parmak uçları mesh'e < 5mm olan uç sayısı) / 5
```

**Hedef:** > 0.7

### 5. Penetration Depth

```
penetration = mean(max(0, -SDF(fingertip_pos)))   [mm]
```

**Hedef:** < 3mm ortalama

### 6. Güven Skoru Kalibrasyonu

- Tahmin edilen güven skoru vs. Unity simülasyon başarısı için AUC-ROC
- Kalibrasyon eğrisi: perfect calibration'dan sapma (reliability diagram)

**Hedef:** AUC > 0.80

### 5. Diversity Score (CVAE için)

Aynı bilek pozu + obje için K=5 farklı örnek alınır:
```
diversity = mean pairwise distance between K samples
```
Daha yüksek diversity = daha fazla kavrama stratejisi çeşitliliği. Diversity vs. quality trade-off'u raporlanır.

---

## B11. Loss Fonksiyonu Özet

**Aşama 1 (OakInk):**
```
L = 1.0*(L_recon + β*L_KL) + λ_c*L_contact + λ_p*L_penetration + λ_q*L_quality
```

**Aşama 2 (HOT3D fine-tune, T>1 batch):**
```
L = Aşama_1_loss + λ_vel*||pose_t - pose_{t-1}|| + λ_acc*||pose_t - 2*pose_{t-1} + pose_{t-2}||
```

**Aşama 3 (confidence kalibrasyon, backbone frozen):**
```
L = BCE(success_prob, unity_label)
```

| Loss terimi | Açıklama | Ne zaman aktif |
|---|---|---|
| `L_recon + β*L_KL` | CVAE pose reconstruction | Her zaman |
| `L_contact` | Parmak uçları → yüzey imzalı mesafe → 0 | Her zaman |
| `L_penetration` | `mean(relu(-SDF(pos)))` — nüfuzu penalize eder | Her zaman |
| `L_quality` | MSE(quality_score, heuristic_label) | Her zaman |
| `L_vel` | `‖(p̂_t−p̂_{t−1}) − (p_t−p_{t−1})‖` — GT hız farkı | Yalnızca T>1 (HOT3D) |
| `L_acc` | `‖(p̂_t−2p̂_{t−1}+p̂_{t−2}) − (p_t−2p_{t−1}+p_{t−2})‖` | Yalnızca T>1 (HOT3D) |
| `L_success` | BCE(success_prob, unity_label) | Yalnızca Aşama 3 |
