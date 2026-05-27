# VR Controller ile Nesne Tabanlı El Pozu Tahmini

## Proje Özeti

Bu proje, VR ortamında controller kullanan kullanıcının el şeklini, yaklaştığı nesnenin türüne ve şekline göre tahmin eden bir yapay zeka modeli geliştirmeyi amaçlamaktadır.

---

## Problem Tanımı

### Mevcut Durum
- VR controller kullanırken kullanıcının gerçek eli görünmez
- Controller sadece pozisyon ve rotasyon bilgisi verir
- Sanal ortamda el, genellikle sabit bir poz olarak gösterilir
- Bu durum immersiyonu (sürükleyiciliği) azaltır

### Hedef
- Controller verisinden el şekli tahmini
- Nesneye yaklaşırken doğal el hareketi
- Nesne türüne göre uygun kavrama pozisyonu
- Gerçekçi ve akıcı el animasyonu

---

## Yaklaşım: Proximity-Based Anticipatory Grasp

İnsan eli nesneye uzanırken önceden şekillenir. Bu doğal davranışı taklit eden sistem 4 durumu kapsar:

### Durum 0: Varsayılan (> 40cm)

**Durum:** El nesneden uzak

**Davranış:**
- Model çalışmaz
- El sabit varsayılan pozda durur — `FIXED_DEFAULT_POSE` (bir kez ölçülerek sabitlenir: controller tutarken doğal, hafif bükümlü parmak pozisyonu)

---

### Aşama 1: Pre-Shape (10cm - 40cm)

**Durum:** El nesneye yaklaşmaya başlıyor

**Davranış:**
- El varsayılan pozdan çıkıyor
- Nesne türüne göre hazırlık başlıyor
- Örnek: Bardak için parmaklar açılıyor, avuç yuvarlaklaşıyor
- Örnek: Kalem için başparmak-işaret parmağı hazırlanıyor

**Model Görevi:** Nesne türü ve boyutuna göre "hazırlık pozu" tahmini

---

### Aşama 2: Approach (< 10cm)

**Durum:** El nesneye çok yakın, kavramaya hazır

**Davranış:**
- El tam grip pozisyonuna geçiyor
- Parmaklar nesne yüzeyine doğru yöneliyor
- Mesafe azaldıkça poz kesinleşiyor
- Smooth interpolasyon ile geçiş

**Model Görevi:** Nesne şekline uygun "kavrama pozu" tahmini

---

### Aşama 3: Grasp (Temas + Grip)

**Durum:** El nesneye temas etti, kullanıcı grip butonuna bastı

**Davranış:**
- Parmaklar nesneyi sarıyor
- Nesne şekline göre parmak açıları ayarlanıyor
- Fiziksel bağlantı kurulur (nesne ele yapışır)
- Tutma sırasında poz korunur

**Model Görevi:** Nesne konumu ve grip kategorisine göre "final kavrama pozu" tahmini — poz sabitlenir ve tutma boyunca korunur

---

## Veri Seti: HOT3D

### Neden HOT3D?

| Özellik | HOT3D Avantajı |
|---------|----------------|
| El formatı | UmeTrack (VR uyumlu) |
| Nesne verisi | 33 farklı nesne, 6DoF poz |
| Koordinat sistemi | El ve nesne aynı sistemde |
| Kayıt cihazı | Quest 3 (hedef platformla aynı) |
| Veri miktarı | 3832 klip, 1.5M+ frame |

### Mevcut Veriler

**El Verisi (Her Frame):**
- UmeTrack formatı: 22 eklem açısı
- Bilek pozisyonu: Quaternion + Translation (7 değer)
- Sol ve sağ el ayrı anotasyonlu

**Nesne Verisi (Her Frame):**
- Nesne türü ve ID
- 6DoF poz (konum + rotasyon)
- Segmentasyon maskesi
- Görünürlük bilgisi

**Nesne Modelleri:**
- 33 farklı rijit nesne
- 3D mesh (.glb formatı)
- Boyut bilgisi (X, Y, Z, çap)
- PBR materyalleri

---

## Model Mimarisi

### Girdi

| Parametre | Boyut | Açıklama |
|-----------|-------|----------|
| Nesnenin göreli konumu | 3 | Bilek frame'inde x, y, z offset — hem yönü hem mesafeyi kodlar |
| Grip kategorisi | 4 | Power / Precision / Palmar / Pinch — one-hot |
| Nesne boyutu | 3 | Bounding box half-extents (x, y, z metre) |
| Mesafe | 1 | Skalar, metre |
| **Toplam** | **11** | |

> **Neden bilek pozisyon/rotasyonu yok?** Grip şekli elin nesneden ne kadar uzakta olduğuna ve nesne tipine bağlıdır — elin odada nerede olduğuna değil. Mutlak bilek pozisyonu kayıtlar arasında değişir ve genelleme yapmaz. Bilek rotasyonu ise göreli konumda zaten örtülü olarak kodlanmıştır (göreli konum bilek frame'inde hesaplandığından). İkisini de çıkarmak modeli sade ve koordinat sisteminden bağımsız tutar.

> **Neden 33 one-hot değil?** Nesneyi ID ile değil fiziksel özellikleriyle temsil etmek modeli görülmemiş nesnelere genelleştirir. "Büyük silindir 12cm önümde" öğrenmek, "object_id=17" ezberlemekten çok daha anlamlı. 33 nesne zaten 4 grip kategorisine indirgenebilir.

### Çıktı

| Parametre | Boyut | Açıklama |
|-----------|-------|----------|
| Joint angles | 22 | UmeTrack formatı — yalnızca parmak açıları |

> **Bilek modelden gelmez.** Bilek controller'a sabit fiziksel ofsetle bağlanır, model yalnızca parmakları tahmin eder. Bkz. HOT3D → Unity Köprüsü.

### Yapı — İki Branch MLP

Uzamsal yaklaşma bilgisi ile nesne tipi/şekli bilgisi farklı türde girdilerdir. İki ayrı encoder kullanılır:

```
Göreli konum (3) + Mesafe (1)       →  [Spatial Encoder  2×FC]  →  spatial_emb (32)
Grip kategorisi (4) + Boyut (3)     →  [Object Encoder   2×FC]  →  obj_emb     (32)
                                                                          ↓
                                               [Concat → Prediction Head 2×FC]
                                                                          ↓
                                                           Joint angles (22)
```

- Her encoder: 2 tam bağlantılı katman, ReLU aktivasyon
- Prediction head: 2 tam bağlantılı katman, son katmanda Tanh
- Tanh çıktısı [-1, 1] → denormalize edilerek gerçek açılara dönüştürülür
- Normalizasyon parametreleri model metadata dosyasında saklanır

### Neden İki Ayrı Model?

- Sol el için ayrı model
- Sağ el için ayrı model
- Avantaj: Bağımsız eğitim, kolay debug
- Aynı mimari, farklı ağırlıklar

---

## Unity Entegrasyonu

### Platform
- Unity 2021.3+ LTS
- Meta XR SDK
- Quest 2/3/Pro

### Çalışma Mantığı

1. **Her frame:**
   - Controller pozisyonunu al
   - En yakın nesneyi tespit et
   - Mesafeyi hesapla

2. **Mesafeye göre karar:**
   - > 40cm: Model çalışmaz, FIXED_DEFAULT_POSE
   - 10–40cm: Model → pre-shape tahmini + interpolasyon
   - < 10cm: Model → grip tahmini
   - Temas + Grip butonu: Final grip pozu sabitlenir

3. **Görselleştirme:**
   - UmeTrack joint angles → Unity el modeline uygula
   - Smooth transition (Lerp/Slerp)
   - Her iki el bağımsız güncellenir

### Temas Algılama

- Her parmak için collider
- Nesne yüzeyi ile collision tespiti
- Grip butonu durumu kontrolü

### Poz Düzleştirme (Pose Smoothing)

Model stateless bir MLP olduğu için ardışık frame'ler arasında büyük sıçramalar olabilir. Bunu önlemek için katmanlı bir düzleştirme sistemi:

**Adım 1 — Eşik Geçişi Blend Zonu**
40cm'de modeli aniden açıp kapamak sıçrama yaratır. Bunun yerine 30–40cm arasında model çıktısı ile FIXED_DEFAULT_POSE mesafeye göre karıştırılır:
```
α = clamp((40 - distance) / 10, 0, 1)
pose = lerp(FIXED_DEFAULT_POSE, model_output, α)
```
- 40cm'de α=0 → saf varsayılan poz
- 30cm'de α=1 → saf model çıktısı
- Arası yumuşak geçiş

**Adım 2 — Input EMA (Giriş Düzleştirme)**
Model çalıştırılmadan önce giriş verileri düzleştirilir. Controller tracking gürültüsünü ve en yakın nesnenin ani değişimini engeller.
```
smooth_input[t] = α_in × raw_input[t] + (1 - α_in) × smooth_input[t-1]
```
- Uygulanır: nesne göreceli konumu, nesne mesafesi
- Önerilen α_in: 0.3–0.5

**Adım 3 — Model Çıkarımı (Her N Frame'de)**
`Inference Every N Frames = 2` ile model her frame çalışmaz. Ara frame'leri Unity lerp doldurur.

**Adım 4 — Output EMA (Çıktı Düzleştirme)**
Model output'u doğrudan uygulamak yerine önceki smooth pose ile karıştırılır:
```
smooth_pose[t] = α_out × model_output[t] + (1 - α_out) × smooth_pose[t-1]
```
- Uygulanır: 22 joint angle değerinin tamamına
- Önerilen α_out: 0.2–0.4

**Adım 5 — Delta Clamp (Ani Sıçrama Engeli)**
Tek frame'de bir joint angle'ın ne kadar değişebileceğini fiziksel sınırla:
```
delta = smooth_pose[t] - current_pose[t-1]
new_pose = current_pose[t-1] + clamp(delta, -max_delta, max_delta)
```
- max_delta değeri joint başına deneysel ayarlanır

**Uygulama Önceliği:**
1. Adım 1 (blend zonu) — eşik geçişini çözer
2. Adım 2 + 4 (EMA input/output) — genel gürültüyü azaltır
3. Yetmezse Adım 5 (delta clamp) ekle

---

## Nesne Kategorileri

### Grip Türlerine Göre

| Kategori | Nesneler | Grip Türü |
|----------|----------|-----------|
| Silindirik | Bardak, şişe, kutu | Power grip (yumruk kavrama) |
| İnce/Uzun | Kaşık, kalem, spatula | Precision grip (parmak ucu) |
| Düz/Geniş | Tabak, klavye, telefon | Palmar grip (avuç) |
| Küçük | Mouse, puzzle | Pinch grip (tutam) |

### Boyuta Göre

| Kategori | Çap | Örnek Nesneler |
|----------|-----|----------------|
| Küçük | < 12cm | Mouse, puzzle, kutu |
| Orta | 12-22cm | Bardak, şişe, telefon |
| Büyük | > 22cm | Tabak, klavye, vazo |

---

## Test Süreci

Model dört aşamada test edilir. Her aşama bir sonrakinin önkoşuludur.

### Aşama 1 — Python Offline Değerlendirme

HOT3D val split'indeki frame'lere model çalıştırılır, tahmin ile ground truth karşılaştırılır. Smoothing yoktur — ham model performansı ölçülür.

**Metrikler:**

| Metrik | Açıklama | Hedef |
|--------|----------|-------|
| Joint Angle MAE | Eklem açısı ortalama hatası (derece) | < 5° |
| MPJPE | Eklem başına konum hatası (mm) | < 20 mm |
| Nesne bazında hata | Bardak ve şişe için ayrı MAE | Dengeli olmalı |

> Not: Sayısal hata düşük olsa da VR'da kötü görünebilir. Asıl kriter subjektif doğallıktır.

### Aşama 2 — Python Simülasyon Testi

Gerçek veri yerine sentetik bir yaklaşma trajektorisi oluşturulur:

```
mesafe: 40cm → 30cm → 20cm → 10cm → 5cm → 2cm
nesne: sabit (örn. Bottle)
```

Her mesafe adımında tahmin edilen joint angle'lar çizilir. Beklenen davranış:
- 40cm'de parmaklar açılmaya başlamalı
- 10cm'de grip şekli netleşmeli
- 2cm'de tam kavrama pozu olmalı

Bu aşamada sıçrama veya anlamsız geçiş görülürse sorun Unity'ye taşınmadan yakalanır.

### Aşama 3 — Unity Editor Testi (Quest Olmadan)

ONNX export sonrası Unity Play modunda test edilir. Sanal bir controller path simüle edilir, cihaz gerekmez.

**Kontrol edilecekler:**
- Bottle'a yaklaşırken el pozu değişiyor mu?
- Console'da inference süresi makul mi? (Quest'te hedef < 5ms)
- EMA + delta clamp smoothing çalışıyor mu?
- Bottle'dan Cup'a geçişte poz bozuluyor mu?

### Aşama 4 — Quest Üzerinde Tam Test

`SessionDataLogger` ile her frame kayıt altına alınır. Oturum sonrası kayıtlar Python'da analiz edilir.

**Kontrol edilecekler:**
- Gerçek kullanımda joint angle dağılımı beklenen aralıkta mı?
- Grip fazında doğru açılar çıkıyor mu?
- Üç koşul karşılaştırması: VirtualHands vs Controller vs StaticPose

**Değerlendirme protokolü (minimum):**

| Ölçüt | Yöntem |
|-------|--------|
| Doğallık | 5'li Likert ("el hareketi doğal göründü mü?") |
| Görev süresi | SessionDataLogger ile otomatik ölçüm |
| Yük | NASA-TLX anketi |
| Karşılaştırma | 3 koşul, Latin kare sırası |
| Katılımcı sayısı | ~15–20 kişi (orta etki büyüklüğü, %80 güç) |
| Etik kurul | Veri toplamadan önce etik kurul onayı alınmalı |

> Sayısal hata (MAE) düşük olsa da VR'da kötü görünebilir — subjektif değerlendirme zorunludur.

---

## HOT3D → Unity Köprüsü

HOT3D'de controller verisi yoktur. Bunun yerine her frame'de **bilek pozisyonu ve rotasyonu** mevcuttur. Unity'de ise tam tersi: controller vardır, bilek yoktur.

### Neden Sorun Değil?

Model girişi, nesnenin **elin kendi frame'ine göre göreceli konumudur** — elin dünyadaki mutlak pozisyonu değil. Bu göreceli konum eğitimde ve çalışma zamanında farklı biçimlerde hesaplanır ama sonuç eşdeğerdir:

```
Eğitimde →  rel_pos = R_bilek^T × (nesne_world − bilek_world)      (HOT3D bilek verisi)
Unity'de →  rel_pos = R_ctrl^T × (nesne_world − ctrl_world)         (gerçek controller verisi)
```

Quest controller'ı elde tuttuğunda controller pozisyonu ≈ bilek pozisyonu ve controller rotasyonu ≈ bilek rotasyonudur. Dolayısıyla her iki kaynaktan hesaplanan göreceli konum neredeyse aynıdır.

### Bilek Görselleştirmesi

Bilek controller'a sabit fiziksel ofsetle bağlıdır. Model yalnızca parmak açılarını tahmin eder, bilek konumuna dokunmaz:

```csharp
// Bilek — her zaman controller ile çakışır
anchor.position = controller.position + FIXED_GRIP_OFFSET;
anchor.rotation = controller.rotation * FIXED_GRIP_ROTATION;

// Model sadece parmakları günceller
ApplyFingerAngles(modelOutput.jointAngles);
```

`FIXED_GRIP_OFFSET` ve `FIXED_GRIP_ROTATION` bir kez fiziksel olarak ölçülür: Quest controller'ı doğal tuttuğunda bilek merkezinin controller takip noktasına göre konumu.

### Koordinat Sistemi

Hem HOT3D hem Unity dünya uzayı kullanır. HOT3D'nin dünya uzayı kayıt bazlıdır (her sekansın kendi başlangıç noktası vardır). Unity'nin dünya uzayı ise sahne koordinatlarıdır. Bu fark önemsizdir çünkü model girişi her zaman bilek/controller frame'inde göreceli bir konumdur — yapı gereği koordinat sisteminden bağımsızdır.

### Özet

| | Eğitim (HOT3D) | Unity Çalışma Zamanı |
|---|---|---|
| Bilek/controller verisi | HOT3D wrist_transform | Gerçek controller transform |
| Göreceli konum | `R_bilek^T × (nesne − bilek)` | `R_ctrl^T × (nesne − ctrl)` |
| Nesne bilgisi | Ground truth annotation | ProximityDetector |
| Bilek görselleştirmesi | — | Controller + FIXED_GRIP_OFFSET |

---

## Eğitim Stratejisi

### Veri Hazırlama

1. HOT3D kliplerinden frame çıkar
2. **Yalnızca 0–40cm arası frame'leri tut** — Unity'de model 40cm'den itibaren çalışır, uzak frame'ler anlamsız
3. Her frame için:
   - El pozu (UmeTrack 22 joint)
   - En yakın nesne türü ve boyutu
   - El-nesne mesafesi
4. Mesafe eşiklerine göre etiketle:
   - Pre-shape (10–40cm): Hazırlık pozu örneği
   - Grip (< 10cm): Kavrama pozu örneği

### Eğitim

- Loss: **Ağırlıklı MSE** — 0–10cm frame'lerine daha yüksek ağırlık (yakın frame'ler az ama kritik)
- Optimizer: Adam
- Batch size: 64
- Epochs: 50–100

### Export

- PyTorch → ONNX
- UmeTrack formatı korunur
- Unity Sentis ile inference

---

## Avantajlar

| Yaklaşım | Avantaj |
|----------|---------|
| HOT3D + UmeTrack | Quest SDK ile doğal uyum |
| Proximity-based | Doğal insan davranışı |
| Nesne-aware | Farklı nesnelere farklı grip |
| İki el bağımsız | Gerçekçi iki elle etkileşim |
| Smooth transition | Sıçrama yok, akıcı animasyon |

---

## Potansiyel Zorluklar

### Kritik — Şimdi Ele Alınmalı

| Zorluk | Çözüm |
|--------|-------|
| Uzak frame'ler eğitimi bozar | Sadece 0–40cm frame kullan |
| Yakın frame'ler az, loss baskılanır | Ağırlıklı MSE, yakın frame'lere yüksek ağırlık |
| Değerlendirme protokolü yok | Likert + NASA-TLX + görev süresi tanımla |

### Orta — Gerekirse Düzelt

| Zorluk | Çözüm |
|--------|-------|
| En yakın nesne hızla değişir | Histerezis: yeni nesne belirgin şekilde yakın oluncaya kadar geçiş yapma |
| Nesne algılanmadığında model ne alır | Nesne yoksa model çalışmasın, varsayılan poza lerp |
| Anatomik imkânsız poz | Önce dene, sorun çıkarsa joint angle sınırları ile clamp |

### Bilinen Sınırlama — Tez Kapsamı Dışı

| Zorluk | Durum |
|--------|-------|
| El-nesne içinden geçme | IK/fizik simülasyonu gerektirir, gelecek çalışma |
| Controller tutan el ile serbest el farkı | Sanal el tahmini olduğu için kritik değil |
| Koordinat dönüşümü | AuraXRFeatureAssembler hallediyor |
| Quest performansı < 5ms | MLP küçük, ölçülmeli |

---

## Proje Aşamaları

### Faz 1: Veri Hazırlama
- [ ] HOT3D'den eğitim verisi çıkar
- [ ] Nesne-el çiftlerini oluştur
- [ ] Mesafe eşiklerine göre etiketle

### Faz 2: Model Eğitimi
- [ ] UmeTrack tabanlı MLP modeli
- [ ] Sol ve sağ el için ayrı eğitim
- [ ] Validasyon ve test

### Faz 3: ONNX Export
- [ ] PyTorch → ONNX dönüşümü
- [ ] Girdi/çıktı format doğrulama

### Faz 4: Unity Entegrasyonu
- [ ] Meta XR SDK kurulumu
- [ ] ONNX model yükleme (Sentis)
- [ ] Proximity sistemi implementasyonu
- [ ] El modeli görselleştirme

### Faz 5: Test ve İyileştirme
- [ ] Quest üzerinde test
- [ ] Performans optimizasyonu
- [ ] Kullanıcı deneyimi değerlendirmesi

---

## Beklenen Sonuçlar

1. **Doğal görünen el hareketi** - Nesneye yaklaşırken
2. **Nesne-spesifik grip** - Her nesneye uygun kavrama
3. **Smooth geçişler** - Aşamalar arası akıcılık
4. **Real-time performans** - Quest'te 72+ FPS
5. **İki el desteği** - Bağımsız ve koordineli

---

## Referanslar

- HOT3D Dataset: Meta Reality Labs, CVPR 2025
- UmeTrack: Meta, SIGGRAPH Asia 2022
- MANO: Max Planck Institute, 2017
- Anticipatory Grasp: Literatürde yaygın konsept

---

*Bu doküman proje mantığını açıklar. Teknik implementasyon detayları ayrı dokümanlarda yer alacaktır.*
