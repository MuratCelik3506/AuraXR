# AuraXR Thesis Completion Analysis
# Makale Bitiş Yol Haritası — 5 Döngülü Kapsamlı İnceleme

**Tarih:** 2026-06-06  
**Kapsam:** Veri setinden Unity entegrasyonuna tüm pipeline  
**Durum:** Sadece analiz ve yönlendirme — hiçbir kod değiştirilmedi

---

## Özet Tablo — Kritik Göstergeler

| Boyut | Hedef | Mevcut Durum | Boşluk |
|-------|-------|-------------|--------|
| Joint Angle MAE | < 5° | **14.0° (sağ), 13.8° (sol)** | 2.8× hedefin üzerinde |
| Quest 3 cihaz testi | Tamamlandı | **Hiç yapılmadı** | KRİTİK BLOK |
| Kullanıcı çalışması | ~15–20 katılımcı | **Hiç yapılmadı** | Makale için zorunlu |
| ONNX cihaz doğrulaması | Çalışıyor | **Sadece Mac'te test** | KRİTİK BLOK |
| Performans profili | < 5ms/frame | **Ölçülmedi** | KRİTİK BLOK |
| Ablasyon tablosu | Var | **Yok** | Makale için gerekli |

---

## DÖNGÜ 1 — Veri Seti ve Feature Mühendisliği Analizi

### 1.1 Mevcut Durum

`build_dataset.py` HOT3D Quest3 ZIP dosyalarından 15 boyutlu feature vektörü çıkarır:

```
[dir_world(3), dir_obj_local(3), dist(1), approach_speed(1)] → spatial (8)
[grip_oh(4), bbox(3)]                                         → object (7)
```

**Mevcut dataset boyutları:**
- Sağ el: 1,021,853 toplam frame, ~63,233 val
- Sol el: 877,985 toplam frame, ~62,245 val
- Grip frames (< 10cm): sağ=1,405, sol=1,061 (val setinde)

### 1.2 Tespit Edilen Problemler

#### PROBLEM D1 — Grip frame oranı çok düşük (val setinde)
**Ne:** Val setinde sağ elde 63,233 frame içinde sadece 1,405 grip frame (%2.2). Train setinde 10× oversample uygulanıyor ama val setinde yok.  
**Neden önemli:** Model grip fazında (< 10cm) 16.1° MAE yapıyor — pre_shape fazından (14.0°) daha kötü. Bu, val setindeki yetersiz grip örneğiyle de ilişkili olabilir.  
**Çözüm yolu:** Val setinde de grip frame'leri ayrıca raporla. Hem oversampled hem ham val değerlendirmesi yap. Grip-sadece bir eval seti oluştur ve grip MAE'yi ayrı bir metrik olarak sun.

#### PROBLEM D2 — approach_speed gürültülü bir feature
**Ne:** `approach_speed = dot(wrist_velocity, dir_world)` değeri sadece iki ardışık frame arasındaki farka bakarak hesaplanıyor. HOT3D 30fps @ nanosecond zaman damgasıyla kaydedilmiş ama bazı frame'ler arası dt çok küçük veya tracking glitch var.  
**Neden önemli:** Yaklaşma hızı teorik olarak güçlü bir feature, ancak gürültülü hesaplandığında pre-shape fazında net sinyal vermiyor. Model'in feature kullanımı ablasyon olmadan bilinmiyor.  
**Çözüm yolu:** approach_speed olmadan bir ablasyon eğitimi yap. Eğer MAE değişmiyorsa bu feature'ı çıkarmayı değerlendir — Unity'de wrist velocity güvenilir değil.

#### PROBLEM D3 — Sadece 2 phase label, geçiş bölgesi modellenmemiş
**Ne:** Frameler ya `grip` (< 10cm) ya `pre_shape` (10–40cm) olarak etiketleniyor. 10–40cm aralığının kendi içindeki temporal ilerleyiş (30cm nasıl farklı 15cm'den?) temsil edilmiyor.  
**Neden önemli:** Pre-shape fazı kendi içinde ~10° büyük bir aralık kapsıyor. Uzak pre-shape (35cm) ile yakın pre-shape (12cm) arasında el şekli çok farklı olmalı ama model bunları aynı "sınıf" olarak görüyor.  
**Çözüm yolu:** Distance'ı sürekli bir feature olarak kullan (zaten feature vektöründe var). Eğitimde distance-ağırlıklı loss'u 3 bölgeye böl: yakın (< 15cm) → ağırlık 3×, orta (15–30cm) → 1.5×, uzak (30–40cm) → 1×.

#### PROBLEM D4 — HOT3D'deki obje çeşitliliği sınırlı
**Ne:** 33 obje, 4 grip kategorisi. Power grip, 18/33 objeyi kapsıyor — veri seti bu kategoriye aşırı yatıklı.  
**Neden önemli:** Precision grip (çatal, kalem) ve Palmar grip (tablet, telefon) daha az örneklenmiş → Precision grip MAE ~16°, diğerlerinden yüksek.  
**Çözüm yolu:** Per-category frame sayılarını belgele. Precision için ek oversample veya ayrı ağırlıklı loss ekle. Makalede bu kısıtı açıkça belirt: "33 obje, 4 kategori — HOT3D sınırı."

#### PROBLEM D5 — Temporal bağlam yok
**Ne:** Her frame bağımsız. Model, el önceki frame'de neredeydi, ne kadar süredir yaklaşıyor bilmiyor.  
**Neden önemli:** İnsan eli nesneye yaklaşırken sürekli ve monoton bir hareket yapıyor. Stateless MLP bu yapıyı göremez — tüm temporal smoothing Unity'deki EMA'ya bırakılmış.  
**Çözüm yolu:** Bu bir mimari sınır. Makalede açıkça belirt: "Stationary MLP with post-hoc temporal smoothing." Gelecek çalışma olarak LSTM/TCN tabanlı model öner.

---

## DÖNGÜ 2 — Model Mimarisi Analizi

### 2.1 Mevcut Mimari

```
spatial(8) → FC(256)+LN+ReLU+Drop(0.2) → FC(128)+ReLU → emb(128)
object(7)  → FC(128)+LN+ReLU → FC(128)+ReLU             → emb(128)
concat(256) → FC(256)+ReLU+Drop(0.4) → FC(256)+ReLU+Drop(0.2) → trunk(256)
trunk → 5× FingerHead → joints(20) + zeros(2) = joints(22)
trunk → WristRotHead → rot6d(6)
trunk → GripClassifier (train only) → logits(4)
~294k parametre
```

### 2.2 Tespit Edilen Problemler

#### PROBLEM M1 — MAE 14° vs Hedef 5° — Kritik Performans Boşluğu
**Ne:** Sağ el genel MAE = 14.0°, sol el = 13.8°. Hedef < 5°. Boşluk 2.8× büyük.  
**Detay — En kötü jointler:**
```
Pinky.MCP:  23.6° (sağ), 22.0° (sol)
Pinky.PIP:  20.0° (sağ), 19.8° (sol)
Pinky.DIP:  20.4° (sağ), 21.1° (sol)
Index.MCP:  16.7° (sağ), 17.5° (sol)
Ring.MCP:   17.4° (sağ), 17.4° (sol)
```
Serçe parmak sistematik olarak en kötü — 5 finger head yaklaşımına rağmen.  
**Neden önemli:** 14° gerçek anlamda görsel olarak yanlış görünen pozlar üretiyor. 5° hedefi ise küçük sapmaların fark edilmeyeceği eşik.  
**Çözüm yolu:**

1. **Mimari boyut artışı:** hidden_dim 256→512, embedding_dim 128→256 dene. Parametre sayısı ~1.1M'e çıkar ama Quest 3 hala rahatlıkla işler.
2. **Serçe için ayrı ağırlık:** `_JOINT_WEIGHTS_RAW` içinde Pinky.MCP ağırlığı 3.0 (halihazırda en yüksek). Ama normalizasyon bunu siliyor olabilir. Serçe loss'unu 2× daha artır (3.0→6.0) denemesi yap.
3. **Grip fazı için distance encoding:** 0–10cm aralığını daha ince kodla. `dist` feature'ına ek olarak `1/dist` (yakınlık) veya sinüs encoding dene.
4. **Feature ablation:** approach_speed, dir_obj_local çıkarıldığında ne oluyor? Tek baseline olmadan modelin hangi feature'ları kullandığı bilinmiyor.

#### PROBLEM M2 — Wrist Rotation Kalitesi Ölçülmüyor
**Ne:** Wrist rotation loss = 0.3 × MSE(rot6d). Ama wrist rotation için hiçbir MAE metriği hesaplanmıyor ve raporlanmıyor.  
**Neden önemli:** Wrist orientation, görsel kalitede kritik. Parmak açıları doğru olsa bile bilek yanlış döndüyse el "ters bakıyor" gibi görünür. Makalede wrist rotation kalitesini ölçüp raporlamak gerekiyor.  
**Çözüm yolu:** `evaluate_onnx.py`'ye wrist rotation angular error (derece cinsinden) ekle. Geodesic distance veya euler angle MAE kullan. Hedef: < 10°.

#### PROBLEM M3 — Anatomik Kısıtlar Yeterli Değil
**Ne:** DIP ≈ 0.67 × PIP coupling loss var ama bu tek anatomik kısıt. MCP > PIP > DIP sıralaması, abduction açı sınırları, thumb opposition kısıtları yok.  
**Neden önemli:** Model anatomik olarak imkansız pozlar üretebilir. Özellikle Pinky.MCP 23° hata yapıyorken gerçek-dünya anatomisinden uzaklaşıyor.  
**Çözüm yolu:** Şu eklenebilir (range penalty zaten var):
- MCP > PIP soft coupling: `relu(PIP - MCP)` loss
- Abduction angle clamping: ±15° kısıtı
Bu eklemeler compute maliyeti yok, sadece loss term.

#### PROBLEM M4 — Grip Classifier Auxiliary Task Etkinliği Ölçülmüyor
**Ne:** Grip classifier auxiliary head eğitimde var ama onun regularization etkisi hiç test edilmemiyor.  
**Neden önemli:** Bu head, trunk'ın grip-category aware kalmasını sağlamak için var. Ama gerçekten yardımcı oluyor mu?  
**Çözüm yolu:** Ablasyon: grip_classifier olmadan eğit, MAE'yi karşılaştır. Sonucu makalede raporla.

---

## DÖNGÜ 3 — Eğitim ve Değerlendirme Analizi

### 3.1 Mevcut Eğitim Durumu

Checkpointlar mevcut: `checkpoints/right/`, `checkpoints/left/`  
ONNX dosyaları mevcut: `onnx/auraxr_right.onnx`, `onnx/auraxr_left.onnx`  
Eval sonuçları mevcut: `results/onnx_eval_right.json`, `onnx_eval_left.json`

### 3.2 Tespit Edilen Problemler

#### PROBLEM T1 — 5° Hedefine Ulaşmak İçin Net Bir Yol Yok
**Ne:** Hedef < 5°, mevcut 14°. Bu boşluğu kapatmak için ne yapılacağı belgelenmemiş.  
**Neden önemli:** Makale yazımında "model performansı" bölümü ya bu boşluğu kapatmış ya da açıkça tartışmış olmalı. Şu an ne biri ne diğeri yapılmış.  
**Çözüm yolu — Öncelik sırası:**
1. **Hızlı kazanç:** hidden_dim 256→512 ile yeniden eğit (30–60dk) → beklenen ~2–4° iyileşme
2. **Veri kalitesi:** HOT3D `train` + `test` split'lerini birleştir (val için ayrı tut). Dataset boyutunu artır.
3. **Hedefi revize et:** Eğer 5° ulaşılamıyorsa, makalede 5° hedefini "yumuşat" ve "görsel kalite eşiği" olarak yeniden çerçeve: "< 5° fark anlamlı — insan algısı eşiği". Bunu literature'dan destekle.
4. **Karşılaştırmalı baseline:** Statik pose (sabit el şekli) vs model → model ne kadar daha iyi? 14° MAE sayı olarak kötü görünse bile bir baseline'a göre anlamlı bir iyileşme sunabilir.

#### PROBLEM T2 — Eğitim Süreci Tekrar Edilebilir Değil
**Ne:** `train.py --resume` var ama hangi checkpoint'tan, hangi hyperparameterla eğitildiği belgeli değil. `checkpoints/right/model_meta.json` var ama training_log yeterince analiz edilmemiş.  
**Neden önemli:** Makale için reproducibility şart. Başkasının aynı sonucu alabilmesi gerekiyor.  
**Çözüm yolu:** `checkpoints/right/model_meta.json` ve `training_log.json` dosyalarını kontrol et. Eğitim komutunu, epoch sayısını, hangi loss'u olduğunu makalede açık şekilde yaz.

#### PROBLEM T3 — Ablasyon Tablosu Yok
**Ne:** Farklı feature setleri (V1 4-dim, V2 11-dim, V5, V6 15-dim) hiçbir zaman yan yana karşılaştırılmamış.  
**Neden önemli:** Makalenin katkısı "approach_speed + dir_obj_local feature'larını ekledik, X°'lik iyileşme sağladık" iddiasına ihtiyaç duyuyor. Bu tablo olmadan katkı kanıtlanamaz.  
**Çözüm yolu:**

```
Ablasyon tablosu (Tablo X):
┌─────────────────────────────────┬──────────┬──────────┬──────────┐
│ Feature Set                     │ Dim │ Sağ MAE° │ Sol MAE° │
├─────────────────────────────────┼─────┼──────────┼──────────┤
│ Baseline: dir_world + dist      │  4  │   ?°     │   ?°     │
│ + grip_oh + bbox                │ 11  │   ?°     │   ?°     │
│ + dir_obj_local                 │ 14  │   ?°     │   ?°     │
│ + approach_speed (V6, current)  │ 15  │  14.0°   │  13.8°   │
└─────────────────────────────────┴─────┴──────────┴──────────┘
```

Önceki versiyonlar silinmiş bile olsa, şu an 15-dim model üzerinde feature'ları sıfırlayarak (dir_obj_local=0, approach_speed=0) "ablated" inference yapılabilir.

#### PROBLEM T4 — Evaluation Sadece MAE — MPJPE Hesaplanmıyor
**Ne:** `PROJECT_LOGIC.md`'de MPJPE (Mean Per-Joint Position Error, mm cinsinden) hedef metrik olarak belirtilmiş ama hiçbir evaluate script'i bunu hesaplamıyor.  
**Neden önemli:** Computer vision alanında MPJPE standart metrik. MAE radyan/derece cinsinden sezgisel değil — mm cinsinden hata daha anlaşılır.  
**Çözüm yolu:** `evaluate_onnx.py`'ye MPJPE ekle. HOT3D'nin bone length bilgisi kullanılarak joint açı → joint pozisyona dönüştürme yapılabilir. Alternatif: makale açıkça "MAE cinsinden değerlendiriyoruz çünkü..." diyerek metrik seçimini savunur.

#### PROBLEM T5 — Cross-Validation Yok
**Ne:** Tek bir sequence-level train/val split (%15 val). Variance bilinmiyor.  
**Neden önemli:** Sonuçlar "şanslı bir split" olabilir. Makale için single-seed sonuç yeterli olabilir ama reviewer "3 run ortalaması ver" diyebilir.  
**Çözüm yolu:** En az 3 farklı seed ile eğit, ortalama ± std raporla. Bu işlem 3× süre ama makale için güven artırır.

---

## DÖNGÜ 4 — ONNX Export ve Unity Entegrasyonu Analizi

### 4.1 Mevcut Durum

ONNX dosyaları export edilmiş ve Mac'te ONNX Runtime ile doğrulanmış.  
Unity'de `AuraXRInferenceManager.cs` tam entegre — feature assembly, normalization, Gram-Schmidt decode, EMA smoothing hepsi var.  
`HandRigController.cs` OVRSkeleton'dan otomatik bone yükleme yapıyor.

### 4.2 Tespit Edilen Kritik Problemler

#### PROBLEM U1 — QUEST 3'TE HİÇ TEST EDİLMEDİ (BLOCKER)
**Ne:** `11_known_issues_gaps.md` C1–C3 numaralı kritik issue'lar: ONNX cihaz testi yok, GPUCompute Snapdragon XR2 üzerinde çalışmıyor olabilir, ReadbackAndClone() render thread'i blokluyor olabilir.  
**Neden önemli:** Makalede "Quest 3'te çalışıyor" iddiası ancak cihaz testi ile desteklenebilir. Bu olmadan paper "simulation only" kısıtını taşır.  
**Çözüm yolu — adım adım:**
1. Quest 3'e build al (Development Build + IL2CPP)
2. İlk test: `debugBypassModel=true` ile çalıştır — sadece sabit pose göster. Hand rig çalışıyor mu doğrula.
3. İkinci test: `BackendType.CPU` ile başla (GPU Compute yerine). CPU daha yavaş ama güvenilir.
4. `adb logcat` ile inference time'ı ölç. Hedef < 5ms. Gerçeklik < 15ms'de de kabul edilebilir.
5. Eğer GPU Compute çalışıyorsa geç, değilse CPU ile devam et.
6. ReadbackAndClone() blokluyorsa async inference geç: Unity Sentis `RunAsync` API'si var.

#### PROBLEM U2 — BackendType.GPUCompute Snapdragon XR2'de Güvensiz
**Ne:** `AuraXRInferenceManager.cs` satır 616: `new Worker(model, BackendType.GPUCompute)`. Snapdragon XR2 Gen 2 Vulkan Compute destekliyor ancak Unity Sentis'in Snapdragon üzerindeki GPUCompute davranışı dokümente değil.  
**Neden önemli:** Cihazda crash veya yanlış sonuç üretebilir. 294k parametreli MLP için CPU genellikle 2–5ms — GPUCompute'a geçmek gerekli mi bile belirsiz.  
**Çözüm yolu:**
```csharp
// AuraXRInferenceManager.cs içinde:
BackendType backend = Application.platform == RuntimePlatform.Android 
    ? BackendType.CPU 
    : BackendType.GPUCompute;
worker = new Worker(model, backend);
```

#### PROBLEM U3 — ReadbackAndClone() Render Thread Bloğu
**Ne:** `using var angleCpu = anglesTensor.ReadbackAndClone()` senkron GPU→CPU transfer. Quest 3'te GPU'dan CPU'ya kopyalama frame'i blokluyor olabilir.  
**Neden önemli:** 72Hz rendering'de her frame 13.9ms var. Senkron tensor transfer bunu aşabilir → frame drop.  
**Çözüm yolu:**
1. Kısa dönem: CPU backend kullan → ReadbackAndClone zaten gerekmiyor (tensor zaten CPU'da).
2. Uzun dönem: Unity Sentis `IEnumerator` based async scheduling ile bir frame gecikmeli sonuç al.

#### PROBLEM U4 — EMA Parametreleri Kalibre Edilmemiş
**Ne:** `emaAlpha=0.35` (joint angles), `rotEmaAlpha=0.25` (wrist rotation). Bu değerler ampirik olarak seçilmiş, gerçek kullanıcı hareketi verisiyle doğrulanmamış.  
**Neden önemli:** Yanlış EMA değerleri ya "sticky hands" (çok yavaş tepki) ya da "jittery hands" (çok hızlı tepki) üretir. Her ikisi de naturalness skorunu düşürür.  
**Çözüm yolu:** Quest 3 loglarından (AuraXRLogger zaten var) gerçek frame-to-frame değişim dağılımını analiz et. Optimal EMA'yı bu dağılımdan türet: `alpha ≈ 1 - exp(-dt/tau)` formülü.

#### PROBLEM U5 — Pivot Offset Tek Ölçümden Geliyor
**Ne:** `handPivotOffset = new Vector3(0.1685f, 0f, 0.0351f)` tek bir oturumdan ölçülen değer.  
**Neden önemli:** Controller tutma şekli, el boyutu, headset ayarına göre değişir. Sabit offset kullanıcı çalışmasında her katılımcı için yanlış olabilir.  
**Çözüm yolu:** Mevcut `LogWristOffset()` fonksiyonu zaten bu değeri ölçüyor. 3–5 farklı kullanıcıyla ölç ve ortalamasını al. Veya kullanıcıya kalibrasyon adımı ekle (bu thesis scope'unu aşabilir — makalede limitation olarak belirt).

#### PROBLEM U6 — ProximityDetector: En Yakın Nesne Histerezis Yok
**Ne:** `ProximityDetector.cs` her frame'de `FindNearest()` çağırıyor ve O(n) linear search yapıyor. Eğer iki nesne eşit mesafede ise frame-to-frame switching olabilir.  
**Neden önemli:** Nesne switching'i anında grip/bbox değişimine yol açar → EMA bunu temizleyemez → "model özellikleri aniden değişiyor" etkisi → görsel jitter.  
**Çözüm yolu:**
```csharp
// Hysteresis: yeni nesneyi ancak şu an takip edilenden %20 daha yakınsa kabul et
float switchThreshold = currentDist * 0.80f;
if (candidateDist < switchThreshold) SwitchTarget(candidate);
```

#### PROBLEM U7 — Bone Wiring Doğrulanmamış
**Ne:** `HandRigController.cs` OVRSkeleton'dan otomatik bone alıyor ama "fingerJoints[15] not confirmed all 15 joints wired for both hands" (issue U1 in 11_known_issues_gaps.md).  
**Neden önemli:** Yanlış bone → parmaklar yanlış yerde veya hiç bükülmüyor. Bunun cihaz testi olmadan doğrulanması mümkün değil.  
**Çözüm yolu:** Quest 3 build'inde `debugForceTestPose=true` ile çalıştır. Tüm parmaklar 28° açıyla bükülüyor mu gözlemle. Bükülmüyorsa bone mapping sorunu var demektir.

---

## DÖNGÜ 5 — Makale İçin Pipeline Tamamlanması

### 5.1 Makale İçin Zorunlu Olanlar

Bir konferans/dergi makalesi için aşağıdakiler **zorunlu**:

#### ZORUNLU 1 — Cihaz Üstü Performans Ölçümü
**Durum:** Yok  
**Gereksinim:** Quest 3'te ms cinsinden inference süresi. Her yerde "real-time" iddiası var ama ölçüm yok.  
**Nasıl:** AuraXRLogger zaten var. Buna inference start/end timestamp ekle, logları Mac'e çek, analiz et.

#### ZORUNLU 2 — Kullanıcı Çalışması
**Durum:** Hiç yapılmamış  
**Gereksinim:** Minimum 15 katılımcı, 3 koşul (Controller only / Static Pose / AuraXR), Latin square order, Likert naturalness rating, NASA-TLX workload.  
**Timeline:** 
- IRB başvurusu → 2–4 hafta
- Katılımcı toplama → 2 hafta
- Veri toplama (her katılımcı ~30dk) → 2 hafta
- Analiz → 1 hafta
- Toplam minimum: 7–9 hafta

**Bu makale bitiş için kritik yol üzerinde.**

#### ZORUNLU 3 — Ablasyon Tablosu
**Durum:** Yok  
**Gereksinim:** En az 2–3 feature seti karşılaştırması.  
**Hızlı yol:** Mevcut modelle 15-dim, 11-dim (approach_speed=0, dir_obj_local=0), 8-dim (sadece dir_world+dist+grip+bbox) inference yapıp MAE karşılaştır. Yeniden eğitim gerekmez — feature maskeleme yapılır.

#### ZORUNLU 4 — Kaliteli Sonuç Görselleri
**Durum:** Yok  
**Gereksinim:** 
- Simülasyon grafikler: `simulate.py` çıktıları (her nesne kategorisi için)
- Quest 3 ekran kaydı: el şekli değişimini gösteren video frame'leri
- Karşılaştırma figürü: Ground truth vs model prediction (HOT3D frame üzerinde)

#### ZORUNLU 5 — Baseline Karşılaştırması
**Durum:** Tanımlanmamış  
**Gereksinim:** "Modelimiz X'ten ne kadar iyi?" sorusuna cevap.  
**Önerilen baselines:**
1. **Zero baseline:** Tüm jointleri sıfır predict et → MAE'si nedir? (~sabit ortalama)
2. **Mean baseline:** Training setinin mean joint angle'larını predict et → MAE?
3. **Object-only baseline:** Distance olmadan, sadece grip category ile predict et
4. **Static pose:** Unity'de sabit el şekli (controller rotasyonu)

Bu baselines'ları elde etmek 1 gün çalışma.

### 5.2 Opsiyonel (Ama Güçlendirir)

#### OPSİYONEL A — Model Geliştirmesi
Eğer MAE 5°'ye indirilebilirse makale çok güçlenir. Bunun için:
1. hidden_dim 512'ye çıkar → yeniden eğit (~2 saat)
2. Approach speed için daha iyi velocity estimation (windowed average, 3 frame ortalaması)
3. Temporal input: son 3 frame'in feature'larını concat et (15×3=45 dim input)

#### OPSİYONEL B — Görsel Kalite Metriği
MAE'nin yanı sıra bir "naturalness" metriği tanımla:
- Anatomik geçerlilik skoru: kaç frame'de joint angle limitleri ihlal ediliyor?
- Temporal smoothness: frame-to-frame açı değişiminin std'si

#### OPSİYONEL C — Quantization
ONNX Float32 → Float16 quantization → boyut ~%50 küçülür, inference hızı artabilir.  
`onnxruntime.quantization` ile 1 saatte yapılır.

---

## Tam Pipeline — Baştan Sona Yapılması Gerekenler

```
ADIM 1: Model Geliştirmesi [1–2 gün]
  ├── hidden_dim=512 ile yeniden eğit
  ├── Ablasyon tablosu için feature maskeleme inference
  └── Wrist rotation MAE metriği ekle

ADIM 2: Simulation ve Visualizations [1 gün]
  ├── python simulate.py --object bottle/cup/spoon/mouse çalıştır
  ├── Grafikleri makale figürü olarak kaydet
  └── Test sonuçlarını docs'a ekle

ADIM 3: Quest 3 Cihaz Testi [2–3 gün]
  ├── BackendType.CPU ile build al
  ├── debugBypassModel=true → bone wiring doğrula
  ├── Gerçek inference ile çalıştır
  ├── Inference süresini logla (< 5ms hedef)
  └── ReadbackAndClone timing ölç

ADIM 4: Baseline Karşılaştırması [1 gün]
  ├── Zero/Mean/Object-only/Static-pose baselines
  └── MAE tablosu oluştur

ADIM 5: Kullanıcı Çalışması [7–9 hafta — paralel yürüt]
  ├── IRB başvurusu (hemen başla!)
  ├── Protokol yaz (PROJECT_LOGIC.md'deki taslak var)
  ├── Unity'de 3 koşul sistemi kur
  ├── SessionDataLogger doğrula
  └── Veri topla ve analiz et

ADIM 6: Makale Yazımı [3–4 hafta]
  ├── Methodology: feature engineering açıkla
  ├── Results: ablasyon tablosu, MAE tablo, kullanıcı çalışması
  ├── Discussion: 14° MAE'yi açıkla, 5° hedefine neden ulaşılamadı
  ├── Limitations: single-frame MLP, 33 object sınırı, pivot offset
  └── Future work: LSTM/TCN, visual features, multi-object

ADIM 7: Son Kontroller [1 hafta]
  ├── onnx_eval sonuçlarını makaleye işle
  ├── Tüm claims'i data ile destekle
  └── Video demo hazırla
```

---

## Süreç Yönetimi — Risk ve Öncelik Matrisi

### Kritik Yol (Bunlar blokluyor)

| Görev | Bağımlılık | Süre Tahmini | Risk |
|-------|-----------|-------------|------|
| Quest 3 cihaz testi | Hardware erişimi | 2–3 gün | YÜKSEK (ilk build problem olabilir) |
| IRB başvurusu | Tez danışmanı onayı | 2–4 hafta bekleme | YÜKSEK (zaman alıyor) |
| Kullanıcı toplama | IRB onayı | 2–3 hafta | ORTA |
| Makale draft | Tüm veriler | 3–4 hafta | DÜŞÜK |

### Paralel Yapılabilecekler

Şu an **hemen** yapılabilecekler (Quest 3 beklemeden):
1. Ablasyon tablosu (1 gün)
2. Baseline karşılaştırması (1 gün)
3. Wrist rotation MAE metriği (yarım gün)
4. hidden_dim=512 eğitimi (2 saat)
5. Simulation grafikleri (1 saat)
6. IRB başvurusu taslağı (danışmanla birlikte)

### Eğer Süre Kısıtlı İse — Minimum Viable Paper

Eğer tez için minimum yeterli sonuç gerekiyorsa:

```
Minimum set:
✓ Mevcut MAE (14°) → baseline'a göre karşılaştır
✓ Quest 3 cihaz testi (inference süresi)
✓ Ablasyon tablosu (maskeleme ile)
✓ Simulation grafikleri
✗ Kullanıcı çalışması → "future work" olarak ertele
    (ama bu makale türüne bağlı — sistem makalesi vs kullanıcı çalışması makalesi)
```

---

## Bilinen Eksikler — 11_known_issues_gaps.md'den Güncel Durum

### Hala Açık (2026-06-06 itibariyle)

| ID | Sorun | Kritiklik |
|----|-------|----------|
| C1 | ONNX Quest 3'te test edilmedi | KRİTİK |
| C2 | BackendType.GPUCompute Snapdragon'da onaylanmadı | KRİTİK |
| C3 | ReadbackAndClone() async edilmedi | YÜKSEK |
| M3 | Ablasyon tablosu yok | YÜKSEK |
| U1 | fingerJoints[15] wiring onaylanmadı | YÜKSEK |
| U2 | Pivot offset tek ölçümden | ORTA |
| U3 | Sign convention validation | ORTA |
| P2 | MAE sonuçları 05_training_evaluation.md'e işlenmedi | DÜŞÜK |
| P3 | Model quantization yapılmadı | DÜŞÜK |
| P4 | Kullanıcı çalışması protokolü yok | KRİTİK |

### Çözülenler
- M1: V6 architecture (15-dim input, 294k params) — teyit edildi
- M2: BopToGrip + BopToBbox hataları — düzeltildi (2026-06-03)
- P1: Frame sayıları — belgelendi

---

## Sonuç — Makaleye Götürmek İçin Özet

### Bugün Yapılabilecekler (Hemen Başla)

1. **`python simulate.py`** — bottle, cup, spoon için çalıştır, grafikleri kaydet
2. **Wrist rotation MAE** — `evaluate_onnx.py`'ye ekle
3. **Ablasyon inference** — approach_speed=0, dir_obj_local=0 ile mevcut modelden MAE hesapla
4. **IRB taslağı** — tez danışmanınla konuş, kullanıcı çalışması süreci başlat

### Bu Hafta

5. **hidden_dim=512 eğitimi** — `train.py --hidden_dim 512`
6. **Baseline tablosu** — zero, mean, object-only baselines

### Önümüzdeki Ay

7. **Quest 3 cihaz testi** — BackendType.CPU ile başla
8. **Kullanıcı çalışması** (IRB onayı bekleniyor)

### Makale Zaman Çizelgesi (Gerçekçi)

Kullanıcı çalışması dahil tam makale: **2026-10 hedefe uygun** eğer IRB başvurusu hemen yapılırsa.  
Cihaz testi + ablasyon + system makale (kullanıcı çalışması olmadan): **2026-08 mümkün.**

---

*Bu analiz 2026-06-06 tarihinde 5 döngülü okuma ile oluşturulmuştur.*  
*Hiçbir kod değiştirilmemiştir — sadece gözlem ve yönlendirme içerir.*
