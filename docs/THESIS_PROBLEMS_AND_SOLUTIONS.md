# Tez Sorunları ve Çözüm Planları — Sade Anlatım

Bu belge, teknik değerlendirmede (THESIS_EVALUATION.md) tespit edilen sorunları daha az teknik bir dille açıklar ve her biri için uygulanabilir bir çözüm planı sunar.

---

## BÖLÜM A — Mimari ve Tasarım Sorunları

---

### A1. Model Objeye Yaklaşırken Eli Zaten Kapatıyor

**Ne oluyor, neden yanlış?**

Hayal edin bir masanın üzerindeki bardağa uzanıyorsunuz. Normalden eliniz açık, bardağa yaklaştıkça parmaklarınız kapanmaya başlıyor, tam temas anında kavramayı tamamlıyorsunuz. Bu doğal.

Bu sistemde ise el bardağa 10 cm uzakta bile olsa model zaten kapalı parmak pozu üretiyor. Unity bardağa yaklaşıldığında bu kapalı pozu karıştırmaya başladığı için kullanıcı henüz bardağa değmeden parmaklarının kapandığını görüyor. XR'da bu çok rahatsız edici görünüyor.

**Neden böyle oluyor?**

Modelin tek bir çıkış modu var: "şu an parmaklar nerede olmalı?" sorusunu cevaplıyor, ama aslında hep "final kavrama pozu nerede?" sorusunu cevaplıyor. Yaklaşım aşamasında da, temas anında da aynı decoder aynı şeyi yapıyor. Aralarındaki farkı göremiyoru.

Üstüne üstlük, eğitim verisinin %79'u yaklaşım fazı (el açık) ama model bunu yeterince öğrenememiş. Sonuç: her durumda yarı kapalı bir el üretiyor — ne tam açık, ne tam kapalı.

**Çözüm Planı:**

Kısa vadeli (**runtime mitigation — bilimsel model problemini çözmez, tezde böyle etiketle**):
1. Unity tarafında blend başlangıç mesafesini 10cm'den 4cm'e çek. Modelin etkisi sadece çok yakında başlasın.
2. Blend formülünü güncelle: `smoothstep(10cm, 2cm)` yerine `smoothstep(4cm, 1cm)`.

Orta vadeli (doğru mimari düzeltme):
1. Modele `dist` (bilek-obje mesafesi) değerini sürekli bir faz sinyali olarak ver. Keskin iki decoder yerine decoder'a sürekli bir `p = smoothstep(d_far, d_near, dist)` skaleri ver; model hem yaklaşım hem temas bağlamını aynı ağırlıklarla öğrensin, eşik çevresinde süreksizlik olmaz.
2. Alternatif: model sadece "grasp residual" üretsin, yaklaşımda el gerçek tracker pozisyonunda açık kalsın, sadece temas anında model devreye girsin.
3. Eğitim sırasında yaklaşım framelerini (contact_flag=0) ve temas framelerini (contact_flag=1) ayrı ayrı ağırlıklandır.

---

### A2. CVAE Çeşitlilik Üretemiyor — K Aday Seçimi İşe Yaramıyor

**Ne oluyor, neden yanlış?**

Sistemin tasarımı şu: modelden 5 farklı kavrama alternatifi üret (K=5), bunların arasından en iyisini seç. Fikir güzel — aynı bardağı farklı açılardan tutma önerileri sun, en uygununu al.

Ama gerçekte K=1, K=3, K=5 üretilen pozlar neredeyse aynı. "5 farklı öneri" aslında 5 kopya. Bu yüzden seçim yapmak da anlamsız.

**Neden böyle oluyor?**

CVAE modeli eğitirken "latent space ne kadar çeşitli olsun?" diye bir parametre var (KL weight). Bu parametre çok düşük (0.01) ayarlanmış. Bu, modelin "güvenli" tek bir orta yol pozu öğrenmesine, çeşitlilik üretmeyi görmezden gelmesine yol açıyor.

Üstüne, Unity dışa aktarımı zaten `z=0` (sıfır gürültü) kullanıyor — yani production ortamında her zaman tek deterministik poz üretiliyor. CVAE'nin varlığı gereksiz hale geliyor.

**Çözüm Planı:**

Kısa vadeli (sweep — tek değer deneme değil):
1. `kl_weight ∈ {0.001, 0.01, 0.05, 0.1}` için ayrı çalıştır. KL weight'i doğrudan 0.1'e çıkarmak posterior collapse'ı kötüleştirebilir; decoder latent değişkeni görmezden gelmeye başlayabilir.
2. Her çalıştırmada `mu` ve `logvar` dağılımını izle, her latent boyutun aktif kullanımını ölç.
3. Diversity artarken reconstruction error ve joint-limit violation da raporla. Sadece diversity'yi artırmak kolaydır — rastgele ve kötü pozlar da yüksek diversity üretir.

Orta vadeli:
1. **Oracle-best-of-K karşılaştırması yap:** Deterministik decoder / CVAE mean (`z=0`) / CVAE rastgele K / CVAE oracle-K (ground truth'a en yakın aday) / CVAE quality-selected-K. Oracle bile K=1'i geçemiyorsa sorun seçim mekanizmasında değil, adayların çeşitlilik üretmemesindedir.
2. Eğer sweep sonrası diversity hâlâ yetersizse, CVAE'yi kaldırıp **basit deterministik decoder** koy. Bunu ablation olarak yayınla: "CVAE fazladan karmaşıklık katıyor ama fayda sağlamıyor."

---

### A3. Modelin Her Bileşeni Gerçekten Gerekli mi? Kimse Sınamadı

**Ne oluyor, neden yanlış?**

Bu mimari beş farklı parçadan oluşuyor: PointNet (obje geometrisi), FiLM (geometriyi modele enjekte etme), GRU (zamansallık), Self-Attention (eklemler arası ilişki), CVAE (çeşitlilik). Her biri için "bu olmasa ne olur?" sorusu hiç sorulmadı.

Bir yemek tarifi hayal edin: tarife 5 malzeme koyuyorsunuz ama hangisinin lezzete katkı sağladığını test etmeden "bu 5 malzemenin hepsi gerekli" diyemezsiniz.

**Neden önemli?**

Jüri şunu soracak: "PointNet yerine basit bir kutu (bounding box) kullansaydınız ne olurdu?" Cevap şu an yok. Hatta belki daha basit bir mimari benzer sonuç verebilir — bunu bilmeden mimarinin "özgün katkısı" iddiası kanıtsız kalıyor.

**Çözüm Planı — Öncelik Sırasıyla:**

**1. Temporal Baseline (en önemli, ~1 gün):**

Sadece T=1 vs T=16 karşılaştırmak yeterli değil — T=1 ile GRU hâlâ tek frame'i nonlinear encoder gibi işleyebilir. İki farklı soruyu ayırarak üç karşılaştırma yapılmalı:

- **SingleFrame-MLP:** Son frame → MLP → decoder (GRU tamamen yok)
- **GRU-T1:** Tek frame → GRU → decoder
- **GRU-T16:** 16 frame → GRU → decoder (mevcut model)

Bu şekilde "GRU mimarisi faydalı mı?" ve "geçmiş frame'ler faydalı mı?" soruları ayrışır.

Yalnızca jitter score ile ölçme; temporal başarıyı şu metriklerle birlikte raporla: pose error, velocity error, contact stability. Çünkü aşırı düzleştirilmiş bir el düşük jitter üretebilir ama hareketi takip etmeyebilir.

**2. Geometri Encoding Baseline (~1 gün):**

Üç seviyeli karşılaştırma jüriyi daha ikna eder:
- Pose only (obje bilgisi yok)
- Pose + bbox boyutları (en, boy, derinlik + aspect ratio)
- Pose + PointNet feature (mevcut model)

Sadece "bbox vs PointNet" yerine bu sıralama, her adımın katkısını gösterir.

**3. Deterministik Decoder Baseline (~2 saat):**

Zaten Unity export'un yaptığı şey. Karşılaştırmayı A2'deki oracle-K tablosuyla birleştir.

**4. Self-Attention Ablation (~1 gün):**
- JointSelfAttention yerine per-joint bağımsız MLP koy (eklemler birbirini görmüyor).
- Geodesic error ve joint limit violation karşılaştır.

**5. FiLM Ablation (~1 gün):**
- FiLM yerine `concat(temporal_feat, obj_emb)` → Linear kullan.
- Sonuç farkı varsa FiLM'in katkısı kanıtlanmış, yoksa daha basit mimari tercih edilmeli.

---

## BÖLÜM B — Eğitim ve Veri Sorunları

---

### B1. Model Temas Etmeyi Öğrenmedi — Metrik Oyunu

**Ne oluyor, neden yanlış?**

Hedef: parmak uçlarının objeye değmesi (contact ratio > %70). Gerçek: %13-23.

Ama ilginç olan şu: penetrasyon da düşük (~0.5mm). Yani parmaklar objeye değmiyor ama objenin içine de girmiyor.

Bu bir çelişki değil aslında — tam aksine tutarlı ve daha derin bir problemi gösteriyor: **model objeye hiç yaklaşmıyor.** Uzakta durarak hem "temas etmeme" hem "içine girmeme" cezalarından kaçınıyor. Öğrenilen strateji: "hiçbir şeye dokunma, loss küçük olsun."

Buna istatistikte "Goodhart's Law" deniyor: ölçüt hedefe dönüşünce, ölçütü optimize etmek gerçek hedefi optimize etmekten ayrılabiliyor.

**Neden böyle oluyor?**

Birden fazla eşik aynı sabit üzerinden tanımlanmış ve görevleri karışıyor. Kodda `CONTACT_THRESHOLD_M = 0.030` (30mm) hem quality label üretimi hem de eval metriği için kullanılıyor; eğitim contact loss hinge'i ise ayrıca 15mm'de. Bu üç amacın tek sabite bağlı olması hem eğitim sinyalini hem de metriği yanıltıcı hale getiriyor.

Ayrıca yüzey mesafesi gerçek mesh üzerinden değil, objenin merkezine olan mesafeyle yaklaşık hesaplanıyor (centroid-proxy). Karmaşık şekilli objeler için bu tamamen yanlış bir ölçüm.

**Eşik Tablosu — Önce Bunları Ayır:**

| Amaç | Önerilen Eşik | Açıklama |
|------|--------------|----------|
| Phase label (approach/contact) | ~30mm | Gürültülü temas fazını belirler, büyük tolerans mantıklı |
| Eğitim contact loss | ~5mm | Öğrenme sinyali, sıkı tutulmalı |
| Eval contact metriği | ~2–5mm | Nihai geometrik temas standardı |
| Unity physics collision | collider teması | Gerçek simülasyon olayı |

Bu dört amacı aynı sabite bağlamak döngüsel bağımlılık yaratır.

**Çözüm Planı:**

**1. Eşikleri ayır:**
- `model_io.py`'da `CONTACT_THRESHOLD_M` yalnızca eval metriği için kullan (önerilen: 5mm).
- Eğitim loss'undaki hinge'i ayrı bir sabit olarak tanımla (önerilen: 5mm).
- Phase label üretimi mevcut 30mm toleransında kalabilir.

**2. Contact loss ağırlığını sweep ile bul:**
- `contact_weight ∈ {0.3, 0.7, 1.0, 2.0}` ile aynı seed ve eğitim bütçesinde dört çalıştırma yap.
- Her birinde contact ratio, pose error ve penetration'ı birlikte raporla.
- 0.3'ten doğrudan 2.0'a atlamak reconstruction veya anatomik doğruluğu bozabilir; sweep bu trade-off'u gösterir.

**3. Gerçek yüzey mesafesi kullan:**
- Centroid-proxy yerine point cloud üzerinde differentiable nearest-neighbor distance hesapla.
- Yüzey normalleriyle penetration yönünü doğrula.
- Eğer zaman yeterse `trimesh` ile SDF grid (32³) ekle — ama SDF için non-watertight mesh kontrolü, grid sınırı dışı noktalar ve differentiable interpolasyon gerekliliğini göz önünde bulundur; bu bir günlük iş değil.

---

### B2. OakInk Veri Bölümlemesi Yanlış

**Ne oluyor, neden yanlış?**

OakInk veri seti ~1800 farklı obje içeriyor. Mevcut bölümleme şu şekilde yapılmış: 11,151 kavrama örneğinin %80'i eğitim, %10'u doğrulama, %10'u test — ama rastgele.

Sorun şu: aynı bardak için 10 farklı kavrama örneği var ve bunların bir kısmı eğitimde, bir kısmı testte. Model eğitimde bu bardağı görmüş, testte de aynı bardakla değerlendiriliyor. Bu "gerçek genelleme" testi değil.

Gerçek soru şu olmalı: "Model hiç görmediği bir objeyi kavrayabilir mi?" Bunu test etmek için test setindeki objelerin eğitimde hiç görünmemiş olması gerekiyor.

**Çözüm Planı:**

1. `build_oakink_canonical.py` içinde bölümlemeyi obje bazlı ve kategori-dengeli yap:
   - Objeleri kategorilere göre grupla
   - Her kategoriden %80/%10/%10 oranında obje seç (kategori dağılımını korumak için stratified)
   - Aynı objenin tüm örnekleri aynı sette kalır
   - Sabit seed kullan ve split.json olarak kaydet

2. `dataset.npz`'yi yeniden üret.

3. **İki ayrı test sonucu raporla — eskisini silme:**
   - *Seen-object test* (mevcut sample-level split sonucu): geçmiş sonuçlarla karşılaştırılabilirliği korur
   - *Unseen-object test* (yeni object-level split): gerçek genellemeyi gösterir
   - Bu iki sonucu aynı tabloda "upper bound" ve "generalization" olarak sun

---

### B3. Phase 2 Çok Kısa Eğitildi

**Ne oluyor, neden yanlış?**

Temporal modelin HOT3D üzerinde ince ayarı (Phase 2) yalnızca 14 epoch sürdü. Eğitim eğrileri incelendiğinde model hâlâ düzeliyor — daha fazla epoch ile daha iyi sonuç çıkabilirdi. 14 epoch muhtemelen bir zaman kısıtı nedeniyle durduruldu.

**Çözüm Planı:**

1. Phase 2'yi 14 epoch yerine 50+ epoch çalıştır — ama önce validation eğrilerini kontrol et; sadece train loss'un düşmesi daha uzun eğitimin faydalı olacağını garanti etmez.
2. Erken durdurma için tek metriğe bağlanma. `contact_ratio`'ya göre durdurmak modeli parmakları objeye yapıştırmaya iterebilir — pose error artar, penetrasyon artar. Bunun yerine birincil kriter `val_rec` olsun; contact ratio'yu ayrıca izle ve ikisi birlikte iyileşiyorken checkpoint al.
3. Phase 2 epoch süresi ~5 dakika, 50 epoch = ~4 saat. Makul süre.

---

### B4. Augmentasyon Etkisi Ölçülmedi

**Ne oluyor, neden önemli?**

Augmentasyon implement edilmiş ve eğitimde aktif olarak kullanılıyor: `src/preprocessing/augment.py` içinde yaw rotasyonu (±180°, Z ekseni), point cloud jitter (σ=3mm) ve `rel_vel` dönüşümü mevcut; Phase 1 OakInk eğitimi `augment=True` ile çalışıyor.

Eksik olan şu: **augmentasyonun modele gerçekten katkı sağlayıp sağlamadığı hiç ölçülmemiş.** "Augmentasyon var" demek yeterli değil — jüri "ne kadar faydalı?" diye soracak.

Özellikle HOT3D'nin 33 objeyle sınırlı olduğu düşünüldüğünde augmentasyonun katkısını kanıtlamak kritik.

**Çözüm Planı:**

**1. Augmentasyon ablation'ı (~yarım gün):**
- `augment=False` ile modeli yeniden eğit (sadece Phase 1 veya Phase 2, tam eğitim gerekmez).
- Geodesic error ve contact ratio'yu karşılaştır.
- Sonuç: "Augmentasyon geodesic error'ı X°'den Y°'ye düşürüyor" — katkı kanıtlanmış olur.

**2. HOT3D Phase 2'de augmentasyon kontrolü (~1 saat):**
- `dataset_hot3d.py` içinde `augment` flag'inin Phase 2 eğitiminde de aktif olduğunu doğrula.
- Temporal sekanslar için yaw augmentasyonu tüm frame'lere tutarlı uygulanmalı (zaten `augment_frame_feat` bunu yapıyor, kullanımı doğrula).

**Not:** Canonical frame objeye bağlı değil dünya koordinatlarına göreyse yaw augmentasyonu gerçek çeşitlilik katıyor. Eğer canonical frame zaten obje rotasyonunu kaldırıyorsa augmentasyonun katkısı sınırlı olabilir; ablation bunu da ortaya çıkarır.

Bu ablation sonucu tezde "Veri Artırma" alt başlığı altında raporlanabilir.

---

## BÖLÜM C — Değerlendirme Eksiklikleri

---

### C1. Hiç Görsel veya Video Yok

**Ne oluyor, neden yanlış?**

Tez el hareketi üretiyor. Üretilen hareketlerin neye benzediği hiç gösterilmemiyor — ne resim, ne video, ne Unity demo kaydı. Sadece sayılar var: "geodesic error 9.7°", "contact ratio 0.13" gibi.

Bu sayılar teknik anlamda doğru ama sezgisel olarak hiçbir şey anlatmıyor. Jüri "parmaklar objenin neresinde duruyor, nasıl görünüyor?" sorusunu soracak ve cevap "sayısal olarak şöyle" olmamalı.

**Çözüm Planı:**

**1. Statik görselleştirme (~1 gün):**
- Yalnızca "en iyi ve en kötü 3 örnek" seçmek selection bias yaratır. Bunun yerine: rastgele örnekler + median performans + en iyi + en kötü + en az bir başlıca failure category.
- MANO mesh'i (parmaklar) + obje point cloud'u 3D görselleştir, PNG olarak kaydet.
- Göster: predicted pose (kırmızı) vs. ground truth (yeşil), yanyana.
- Mümkünse contact points, penetrating vertices ve joint-limit ihlallerini renk kodla.

**2. Temporal animasyon (~1 gün):**
- HOT3D test setinden 3-5 sekans seç.
- Frame-by-frame model çıktısını kaydet, GIF veya MP4 üret.
- Aynı kamera açısı ve eşit hız kullan. Sadece başarılı demolar değil en az bir failure case göster — tez daha güvenilir görünür.

**3. Unity demo kaydı (~yarım gün):**
- Demo scene çalışıyor, sadece ekran kaydı al.
- 3 farklı obje (mug, bowl, pot), her biri için kavrama göster.
- 30-60 saniyelik video tezin en ikna edici belgesi olacak.

```python
import open3d as o3d
import matplotlib.pyplot as plt
import imageio  # GIF üretimi
import cv2      # MP4 üretimi
```

---

### C2. SOTA ile Karşılaştırma Yok

**Ne oluyor, neden yanlış?**

Geodesic error 9.7° iyi mi kötü mü? MPJPE 5.7mm nerede duruyor? Bu soruların cevabı için referans nokta gerekiyor.

GrabNet OakInk üzerinde yayınlanmış sonuçlar var. ContactOpt benzer metrikleri raporluyor. Bu sistemin o çalışmalarla nasıl karşılaştırıldığı bilinmiyor.

**Çözüm Planı — İki Ayrı Tablo:**

Dataset split, input modality, MANO representation, metric implementation ve evaluation sample count aynı değilse yayınlanmış sayıları doğrudan karşılaştırmak yanıltıcı olabilir.

**Tablo 1 — Literatür Bağlam:**
- GrabNet, ContactOpt gibi çalışmaların yayınlanmış metriklerini listele.
- Yanına "not directly comparable — farklı split/implementation" notu ekle.
- Bu tablo "biz bu aralıkta duruyoruz" demek içindir.

**Tablo 2 — Reproduced Baseline (asıl karşılaştırma):**
- Aynı split ve aynı eval kodu üzerinde çalıştırabileceğin bir baseline (örn. deterministik decoder veya BBox modeli).
- Bu tablo ablation sonuçlarıyla birleşir.

GrabNet'i aynı pipeline'da çalıştırmak mümkün değilse "referans aralık" olarak sun; "biz daha iyiyiz" sonucu çıkarma.

---

### C3. Gerçek Zamanlı Çalışıp Çalışmadığı Bilinmiyor

**Ne oluyor, neden yanlış?**

Tezin iddialarından biri "gerçek zamanlı XR'da kullanılabilir." Hedef inference süresi <5ms. Ama bu hiç ölçülmemiş.

90 FPS VR için her kare 11ms'de tamamlanmalı. Model inference + Unity render + Air Link gecikme = toplam <30ms hedefi. Modelin ne kadar sürdüğü bilinmeden bu hedefin karşılanıp karşılanmadığı söylenemez. Ayrıca <5ms modelin tek forward geçişine ait olabilir; end-to-end XR latency değildir.

**Çözüm Planı (~2 saat):**

```python
# src/eval/benchmark_latency.py
import torch, time
from statistics import mean
import numpy as np

model.eval()
device = next(model.parameters()).device

# Warm-up (50 tekrar)
for _ in range(50):
    _ = model(frame_feat, obj_pts, prev_pose)

# Gerçek ölçüm (500 tekrar)
times = []
with torch.no_grad():
    for _ in range(500):
        start = time.perf_counter()
        _ = model(frame_feat, obj_pts, prev_pose)
        # MPS için: torch.mps.synchronize() | CUDA için: torch.cuda.synchronize()
        if device.type == "mps":
            torch.mps.synchronize()
        elif device.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - start)

print(f"Median: {np.median(times)*1000:.2f} ms")
print(f"Mean:   {mean(times)*1000:.2f} ms")
print(f"P95:    {np.percentile(times, 95)*1000:.2f} ms")
print(f"P99:    {np.percentile(times, 99)*1000:.2f} ms")
```

Raporda şunları belirt: donanım (MPS / CPU / ONNX), K değeri, input shapes, thread ayarları. Ölçüme nelerin dahil olduğunu açık yaz: point cloud preprocessing, model forward, candidate scoring — sadece forward geçişi değil.

Unity tarafında da `AuraXRModelRuntime.latencyMs` zaten loglanıyor — demo sırasında end-to-end latency'yi de ölç ve tabloya koy.

---

### C4. Tek Seed — Sonuçlar Güvenilir Değil

**Ne oluyor, neden yanlış?**

Tüm sonuçlar tek bir rastgele başlangıç noktasından (random seed) üretilmiş. Farklı başlangıç noktalarında model farklı sonuçlar verebilir. "Geodesic 9.7°" bu tek denemenin sonucu; başka bir seedle 8° ya da 11° çıkabilir.

Akademik standartta en az 3 farklı seedle çalıştırıp ortalamayla standart sapmayı rapor etmek gerekiyor: `9.7 ± 0.3°` gibi.

**Çözüm Planı:**

1. Phase 1 eğitimini seed=42, 123, 456 ile 3 kez çalıştır (~3 × 50 epoch × 11 sn = ~30 dakika).
2. Phase 2'yi de 3 kez çalıştır — 50 epocha uzatılacaksa ~12 saat.
3. Tüm metrik tabloları `mean ± std` formatına dönüştür.
4. HOT3D'de obje sayısı çok az (33) olduğundan sample-level std yanıltıcı olabilir. Mümkünse object-level bootstrap confidence interval da hesapla.

Kaynak kısıtlıysa her ablation'ı 3 seed ile çalıştırmak gerekmez — önce 1 seed ile hiperparametre tara, seçilen en iyi varyantı 3 seed ile tekrarla.

---

## BÖLÜM D — Sistem Tamamlanmamışlıkları

---

### D1. Unity Fizik Değerlendirmesi Hiç Yapılmadı

**Ne oluyor, neden yanlış?**

Tezin iddiası: "gerçek XR ortamında çalışan bir kavrama sistemi." Bunu kanıtlamanın yolu: Unity'de sanal objeye kavramayı uygula, üstüne kuvvet uygula, obje düştü mü düşmedi mi kaydet.

Bu değerlendirme hiç yapılmadı. Yani "sistem gerçekten çalışıyor mu?" sorusu yanıtsız.

**Çözüm Planı:**

Eğer kalan zaman kısıtlıysa alternatif:

**Offline karşılaştırma (hızlı, 1-2 gün):**
- Unity fizik eval yerine OakInk üzerinde GrabNet ile sayısal karşılaştırma yap (C2 tablosuyla birleşir).
- "Fizik simülasyonu yapamadık ama offline metrikte mevcut SOTA ile karşılaştırıldığımızda X durumundayız" dürüst bir yaklaşım.

**Minimal Unity eval (1 hafta):**
1. Unity'de 3 obje için (mug, bowl, pot) model kavrama pozu uygula.
2. Parmaklar objeye oturdu mu görsel olarak kaydet.
3. Basit kuvvet testi: objeyi yukarıdan it, kavrama tutuyor mu?
4. 3 obje × 10 deneme = 30 veri noktası. Az ama hiç yoktan iyi.

---

### D2. success_prob Head'i İşlevsiz

**Ne oluyor, neden yanlış?**

Sistem tasarımında K=5 aday üretilecek, "success_prob" başlığı bunlardan hangisinin başarılı kavrama yapacağını tahmin edecek, en yüksek olasılıklı seçilecekti.

Ama success_prob eğitilmedi çünkü Unity'den etiket gelmedi. Eğitilmemiş bir baş rastgele sayı üretiyor. K=5 ile K=1 arasında fark olmamasının sebebi bu: seçim mekanizması çalışmıyor.

**Çözüm Planı:**

**Seçenek 1 — Heuristic ile değiştir (hızlı):**
- success_prob yerine quality_score kullan (OakInk'te Spearman 0.72 var).
- K=5 aday üret, quality_score en yüksek olanı seç.
- Önce random selection ve first candidate ile karşılaştır; quality selection random'dan kötü çıkarsa kullanma.

**Seçenek 2 — Offline geometrik etiket (dürüst isimle):**
- OakInk üzerinde contact ratio > 0.5 olan örnekleri "geometrik olarak geçerli", olmayanları "geçersiz" olarak etiketle.
- success_head'i bu etiketle fine-tune et — ama bunu `geometric_validity_prob` olarak adlandır, `success_prob` değil.
- Gerçek fiziksel başarı (friction, force closure, mass, collider stability) bu etiketle ölçülemiyor; yanıltıcı isim kullanmak jüride soru yaratır.

**Seçenek 3 — Head'i kaldır (en dürüst):**
- Unity etiketi olmadan success_head'i eğitmek mümkün değil. Head'i tezden çıkar, "gelecek çalışma: Unity fizik eval ile success etiketi üretimi" olarak sun.
- Zayıf proxy ile "çalışıyor" göstermeye çalışmaktan daha bilimsel.

---

## Özet — Ne Yapılmalı, Hangi Sırayla?

Kalan zamanı verimli kullanmak için öncelik sırası:

| # | Görev | Süre | Etki |
|---|-------|------|------|
| 1 | OakInk object-level + category-stratified split + seen/unseen ikili test | 1 gün | Metodolojik güvenilirlik |
| 2 | Temporal baseline (SingleFrame-MLP / GRU-T1 / GRU-T16) | 1 gün | Temporal katkısını kanıtlar |
| 3 | Geometri baseline (pose-only / BBox / PointNet) | 1 gün | Geometri katkısını kanıtlar |
| 4 | Deterministik decoder + oracle-K karşılaştırması (A2 ile birleşir) | 2 saat | CVAE gerekliliğini belirler |
| 5 | Contact eşiklerini ayır + contact_weight sweep | 1 gün | Kavramsal tutarlılık + metrik iyileşmesi |
| 6 | Nearest-surface contact loss (centroid-proxy yerine) | 1 gün | En kritik performans sorunu |
| 7 | Ana varyantları 3 seed ile çalıştır | 1 gün | İstatistiksel güvenilirlik |
| 8 | Latency benchmark (MPS + ONNX + Unity end-to-end) | 2 saat | "Gerçek zamanlı" iddiasını doğrular |
| 9 | Görselleştirme: rastgele + median + failure case | 1 gün | Savunmada ikna edici kanıt |
| 10 | Demo videosu kaydet | 2 saat | En ikna edici materyal |
| 11 | Phase 2 uzun eğit (50 epoch, erken durdurma val_rec birincil) | 4 saat | Temporal kalite iyileşmesi |
| 12 | Augmentasyon ablation | yarım gün | Augmentasyon katkısını kanıtlar |
| 13 | SOTA literatür bağlam tablosu | yarım gün | Bağlam sağlar |

**İlk 6 görev tezin bilimsel zeminini ve metodolojik güvenilirliğini kuruyor. 7-10 sonuçları güçlendiriyor ve sunumu hazırlıyor. 11-13 iyileştirme ve bağlam.**
