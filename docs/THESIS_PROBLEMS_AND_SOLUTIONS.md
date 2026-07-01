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

Kısa vadeli (hızlı düzeltme):
1. Unity tarafında blend başlangıç mesafesini 10cm'den 4cm'e çek. Modelin etkisi sadece çok yakında başlasın, kullanıcı farkı daha az hisseder.
2. Blend formülünü güncelle: mevcut `smoothstep(10cm, 2cm)` yerine `smoothstep(4cm, 1cm)` kullan. Kullanıcı objeye neredeyse değdiğinde model devreye girsin.

Orta vadeli (doğru mimari düzeltme):
1. Modele `dist` (bilek-obje mesafesi) değerini explicit bir "faz sinyali" olarak ver. Bu zaten `frame_feat` içinde var ama decoder onu yeterince kullanmıyor.
2. Decoder'ı ikiye böl: `dist > 5cm` iken "açık el pozu" çıkışı, `dist <= 5cm` iken "grasp pozu" çıkışı. Aralarında yumuşak geçiş (lerp).
3. Eğitim sırasında yaklaşım framelerini (contact_flag=0) ve temas framelerini (contact_flag=1) ayrı ayrı ağırlıklandır; modelin faz ayrımını görmesini sağla.

---

### A2. CVAE Çeşitlilik Üretemiyor — K Aday Seçimi İşe Yaramıyor

**Ne oluyor, neden yanlış?**

Sistemin tasarımı şu: modelden 5 farklı kavrama alternatifi üret (K=5), bunların arasından en iyisini seç. Fikir güzel — aynı bardağı farklı açılardan tutma önerileri sun, en uygununu al.

Ama gerçekte K=1, K=3, K=5 üretilen pozlar neredeyse aynı. "5 farklı öneri" aslında 5 kopya. Bu yüzden seçim yapmak da anlamsız.

**Neden böyle oluyor?**

CVAE modeli eğitirken "latent space ne kadar çeşitli olsun?" diye bir parametre var (KL weight). Bu parametre çok düşük (0.01) ayarlanmış. Bu, modelin "güvenli" tek bir orta yol pozu öğrenmesine, çeşitlilik üretmeyi görmezden gelmesine yol açıyor.

Üstüne, Unity dışa aktarımı zaten `z=0` (sıfır gürültü) kullanıyor — yani production ortamında her zaman tek deterministik poz üretiliyor. CVAE'nin varlığı gereksiz hale geliyor.

**Çözüm Planı:**

Kısa vadeli:
1. `kl_weight`'i 0.01'den 0.1'e çıkar. Bu tek değişiklik diversity'yi artırabilir.
2. Diversity score'u training sırasında loglayarak takip et. Hedef: K=5 için diversity > 0.3 (şu an 0.05).

Orta vadeli:
1. Eğer KL weight artışı diversity'yi yeterince artırmazsa, CVAE'yi kaldırıp yerine **basit deterministik decoder** koy. Bunu ablation olarak da yayınla: "CVAE fazladan karmaşıklık katıyor ama fayda sağlamıyor."
2. Alternatif: CVAE yerine farklı obje görüş açılarına karşılık gelen birkaç sabit latent vektör öğren (mixture of experts tarzı).

---

### A3. Modelin Her Bileşeni Gerçekten Gerekli mi? Kimse Sınamadı

**Ne oluyor, neden yanlış?**

Bu mimari beş farklı parçadan oluşuyor: PointNet (obje geometrisi), FiLM (geometriyi modele enjekte etme), GRU (zamansallık), Self-Attention (eklemler arası ilişki), CVAE (çeşitlilik). Her biri için "bu olmasa ne olur?" sorusu hiç sorulmadı.

Bir yemek tarifi hayal edin: tarife 5 malzeme koyuyorsunuz ama hangisinin lezzete katkı sağladığını test etmeden "bu 5 malzemenin hepsi gerekli" diyemezsiniz.

**Neden önemli?**

Jüri şunu soracak: "PointNet yerine basit bir kutu (bounding box) kullansaydınız ne olurdu?" Cevap şu an yok. Hatta belki daha basit bir mimari benzer sonuç verebilir — bunu bilmeden mimarinin "özgün katkısı" iddiası kanıtsız kalıyor.

**Çözüm Planı — Öncelik Sırasıyla:**

**1. SingleFrame Baseline (en önemli, ~1 gün):**
- Mevcut modeli al, sadece `T=1` ile çalıştır (GRU'yu devre dışı bırak, son frame'i kullan).
- HOT3D test üzerinde aynı metrikleri hesapla.
- Sonuç: "GRU ile T=16 kullanmak, T=1'e göre jitter score'u X'ten Y'ye düşürüyor" — bu cümle tezin temporal katkısını kanıtlar.

**2. MLP-BBox Baseline (~1 gün):**
- PointNet encoder yerine objenin bounding box boyutları (3 sayı: en, boy, derinlik) veren basit MLP koy.
- OakInk test üzerinde geodesic error karşılaştır.
- Sonuç: "PointNet, BBox'a göre geodesic error'ı X° düşürüyor" — geometri encoding'in katkısı kanıtlanmış olur.

**3. Deterministik Decoder Baseline (~2 saat):**
- Mevcut modeli al, CVAE encoder'ı devre dışı bırak, `z=zeros` sabit tut.
- Metrikleri karşılaştır. Büyük ihtimalle fark çok az çıkacak — bu zaten Unity export'un yaptığı şey.

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

Contact loss şu an şöyle çalışıyor: parmak ucu objenin yüzeyinden 15mm'den uzaksa ceza ver, yakınsa ceza verme. Ama 15mm büyük bir tolerans — model parmağı 14mm uzakta tutarak cezadan kaçınabiliyor ve bu "temas sayılmıyor."

Ayrıca yüzey mesafesi gerçek mesh üzerinden hesaplanmıyor, objenin merkezine olan mesafeyle yaklaşık hesaplanıyor (centroid-proxy). Karmaşık şekilli objeler için bu tamamen yanlış bir ölçüm.

**Çözüm Planı:**

**1. Contact loss threshold'unu küçült (hızlı düzeltme):**
- 15mm → 5mm. Parmak ucu 5mm içine girdiğinde ödüllendir.
- Bunu yapmak için: `model_io.py`'da `CONTACT_THRESHOLD_M = 0.005` (zaten bu değer vardı, sonra 30mm'ye çıkarıldı, 5mm'ye geri dön ama FK hatasını düzeltilmiş halleriyle).

**2. Contact loss ağırlığını artır:**
- Mevcut: `contact_weight = 0.3`
- Yeni: `contact_weight = 2.0` — contact loss dominant hale gelsin, model bunu görmezden gelemesin.

**3. Gerçek mesh SDF kullan (temel düzeltme):**
- Centroid-proxy yerine her obje için önceden hesaplanmış SDF grid (signed distance field) yükle.
- Her parmak ucu için objenin gerçek yüzeyine olan mesafeyi hesapla.
- Bu değişiklik hem contact_loss'u hem penetration_loss'u anlamlı hale getirir.
- Nasıl: `trimesh` kütüphanesiyle her obje mesh'i için 32³ veya 64³ SDF grid önceden hesapla, eğitimde lookup yap.

**4. Loss ile metriği ayır:**
- Contact metriği loss'tan farklı bir eşikle ölçülmeli. Loss'ta 5mm threshold kullan (öğrenme sinyali), metrikte 2mm threshold kullan (gerçek temas standardı). Döngüsel bağımlılığı kır.

---

### B2. OakInk Veri Bölümlemesi Yanlış

**Ne oluyor, neden yanlış?**

OakInk veri seti ~1800 farklı obje içeriyor. Mevcut bölümleme şu şekilde yapılmış: 11,151 kavrama örneğinin %80'i eğitim, %10'u doğrulama, %10'u test — ama rastgele.

Sorun şu: aynı bardak için 10 farklı kavrama örneği var ve bunların bir kısmı eğitimde, bir kısmı testte. Model eğitimde bu bardağı görmüş, testte de aynı bardakla değerlendiriliyor. Bu "gerçek genelleme" testi değil.

Gerçek soru şu olmalı: "Model hiç görmediği bir objeyi kavrayabilir mi?" Bunu test etmek için test setindeki objelerin eğitimde hiç görünmemiş olması gerekiyor.

**Çözüm Planı:**

1. `build_oakink_canonical.py` içinde bölümlemeyi obje bazlı yap:
   - Tüm objeleri listele (~1800 obje)
   - %80 objeyi (1440 obje ve bunların tüm örnekleri) eğitime al
   - %10 objeyi (180 obje) doğrulamaya, %10'unu (180 obje) teste al
   - Aynı objenin örnekleri tek bir sette kalır

2. `dataset.npz`'yi yeniden üret.

3. Modeli yeniden eğit ve metrikleri yeniden hesapla. Büyük ihtimalle test performansı düşecek — bu normal ve dürüst.

---

### B3. Phase 2 Çok Kısa Eğitildi

**Ne oluyor, neden yanlış?**

Temporal modelin HOT3D üzerinde ince ayarı (Phase 2) yalnızca 14 epoch sürdü. Eğitim eğrileri incelendiğinde model hâlâ düzeliyor — daha fazla epoch ile daha iyi sonuç çıkabilirdi. 14 epoch muhtemelen bir zaman kısıtı nedeniyle durduruldu.

**Çözüm Planı:**

1. Phase 2'yi 14 epoch yerine 50+ epoch çalıştır.
2. Erken durdurma kriterini `val_rec` yerine `contact_ratio` üzerinden belirle — gerçek hedefin iyileştiği yerde dur.
3. Phase 2 epoch süresi ~5 dakika, 50 epoch = ~4 saat. Makul süre.

---

### B4. Augmentasyon Hiç Uygulanmadı

**Ne oluyor, neden yanlış?**

Veri çeşitlendirme (augmentasyon) makine öğrenmesinde modelin farklı koşullara genellemesini sağlayan temel bir teknik. Tezde 7 farklı augmentasyon yöntemi planlanmış ama hiçbiri implement edilmemiş.

Özellikle HOT3D'nin 33 objeyle sınırlı olduğu düşünüldüğünde augmentasyon kritik — az sayıda objeyi farklı açılardan, farklı hızlarda göstererek modelin daha iyi genellemesi sağlanabilir.

**Çözüm Planı — Sırayla Uygula:**

**1. Yaw augmentasyon (en kolay, en faydalı, ~1 gün):**
- Her HOT3D sekansını Z ekseni etrafında rastgele ±45° döndür.
- Bu, modelin "bardağa her yönden yaklaşmayı" öğrenmesini sağlar.
- Nasıl: `rel_pos` ve `rel_rot6d`'yi yaw rotation matrix ile döndür, `rel_vel`'i de dönüştür.

**2. Bilek hız gürültüsü (~2 saat):**
- `rel_vel`'e σ=0.02 m/s Gaussian gürültü ekle.
- Modelin hız sinyalindeki küçük bozulmalara karşı dayanıklı hale gelmesini sağlar.

**3. Obje konumu jitter (~2 saat):**
- `rel_pos`'a ±5mm Gaussian gürültü ekle.
- Gerçek kullanımda obje kesin konumu bilinmeyeceği için.

Bu üç augmentasyon birbiriyle birleştirilerek eğitim setini ~3-4 kat çeşitlendirilebilir.

---

## BÖLÜM C — Değerlendirme Eksiklikleri

---

### C1. Hiç Görsel veya Video Yok

**Ne oluyor, neden yanlış?**

Tez el hareketi üretiyor. Üretilen hareketlerin neye benzediği hiç gösterilmemiyor — ne resim, ne video, ne Unity demo kaydı. Sadece sayılar var: "geodesic error 9.7°", "contact ratio 0.13" gibi.

Bu sayılar teknik anlamda doğru ama sezgisel olarak hiçbir şey anlatmıyor. Jüri "parmaklar objenin neresinde duruyor, nasıl görünüyor?" sorusunu soracak ve cevap "sayısal olarak şöyle" olmamalı.

**Çözüm Planı:**

**1. Statik görselleştirme (~1 gün):**
- Her test objesinden en iyi ve en kötü 3 kavrama örneğini seç.
- MANO mesh'i (parmaklar) + obje point cloud'u 3D görselleştir, PNG olarak kaydet.
- Matplotlib veya Open3D kullan.
- Göster: predicted pose (kırmızı) vs. ground truth (yeşil), yanyana.

**2. Temporal animasyon (~1 gün):**
- HOT3D test setinden 3-5 sekans seç.
- Frame-by-frame model çıktısını kaydet, GIF veya MP4 üret.
- Göster: el objeye yaklaşırken parmaklar nasıl hareket ediyor.

**3. Unity demo kaydı (~yarım gün):**
- Demo scene çalışıyor, sadece ekran kaydı al.
- 3 farklı obje (mug, bowl, pot), her biri için kavrama göster.
- 30-60 saniyelik video tezin en ikna edici belgesi olacak.

**Kullanılacak araçlar:**
```python
# Statik görselleştirme için
import open3d as o3d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Video için
import imageio  # GIF üretimi
import cv2      # MP4 üretimi
```

---

### C2. SOTA ile Karşılaştırma Yok

**Ne oluyor, neden yanlış?**

Geodesic error 9.7° iyi mi kötü mü? MPJPE 5.7mm nerede duruyor? Bu soruların cevabı için referans nokta gerekiyor.

GrabNet OakInk üzerinde yayınlanmış sonuçlar var. ContactOpt benzer metrikleri raporluyor. Bu sistemin o çalışmalarla nasıl karşılaştırıldığı bilinmiyor.

**Çözüm Planı:**

1. GrabNet'in orijinal paper'ındaki OakInk metriklerini bul (zaten yayınlanmış).
2. Kendi sonuçlarını aynı tablo formatında yan yana koy.
3. Eğer sistem daha kötü çıkarsa dürüstçe "neden daha kötü ve biz ne ekliyoruz (temporal)" tartış.
4. Eğer daha iyi çıkarsa güçlü bir katkı kanıtı elde edilmiş olur.

Bu karşılaştırmayı yapmak için yeni deney gerekmez — sadece literatür araştırması ve tablo oluşturma.

---

### C3. Gerçek Zamanlı Çalışıp Çalışmadığı Bilinmiyor

**Ne oluyor, neden yanlış?**

Tezin iddialarından biri "gerçek zamanlı XR'da kullanılabilir." Hedef inference süresi <5ms. Ama bu hiç ölçülmemiş.

90 FPS VR için her kare 11ms'de tamamlanmalı. Model inference + Unity render + Air Link gecikme = toplam <30ms hedefi. Modelin ne kadar sürdüğü bilinmeden bu hedefin karşılanıp karşılanmadığı söylenemez.

**Çözüm Planı (~2 saat):**

```python
# src/eval/benchmark_latency.py

import torch, time
model.eval()

# GPU warm-up
for _ in range(10):
    _ = model(frame_feat, obj_pts, prev_pose)

# Gerçek ölçüm
times = []
with torch.no_grad():
    for _ in range(100):
        start = time.perf_counter()
        _ = model(frame_feat, obj_pts, prev_pose)
        torch.cuda.synchronize()  # GPU işinin bitmesini bekle
        times.append(time.perf_counter() - start)

print(f"Ortalama: {mean(times)*1000:.2f} ms")
print(f"P95: {percentile(times, 95)*1000:.2f} ms")
print(f"P99: {percentile(times, 99)*1000:.2f} ms")
```

K=1, K=3, K=5 için ayrı ayrı ölç ve tabloya koy. Unity tarafında da `AuraXRModelRuntime.latencyMs` zaten loglanıyor — demo sırasında gerçek latency'yi ölç.

---

### C4. Tek Seed — Sonuçlar Güvenilir Değil

**Ne oluyor, neden yanlış?**

Tüm sonuçlar tek bir rastgele başlangıç noktasından (random seed) üretilmiş. Farklı başlangıç noktalarında model farklı sonuçlar verebilir. "Geodesic 9.7°" bu tek denemenin sonucu; başka bir seedle 8° ya da 11° çıkabilir.

Akademik standartta en az 3 farklı seedle çalıştırıp ortalamayla standart sapmayı rapor etmek gerekiyor: `9.7 ± 0.3°` gibi.

**Çözüm Planı:**

1. Phase 1 eğitimini seed=42, 123, 456 ile 3 kez çalıştır (~3 × 50 epoch × 11 sn = ~30 dakika).
2. Phase 2'yi de 3 kez çalıştır (~3 × 14 epoch × 5 dk = ~3.5 saat, ama bu 50 epocha uzatılacaksa ~12 saat).
3. Tüm metrik tabloları `mean ± std` formatına dönüştür.

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
- Unity fizik eval yerine OakInk üzerinde GrabNet ile sayısal karşılaştırma yap.
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
- HOT3D'de zayıf ama sıfırdan iyi.

**Seçenek 2 — success_prob'u offline etiketle:**
- Unity fizik eval yapmak yerine, OakInk üzerinde contact ratio > 0.5 olan örnekleri "başarılı", olmayanları "başarısız" olarak etiketle.
- success_head'i bu etiketle fine-tune et.
- Sınırlı ama çalışan bir aday seçimi elde edilmiş olur.

---

## Özet — Ne Yapılmalı, Hangi Sırayla?

Kalan zamanı verimli kullanmak için öncelik sırası:

| # | Görev | Süre | Etki |
|---|-------|------|------|
| 1 | SingleFrame baseline çalıştır | 1 gün | Temporal katkısını kanıtlar |
| 2 | MLP-BBox baseline çalıştır | 1 gün | Geometri katkısını kanıtlar |
| 3 | OakInk object-level split düzelt + yeniden eğit | 1 gün | Metodolojik hatayı düzeltir |
| 4 | Contact loss redesign (threshold küçült + ağırlık artır) | 1 gün | En kritik metrik iyileşmesi |
| 5 | Yaw augmentasyon ekle | 1 gün | Genelleme iyileşmesi |
| 6 | Statik görselleştirme üret | 1 gün | Jüriye kanıt |
| 7 | Latency benchmark çalıştır | 2 saat | "Gerçek zamanlı" iddiasını doğrular |
| 8 | Phase 2 uzun eğit (50 epoch) | 4 saat | Temporal kalite iyileşmesi |
| 9 | 3 seed ile yeniden çalıştır | 1 gün | İstatistiksel güvenilirlik |
| 10 | Unity blend mesafesini ayarla (10→4cm) | 1 saat | Approach-to-grasp görsel iyileşme |
| 11 | Demo videosu kaydet | 2 saat | En ikna edici materyal |
| 12 | SOTA karşılaştırma tablosu | yarım gün | Bağlam sağlar |

**İlk 4 görev tezin bilimsel zeminini kuruyor. 5-8 performansı artırıyor. 9-12 sunumu güçlendiriyor.**
