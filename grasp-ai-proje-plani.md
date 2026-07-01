# Unity XR Hands Tabanlı Yapay Zekâ Destekli Grasp Sistemi — Proje Planı

## 1. Amaç

VR controller ile kontrol edilen sanal insan elinin, objelere doğal bir şekilde yaklaşmasını ve objenin geometrisine uygun, stabil bir biçimde kavramasını sağlayan gerçek zamanlı bir yapay zekâ tabanlı grasp sistemi geliştirmek.

Kullanıcının controller hareketi elin bilek pozisyonu ve temel yönelimini belirler. Yapay zekâ kullanıcının kontrolünü devralmaz; yalnızca objeye yaklaşıldığında parmakların uygun kavrama pozisyonuna geçmesini ve gerekirse küçük yön düzeltmelerini sağlar.

**Çalışma ortamı:** Sistem standalone Quest üzerinde değil, **bilgisayar (PC) üzerinde çalıştırılacak**; Quest sadece display/controller olarak Air Link veya Link Cable üzerinden bağlanıp test edilecektir. Bu nedenle model boyutu/karmaşıklığı mobil çip kısıtlarına göre değil, **PC GPU'da elde edilebilecek en iyi sonuca göre** belirlenir — compute, kısıtlayıcı faktör değildir.

---

## 2. Sistem Mimarisi

Sistemde **toplam 2 öğrenen yapay zekâ modeli** ve **2 destekleyici (kural tabanlı / deterministik) modül** bulunur.

### 2.1. Öğrenen Modeller

| | Approach Model | Grasp Model |
|---|---|---|
| **Devreye girdiği faz** | Elin objeye yaklaşma süreci | Yaklaşım tamamlandıktan, temas başladıktan sonra |
| **Temel görev** | Yaklaşım yönü ve bilek yönelimini optimize etmek | Objenin geometrisine uygun parmak pozisyonlarını üretmek |

### 2.2. Destekleyici Modüller (model değil)

- **Joint Mapping / Retargeting** — MANO/UmeTrack eklem temsilini Unity XR Hands `XRHandJointID` yapısına eşleyen deterministik dönüşüm katmanı.
- **Approach/Grasp Segmentasyonu** — Sürekli hareket sekanslarını approach ve grasp fazlarına ayıran kural tabanlı ön işleme adımı (eğitilen bir model değil).

> **Not:** Kavrama güven skoru ayrı bir model değildir — Grasp Model'in **ek bir output head'i** olarak üretilir (multi-task öğrenme: ortak gövde + iki çıkış katmanı: eklem rotasyonları + güven skoru).

---

## 3. Model Girdi/Çıktıları

### 3.1. Approach Model

**Girdiler:**
- Controller'dan bilek pozisyonu ve rotasyonu
- Obje konumu ve rotasyonu
- Obje boyutu
- Obje 3B geometrisi (mesh / mesh türevi özellikler)
- El–obje mesafesi

**Çıktılar:**
- Yaklaşım yönü
- Hedef bilek yönelimi (gerekirse küçük düzeltmelerle)
- Pre-grasp pozisyonu

### 3.2. Grasp Model

**Girdiler:**
- Bilek pozisyonu/rotasyonu (yaklaşım tamamlandığındaki)
- Obje konumu ve rotasyonu
- Obje boyutu
- Obje 3B geometrisi
- El–obje mesafesi (yakın/temas eşiğinde)

**Çıktılar:**
- XR Hands iskeletindeki parmak eklem rotasyonları (3x15)
- Kavrama güven skoru (ek output head)
- Gerekirse küçük bilek yön düzeltmeleri

> İki model **benzer obje bilgisini girdi olarak paylaşır**, ancak çıktıları ve devrede olduğu faz farklı olduğu için ayrı ağlar olarak tasarlanmıştır.

---

## 4. Veri Setleri ve Roller

### 4.1. Genel Roller

| Veri Seti | Rol | Not |
|---|---|---|
| **HOT3D** | Hem approach hem grasp için kaynak | Egocentric (Aria/Quest 3), controller verisi değil; thumb DOF hatası bilinen bir sınırlama; sadece 33 obje |
| **OakInk** | Grasp Model'in ana eğitim kaynağı | MANO formatı, 1.800 obje, 50K grasp pozu — obje çeşitliliği yüksek |
| **Unity Sentetik Veri** | Controller davranışı + XR Hands fine-tuning + güven skoru etiketleme | Domain gap'i kapatan ana aşama |

### 4.2. Model Bazında Veri Dağılımı

**Approach Model:**
- **Ana kaynak:** HOT3D — segmentasyon sonrası ilk temastan önceki frame'ler
- **Destekleyici:** OakInk-Image (varsa, sınırlı hareket verisi)
- **Fine-tuning:** Unity sentetik veri — controller'a özel trajectory'ler, farklı yaklaşım açıları

**Grasp Model:**
- **Ana kaynak:** OakInk — 1.800 obje, MANO grasp pozları → 3x15 formatına dönüştürülecek
- **Destekleyici:** HOT3D — segmentasyon sonrası grasp frame'leri (gerçek, geçişli kavrama örnekleri; OakInk'in statik/transfer edilmiş pozlarında olmayan bir gerçekçilik katkısı sağlar)
- **Fine-tuning + etiketleme:** Unity sentetik veri — fizik simülasyonuyla başarılı/başarısız grasp etiketleri üretimi

### 4.3. Önemli Düzeltme: HOT3D'de Grasp Var

HOT3D sekansları sadece yaklaşımı değil, **gerçek obje manipülasyonunun tamamını** (pick-up/observe/put-down, mutfak/ofis/oturma odası senaryoları) içerir. Yani HOT3D'nin grasp kısmı atılmaz, Grasp Model'e de destekleyici kaynak olarak girer. HOT3D'nin grasp katkısının "ana kaynak değil, destekleyici" olmasının nedeni:
- Obje çeşitliliği düşük (33 obje, OakInk'te 1.800)
- Başparmak DOF hatası grasp pozlarının hassasiyetini düşürüyor

---

## 5. Kritik Ek Milestone'lar

### Milestone 1 — MANO/UmeTrack → XR Hands Joint Mapping
HOT3D ve OakInk'in MANO/UmeTrack eklem temsili, Unity XR Hands'in `XRHandJointID` yapısına (26 joint/el) dönüştürülecek retargeting katmanı kurulacak.
- Başparmak eklemleri için ayrıca hata payı değerlendirilecek (HOT3D'nin bilinen thumb DOF sorunu nedeniyle).
- Sayısal doğrulamanın yanı sıra, dönüştürülen sekanslar Unity'de **görsel olarak** da kontrol edilecek (özellikle başparmak ve eklem limitleri için distorsiyon kontrolü).

### Milestone 2 — Approach/Grasp Segmentasyon Pipeline'ı
HOT3D'nin sürekli sekanslarında "yaklaşım bitti, grasp başladı" ayrımı şu kriterlerle belirlenecek:
- İlk temas anı
- Bilek-obje mesafesi
- Parmak kapanma miktarı
- Temas noktalarının oluşumu

Kriterlerin tam frame'de çakışmama olasılığına karşı, sert bir sınır yerine küçük bir **geçiş penceresi (transition window)** tanımlanacak; bu pencerede iki modelin çıktısı yumuşak geçişle (blend) birleştirilecek.

### Milestone 3 — Grasp Güven Skoru Üretimi
Veri setlerinde doğrudan bulunmayan güven skoru, şu metriklerden sonradan üretilecek:
- Temas noktaları
- Parmak-obje mesafesi
- Penetrasyon miktarı
- Temas sayısı
- Unity fizik simülasyonunda stabilite (objenin tutulup tutulmaması)

Bu metriklerin (özellikle birbiriyle ters korelasyonlu olabilen penetrasyon/stabilite gibi) tek bir skalere indirgenme yöntemi (ağırlıklı toplam veya küçük bir sınıflandırıcı) ayrıca tanımlanacak.

---

## 6. Eğitim Sıralaması

1. **OakInk ile Grasp Model ön eğitimi** — format ve obje çeşitliliği nedeniyle önce yapılır.
2. **HOT3D ile approach/grasp segmentlerinin çıkarılması** — Approach Model eğitimi + Grasp Model'e destekleyici veri eklenmesi.
3. **Unity sentetik veri ile her iki modelin XR Hands iskeletine fine-tune edilmesi** — controller tabanlı VR senaryosundaki domain gap'in kapatıldığı asıl aşama; aynı zamanda güven skoru etiketlemesi burada yapılır.

---

## 7. Çalışma Prensibi (Runtime Akışı)

1. Kullanıcı controller ile elini serbestçe hareket ettirir.
2. Sistem, kullanıcı objeye yeterince yaklaştığında yapay zekâyı devreye alır.
3. **Approach Model** devreye girer: yaklaşım yönünü ve bilek yönelimini hassaslaştırır, pre-grasp pozisyonu üretir.
4. Geçiş penceresinde (transition window) Approach Model çıktısından Grasp Model çıktısına yumuşak geçiş yapılır.
5. **Grasp Model** devreye girer: objenin geometrisine uygun parmak eklem rotasyonlarını (3x15) ve güven skorunu üretir.
6. Kullanıcı elin genel hareketini kontrol etmeye devam ederken, kavrama otomatik ve gerçekçi şekilde gerçekleşir.

---

## 9. Model Mimarisi (Network Tasarımı)

PC üzerinde GPU ile çalışılacağı için latency bütçesi rahat; model boyutu konusunda çekinilmeyecek, hedef en iyi sonucu yakalamak.

### 9.1. Approach Model

- **Mimari:** Küçük-orta ölçekli **Transformer encoder** (4-6 katman, 256-512 hidden dim). Son N frame'lik bilek trajectory penceresi (pozisyon, rotasyon, hız/ivme) input olarak alınır, pozisyonel encoding ile zaman bilgisi korunur.
- **Alternatif/karşılaştırma:** **GRU/LSTM** (1-2 katman) ile paralel eğitilip karşılaştırılacak — kısa-vadeli bağımlılıklarda LSTM daha az veriyle Transformer'a yakın/iyi sonuç verebilir.
- **Çıktı:** Yaklaşım yönü, hedef bilek yönelimi, pre-grasp pozisyonu.

### 9.2. Grasp Model

- **Obje encoder:** Basit shape embedding yerine **PointNet/PointNet++ veya küçük bir Point Transformer** ile obje mesh/point cloud'undan zengin geometri embedding'i çıkarılır (ince geometrik farkları — örn. bardak kulpu — ayırt etmek için).
- **Parmak-obje etkileşimi:** Parmaklar/eklemler **query**, obje yüzeyinden örneklenen noktalar **key-value** olarak verilen bir **cross-attention / Transformer decoder** bloğu (ContactOpt/GraspTTA tarzı) — her parmak objenin kendine uygun bölgesine attention ile odaklanır.
- **Obje şekli conditioning:** **FiLM (feature-wise linear modulation)** ile obje embedding'i ana gövdeye enjekte edilir.
- **Çok-modluluk:** **CVAE tabanlı iki aşamalı yapı** (GrabNet mantığı): CoarseNet → kaba grasp önerisi, RefineNet → temas/penetrasyon iyileştirmesi. Aynı obje için birden fazla geçerli grasp stratejisi (üstten/yandan) öğrenilmesini sağlar.
- **İleri seviye opsiyon:** Diffusion-based grasp generation (iteratif denoising, 10-20 step) — daha yüksek kalite/çeşitlilik, PC'de gerçek zamanlı denenebilir.
- **Çıktı başlıkları (multi-task):** 3x15 eklem açısı + güven skoru (sigmoid head).
- **Jitter önlemi:** Önceki frame'in parmak açıları opsiyonel ek girdi olarak verilir.

### 9.3. Önemli Kısıt: Veri, Compute Değil

Compute kısıtlayıcı değil ama **veri hacmi/çeşitliliği** hâlâ kısıtlayıcı. Büyük bir Transformer/CVAE, OakInk (50K grasp) + HOT3D (33 obje) ile overfit olabilir. Bu nedenle:
- Model kapasitesi büyütülürken **Unity sentetik veri üretimi de ölçeklenmeli** (daha fazla obje/yön kombinasyonu).
- Eval'de mutlaka **görülmemiş obje** test seti tutulmalı (sadece görülmemiş poz değil).

---

## 10. Eval Adımları

### 10.1. Offline Metrikler

**Approach Model:**
- Bilek yönelimi/pre-grasp pozisyonu için MAE/RMSE (held-out test split)
- **Trajectory smoothness** — ardışık frame'ler arası değişim varyansı (jitter ölçümü)
- Görülmemiş obje boyutu/şekli üzerinde genelleme testi

**Grasp Model:**
- Açı MAE/RMSE (referans, tek başına yeterli değil)
- **Contact ratio** — forward kinematics ile hesaplanan parmak segment pozisyonlarının obje mesh'ine mesafesi
- **Penetration depth** — parmakların obje içine girme miktarı
- **Confidence score kalibrasyonu** — tahmin edilen güven skoru ile gerçek (fizik simülasyonu) başarı/başarısızlık arasında AUC/kalibrasyon eğrisi

### 10.2. Simülasyon Tabanlı (Unity Fizik Motoru)

- Tahmin edilen poz Unity'de objeye uygulanır, küçük bir bozucu kuvvet/yerçekimi verilir.
- **Başarı kriteri:** Obje N saniye boyunca X cm'den fazla kaymadan tutuluyor mu?
- Test, **farklı obje şekilleri ve farklı yaklaşım yönleri** (üstten, yandan, alttan) için ayrı ayrı çalıştırılır → **yön-bazlı başarı oranı tablosu** çıkarılır.
- Başarısız örnekler kategorize edilir: penetrasyon mu, temassızlık mı, yanlış yön mü, jitter mi — hata analizi sonraki iterasyonu yönlendirir.

### 10.3. Runtime / Performans

- **Inference latency** PC GPU'da ölçülür; ayrıca **Quest üzerinden gerçek streaming test oturumunda** (Air Link/Link Cable) motion-to-photon gecikmesi ölçülür — hedef genelde <20ms toplam.
- Model boyutu PC'de sorun olmasa da, streaming gecikmesiyle birleşince fark edilebilir; bu yüzden sadece PC'de değil, Quest üzerinden de doğrulanmalı.

### 10.4. Uçtan Uca / Kullanıcı Testi

- **Genel başarı oranı** = obje sayısı × yaklaşım yönü kombinasyonları üzerinden kaçı fizik simülasyonunda "başarılı" sayıldı
- **Doğallık/UX skoru** — kullanıcı Likert ölçeğinde "el doğal hissettirdi mi" değerlendirmesi
- **Baseline karşılaştırması** — sabit/canned grasp animasyonlarına karşı başarı oranı ve doğallık farkı

---

## 11. Beklenen Katkı

Klasik önceden tanımlanmış el animasyonlarından farklı olarak, objenin geometrisini dikkate alan, gerçek zamanlı, farklı şekil ve boyutlardaki objelere uyum sağlayabilen, Unity/OpenXR XR Hands altyapısıyla çalışan genellenebilir bir yapay zekâ tabanlı grasp sistemi.
