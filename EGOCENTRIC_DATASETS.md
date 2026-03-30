# Egosentrik (HMD/VR/AR Gözlük) Veri Setleri Kataloğu

Bu belge, XR/AR/VR cihazları **giyilerek** ya da başa takılan kameralarla kaydedilmiş
egosentrik (1. şahıs bakış açısı) veri setlerini kapsamlı biçimde listeler.  
AuraXR — Intent-Aware XR Framework projesi kapsamında hazırlanmıştır.

---

## 📊 Özet Karşılaştırma

| Dataset        | Cihaz              | Ölçek      | El   | Nesne | Göz  | IMU  | Depth | Erişim |
|----------------|--------------------|------------|------|-------|------|------|-------|--------|
| **HOT3D**      | Quest3 + Aria      | 833 dk     | ✅   | ✅    | ✅ * | ❌   | ❌    | Açık   |
| **Ego4D**      | Çeşitli wearable   | 3.670 sa   | ✅ ± | ✅ ±  | ❌   | ✅   | ❌    | Açık   |
| **Ego-Exo4D**  | Aria + sabit       | 740 sa     | ✅   | ✅    | ✅   | ✅   | ❌    | Açık   |
| **HoloAssist** | HoloLens 2         | 166 sa     | ✅   | ✅    | ✅   | ✅   | ✅    | Açık   |
| **Aria Pilot** | Project Aria       | —          | ✅   | ✅    | ✅   | ✅   | ❌    | Açık * |
| **FPHA**       | Baş kamerası       | 1.175 vid  | ✅   | ✅    | ❌   | ❌   | ✅    | Açık   |
| **AssemblyHands** | Baş kamerası   | 3.0 M kare | ✅   | ❌    | ❌   | ❌   | ❌    | Açık   |
| **ARCTIC**     | MoCap + ego cam    | 2.1 M kare | ✅   | ✅    | ❌   | ❌   | ❌    | Açık   |
| **EgoPressure**| Baş kamerası + RGB-D | 5 sa    | ✅   | ✅    | ❌   | ❌   | ✅    | Açık   |

> `*` Göz takibi yalnızca Aria cihazında. `±` Kısmi annotasyon.

---

## 🏆 Tier 1 — Gerçek XR Donanımı ile Kaydedilmiş

### 1. HOT3D — Hand and Object Tracking in 3D
> **Meta · CVPR 2025 Highlight**

- **Cihaz:** Meta Project Aria (AR gözlük) + Meta Quest 3 (VR kulaklık)
- **Ölçek:** 833 dakika · 3.7M+ kare · 19 katılımcı · 33 nesne · 30 FPS
- **Senaryolar:** Mutfak, ofis, oturma odası
- **El Annotasyon:** UmeTrack (22 açı + bilek 6DoF) + MANO (45 poz parametresi)
- **Nesne Annotasyon:** 6DoF pose · 2D bbox · segmentasyon maskesi (SAM2) · görünürlük skoru
- **Ek Modaliteler (Aria):** 3D göz takibi (`gaze_point_3d`) · SLAM nokta bulutu
- **Kamera Akışları:** RGB 1408×1408 (Aria) · 2× Mono 640×480 (Aria) · 2× Mono 1280×1024 (Quest3)
- **Format:** WebDataset `.tar` arşivleri (HOT3D-Clips: 150 kare/clip)
- **Erişim:** https://huggingface.co/datasets/bop-benchmark/hot3d
- **Paper:** https://arxiv.org/abs/2411.19167

**AuraXR'daki Kullanımı:**
- `src/data/hot3d_dataset.py` — UmeTrack → XYZ FK dönüşümü
- H2O ile birleşik eğitim: `shared_head` fusion modu (3 sınıf)
- Şu anda kullanılmayan ama değerli: göz takibi, MANO beta (el şekli), segmentasyon maskeleri

---

### 2. Ego4D
> **Meta + 13 Üniversite · CVPR 2022**

- **Cihaz:** Çeşitli giyilebilir kameralar (GoPro, Aria prototipi vb.)
- **Ölçek:** **3.670 saat** video · 923 katılımcı · 74 lokasyon · 9 ülke
- **İçerik:** Günlük yaşam aktiviteleri (yemek, alışveriş, spor, sosyal etkileşim, imalat)
- **Alt Benchmark'lar:**
  - `Hands & Objects` — el-nesne durumu değişimi tahmini
  - `Forecasting` — kısa/uzun vadeli eylem tahmini
  - `Episodic Memory` — "Ne zaman gördüm?" sorusu
  - `Social` — sosyal etkileşim analizi
  - `NLQ / MQ` — doğal dil sorgusu
- **Annotasyon:** El bbox · nesne bbox · eylem açıklamaları (narration) · IMU
- **Erişim:** https://ego4d-data.org (kayıt gerekli)

**Özelliği:** Önceki en büyük egosentrik datasetten **20× büyük**.

---

### 3. Ego-Exo4D
> **Meta · CVPR 2024**

- **Cihaz:** Meta Project Aria (egocentric) + sabit kameralar (exocentric) — **eşzamanlı**
- **Ölçek:** 740 saat · 800+ senaryo · 5 ülke
- **İçerik:** Beceri gerektiren aktiviteler (basketbol, pişirme, müzik, tıbbi prosedür, montaj)
- **Eşsizlik:** Hem 1. hem 3. şahıs bakış açısı **aynı anda kaydedilmiş** → beceri öğrenimi ve transfer araştırması
- **Annotasyon:** El pozu · nesne etkileşimi · beceri seviyesi · adım-adım açıklamalar
- **Erişim:** https://ego-exo4d-data.org

---

### 4. HoloAssist
> **Microsoft · ICCV 2023**

- **Cihaz:** **Microsoft HoloLens 2** (mixed reality / karışık gerçeklik)
- **Ölçek:** 166 saat · 350 çift (öğretmen + öğrenci) · 20 görev kategorisi
- **İçerik:** İnsan–yapay zeka işbirlikli prosedürel görevler (tamir, montaj, kurulum)
- **7 Eşzamanlı Modalite:**
  1. RGB görüntü
  2. Derinlik (Depth)
  3. Baş pozu (Head pose / 6DoF)
  4. 3D el pozu (21 eklem)
  5. Göz takibi (Eye gaze)
  6. Ses (Audio)
  7. IMU (İvme + Jiroskop)
- **Benchmark Görevleri:** Hata tespiti · müdahale tipi tahmini · el hareketi tahmini
- **Erişim:** Microsoft Research (açık)
- **Paper:** https://arxiv.org/abs/2310.05165

**AuraXR Relevansı:** "AI tabanlı interaktif asistan" konseptiyle birebir örtüşüyor.  
7 modalite füzyonu, IntentFormer'ın multimodal genişlemesi için referans alınabilir.

---

### 5. Project Aria Everyday Objects (AEO)
> **Meta · Süregelen**

- **Cihaz:** Meta Project Aria (hafif AR araştırma gözlüğü)
- **İçerik:** Günlük ortamda 3D nesne tespiti — ofis, ev, dış mekan
- **Modaliteler:** RGB · 2× Mono · **Göz takibi** · **IMU** · SLAM · GPS
- **Amaç:** Egosentrik 3D nesne tespiti ve foundation model validasyonu
- **Erişim:** https://www.projectaria.com (kayıt gerekli)

---

## 🥈 Tier 2 — Baş/Lab Kamerası ile Egosentrik

### 6. FPHA — First-Person Hand Action Benchmark
> **Garcia-Hernando et al. · CVPR 2018**

- **Cihaz:** Intel RealSense + başa takılan özel kamera rig
- **Ölçek:** 1.175 video · 45 aktivite kategorisi · 6 aktör · 1.175.075 kare
- **İçerik:** Günlük el-nesne eylemleri (içme, yazma, ürün açma, telefon kullanma)
- **Annotasyon:** 21 el eklemi (manyetik sensör) · 6DoF nesne pozu
- **Erişim:** https://github.com/guiggh/hand_pose_action

**Sınırlılık:** Manyetik sensör annotasyonu, modern modeller için zorlu referans oluşturur.

---

### 7. AssemblyHands
> **CVPR 2023**

- **Cihaz:** Başa takılan mono kamera (egocentric) + çok kameralı exo setup
- **Kaynak:** Assembly101 veri setinden türetilmiş
- **Ölçek:** **3.0 milyon annotated kare** (490.000 egocentric)
- **İçerik:** Oyuncak montaj ve demontaj
- **Annotasyon:** 3D el pozu (MANO) — hem ego hem exo
- **Eşsizlik:** Egosentrik için **en büyük 3D el pose benchmark'ı**
- **Erişim:** Açık

---

### 8. ARCTIC — ARticulated objeCTs in InteraCtion
> **MPI · CVPR 2023**

- **Cihaz:** 54 adet Vicon kamera (MoCap) + 1 egocentric kamera
- **Ölçek:** 2.1 milyon yüksek çözünürlüklü kare · 10 katılımcı · 11 eklemli nesne
- **İçerik:** İki elle eklemli nesne manipülasyonu (makas, dizüstü bilgisayar, kutu, şişe)
- **Annotasyon:**
  - **SMPL-X** — tam insan vücudu pozu
  - **MANO** — sağ + sol el pozu
  - **Nesne eklem açıları** — makas açısı, kapak açısı vb.
- **Eşsizlik:** Hem eller + tam vücut + **eklemli nesne** içeren tek dataset
- **Erişim:** https://arctic.is.tue.mpg.de
- **Paper:** https://arxiv.org/abs/2305.00737

---

### 9. EgoPressure
> **2024**

- **Cihaz:** Başa takılı kamera + çok görüşlü RGB-D
- **Ölçek:** 5 saat · 21 katılımcı
- **İçerik:** El parmak temas basıncı + kavrama pozu egosentrik görüntüden
- **Annotasyon:** El pozu · mesh · **temas basınç haritası**
- **Eşsizlik:** **Temas basıncı verisi** — başka hiçbir egosentrik datasette mevcut değil
- **Erişim:** Araştırma amaçlı açık

---

## 🎯 AuraXR Tezi İçin Öncelik Sırası

```
1. HOT3D          ✅  Zaten entegre — Quest3 + Aria, göz takibi henüz kullanılmıyor
2. HoloAssist     🎯  En yüksek modalite çeşitliliği, XR asistan konseptiyle örtüşüyor
3. Ego4D          📊  Devasa ölçek, tahmin benchmark'ı, eylem açıklamaları
4. ARCTIC         🤲  İki el + eklemli nesne → daha zengin manipülasyon
5. Ego-Exo4D      📡  Ego + exo eşzamanlı → cross-view genelleştirme
```

---

## 🔗 Hızlı Referanslar

| Dataset | Paper | Kod / İndirme |
|---|---|---|
| HOT3D | [arXiv 2411.19167](https://arxiv.org/abs/2411.19167) | [HuggingFace](https://huggingface.co/datasets/bop-benchmark/hot3d) |
| Ego4D | [arXiv 2110.07058](https://arxiv.org/abs/2110.07058) | [ego4d-data.org](https://ego4d-data.org) |
| Ego-Exo4D | [CVPR 2024](https://ego-exo4d-data.org) | [ego-exo4d-data.org](https://ego-exo4d-data.org) |
| HoloAssist | [arXiv 2310.05165](https://arxiv.org/abs/2310.05165) | [Microsoft Research](https://github.com/microsoft/HoloAssist) |
| ARCTIC | [arXiv 2305.00737](https://arxiv.org/abs/2305.00737) | [arctic.is.tue.mpg.de](https://arctic.is.tue.mpg.de) |
| AssemblyHands | [CVPR 2023](https://assemblyhands.github.io) | [assemblyhands.github.io](https://assemblyhands.github.io) |
| FPHA | [CVPR 2018](https://github.com/guiggh/hand_pose_action) | [GitHub](https://github.com/guiggh/hand_pose_action) |
| EgoPressure | [arXiv 2024](https://arxiv.org/abs/2405.13820) | Araştırma başvurusu |

---

*Son güncelleme: 2026-03-27 — AuraXR Phase 1 proje dokümantasyonu*
