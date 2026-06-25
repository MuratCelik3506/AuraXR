# AuraXR — Controller-to-Hand Yol Haritası

> **Durum:** Sıfırdan yeniden başlangıç. Bu dosya tezin scope'unu ve fazlarını
> kalıcı olarak tutar. Önceki kod/commit'ler bağlayıcı değildir.

## Tezin çekirdeği

> **Controller-to-Hand:** VR controller'ın 6DoF trajektorisi + hedef objeden,
> gerçek zamanlı ve gerçekçi bir MANO el animasyonu (yaklaşırken ön-şekillenme →
> dokununca kavrama) sentezleyen, ONNX ile Unity'de çalışan bir model.

**Senaryo:** Kullanıcı sadece controller kullanır. Elinde sadece controller'ın
6DoF pozu (= bileğin nerede olduğu) ve sahnedeki objenin pozu vardır. Eksik olan
tek şey **parmakların ne yaptığı**dır. AI modelinin görevi: bilek trajektorisi +
obje verildiğinde gerçekçi parmak hareketini üretmek. Obeye yaklaşıp dokunurken
controller yerine sanal el gösterilir.

**Master tezi katkıları (ana katkı = deneyim + sentez, veri hattı bunu besler):**
1. **Controller-koşullu gerçek zamanlı el sentezini VR'da (Unity) deploy etmek** ve
   prosedürel baseline'lara karşı kullanıcı deneyimiyle değerlendirmek. *(ana katkı)*
2. HOT3D ana kaynak olacak şekilde, yardımcı veri setlerinden gelen el-nesne örneklerini
   ortak MANO temsiline uyarlayan **pratik bir eğitim hattı**. *(destekleyici katkı)*

## Kilitlenen scope kararları

| Karar | Seçim |
|---|---|
| AI görevi | Tam yaklaşma→kavrama animasyonu (sequence model) |
| El sayısı | Tek sağ el, tek controller |
| Veri kaynağı | **HOT3D Quest3 ana**; **DexYCB güçlü yardımcı** (1:1 kategori eşleşmesi), **ARCTIC marjinal** (çoğu obje scope dışı) |
| Runtime | Unity Sentis (ONNX) |
| Obje koşullama (v1) | **Paylaşılan grasp-kategori embedding** (ham obje id değil) + göreli poz; geometri/SDF v2'ye ertelendi |
| Obje sayısı (v1) | **Demo: 3 obje** (farklı kavrama tipleri); **eğitim: bu 3'e karşılık gelen 3 grasp-kategorisindeki eşleşen örnekler** (kategori öğrenir, obje ezberlemez) |
| El sürme | Model çıktısı → **Unity hand rig joint rotation'ları** (rig'i sürer; skinned-mesh paritesi derdi yok) |

**İlke:** Scope büyütmek yok. Ana amaç iyi bir Unity deneyimi; her faz buna hizmet eder.

## Elimizdeki 3 veri seti

| | DexYCB | HOT3D | ARCTIC |
|---|---|---|---|
| Bakış açısı | 8 sabit kamera (3. şahıs) | **Egocentric — Quest3 & Aria** | Egocentric + 8 allocentric mocap |
| El | Tek el (sağ) | İki el (MANO) | İki el (MANO) |
| El formatı | MANO tam 45-DoF axis-angle + global + trans (`pose_m` 51) | MANO **15-DoF PCA** + wrist 6DoF + betas | MANO tam 45-DoF + rot + trans + shape |
| Obje | YCB objeleri, 6DoF poz | 6DoF poz + **GLB mesh** + başlık trajektorisi | **Eklemli (articulated)** objeler |
| İçerik | tutma (grasp) | tutma + kullanma, gerçek Quest3 | grab + use, mocap kalite |
| VR'a yakınlık | orta | **en yüksek** | yüksek (kalite) |

**Veri yolları (`data/raw/`):**
- `dexycb/` — subject klasörleri; her seq'te `pose.npz` (`pose_m` el, `pose_y` obje) + `meta.yml`.
- `hot3d/quest3|aria/{train,test}/` — her seq'te `*_hand_data.zip` (MANO jsonl) + `*_ground_truth.zip` (6DoF obje/headset csv). `assets/` GLB mesh'ler.
- `arctic/raw_seqs/sXX/` — `*.mano.npy` (sağ/sol rot+pose+trans+shape), `*.object.npy` (732×7 6DoF), `*.egocam.dist.npy`.

## Obje seçimi & cross-dataset eşleme

### v1 stratejisi: 3 demo objesi + 3 grasp-kategorisi
- **Demo (Unity):** 3 obje; sıklığa değil **kavrama-tipi ayrışıklığına** göre seçilir.
- **Eğitim:** bu 3 objeye karşılık gelen **3 grasp-kategorisindeki eşleşen tüm örnekler**
  (HOT3D + DexYCB). Model `mug_white/bottle/box` ezberlemez; `hook/power/wide` kategorilerini
  öğrenir → tez açısından daha güçlü, daha çok veri.
- **Hızlandırma değil asıl amaç:** pipeline'ı (veri→train→ONNX→Unity→eval) küçük slice'ta
  hızlı borudan geçirmek + kaliteli model + temas metriklerini erken çivilemek.
  (Eğitim zaten hızlı; darboğaz iterasyon döngüsü.)
- Genişletme bedava: `class_emb` öğrenilen embedding olduğu için kategori/obje eklemek =
  embedding'e satır + daha çok veri; **mimari değişmez.**

### Kilit içgörü: ortak dil = grasp-kategori (obje id değil)
Üç veri setinin obje kümeleri farklı; ham obje id transfer olmaz. `class_emb`'i
**paylaşılan grasp-affordance taksonomisiyle** indeksliyoruz:
`power/silindir · kutu/wide · pinch/ince · hook/kulp`. Böylece DexYCB'nin
`mustard_bottle`'ı HOT3D'nin `bottle`'ını besler (cross-dataset augmentasyon).

### HOT3D Quest3 aday üçlü (33 objeden, coverage = #seq)
| Kavrama tipi | Aday (coverage) |
|---|---|
| Hook/kulp | `mug_white` (33) |
| Power/silindir | `bottle_ranch` (23) / `vase` (31) |
| **Wide/kutu** | `bowl` (28) / carton (`carton_milk`/`carton_oj`) |
> **Neden wide, pinch değil (v1):** (1) küçük obje pinch'i VR'da zor, hata daha görünür;
> wide grasp daha affedici. (2) **Wide/kutu, 3 veri setinde de bulunan tek kova** (HOT3D
> bowl/carton, DexYCB cracker/sugar_box, ARCTIC box) → cross-dataset augmentasyon için en
> zengin. Marker yalnız DexYCB'de.
> **Pinch/marker → v1.1 stretch:** kavrama çeşitliliği anlatısını güçlendirmek için sonra eklenir.
>
> Coverage = sahnede bulunma sıklığı; gerçek *grasp* sıklığı Faz 1A'da el-obje yakınlığıyla
> doğrulanacak (bulunma ≠ kavranma). HOT3D'de grasp etiketi çıkarım gerektirir.

### DexYCB — güçlü yardımcı (neredeyse 1:1 kategori eşleşmesi)
21 YCB objesi tamamı el-içi kavranabilir; **`ycb_grasp_ind` ile hangi objenin kavrandığı kesin belli.**
| Kova | DexYCB | HOT3D karşılığı |
|---|---|---|
| Power/silindir | `master_chef_can`, `tomato_soup_can`, `mustard_bottle` | `bottle_ranch` |
| Kutu/wide | `cracker_box`, `sugar_box`, `pudding_box`, `wood_block` | (box) |
| Pinch/ince | `large_marker`, `scissors` | `whiteboard_marker` |
| Hook/kulp | `mug` | `mug_white` |

### ARCTIC — marjinal yardımcı (çoğu scope dışı)
11 objenin 6'sı büyük **eklemli cihaz** (microwave, espressomachine, capsulemachine, mixer,
waffleiron, laptop) → iki-elli "use" + articulation, tek-el kavramaya uymaz, **v1'de atlanır.**
Kovalara giren sadece: `ketchup`→power, `box`→kutu, `phone`/`notebook`→flat. Ek dert: büyük
ölçek + mm birim → ölçek/handedness normalizasyonu dikkatli. → **düşük öncelik, Faz 1C-opsiyonel.**

### Cross-dataset harmonizasyon iskeleti (DexYCB & ARCTIC, aynı adımlar)
1. Kanonik sağ-el MANO örneği çıkar (ikisi de tam 45-DoF axis-angle → doğrudan).
2. Obje → paylaşılan grasp-kategori etiketi map'le.
3. Filtrele: yalnız seçilen 3 HOT3D kategorisine düşen objeler.
4. Normalize: Unity-ready (metre, obje-merkezli, göreli) çerçeve. Göreli-poz + kategoriyle
   çalıştığımız için tam mesh hizalama gerekmez; obje lokal çerçevesi kabaca merkezli yeter.
5. Eğitime aynı kategori etiketiyle ekle.

**Öncelik sırası:** HOT3D Quest3 (1A) → DexYCB eşleşen objeler (1C-öncelikli) → ARCTIC (1C-opsiyonel).

## Model & Eğitim Detayı

### Unity'de her frame ne var, ne istiyoruz?

**Eldeki ham veri (controller-only, Quest3):**
- `controllerPose` — dünya uzayında pozisyon (Vector3) + rotasyon (Quaternion), ~72–90 Hz.
- `objectPose` — sahnedeki etkileşimli objenin 6DoF pozu (yazım zamanı bilinir).
- `objectClassId` — obje sınıfı (sahne authoring'de atanır).

**Türetilenler (Unity tarafında hesaplanır):**
- `wristPose = controllerPose * T_wristOffset` — controller bileğin biraz önündedir;
  sabit bir rigid offset (`T_wristOffset`, bir kez kalibre edilir) ile bilek pozuna çevrilir.
  **Controller = bilek proxy'si.** Bileği model TAHMİN ETMEZ; controller'dan gelir.
- `relPose = inverse(objectPose) * wristPose` — bileğin **objeye göreli** pozu.
  Göreli koşullama, objenin sahnedeki konumundan/yöneliminden bağımsız genelleme sağlar.
- `relVel` — son birkaç frame'den sonlu farkla yaklaşma hızı.

### Model girdisi (her frame, obje çerçevesinde)

| Alan | Boyut | Açıklama |
|---|---|---|
| `rel_pos` | 3 | bileğin obje çerçevesindeki konumu (metre) |
| `rel_rot6d` | 6 | bileğin yönelimi — **6D sürekli rotasyon** (quaternion yerine; NN için daha stabil) |
| `rel_vel` | 3 | obje çerçevesinde bilek lineer hızı |
| `dist` | 1 | bilek–obje mesafesi (yaklaşma sinyali) |
| `class_emb` | E | **grasp-kategori** embedding (paylaşılan taksonomi; cross-dataset augmentasyonun anahtarı) |
| `prev_pose` | P | modelin bir önceki frame çıktısı (autoregressive geri besleme) |

`prev_pose` geri beslemesi pürüzsüzlüğü artırır ve scheduled sampling'i mümkün kılar.
Model ayrıca recurrent hidden state taşır.

### Model çıktısı (her frame)

| Alan | Boyut | Açıklama |
|---|---|---|
| `finger_pca` | 15 | MANO parmak pozu **PCA katsayıları** (v1; 45-DoF'a göre az çıktı + pürüzsüz) |
| `contact` | 1 | (opsiyonel) kavrama/temas güveni — controller↔el blend'i için |

> v1: 15-DoF PCA → MANO bileşen matrisiyle 45-DoF axis-angle'a açılır.
> v2 alternatifi: doğrudan 45-DoF axis-angle. Bilek pozu çıktıda **yok** (controller'dan gelir).

### Unity inference döngüsü (pseudo)

```
each frame:
  wristPose = controllerPose * T_wristOffset
  obj = NearestInteractable(within radius)
  if obj:
    relPose = inverse(obj.pose) * wristPose
    obs = [rel_pos, rel_rot6d, rel_vel, dist, class_emb(obj.id), prevPose]
    obsN = normalize(obs, stats)                      // eğitimdeki mean/std
    finger_pca, contact, hidden = Sentis.Run(obsN, hidden)   // recurrent state korunur
    finger45 = PCA_expand(finger_pca)                 // 15 PCA -> 45 axis-angle
    jointRots = ToJointRotations(finger45)            // MANO eklem lokal rotasyonları
    DriveHandRig(wristPose, jointRots)                // Unity hand rig'ini sür (mesh paritesi yok)
    blend = f(dist, contact)                          // controller <-> el crossfade
    prevPose = finger_pca
```

### Mimari (v1 önerisi)

- **Causal/tek-yönlü LSTM (veya GRU)**, 1–2 katman, hidden 128–256.
  Girdi MLP encoder → LSTM → çıktı MLP → PCA katsayıları.
- **Autoregressive:** önceki tahmini girdiye geri besle.
- Gerçek-zaman dostu, ONNX'e temiz export, Sentis'te recurrent state ile çalışır.
- **v2 alternatifi:** küçük causal Transformer (pencere üzerinde temporal attention) — daha çok
  kapasite ama daha ağır ONNX/Sentis. Şimdilik scope dışı.

### Koordinat & fps uyumu (kritik gotcha'lar)

- **Handedness/birim:** Unity sol-elli, Y-up, metre. Veri setleri sağ-elli, bazıları mm
  (ör. ARCTIC obje trans ~1035 = mm). **Dönüşüm Faz 1'de yapılır** → model "Unity-ready"
  obje-merkezli, metre, göreli koordinatlarla eğitilir; Unity tarafı minimum iş yapar.
- **fps:** veri ~30 Hz, Unity ~72 Hz. Model sabit oranda koşar, aradaki frame'ler
  interpolasyonla doldurulur (veya çıktı yumuşatma filtresi).

### Eğitim süreci (Faz 2)

1. **Veri:** Faz 1 kanonik örnekleri → `(input_obs, target_finger_pose)` sekansları.
2. **Segmentasyon:** reach-to-grasp pencereleri (yaklaşma başı → kavrama tamam).
3. **Normalizasyon:** girdileri standardize et (mean/std), istatistikleri Unity için sakla.
4. **Kayıp fonksiyonu:**
   - *Poz kaybı:* PCA katsayıları üzerinde MSE **veya** eklem rotasyonlarında geodesic loss.
   - *FK pozisyon kaybı:* türevlenebilir MANO FK ile 3B eklem/parmak-ucu L2 (MPJPE) —
     algısal olarak en önemli terim.
   - *Pürüzsüzlük:* ardışık frame'ler arası hız/jerk cezası.
   - *(v2) Temas kaybı:* kavramada parmak uçları obje yüzeyine değsin.
5. **Scheduled sampling:** eğitimde `prev_pose`'u kademeli olarak ground-truth yerine modelin
   kendi tahminiyle besle → autoregressive drift'i (train/inference farkı) kapat.
6. **Augmentasyon:** obje yaw rotasyonu, hafif bilek gürültüsü, (dikkatli) sol↔sağ ayna ile
   sağ-el verisini çoğaltma.
7. **Split:** subject + obje bazında train/val/test — gerçek genellemeyi ölçmek için.
8. **Değerlendirme:** MPJPE, parmak-ucu hatası, smoothness + nitel render'lar.

> **Hesaplama notu:** Donanım M2 Max (30-core GPU, 32 GB, torch MPS). Model küçük
> (<1M param), eğitim yalnız poz parametreleriyle (görüntü decode yok), veri ~0.5–0.8M
> frame. Bu sayede tam eğitim tipik olarak birkaç dakika–birkaç saat arasında kalır;
> donanım darboğaz değil. Hız için loss'ta **joint-only MANO FK (16 eklem)** kullanılır
> (full-mesh değil). **Öncelik daima model kalitesi** — hız bir kısıt değil, rahat bir pay.

## Fazlar

> **Sıralama ilkesi:** Ana amaç iyi Unity deneyimi → deneyim iskeleti **modelden önce**
> doğrulanır (Faz 0.5). Veri ise küçükten büyüğe, ablation olarak eklenir.

### Faz 0 — Temsil & arayüz tasarımı (kağıt üstü)
Modelin girdi/çıktısını sabitle (yukarıdaki "Model & Eğitim Detayı"). Kod yok.

### Faz 0.5 — Unity vertical slice (model YOK, erken de-risk)
Deneyimin iskeletini prosedürel elle, AI gelmeden ayağa kaldır:
- controller'ı gizle, sanal el göster
- **1 obje**, distance-based open/close (prosedürel grasp)
- bırakınca/uzaklaşınca controller'a geri dön
- Amaç: blend/trigger/rig akışını model gelmeden doğrulamak. Bu prosedürel el **aynı zamanda
  bir baseline** (aşağı bak).
- **Bitti sayılır:** (1) VR'da controller gizlenir, (2) virtual hand stabil takip eder,
  (3) 1 objede approach/contact/release çalışır, (4) prosedürel baseline kayıt altına alınır.

### Faz 1 — Veri harmonizasyonu (foundation, kademeli)
Ortak "sağ-el MANO" kanonik formatına indir. Aynı anda hepsini değil, kademeli:
- **Faz 1A — sadece HOT3D Quest3 sağ el:** 15-DoF PCA → 45-DoF axis-angle, `wrist_xform`'dan 6DoF, sağ eli seç.
- **Faz 1B — HOT3D Aria ekle:** aynı şema, ek kaynak.
- **Faz 1C — DexYCB / ARCTIC (opsiyonel, ablation):** DexYCB `pose_m`, ARCTIC `right/{rot,pose,trans,shape}` → doğrudan. Yalnız HOT3D kalitesi yetmezse veya genelleme ablation'ı için.
- Çıktı: kanonik frame'ler `{ wrist (objeye göreli), parmak pozu, obje pozu, grasp-kategori }`.
- **Doğrulama:** birkaç reach-grasp sekansını matplotlib/Open3D ile çiz, mantıklı mı bak.
- **Bitti sayılır (1A):** (1) HOT3D Quest3 sağ-el frame'leri çıkarılır, (2) objeye göreli wrist
  pose hesaplanır, (3) finger PCA/pose target hazır, (4) 3 grasp-kategorisine göre filtreleme
  yapılır, (5) birkaç sekans görselleştirilip doğrulanır.

### Faz 2 — Model + eğitim
Detaylar "Model & Eğitim Detayı" bölümünde. Özet: causal LSTM, autoregressive +
scheduled sampling, FK pozisyon + pürüzsüzlük kayıpları, subject/obje bazlı split,
MPJPE + nitel değerlendirme. **Baseline'larla karşılaştırma zorunlu** (aşağı bak).

### Faz 3 — ONNX + Unity inference
- ONNX export, Unity Sentis ile çalıştır, model çıktısını (PCA→45-DoF→) **Unity hand rig** joint rotasyonlarına retarget et.
- Python↔Unity çıktı paritesini doğrula (aynı girdi → aynı poz).
- **Retarget başarı kriteri:** aynı örnek poz, Python MANO görselleştirmesinde ve Unity rig'inde
  **benzer parmak konfigürasyonu** üretmeli. (Model iyi olsa bile sessiz retarget bozulması
  Unity'de kötü göstertebilir — bu kontrol şart.)

### Faz 4 — Unity deneyimi
- Faz 0.5 slice'ı AI modeliyle besle: proximity trigger → controller'dan sanal ele blend, modeli her frame çalıştır.
- Uzaklaşınca/bırakınca → controller'a geri blend.
- **3 obje** (kutu, silindir/şişe, küçük obje). Demo için sonradan genişletilir.

### Faz 5 — Değerlendirme & yazım
**Nicel (model):** held-out MPJPE / parmak-ucu hatası, smoothness, FPS/latency, ablasyonlar.

**Temas/kavrama kalitesi (MPJPE tek başına yetmez):**
MPJPE pozisyon doğruluğunu ölçer ama fiziksel makullüğü ölçmez — el "ortalama doğru" olsa bile
objeye gömülebilir veya havada kalabilir. Katmanlı plan:
- **v1 zorunlu (nicel):**
  - *Fingertip-object distance at contact* — kavrama anında parmak uçlarının obje yüzeyine
    mesafesi (ideal ≈ 0; pozitif = havada kalma, negatif = penetrasyon).
  - *Contact timing error* — modelin kavrama anı ile ground-truth kavrama anı arasındaki gecikme.
- **v1 nitel:** *Penetration visual / collision sanity check* — 3 demo objesinin mesh'i zaten
  elimizde (HOT3D GLB), bu objelerde gözle/collider ile gömülme kontrolü.
- **v2:** *Gerçek mesh/SDF penetration rate* — veri genelinde hacimsel penetrasyon derinliği/oranı.

> Bu sinyaller aynı zamanda Faz 2'deki **(v2) temas kaybı** için doğal hedeflerdir:
> fingertip-object distance → temas loss'u, penetration → ceza terimi.

**Kullanıcı çalışması (deneyim) — 3 koşullu, küçük N:**
- Koşullar: (1) controller görünür, (2) prosedürel sanal el, (3) AI sanal el.
- Ölçümler: realism, agency, presence, preference, discomfort + objektif FPS/latency.
- Valide edilmiş kısa anket kullan (ör. embodiment: Gonzalez-Franco & Peck; presence: kısa ölçek) — hafif ama savunulabilir.

## Baseline'lar (tez savunması için zorunlu)
AI modeli mutlaka bunlarla kıyaslanır; yoksa "model iyi mi?" sorusu çürük kalır:
1. **Controller-following static hand** — sabit poz, parmaklar oynamaz.
2. **Distance-based pose blending** — mesafeye göre açık↔kapalı el arası blend (prosedürel).
3. **Basit IK / prosedürel grasp** — obje başına önceden tanımlı grasp pozuna IK ile yaklaş.

> Baseline 2/3 zaten Faz 0.5 vertical slice'ının içeriği — bedavaya hem de-risk hem kıyas.

## Açık sorular / sonraki adım
- Sıradaki iş: **Faz 0.5 (Unity vertical slice)** + **Faz 1A (HOT3D Quest3 harmonizasyonu)** paralel.
- v2 / scope dışı: iki el, obje geometrisi/SDF, eklemli objeler, görüntü-bazlı tracking, tam fiziksel grasp simülasyonu.
