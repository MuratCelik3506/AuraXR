# A. Veri Hazırlık Pipeline'ı

Tüm modellerin eğitimi bu aşamaya dayanır. Veri hazırlama; ham veri setlerini indirmekten, format dönüşümü, segmentasyon ve güven skoru etiketlemeye kadar uzanan ardışık bir süreçtir. Downstream adımlar (model eğitimi, eval) bu pipeline'ın çıktılarını kullanır; dolayısıyla hataların erken yakalanması kritiktir.

> **Durum (2026-07-01):** A1–A5 adımları tamamlandı. Ham veri koordinat uyuşmazlıkları incelendi, giderildi ve processed veri üretildi. OakInk split hatası (per-kategori → global 70/15/15) giderildi; HOT3D `dist` alanı AABB'den Euclidean nearest-surface mesafesine dönüştürüldü. Mevcut sıkıntılar A8'de belgelenmiştir.

---

## A1. HOT3D İndirme & Format İnceleme

**Kaynak:** [HOT3D Dataset](https://facebookresearch.github.io/hot3d/)

HOT3D, egocentric (Aria + Quest 3) perspektiften çekilmiş, 33 farklı mutfak/ofis/oturma odası objesiyle gerçekleştirilen el manipülasyonu sekanslarını içerir. Her sekans pick-up → observe → put-down gibi tam bir manipülasyon döngüsünü kapsar. Bu tezde HOT3D'nin ana rolü, **temporal kapanış davranışını** ve frame-to-frame el hareketini öğretmektir.

**İndirilecekler:**
- Quest 3 kameralarına ait görüntü sekansları (isteğe bağlı, pose için kullanılmaz)
- **Hand pose verileri:** UmeTrack formatı — bilek + 15 eklem rotasyonu, per-frame
- **Obje pose verileri:** 6-DoF obje konumu ve rotasyonu, per-frame
- **Obje 3D mesh'leri:** Her obje için .obj / .glb formatında

**Format notları (ham veri incelemesinden doğrulandı):**
- `mano_hand_pose_trajectory.jsonl` (zip içinde): her satır bir frame, `timestamp_ns` + `hand_poses` dict. Sağ el `id="1"`, sol el `id="0"`. **`test` split'teki tüm bu dosyalar boş** — el pozu yalnızca `train` split'te mevcut.
- `wrist_xform.t_xyz`: metre cinsinden, world frame, per-timestamp bilek translation.
- `wrist_xform.q_wxyz`: world frame bilek quaternion (wxyz sırası).
- `dynamic_objects.csv` (zip içinde): `object_uid, timestamp[ns], t_wo_x[m], t_wo_y[m], t_wo_z[m], q_wo_w, q_wo_x, q_wo_y, q_wo_z` — per-frame, per-object, world frame.
- `instance.json`: `instance_id → instance_name` mapping. `dynamic_objects.csv`'deki `object_uid` bu `instance_id` ile eşleşir.
- `.glb` mesh'leri object-local frame'de, metre cinsinden. `instance.json` üzerinden `uid → name → glb` zinciri kurulur.
- UmeTrack eklem sırası: bilek → başparmak (4 eklem) → işaret (4) → orta (4) → yüzük (4) → serçe (4) → toplam 21 nokta (5 uç nokta + 16 eklem)
- HOT3D'nin bilinen **thumb DOF hatası**: başparmak pronasyon/supinasyon ekseni yanlış modellendi — bu eklemlerin rotasyonları yüksek gürültülü; eğitimde bu eklemlere daha düşük ağırlık verilmeli veya ayrıca normalize edilmeli.

**Kontrol listesi:**
- [x] Tüm sekans dosyaları eksiksiz indirildi
- [x] Frame sayısı × obje sayısı tutarlı
- [ ] En az 5 sekans görsel olarak gözden geçirildi (özellikle başparmak hareketi)

---

## A2. OakInk İndirme & Format İnceleme

**Kaynak:** [OakInk Dataset](https://oakink.net/)

OakInk, 1.800 farklı obje ve 50.000+ grasp pozu içeren MANO tabanlı el kavrama veri setidir. HOT3D'den farklı olarak hareket sekansları değil, **statik grasp pozları** sunar. Bu tezde OakInk'in ana rolü, objenin geometrisi ile final kavrama pozu arasındaki ilişkiyi öğretmektir.

**İndirilecekler:**
- `OakInk-Shape`: Obje mesh'leri (SDF ve point cloud)
- MANO parametreleri: shape (β, 10 boyut) + pose (θ, 48 boyut = 16 eklem × 3 eksen)

**Format notları (ham veri incelemesinden doğrulandı):**
- `anno/general_info/*.pkl`: her sample için `hand_anno.hand_tsl (3,)`, `hand_anno.hand_pose (16,4)` wxyz quat, `hand_anno.hand_shape (10,)`, `obj_anno (4,4)` object-to-world rigid transform, `cam_extr (4,4)`, `cam_intr (3,3)`.
- `hand_tsl` = wrist'in **world frame'deki** gerçek pozisyonudur (`cam_extr⁻¹ @ hand_j[0]_cam = hand_tsl` doğrulandı). MANO `transl` parametresiyle karıştırılmamalı.
- `obj_anno` translation değerleri ~2–13 cm aralığında, world frame, metre cinsinden.
- `anno/hand_j/*.pkl`: `(21,3)` MANO eklem pozisyonları — **kamera frame'inde**. `cam_extr⁻¹` ile world frame'e taşınır. Parmak ucu indeksleri: thumb=4, index=8, middle=12, ring=16, pinky=20.
- `OakBase/OakBase/<category>/<instance>/part_*.ply`: binary PLY, vertex koordinatları **metre cinsinden**, object-local frame. Ek scale/rotation gerektirmez.
- `shape/OakInkObjects*/align/model_scale.json`: Bu dosyalar **mevcut pipeline'da kullanılmıyor** — `OakInkVirtualObjects` adlı farklı bir asset kütüphanesine ait, `OakBase` mesh'leri zaten doğru ölçekte.
- Obje mesh'leri canonical (object-centric) frame'de; skalanın korunması gerekir.
- Bazı objelerde temas bölgesi (contact map) label'ı mevcuttur — güven skoru hesaplamasında kullanılabilir.

**Kontrol listesi:**
- [x] MANO modeli (.pkl) ve OakInk pose verisi uyumlu yükleniyor
- [ ] 10 farklı obje × 5 farklı grasp pozu görsel olarak incelendi
- [x] Obje mesh ölçekleri birbirine normalize edildi mi? → OakBase zaten metre, normalize gerekmez

---

## Veri Kaynaklarının Rolleri

Bu tezde tek ana öğrenilmiş sistem vardır: **Temporal Geometry-Conditioned Grasp Model**. OakInk ve HOT3D bu modeli farklı yönlerden besler.

| Veri Kaynağı | Veri Tipi | Modele Öğrettiği Şey |
|---|---|---|
| OakInk | Statik grasp pozu | "Bu obje hangi parmak pozu ile tutulabilir?" |
| HOT3D | Zaman sıralı el-obje sekansı | "El bu poza zaman içinde nasıl kapanır?" |
| Unity | Fizik simülasyonu/eval | "Üretilen grasp fiziksel olarak başarılı mı?" |

OakInk temporal model eğitmek için tek başına yeterli değildir, çünkü frame dizisi sağlamaz. HOT3D ise temporal davranış için gereklidir, ancak obje çeşitliliği OakInk kadar geniş değildir. Bu yüzden iki veri seti aynı ana modeli tamamlayıcı biçimde eğitir.

---

## Canonical Veri Sözleşmesi

Modelin veri setlerinden beklediği alanlar aşağıdaki gibi sabitlenir.

### OakInk Canonical Alanları

OakInk statik grasp pre-training için kullanılır.

| Alan | Boyut | Açıklama |
|---|---:|---|
| `pose` | `(N, 48)` | MANO global orient `(3)` + finger axis-angle `(45)` |
| `shape` | `(N, 10)` | MANO betas |
| `tsl` | `(N, 3)` | Bilek world frame translation (= `hand_tsl` = `cam_extr⁻¹ @ wrist_cam`) |
| `obj_anno` | `(N, 12)` | Object-to-world transform: `R(9, row-major) + t(3)` flat |
| `fingertips_world` | `(N, 5, 3)` | GT parmak ucu pozisyonları — world frame (`hand_j` cam→world dönüşümünden) |
| `obj_name` | `(N,)` | Obje kimliği |
| `category` | `(N,)` | Obje kategorisi |
| `obj_pts/{obj_name}.npy` | `(1024, 3)` | OakBase'den örneklenmiş obje point cloud (canonical/object-local frame, metre) |

**Koordinat sözleşmesi:** `obj_pts_contact` (contact/penetration loss girdisi) şu zincirle wrist frame'e taşınır:
```
pts_world = obj_pts_canonical @ R_obj.T + t_obj    # canonical → world
pts_wrist = (pts_world - wrist_tsl) @ R_wrist       # world → wrist
```
`R_wrist = axis_angle_to_matrix(global_orient_aa)` — wrist-local→world dönüşümü.

### HOT3D Canonical Alanları

HOT3D temporal training için kullanılır.

| Alan | Boyut | Açıklama |
|---|---:|---|
| `rel_pos` | `(F, 3)` | Bileğin objeye göre relatif konumu (object frame) |
| `rel_rot6d` | `(F, 6)` | Bileğin objeye göre 6D rotasyon temsili |
| `rel_vel` | `(F, 3)` | Relatif bilek hızı |
| `dist` | `(F, 1)` | Bilek → obje nearest-surface Euclidean mesafesi (metre). **AABB değil** — `trimesh.proximity.closest_point` ile hesaplanır; stats.json mean=0.175, std=0.108. |
| `finger_aa45` | `(F, 45)` | MANO parmak axis-angle pozu (wrist-relative) |
| `fk_joints` | `(F, 16, 3)` | Gerçek MANO FK eklem pozisyonları — **world frame** |
| `wrist_world_t` | `(F, 3)` | Bilek world frame translation (metre) |
| `wrist_world_q` | `(F, 4)` | Bilek world frame quaternion (wxyz) |
| `obj_world_t` | `(F, 3)` | Obje world frame translation (metre) |
| `obj_world_q` | `(F, 4)` | Obje world frame quaternion (wxyz) |
| `contact_flag` | `(F,)` | Temas var/yok (3cm AABB eşiğiyle tespit edildi) |
| `segment_id` | `(F,)` | Grasp segment kimliği |
| `obj_name` | `(F,)` | Obje adı |

**`fk_joints` kullanımı:** Parmak ucu indeksleri `[3, 6, 9, 12, 15]` — her parmak için 3-eklemli zincirin son elemanı. World→wrist dönüşümü: `(fk_joints[tip_idx] - wrist_world_t) @ R_wrist`.

**`obj_pts_contact` (HOT3D):**
```
pts_world = obj_pts_canonical @ R_obj.T + obj_world_t    # canonical → world
pts_wrist = (pts_world - wrist_world_t) @ R_wrist         # world → wrist
```

**Model input (batch alanları):**

| Alan | Boyut | Açıklama |
|---|---:|---|
| `frame_feat` | `(B, T, 13)` | Ana girdi penceresi — `concat(rel_pos, rel_rot6d, rel_vel, dist)` |
| `prev_frame_feat` | `(B, T, 13)` | 1 frame geriye kaydırılmış pencere — Phase 2 `L_vel` için |
| `prev2_frame_feat` | `(B, T, 13)` | 2 frame geriye kaydırılmış pencere — Phase 2 `L_acc` için |
| `target_pose` | `(B, 45)` | t anındaki GT parmak pozu |
| `prev_pose` | `(B, 45)` | t-1 anındaki GT parmak pozu |
| `prev2_pose` | `(B, 45)` | t-2 anındaki GT parmak pozu |

`prev_frame_feat` ve `prev2_frame_feat`, Phase 2 eğitiminde `no_grad` forward pass ile `pred_{t-1}` ve `pred_{t-2}` üretmek için kullanılır; gradient yalnızca `pred_t` (ana forward pass) üzerinden akar. Başlangıç için `T=8` ve `T=16` denenir.

---

## A3. MANO / UmeTrack → XR Hands Joint Mapping

Bu adım, HOT3D (UmeTrack) ve OakInk (MANO) verilerinin Unity XR Hands iskeletine dönüştürüldüğü **Milestone 1**'dir.

### Hedef Format

Unity XR Hands `XRHandJointID` yapısı: **26 eklem/el**

| XR Hands Bölgesi | Eklem Sayısı |
|---|---|
| Bilek (Wrist) | 1 |
| Başparmak (Thumb) | 5 |
| İşaret (Index) | 5 |
| Orta (Middle) | 5 |
| Yüzük (Ring) | 5 |
| Serçe (Little) | 5 |

**Toplam:** 26 eklem. MANO/UmeTrack'te 21 nokta var; fark: her parmakta metacarpal (palm'e en yakın kemik) eklenmiş.

### Dönüşüm Adımları

1. **UmeTrack → XR Hands:**
   - UmeTrack'in 21 noktası (bilek + 20 eklem) XR Hands'in 26 eklemine map edilir
   - Eksik metacarpal pozisyonları: bilek ve proximal phalange arasına doğrusal interpolasyon
   - Rotasyonlar: UmeTrack'in local quaternion'ları parent bone yönüne göre yeniden hesaplanır

2. **MANO → XR Hands:**
   - MANO'nun 16 eklem rotasyonu (axis-angle) önce FK ile 3D pozisyona çevrilir
   - Sonra XR Hands'in 26 eklemine map edilir
   - MANO shape parametresi (β) el boyutunu etkiler; canonical boyuta normalize et

3. **Thumb DOF düzeltmesi (HOT3D özelinde):**
   - Başparmak CMC ekleminin yaw ekseni hatalı; bu ekseni sabit/sıfır kabul eden bir override eklenebilir
   - Ya da bu eğitim örneklerine loss maskesi uygulanır (thumb_loss_weight = 0.3)

### Doğrulama

- **Sayısal:** Referans dataset'ten alınan 5 örnek için orijinal ve dönüştürülmüş rotasyonlar arasındaki angular error < 5°
- **Görsel:** Unity Editor'da dönüştürülen sekanslar oynatılır; özellikle başparmak ve küçük parmak için distorsiyon ve eklem limiti ihlali kontrol edilir

---

## A4. HOT3D Temporal Grasp Segmentasyonu

Bu adım **Milestone 2**'dir. HOT3D'nin sürekli hareket sekansları temporal grasp öğrenimi için fazlara ayrılır.

### Uygulanan Segmentasyon (build_hot3d_canonical_full.py)

| Parametre | Değer | Açıklama |
|---|---|---|
| Temas eşiği | 3 cm (AABB) | Parmak ucu → obje bounding box mesafesi |
| Min segment uzunluğu | 5 frame | Daha kısa segmentler atılır |
| Pre-context | 30 frame | Temas öncesi eklenen bağlam |
| Post-context | 5 frame | Temas sonrası eklenen bağlam |

Temas tespiti world frame'de yapılır: `fingertip_aabb_dist(tips_world, t_obj, q_obj, bb_lo, bb_hi) < 0.03m`. Obje mesh'i object-local AABB, her frame için `obj_world_t/q` ile world'e taşınır.

**Önemli not:** 3 cm AABB eşiği geniş tutulduğu için segmentlerin önemli kısmı pre-grasp "yaklaşım" fazı içerir. Bu A8'de detaylandırılmıştır.

### Çıktı

- `data/processed/hot3d_canonical/seq_*.npz` — per-sequence grasp segment verileri
- `data/processed/hot3d_canonical/manifest.csv` — sequence meta bilgisi
- `data/processed/hot3d_canonical/stats.json` — normalizasyon istatistikleri
- `data/processed/hot3d_canonical/obj_pts/*.npy` — 27 obje için GLB'den örneklenen point cloud'lar
- `results/hot3d_canonical_summary.json` — genel istatistikler

**Üretilen veri özeti:** 157 sequence, 297.248 frame, 4.113 grasp segmenti.

---

## A5. Label Üretimi

Bu adım **Milestone 3**'tür. Model iki ayrı label kaynağı kullanır; bunlar karıştırılmamalıdır.

### Heuristic quality_label (OakInk + HOT3D'den, runtime'da hesaplanır)

| Metrik | Formül | Aralık |
|---|---|---|
| **Contact Ratio** | 3 cm'den yakın parmak sayısı / 5 | [0, 1] |
| **Penetration Depth** | Parmak uçlarının mesh içine girme — normalize | [0, 1] |
| **Wrist–Object Distance** | Bilek-obje mesafesi — normalize | [0, 1] |

```python
quality_raw = (
    w1 * contact_ratio
    - w2 * clip(penetration_depth / max_pen, 0.0, 1.0)
    - w3 * clip(dist / max_dist, 0.0, 1.0)
)
quality_label = clip(quality_raw, 0.0, 1.0)
```

Ağırlıklar: `w1=1.0, w2=0.3, w3=0.2`. Contact eşiği: `CONTACT_THRESHOLD_M = 0.030` (3 cm — build scriptindeki AABB eşiğiyle tutarlı).

> **Penetration metriği hakkında:** quality_label içindeki penetration terimi centroid-proxy tabanlıdır — parmak ucu obje merkezine yüzey noktasından daha yakınsa "içeride" sayılır. Küresel/silindirikal objeler için makul, düz/hollow objeler için yanıltıcı olabilir. Bu değer bir **eğitim yönlendiricisidir**; gerçek penetration ölçümü Unity PhysX'ten (`Physics.ComputePenetration`) gelir ve `success_label`'a dahildir. Tezde "modelimiz penetrasyonu önler" iddiası offline metriğe değil Unity eval'e dayandırılmalıdır.

**Fingertip pozisyonu kaynağı (kritik):**
- `mano_fk.py` simplified FK **kullanılmaz** — gerçek MANO'dan 18–20 cm sapma gösterdiği ölçüldü.
- HOT3D: `fk_joints[t, [3,6,9,12,15]]` (stored world frame) → `(fk_world - wrist_t) @ R_wrist` → wrist frame.
- OakInk: `fingertips_world[idx]` (hand_j cam→world dönüşümünden, dataset.npz'de saklanıyor) → `(tips_world - tsl) @ R_wrist` → wrist frame.
- Sonuç: HOT3D `mean=0.064, max=0.60, >0: %21` | OakInk `mean=0.117, max=0.80, >0: %30`.

`quality_label` [0,1] aralığına zorlanır; sigmoid çıktısıyla MSE hesabı için gerekli.

### Unity success_label (Unity eval'den, Aşama 3'te kullanılır)

```
success_label ∈ {0, 1}
```

Unity fizik simülasyonundan binary başarı etiketi olarak gelir. Stability bilgisi yalnızca bu etikette yer alır; `quality_label`'a dahil edilmez. `quality_label` ve `success_label` farklı head'lerde kullanılır (B7).

---

## A6. Veri Augmentasyon Stratejisi

Modellerin görülmemiş obje/yön kombinasyonlarına genellemesi için offline augmentasyon uygulanır.

### Temporal Context Augmentasyonu

| Augmentasyon | Yöntem |
|---|---|
| Yaklaşım yönü çeşitlendirme | Bilek trajectory'sini global koordinatlarda döndür (±45° yaw/pitch) |
| Bilek hız pertürbasyon | Trajectory hızına Gaussian gürültü ekle (σ = 0.02 m/s) |
| Obje konum jitter | Obje pozisyonuna ±5mm Gaussian gürültü ekle |
| Temporal flip | Sekansı tersine çevir (put-down → pick-up) — yaklaşım profili benzer |

### Ana Model Grasp Augmentasyonu

| Augmentasyon | Yöntem |
|---|---|
| Bilek rotasyon çeşitlendirme | Bilek yönelimini ±30° döndür, FK ile parmak pozisyonlarını güncelle |
| Obje yönelim çeşitlendirme | Objeyi kendi ekseni etrafında döndür (obje simetrikse her açı geçerli) |
| Ölçek pertürbasyon | Obje mesh'ini ±10% ölçekle, eklem pozisyonlarını güncelle |
| Gürültü ekleme | Eklem açılarına ±2° Gaussian gürültü |

### Oranlar

Augmentasyon sonrası veri dağılımı hedefi: OakInk 11K × 4 = ~44K, HOT3D grasp ~297K (zaten geniş).

---

## A7. Train / Val / Test Split — Görülmemiş Obje Seti

**Temel kural:** Test seti "görülmemiş obje" içermelidir.

### OakInk Split (mevcut)

**Global 70/15/15 obje bazlı split** — aynı `obj_name`'e ait tüm sample'lar aynı partition'da kalır (seen/unseen test ayrımı mümkün). Eski "per-kategori stratification" stratejisi, 1 objeli kategorilerin tümünün test'e düşmesi sorununu üretiyordu — bu hata `build_oakink_canonical.py`'de global shuffle ile giderildi.

| Partition | Sample | Obje sayısı |
|---|---|---|
| `train` (sample-level) | 8.921 | — |
| `val` (sample-level) | 1.115 | — |
| `seen_test` (sample-level) | 1.115 | — |
| `obj_train` (obje bazlı) | 7.466 | 17 |
| `obj_val` (obje bazlı) | 2.002 | 4 |
| `unseen_test` (obje bazlı) | 1.683 | 4 |

Toplam dataset: **11.151 sample, 25 benzersiz obje.** `split.json` her iki partition biçimini de içerir.

### HOT3D Split

157 sequence, tümü `train` split'teki kayıtlardır (HOT3D `test` split'te hand pose yok). Split; sequence değil **frame bazında** `obj_name`'e göre yapılır — her frame'in hedef objesine bakarak `obj_split.json`'daki mapping'den split atanır. Aynı sequence'ın farklı frame'leri farklı split'lere düşebilir; bu sayede sequence-level data leakage olmadan obje bazlı genelleme testi mümkün olur.

**Obje bazlı split (`data/processed/hot3d_canonical/obj_split.json`):**

| Split | Obje sayısı | Sequence | Frame |
|---|---|---|---|
| train | 11 obje | 93 seq | ~168k frame |
| val | 2 obje (`bottle_ranch`, `can_soup`) | 31 seq | ~77k frame |
| test | 4 obje (`bottle_mustard`, `flask`, `mug_white`, `puzzle_toy`) | 33 seq | ~52k frame |

Toplam 17 benzersiz obje, 157 sequence. `make_hot3d_split.py` `HOT3D_SPLIT_COUNTS`'u okur; veri setinde 17 obje bulunduğundan hedef sayılar (22/4/3/4) küçültülerek orantılı atama yapılır. Loader `dataset_hot3d.py`, `obj_split.json` dosyasını okur ve `_index_windows` aşamasında frame-level filtreleme uygular.

### Final Test Protokolü

**Offline eval (Unity gerektirmez):**
- HOT3D val sequence'ları: temporal sekanslar → geodesic error, jitter, contact ratio
- OakInk val: statik grasp pozları → geodesic error, contact ratio, penetration

**Unity eval — üç ayrı amaç, ayrı obje setleri:**

| Unity Kullanımı | Obje Seti | Amaç |
|---|---|---|
| Confidence calibration | HOT3D calibration 3 obje | `success_label` üretimi → `success_prob` eğitimi |
| Final temporal test | HOT3D held-out 4 obje | Nihai fizik başarı metriği — tek seferlik |
| Static geometry test (isteğe bağlı) | OakInk test setinden kategori-stratified ~20–30 obje | Statik modelin fizik genellemesi |

---

## A8. Koordinat Uyuşmazlık Analizi ve Giderilen Sorunlar

> **Kaynak:** `docs/claude_coordinate_analysis.md` ve `docs/chatgpt_coordinate_analysis.md` — detaylı analiz ve ham veri doğrulaması bu belgelerde.

### Tespit Edilen Uyuşmazlıklar

#### 1. mano_fk.py FK Convention ve Kemik Geometrisi Hatası (Tamamen Giderildi)

**Sorun (iki katmanlı):**

*Katman 1 — quality_label:* `compute_quality_label`, `mano_fk.py`'nin simplified FK'sını kullanıyordu. Bu FK Y eksenini parmak yönü olarak varsayıyordu; gerçek MANO modelinde ise parmaklar −X yönünde uzanır. Sonuç: fingertip pozisyonları gerçek MANO'dan **18–20 cm** sapıyordu → `quality_label = 0.0` tüm örneklerde.

*Katman 2 — L_contact / L_penetration:* `contact_penetration_loss` da aynı simplified FK'yı kullanıyordu. Parmak uçları `obj_pts_contact`'tan 18–20 cm uzakta hesaplandığından L_contact ve L_penetration loss'ları eğitim boyunca geometrik olarak kör çalışıyordu.

**Giderilme — İki ayrı müdahale:**

*quality_label için (ara fix):*
- `compute_quality_label`'a `fingertips_wrist` opsiyonel parametresi eklendi (`quality_labels.py`).
- HOT3D dataset: `fk_joints` (NPZ'de saklanan gerçek MANO world-frame joints) → `(fk_world - wrist_t) @ R_wrist` → wrist frame → override olarak geçirildi (`dataset_hot3d.py`).
- OakInk dataset: `hand_j` parmak uçları `cam_extr⁻¹` ile world frame'e taşınıp `fingertips_world (N,5,3)` olarak `dataset.npz`'ye kaydedildi → wrist frame'e taşınıp override olarak geçirildi (`dataset_oakink.py`, `build_oakink_canonical.py`).

*mano_fk.py tamamen yeniden yazıldı (L_contact + L_penetration için kalıcı fix):*
- Eski: hardcoded 5 parmak × Y-ekseninde kemik geometrisi, rasgele MCP offset'leri.
- Yeni: `src/model/mano_fk.py` gerçek MANO kinematic tree'yi (`parents = [-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 0, 10, 11, 0, 13, 14]`) ve zero-beta template joint pozisyonlarını (`_J_TEMPLATE`, `utils/mano_right.ManoRight`'tan türetildi) kullanarak differentiable 4×4 matris zinciri olarak yeniden implement edildi.
- FK hatası: **18–20 cm → 0.9 cm** (HOT3D stored `fk_joints` ile doğrulandı, 430 frame).
- Kalan ~0.9 cm hata: betas=0 (template) ile gerçek subject betas arasındaki fark — eğitimde kabul edilebilir.
- `contact_penetration_loss` hinge eşiği: `0.040 m → 0.015 m` — eski 40mm simplified FK'nın ~50mm hatasını maskeliyordu; yeni FK ~9mm hata yaptığından 15mm yeterli.

**Sonuç:**

| Metrik | Öncesi | Sonrası |
|---|---|---|
| FK fingertip hatası | 18–20 cm | 0.9 cm |
| quality_label mean (HOT3D) | 0.000 | 0.065 |
| quality_label >0 (HOT3D) | %0 | %18 |
| quality_label mean (OakInk) | 0.000 | 0.101 |
| quality_label >0 (OakInk) | %0 | %26 |
| L_contact mean (HOT3D) | kör (~sabit) | 0.067 |
| L_penetration mean | kör (~0) | 0.0002 |

#### 2. Contact Eşiği Uyumsuzluğu (Giderildi)

**Sorun:** `CONTACT_THRESHOLD_M = 0.005` (5 mm) — HOT3D build scriptinin kullandığı 3 cm AABB eşiğiyle uyumsuzdu. Gerçek MANO FK ile bile temas olan framelerde quality_label = 0 çıkıyordu.

**Giderilme:** `CONTACT_THRESHOLD_M = 0.030` (30 mm) olarak güncellendi (`model_io.py`).

#### 3. OakInk dataset.npz'de obj_anno Eksikliği (Giderildi)

**Sorun:** Eski `build_oakink_canonical.py` versiyonu `obj_anno`'yu `savez()` çağrısına eklemiyordu. `dataset_oakink.py` fallback olarak zero array döndürüyordu → contact/penetration loss anlamsız.

**Giderilme:** Mevcut `build_oakink_canonical.py` `obj_anno=obj_anno_arr` parametresini doğrudan `savez()` çağrısında içeriyor. `dataset.npz` yeniden üretildi — `obj_anno (11151, 12)` kayıtlı.

#### 4. mano_fk.py Autograd Inplace Op Hatası (Giderildi)

**Sorun:** `finger_aa45_to_joint_positions`'daki 4×4 transform zinciri (`G[..., i, :, :] = ...` ve `T[..., :3, :3] = ...`) inplace tensor atamaları kullanıyordu. Bu, autograd hesaplama grafiğini bozuyordu → Phase 2 `loss.backward()` sırasında `RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation` hatası.

**Giderilme:** `_make_T` yardımcı fonksiyonu eklendi — `torch.cat` ile out-of-place 4×4 transform üretir. G zinciri Python listesi olarak biriktirilip `torch.stack` ile birleştirildi; hiçbir inplace atama yok. FK doğruluğu değişmedi (mean=0.9 cm, max=1.4 cm).

#### 5. Fingertip Pozisyon Loss Eksikliği (Giderildi)

**Sorun:** `grasp_loss` yalnızca `L_recon` (45 boyutlu açı MSE) kullanıyordu. Küçük açı hataları kinematik zincirde birikip parmak ucunda büyük pozisyon hatasına dönüşebilir — model "açılar yakın" derken parmak ucu 2-3 cm yanlış yerde olabilir.

**Giderilme:** `grasp_model.py`'ye `L_tip = MSE(FK(pred), FK(gt))` eklendi. `fingertip_positions(pred)` ve `fingertip_positions(target)` wrist frame'de (5,3) parmak ucu pozisyonu üretir; bunların MSE'si `tip_weight=0.5` ağırlığıyla total loss'a eklenir. `train_grasp.py`'ye `--tip_weight` argümanı eklendi.

**Doğrulama:** Phase 1 `tip=0.0007`, Phase 2 `tip=0.0008`, backward temiz.

#### 6. Phase 2 Temporal Loss Aktif Değildi (Giderildi)

**Sorun:** `train_grasp.py` içinde `grasp_loss(... prev_pred_pose=None, prev2_pred_pose=None)` hardcoded geçiliyordu. `vel_weight` ve `acc_weight` Phase 2 için set edilse de `grasp_loss` içindeki `if prev_pred_pose is not None` koşulu hiç girilmiyordu → `L_vel` ve `L_acc` sıfır, Phase 2 pratikte Phase 1'in tekrarı.

**Giderilme:**
- `dataset_hot3d.py`: `prev_frame_feat` (1 frame geri) ve `prev2_frame_feat` (2 frame geri) batch'e eklendi. Sınır durumları için ilk frame tekrarlanarak padding uygulanıyor.
- `train_grasp.py`: ana forward pass'ten **önce** (inplace op çakışmasını önlemek için) `no_grad` içinde `prev_frame_feat` ve `prev2_frame_feat` üzerinden forward pass yapılıyor; `pred_{t-1}` ve `pred_{t-2}` `.detach()` ile `prev_pred_pose`/`prev2_pred_pose` olarak `grasp_loss`'a geçiliyor.
- Gradient yalnızca `pred_t` (ana forward pass) üzerinden akıyor.

**Doğrulama:**
```
Phase 1: vel=—  acc=—  (kapalı, doğru)
Phase 2: vel=0.0096  acc=0.0398  backward=OK
```

#### 7. HOT3D obj_pts Eksikliği (Giderildi)

**Sorun:** `build_hot3d_canonical_full.py` seq_*.npz dosyalarını üretiyordu ama `obj_pts/*.npy` üretmiyordu. Dataset loader obj_pts olmadan çalışamıyordu.

**Giderilme:** 27 HOT3D objesinin `.glb` mesh'lerinden `trimesh` ile 1024 nokta örneklendi, `data/processed/hot3d_canonical/obj_pts/` altına kaydedildi.

**Yeniden oluşturmak için:**
```bash
python - <<'EOF'
import trimesh, numpy as np
from pathlib import Path

ASSETS = Path("data/raw/hot3d/assets")   # .glb dosyaları burada
OUT    = Path("data/processed/hot3d_canonical/obj_pts")
OUT.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(0)
for glb in sorted(ASSETS.rglob("*.glb")):
    name = glb.stem
    try:
        scene = trimesh.load(str(glb), force="scene")
        geoms = list(scene.geometry.values()) if hasattr(scene, "geometry") else [scene]
        pts = np.concatenate([np.array(g.vertices, dtype=np.float32) for g in geoms if hasattr(g, "vertices") and len(g.vertices) > 0], axis=0)
        idx = rng.choice(len(pts), 1024, replace=(len(pts) < 1024))
        np.save(OUT / f"{name}.npy", pts[idx])
        print(f"  {name}: {len(pts)} verts → 1024")
    except Exception as e:
        print(f"  SKIP {name}: {e}")
EOF
```

#### 8. HOT3D dist Alan Semantiği Uyuşmazlığı (Giderildi)

**Sorun:** `build_hot3d_canonical_full.py`'deki `dist` alanı, obje AABB'nin bilek noktasına en yakın yüzeyine olan mesafeyi hesaplıyordu (object-local frame → world). OakInk'teki `dist` ise `trimesh.proximity.closest_point` ile Euclidean nearest-surface mesafesiydi. İki dataset'in `dist` dağılımları farklı semantiğe sahipti; normalizasyon sonrası |Δmean| = 0.654 std idi.

**Giderilme:** Aşağıdaki script tüm 157 seq_*.npz dosyasındaki `dist` alanını yeniden hesapladı ve `stats.json`'u güncelledi:

```bash
python - <<'EOF'
import numpy as np, json, trimesh
from pathlib import Path

HOT3D_CANON = Path("data/processed/hot3d_canonical")
OBJ_PTS_DIR = HOT3D_CANON / "obj_pts"

paths = sorted(HOT3D_CANON.glob("seq_*.npz"))
print(f"{len(paths)} sequence düzeltiliyor...")
for i, p in enumerate(paths):
    if i % 30 == 0: print(f"  {i}/{len(paths)} done")
    d = dict(np.load(p, allow_pickle=True))
    obj_name = str(d["obj_name"][0])
    pts_path = OBJ_PTS_DIR / f"{obj_name}.npy"
    if not pts_path.exists():
        continue
    obj_pts = np.load(pts_path)                  # (1024, 3) object-local frame
    obj_t = d["obj_world_t"].astype(np.float32)  # (F, 3)
    obj_q = d["obj_world_q"].astype(np.float32)  # (F, 4) wxyz
    wrist  = d["wrist_world_t"].astype(np.float32)  # (F, 3)
    F = len(obj_t)
    new_dist = np.zeros((F, 1), dtype=np.float32)
    from scipy.spatial.transform import Rotation
    for f in range(F):
        R = Rotation.from_quat([obj_q[f,1], obj_q[f,2], obj_q[f,3], obj_q[f,0]]).as_matrix()
        pts_world = (obj_pts @ R.T) + obj_t[f]
        diffs = pts_world - wrist[f]
        new_dist[f, 0] = float(np.min(np.linalg.norm(diffs, axis=1)))
    d["dist"] = new_dist
    np.savez_compressed(p, **d)

# stats güncelle — train split obj'lerinden
obj_split = json.load(open(HOT3D_CANON / "obj_split.json"))
train_objs = set(obj_split.get("train", []))
import csv
manifest = list(csv.DictReader(open(HOT3D_CANON / "manifest.csv")))
all_frame_feats = []
for row in manifest:
    if row["split"] != "train": continue
    npz = np.load(HOT3D_CANON / f"seq_{row['seq_id']}.npz", allow_pickle=True)
    ff = np.concatenate([npz["rel_pos"], npz["rel_rot6d"], npz["rel_vel"], npz["dist"]], axis=1)
    all_frame_feats.append(ff)
ff_all = np.concatenate(all_frame_feats, axis=0)
stats = json.load(open(HOT3D_CANON / "stats.json"))
stats["input_mean"] = ff_all.mean(0).tolist()
stats["input_std"]  = np.maximum(ff_all.std(0), 1e-6).tolist()
stats["dist_note"]  = "Euclidean nearest-surface distance (trimesh proximity), not AABB"
json.dump(stats, open(HOT3D_CANON / "stats.json", "w"), indent=2)
print(f"stats.json güncellendi — dist mean={stats['input_mean'][12]:.3f} std={stats['input_std'][12]:.3f}")
EOF
```

**Sonuç:** dist dağılımı: mean=0.175 std=0.108 (Euclidean, metre). OakInk ile semantik uyuşmazlık giderildi; normalizasyon sonrası |Δmean| = 0.654 std → 0.375 std.

### Hâlâ Açık Olan Sıkıntılar

| Sıkıntı | Dataset | Risk | Açıklama |
|---|---|---|---|
| Mesh-instance uyuşmazlığı | OakInk | Orta | `obj_pts` OakBase genel kategori mesh'inden; grasped edilen spesifik instance farklı boyut/şekle sahip olabilir. quality_label gürültülü. |
| OakInk FK residual ~3 cm | OakInk | Düşük-Orta | Model FK (zero-beta template DIP proxy) ile GT `fingertips_world` (gerçek subject betas, mesh vertex ucu) arasında ~3.28 cm kalıcı fark var. quality_label GT ile hesaplandığı için doğru; contact loss model FK'dan hesaplandığı için bu kadar gürültü taşıyor. |
| Grasp segmentlerinde yaklaşım fazı fazlalığı | HOT3D | Düşük-Orta | 3 cm AABB eşiği geniş; frame'lerin %79'unda quality_label = 0 (yaklaşım fazı). |
| HOT3D val/test split | HOT3D | ✅ Giderildi | Obje bazlı frame-level split uygulandı. `obj_split.json` + loader değişikliğiyle: train 11 obje (~168k frame), val 2 obje (~77k frame), test 4 obje (~52k frame). |
| OakInk split hatası | OakInk | ✅ Giderildi | Per-kategori stratification → global 70/15/15. 4 obje unseen_test, 4 val, 17 train. Eski kod tek objeli kategorilerin hepsini test'e atıyordu. |
| HOT3D dist semantik uyuşmazlığı | HOT3D | ✅ Giderildi | AABB mesafesinden Euclidean nearest-surface mesafesine dönüştürüldü. 157 NPZ yeniden hesaplandı, stats.json güncellendi. |
| HOT3D obj_pts normalizasyon eksikliği | HOT3D | ✅ Giderildi | stats.json'da `pts_mean/pts_std` yoktu → ham metre kalıyordu. `recompute_normalization_stats.py` ile eklendi: mean≈0, std≈[0.044, 0.045, 0.036]. OakInk std≈[0.029, 0.023, 0.049] ile benzer ölçek, normalize sonrası her ikisi de std=1. |
| Penetration proxy (tasarım kararı) | Her ikisi | Kabul edildi | Centroid-proxy eğitim sinyali olarak bırakıldı (`penetration_weight=0.1`, düşürüldü). Gerçek ölçüm Unity PhysX'ten gelir. Tezde açıkça belirtilmeli. |

### Mevcut Processed Veri Durumu

| Veri | Konum | İçerik | Durum |
|---|---|---|---|
| HOT3D seq'ler | `data/processed/hot3d_canonical/seq_*.npz` | 297k frame, 4113 segment | ✓ Üretildi |
| HOT3D obj_pts | `data/processed/hot3d_canonical/obj_pts/` | 27 obje × 1024 nokta | ✓ Üretildi |
| HOT3D istatistikler | `data/processed/hot3d_canonical/stats.json` | Normalizasyon mean/std | ✓ Üretildi |
| OakInk dataset | `data/processed/oakink_canonical/dataset.npz` | 11151 sample, obj_anno + fingertips_world dahil | ✓ Üretildi |
| OakInk obj_pts | `data/processed/oakink_canonical/obj_pts/` | 25 kategori × 1024 nokta | ✓ Üretildi |
| OakInk split/stats | `data/processed/oakink_canonical/split.json`, `stats.json` | 70/15/15 obje bazlı — 11151 sample, 17/4/4 obje | ✓ Üretildi |

### Processed Veriyi Sıfırdan Yeniden Oluşturma

Aşağıdaki sırayla çalıştırıldığında `data/processed/` dizini mevcut haline gelir:

```bash
# 1. OakInk canonical dataset (dataset.npz + split.json + stats.json + obj_pts/)
python src/preprocessing/build_oakink_canonical.py

# 2. HOT3D canonical sequences (seq_*.npz + stats.json)
python src/preprocessing/build_hot3d_canonical_full.py --build

# 3. HOT3D obj_pts (GLB → 1024-pt point cloud, obj_pts/*.npy)
#    build_hot3d_canonical_full.py bunu üretmez — ayrı script gerekir.
#    A8 item 7'deki inline scripti çalıştır.

# 4. HOT3D dist alanını AABB'den Euclidean nearest-surface'e dönüştür
#    A8 item 8'deki inline scripti çalıştır.
#    Bu adım stats.json'u da günceller (dist_note dahil).

# 5. HOT3D obje split + manifest
python src/preprocessing/make_hot3d_split.py

# 6. Normalizasyon istatistiklerini hesapla/güncelle
#    OakInk input_mean/std (frame_feat) ve HOT3D pts_mean/pts_std (obj_pts PointNet normalizasyonu)
python src/preprocessing/recompute_normalization_stats.py
```

**Adım 6 zorunludur** — HOT3D stats.json başlangıçta `pts_mean/pts_std` içermez (A8 item 9). Bu adım olmadan HOT3D `obj_pts` normalize edilmez ve OakInk ile dağılım uyuşmazlığı oluşur.

**Bağımlılıklar:**
- Ham OakInk verisi: `data/raw/oakink/` (anno/, OakBase/, shape/)
- Ham HOT3D verisi: `data/raw/hot3d/` (sequences/ ve assets/ → .glb mesh'leri)
- MANO sağ el modeli: `utils/mano_right.pkl`
- Python paketleri: `trimesh`, `scipy`, `numpy`, `torch`

### .gitignore Notu

`src/data/` dizini (dataset loader kodları) `.gitignore:23`'teki `data/` satırı yüzünden yanlışlıkla git tarafından izlenmiyor. Bu satır `/data/` (kök-göreli) olarak düzeltilmeli; aksi hâlde `dataset_hot3d.py`, `dataset_oakink.py` ve ilgili tüm kod commit edilemiyor.
