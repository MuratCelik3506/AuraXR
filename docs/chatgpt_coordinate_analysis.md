# ChatGPT Coordinate Analysis

Bu dokuman, `data/raw` altindaki veri kaynaklarinin koordinat sistemi, olcek, obje noktasi ve el pozu acisindan durumunu ozetler. Amac, contact ve penetration loss/metriklerinin hangi durumda fiziksel olarak anlamli, hangi durumda riskli oldugunu netlestirmektir.

## Ana Problem

Contact ve penetration hesabi ancak el noktalarinin ve obje noktalarinin ayni 3D uzayda olmasi durumunda anlamlidir.

Riskli hesap:

```text
cdist(fingertip_points, obj_pts)
```

Bu hesap su durumda dogrudur:

```text
fingertip_points ve obj_pts ayni frame'de
fingertip_points ve obj_pts ayni olcekte
fingertip_points ve obj_pts ayni zaman frame'ine ait
```

Bu hesap su durumda yanlistir:

```text
fingertip_points wrist frame'de
obj_pts object-local / canonical / normalized frame'de
```

Bu yanlis durumda contact ratio dusuk, penetration yuksek veya mesafe degerleri anlamsiz cikabilir. Bu, modelin gercekten kotu grasp urettigini kanitlamaz; once metriklerin dogru fiziksel uzayda hesaplandigi kanitlanmalidir.

## Terimler

`world frame`: Sahnenin ortak koordinat sistemi. El, obje ve kamera pozlari bu uzayda ifade edilebilir.

`wrist frame`: El bilegini merkez alan lokal koordinat sistemi. FK ile uretilen parmak noktalarinin bu frame'de olmasi yaygindir.

`object frame`: Objeyi merkez alan lokal koordinat sistemi. Obje mesh veya point cloud noktalarinin dogal frame'i genellikle budur.

`canonical frame`: Dataset veya preprocessing tarafindan standartlastirilmis obje/el uzayi. Obje merkeze alinmis, dondurulmus veya normalize edilmis olabilir.

`normalized`: Koordinatlarin gercek metre/mm olceginden cikarilip standart bir araliga tasinmasi.

`obj_pts`: Obje yuzeyinden orneklenmis 3D point cloud.

`FK`: Forward kinematics. Eklem rotasyonlarindan parmak ucu ve eklem pozisyonlarini hesaplama islemi.

`wrist_tsl`: Bilegin translation bilgisi. Genellikle bilegin world veya object-relative konumunu verir.

## Dataset Durum Ozeti

| Veri | Ham frame durumu | Olcek | Temporal | Contact icin durum |
|---|---|---:|---:|---|
| HOT3D | El world pose + obje world pose + mesh object-local | metre | evet | Uygun, transform sart |
| OakInk | El/world bilgisi + `obj_anno` + mesh/PLY local/canonical | metre gibi | statik | Ham veri uygun, mevcut processed veri eksik |

## HOT3D

### Ham Veri

Ilgili dosyalar:

```text
data/raw/hot3d/quest3/.../*hand_data.zip
data/raw/hot3d/quest3/.../*ground_truth.zip
data/raw/hot3d/assets/Hot3DAssets_assets_assets/*.glb
data/raw/hot3d/assets/Hot3DAssets_assets_assets/instance.json
```

`mano_hand_pose_trajectory.jsonl` icinde gorulen alanlar:

```text
timestamp_ns
hand_poses
pose: 15 PCA MANO finger pose
wrist_xform.t_xyz: wrist world translation, metre
wrist_xform.q_wxyz: wrist world rotation
betas: MANO shape
```

`dynamic_objects.csv` icinde gorulen alanlar:

```text
object_uid
timestamp[ns]
t_wo_x[m], t_wo_y[m], t_wo_z[m]: object -> world translation
q_wo_w, q_wo_x, q_wo_y, q_wo_z: object -> world rotation
```

`*.glb` dosyalari:

```text
obje mesh'i object-local frame'de
```

### Durum

HOT3D ham verisi contact/penetration hesabi icin gerekli bilgiyi tasiyor. Ancak obje mesh noktalari dogrudan parmak FK noktalariyla karsilastirilamaz.

Dogru zincir:

```text
obje mesh noktasi object-local frame'de baslar
object pose ile world frame'e tasinir
el FK noktasi wrist/world zinciriyle ayni frame'e tasinir
sonra mesafe/contact/penetration hesaplanir
```

Yanlis zincir:

```text
FK fingertip wrist frame'de
obj_pts GLB object-local frame'de
cdist(fingertip, obj_pts)
```

Bu yanlis durumda mesafe fiziksel mesafe degildir.

### Mevcut Processed Durum

`data/processed/hot3d_canonical/seq_*.npz` icinde su alanlar var:

```text
rel_pos
rel_rot6d
rel_vel
dist
finger_aa45
wrist_world_t
wrist_world_q
obj_world_t
obj_world_q
obj_name
contact_flag
```

`data/processed/hot3d_canonical/obj_pts/*.npy` dosyalari da mevcut. Ornek point cloud degerleri metre olceginde gorunuyor.

Bu iyi bir durumdur. HOT3D icin dogru contact hesabi su sekilde yapilabilir:

```text
obj_pts object-local
-> obj_world_t / obj_world_q ile world frame
-> wrist_world_t / wrist_world_q ile wrist frame
-> FK fingertip noktalarinin frame'iyle karsilastir
```

Sonuc: HOT3D ham ve processed veri contact duzeltmesine uygundur.

## OakInk

### Ham Veri

Ilgili dosyalar:

```text
data/raw/oakink/anno/general_info/*.pkl
data/raw/oakink/anno/hand_j/*.pkl
data/raw/oakink/OakBase/OakBase/*/*/*.ply
data/raw/oakink/shape/OakInkObjects/OakInkObjectsV2/*/align/*.json
data/raw/oakink/shape/OakInkVirtualObjects/OakInkVirtualObjectsV2/*/align/*.json
```

Ornek `general_info` PKL icinde gorulen alanlar:

```text
hand_anno.hand_tsl: (3,) wrist/world translation
hand_anno.hand_shape: (10,) MANO shape
hand_anno.hand_pose: (16,4) quaternion MANO pose
obj_anno: (4,4) object transform
cam_extr: (4,4)
cam_intr: (3,3)
```

`hand_j` dosyalarinda:

```text
(21,3) MANO joint positions
```

OakBase tarafinda:

```text
PLY mesh/point cloud parcalari object-local veya dataset-local frame'de
```

Shape alignment JSON dosyalarinda:

```text
scale matrix
rot matrix
```

### Durum

OakInk ham verisi dogru contact/penetration hesabi icin gerekli bilgiyi tasiyor. Ama mesh/point cloud noktalarini dogrudan el noktalarina karsilastirmak yanlis olur.

Dogru zincir:

```text
OakBase/OakInk shape point cloud
-> gerekiyorsa align scale/rot uygula
-> obj_anno ile world frame'e tasi
-> hand_tsl ve wrist/global orient ile wrist frame'e tasi
-> FK fingertip ile karsilastir
```

Yanlis zincir:

```text
hand_tsl world frame'de
obj_pts OakBase local/canonical frame'de
cdist(fingertip, obj_pts)
```

Bu yanlis durumda OakInk penetration degerleri yapay olarak yuksek, contact ratio yapay olarak dusuk cikabilir.

### Mevcut Processed Durum

`data/processed/oakink_canonical/dataset.npz` icinde gorulen alanlar:

```text
pose
shape
tsl
category
obj_name
```

Beklenen ama mevcut dosyada olmayan kritik alan:

```text
obj_anno
```

Bu onemli bir bulgu. Guncel `src/data/dataset_oakink.py`, `obj_anno` varsa `obj_pts_contact` uretip obje noktalarini wrist frame'e tasiyor. Fakat mevcut processed `dataset.npz` icinde `obj_anno` olmadigi icin OakInk tarafinda contact/penetration hesabi guvenilir olmayabilir.

Sonuc:

```text
OakInk ham veri uygun.
Mevcut processed OakInk veri eksik.
OakInk canonical yeniden uretilmeli.
```

## Senaryo Bazli Risk Listesi

### Senaryo 1: Frame Uyusmazligi

Durum:

```text
fingertip wrist frame'de
obj_pts object/canonical frame'de
```

Etkisi:

```text
contact ratio anlamsiz dusuk cikabilir
penetration anlamsiz yuksek cikabilir
quality label bozulur
```

Etkilenen veriler:

```text
OakInk: mevcut processed veri icin yuksek risk
HOT3D: transform uygulanmazsa riskli, ama processed veri gerekli alanlari tasiyor
```

### Senaryo 2: Olcek Uyusmazligi

Durum:

```text
el metre cinsinden
obje normalize veya mm cinsinden
```

Etkisi:

```text
mm seviyesindeki contact threshold metre gibi yorumlanabilir
3 cm threshold anlamsiz hale gelebilir
```

Etkilenen veriler:

```text
HOT3D: metre olarak iyi
OakInk: metre gibi gorunuyor ama align/scale dikkat ister
```

### Senaryo 3: Object Mesh Local Kalmasi

Durum:

```text
obj_pts mesh local frame'de kalir
object pose uygulanmaz
```

Etkisi:

```text
obje sahnedeki gercek yerine tasinmadigi icin parmak-obje mesafesi yanlis olur
```

Etkilenen veriler:

```text
HOT3D GLB dosyalari
OakInk PLY/OBJ dosyalari
```

### Senaryo 4: Temporal Frame Uyusmazligi

Durum:

```text
el frame t'den
obje pose frame t-1 veya baska timestamp'ten
```

Etkisi:

```text
el-obje mesafesi hareketli sahnede yanlis olur
temporal contact sinyali bozulur
```

Etkilenen veriler:

```text
HOT3D: timestamp_ns ile eslestirme sart
OakInk: statik oldugu icin temporal risk dusuk
```

## En Kritik Bulgu

HOT3D tarafinda contact/penetration duzeltmesi icin gerekli alanlar hem ham veride hem mevcut processed veride buyuk olcude var.

OakInk tarafinda ham veri gerekli `obj_anno` bilgisini tasiyor, fakat mevcut `data/processed/oakink_canonical/dataset.npz` bu alani icermiyor. Bu nedenle OakInk contact/penetration sonuclarina mevcut processed veriyle guvenilmemeli.

## Onerilen Sonraki Adimlar

1. OakInk canonical veriyi yeniden uret.

```text
dataset.npz icinde obj_anno bulunmali
obj_anno = R_obj(9) + t_obj(3)
```

2. OakInk loader'da `obj_pts_contact` alaninin sifir sentinel degil gercek wrist-frame point cloud oldugunu dogrula.

3. HOT3D icin bir sample secip su gorsel audit'i yap:

```text
obj_pts object-local -> world -> wrist
FK fingertip wrist
aynı 3D sahnede ciz
en yakin mesafeyi mm olarak raporla
```

4. OakInk icin ayni audit'i yap.

5. Contact/penetration metriklerini sadece `obj_pts_contact` uzerinden hesapla.

6. `obj_pts` ve `obj_pts_contact` rollerini ayir:

```text
obj_pts: modelin PointNet girdisi, normalize/canonical olabilir
obj_pts_contact: loss/metrik girdisi, wrist frame + metre olcegi olmali
```

## Kisa Sonuc

Mevcut durumda en guvenilir yorum:

```text
HOT3D: dogru frame bilgisi var, contact duzeltmesi uygulanabilir.
OakInk: ham veri dogru bilgi tasiyor, ama processed veri eski/eksik.
```

Bu nedenle once OakInk processed veri yeniden uretilmeli ve `obj_pts_contact` gorsel olarak dogrulanmalidir. Bundan sonra contact ratio, penetration depth, quality label ve CVAE aday secimi anlamli hale gelir.
