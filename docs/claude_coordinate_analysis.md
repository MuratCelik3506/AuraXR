# AuraXR Veri Koordinat Sistemi ve Dönüşüm Analiz Raporu (Güncellenmiş)

İlgili risk değerlendirmesi: [docs/E-iki-asamali-egitim-oneri-degerlendirmesi.md](E-iki-asamali-egitim-oneri-degerlendirmesi.md) (satır 37 civarı, Öncelik 1).

> **Not (2026-06-30):** Bu rapor, önceki sürümün ve `docs/chatgpt_coordinate_analysis.md` raporunun ham veri üzerinde tekrar doğrulanmasıyla güncellendi. Önceki sürümde OakInk hakkında verilen bir bulgu (`obj_anno`'nun `savez()` çağrısına eklenmediği iddiası) **güncel kodda doğru değildir** — aşağıda düzeltildi. Ayrıca her iki dataset için işlenmiş (`data/processed/*`) çıktıların şu anda diskte **hiç bulunmadığı** doğrulandı.

## Ana Problem

Contact ve penetration hesabı ancak el noktalarının ve obje noktalarının aynı 3D uzayda (frame), aynı ölçekte (scale) ve aynı zaman anında olması durumunda fiziksel olarak anlamlıdır:

```text
cdist(fingertip_points, obj_pts)
```

Bu hesap yalnızca `fingertip_points` ve `obj_pts` aynı frame + aynı ölçek + aynı zaman dilimindeyse doğrudur. Aksi halde contact ratio yapay düşük, penetration yapay yüksek çıkabilir ve bu, modelin gerçekten kötü grasp ürettiğini kanıtlamaz.

## Terimler

- **world frame**: Sahnenin ortak koordinat sistemi.
- **wrist frame**: El bileğini merkez alan lokal koordinat sistemi.
- **object/canonical frame**: Objeyi merkez alan, dataset tarafından standartlaştırılmış lokal uzay.
- **obj_pts**: Obje yüzeyinden örneklenmiş 3D point cloud.
- **obj_pts_contact**: `obj_pts`'in wrist frame'e taşınmış, contact/penetration loss için kullanılan hali (loss girdisi `obj_pts`'ten kavramsal olarak ayrı tutulmalı — `obj_pts` PointNet girdisi olarak canonical kalabilir, `obj_pts_contact` mutlaka wrist frame + metre olmalı).
- **FK**: Forward kinematics — eklem rotasyonlarından parmak ucu pozisyonu hesaplama.
- **wrist_tsl / wrist_world_t**: Bileğin world/object-relative translation bilgisi.

## Dataset Durum Özeti

| Veri | Ham frame durumu | Ölçek | Temporal | Kod durumu | Processed veri var mı |
|---|---|---:|---:|---|---|
| HOT3D | El world pose (per-frame) + obje world pose (per-frame) + mesh object-local | metre | evet | Loader dönüşümü doğru uyguluyor | **Hayır — diskte yok** |
| OakInk | El tsl/pose + `obj_anno` (4×4, per-sample) + mesh object-local | metre | hayır (statik) | Build + loader dönüşümü doğru uyguluyor (önceki rapordaki bug artık geçersiz) | **Hayır — diskte yok** |

## HOT3D

### Ham Veri (doğrudan dosyalardan doğrulandı)

```text
data/raw/hot3d/quest3/{train,test}/<seq>/*hand_data.zip
  -> mano_hand_pose_trajectory.jsonl
  -> umetrack_hand_pose_trajectory.jsonl
data/raw/hot3d/quest3/{train,test}/<seq>/*ground_truth.zip
  -> dynamic_objects.csv
  -> metadata.json
data/raw/hot3d/assets/Hot3DAssets_assets_assets/*.glb
data/raw/hot3d/assets/Hot3DAssets_assets_assets/instance.json
```

**Önemli ve önceki raporlarda atlanan bir detay:** `test` split'indeki tüm `hand_data.zip` dosyalarında `mano_hand_pose_trajectory.jsonl` **boş** (`{"timestamp_ns": ..., "hand_poses": {}}`, satır boyu 0 bayt veri). Gerçek el pozu verisi yalnızca **`train` split**'te mevcut. Örnek (`P0002_1464cbdc`, train):

```json
{"timestamp_ns": 44994166666666, "hand_poses": {
  "0": {"pose": [15 değer], "wrist_xform": {"t_xyz": [0.0149, 0.4976, -0.1989], "q_wxyz": [...]}, "betas": [...]},
  "1": {...}
}}
```

`wrist_xform.t_xyz` doğrudan **metre cinsinden, world frame'de, per-timestamp** wrist translation'dır.

`dynamic_objects.csv` (aynı sequence, ground_truth.zip içinden):

```text
object_uid,timestamp[ns],t_wo_x[m],t_wo_y[m],t_wo_z[m],q_wo_w,q_wo_x,q_wo_y,q_wo_z
228358276546933,44993900000000,0.4118,0.4250,-0.4413,0.7954,-0.0005,-0.6060,0.0130
```

Bu da **per-frame, per-object, metre cinsinden, world frame** object→world transform'dur. `object_uid` değeri `instance.json`'daki `instance_id` ile eşleşir (örn. `106434519822892` → `bottle_bbq`), yani GLB mesh'leri object-local frame'de saklanır ve bu transform ile world'e taşınır. `metadata.json` içindeki `object_names`/`object_uids` listesi sequence başına hangi objelerin sahnede olduğunu doğrular.

**Sonuç:** Ham veri hem el hem obje için **per-frame, metre, world-frame** dönüşüm bilgisini tam olarak taşıyor. Tek risk: test split'te hand pose verisi yok — bu split contact/penetration eğitimi için kullanılamaz, sadece train split kullanılabilir.

### Kod Tarafı

`src/data/dataset_hot3d.py:157-176` (`_obj_pts_contact`) zinciri uyguluyor:

```text
pts_world  = raw_obj_pts @ R_obj.T + obj_world_t      # canonical/object-local -> world
pts_wrist  = (pts_world - wrist_world_t) @ R_wrist     # world -> wrist
```

FK ile üretilen `target_pose`/fingertip çıktısı da wrist frame'de olduğundan, `cdist(fingertip, obj_pts_contact)` aynı frame + aynı ölçekte çalışıyor. **Kod seviyesinde tutarlı.**

### Processed Veri — Durum Değişti

Önceki rapor `data/processed/hot3d_canonical/seq_*.npz` ve `obj_pts/*.npy` dosyalarının var olduğunu ve değerlerinin makul (metre ölçeğinde) olduğunu doğrulamıştı. **Şu an** `data/processed/` klasöründe yalnızca `.DS_Store` var — `hot3d_canonical/` dizini **diskte mevcut değil**. Bu, görüşme başındaki git status'ta görülen büyük temizlik/silme işlemiyle (checkpoints, results, vb. birçok dosyanın `D` olarak işaretlenmesi) tutarlı. Üretim scripti `src/preprocessing/build_hot3d_canonical_full.py` diskte mevcut ve git'e **eklenmemiş** (untracked), yani pipeline yeniden çalıştırılabilir durumda ama henüz çalıştırılmamış/çıktısı silinmiş.

## OakInk

### Ham Veri (doğrudan dosyalardan doğrulandı)

```text
data/raw/oakink/anno/general_info/*.pkl
data/raw/oakink/anno/hand_j/*.pkl
data/raw/oakink/OakBase/OakBase/<category>/<instance>/part_*.ply
data/raw/oakink/shape/OakInkObjects*/.../align/*.json   (kullanılmıyor — aşağıya bakın)
```

`general_info/*.pkl` örneği (`pickle.load`):

```text
hand_anno.hand_tsl   : (3,)    wrist/world translation, torch.Tensor
hand_anno.hand_shape : (10,)   MANO shape
hand_anno.hand_pose  : (16,4)  quaternion MANO pose
obj_anno             : (4,4)   object-to-world rigid transform, torch.Tensor
cam_extr             : (4,4)
cam_intr             : (3,3)
```

Gerçek `obj_anno` örneği (translation sütunu ~2-13 cm aralığında, makul metre ölçeği):

```text
[[ 0.0974, -0.7647,  0.6370,  0.0232],
 [ 0.8237,  0.4212,  0.3797,  0.1300],
 [-0.5587,  0.4877,  0.6709,  0.0498],
 [ 0.0000,  0.0000,  0.0000,  1.0000]]
```

`hand_j/*.pkl` (21×3 MANO joint pozisyonları) örnek değerleri `[0.183, -0.033, 1.036]` gibi — sıfır merkezli değil, büyüklük kamera/world frame ile tutarlı (yaklaşık 1m derinlik), **normalize edilmiş [-1,1] değil**.

`OakBase/*/part_*.ply` (binary PLY, manuel `struct` ile okundu, `stanley_pincer/part_01.ply`): X/Y/Z aralığı yaklaşık `[-0.004, 0.084]` metre — birkaç santimetrelik gerçek obje parçası ölçeğinde, **zaten metre cinsinden**, ek scale gerektirmiyor.

**Önceki ChatGPT raporundaki bir noktanın netleştirilmesi:** `shape/OakInkObjects*/.../align/model_scale.json` dosyaları (`scale` matris + `rot` matris) gerçekten var ve örneğin `mug_s204` için `scale=0.162` gibi belirgin bir ölçek faktörü içeriyor. **Ancak** bu dosyalar `OakInkObjects`/`OakInkVirtualObjects` adlı **ayrı bir ham asset kütüphanesine** ait olup, `build_oakink_canonical.py`'nin `obj_pts` ürettiği `OakBase/` dizini **bu align/scale dosyalarını hiç kullanmıyor** (`grep` ile doğrulandı — kodda `align`/`model_scale` referansı yok). Yani mevcut pipeline için bu align JSON'lar **alakasız**; OakBase mesh'leri zaten doğrudan kullanılabilir ölçektedir. Bu, önceki raporun "align/scale uygula" adımını gerekli gösteren zincirini OakInk'in fiilen kullandığı asset kaynağı için **geçersiz/güncel değil** kılıyor.

### Kod Tarafı — Önceki Rapordaki Hata Düzeltildi

Önceki sürümde şu iddia vardı: *"`build_oakink_canonical.py`'deki `savez()` çağrısına `obj_anno` eklenmemiş, bu yüzden contact/penetration sıfır array ile hesaplanıyor."* Kodu satır satır yeniden okudum — **bu iddia güncel dosya için yanlış**:

`src/preprocessing/build_oakink_canonical.py:161-172` `obj_anno`'yu PKL'den okuyup `R_obj`/`t_obj`'a ayırıyor; `:196` her sample için `obj_annos` listesine ekliyor (`R(9) + t(3)` flat, 12-dim); `:215` `obj_anno_arr` olarak stack'liyor; ve **`:219-224`'teki `np.savez()` çağrısı `obj_anno=obj_anno_arr` parametresini açıkça içeriyor**:

```python
np.savez(
    OUT_DIR / "dataset.npz",
    pose=pose_arr, shape=shape_arr, tsl=tsl_arr,
    obj_anno=obj_anno_arr,
    category=cat_arr, obj_name=name_arr,
)
```

Loader tarafında da `src/data/dataset_oakink.py:68-79` `obj_anno` alanını okuyor, yoksa uyarı veriyor ve `:121-144`'teki `_obj_pts_to_wrist_frame` ile `obj_anno`'dan `R_obj`/`t_obj` çıkarıp `obj_pts`'i canonical → world → wrist zinciriyle taşıyor (`:179-184`). Sadece `obj_anno` alanı npz'de **hiç yoksa** (eski/bozuk bir npz okunursa) `obj_pts_contact` sıfır array'e düşüyor (satır 184) — bu fallback hâlâ kodda var ama **tetiklenme koşulu artık mevcut build scriptiyle oluşmuyor**.

**Sonuç:** OakInk için kod (hem build hem loader) güncel haliyle **doğru tasarlanmış ve doğru bağlanmış**. Önceki raporun belirttiği "hesaplanıp atılıyor" bug'ı bu repodaki güncel `build_oakink_canonical.py`'de mevcut değil.

### Processed Veri — Her İki Raporun da Artık Geçersiz Bir Varsayımı

Hem önceki Claude raporu hem ChatGPT raporu, `data/processed/oakink_canonical/dataset.npz` dosyasını **gerçekten açıp** içindeki alanları inceleyerek sonuca varmıştı. Şu an bu dosya **diskte yok**:

```text
$ find data/processed/oakink_canonical -maxdepth 1
bfs: error: data/processed/oakink_canonical: No such file or directory.
```

`data/processed/` altında yalnızca `.DS_Store` kaldı. Yani önceki raporların dayandığı somut `.npy`/`.npz` kanıtları artık mevcut değil (muhtemelen görüşme başındaki büyük dosya temizliği sırasında silindi). Build scripti (`src/preprocessing/build_oakink_canonical.py`) diskte mevcut ve git'e eklenmemiş (untracked, bugün `Jun 30 21:58`'de değiştirilmiş) — yani bu, üzerinde aktif çalışılan/yeni yazılmış bir versiyon. **Pipeline yeniden çalıştırılmadan elde mevcut hiçbir processed OakInk verisi yok.**

## Ek Bulgu: `.gitignore` `src/data/`'yı Yanlışlıkla Görmezden Geliyor

İnceleme sırasında fark edildi: `.gitignore:23`'teki `data/` satırı, başında `/` olmadığı için repo'daki **her** `data/` adlı dizinle eşleşiyor — sadece kök `data/` ile değil, `src/data/` ile de. Sonuç: `src/data/dataset_hot3d.py` ve `src/data/dataset_oakink.py` (yukarıda incelenen kritik dataset loader kodu) **git tarafından izlenmiyor/ignore ediliyor**. `git status`/`git ls-files` bu dosyaları hiç göstermiyor. Bu, kod kaybı riski taşıyan ayrı bir konudur (commit edilmiyor) ve `.gitignore`'a `/data/` (mutlak, kök-göreli) şeklinde düzeltilmesi önerilir — bu rapor kapsamı dışında ama gözden kaçmaması için not edildi.

## Senaryo Bazlı Risk Listesi (güncellenmiş)

| Senaryo | HOT3D | OakInk |
|---|---|---|
| Frame uyuşmazlığı (wrist vs canonical) | Kod tarafında ele alınmış, ham veri destekliyor | Kod tarafında ele alınmış, ham veri destekliyor |
| Ölçek uyuşmazlığı | Ham veri metre, tutarlı | Ham veri metre, tutarlı (align/scale dosyaları kullanılmıyor — bkz. yukarı) |
| Mesh local kalması (pose uygulanmadan) | Kodda düzeltilmiş (`R_obj`/`obj_world_t` uygulanıyor) | Kodda düzeltilmiş (`R_obj`/`t_obj` uygulanıyor) |
| Temporal uyuşmazlık | Per-frame `timestamp_ns` eşleştirmesi gerekli; mevcut loader bunu sağlıyor (doğrulama: kod okuması, çalışma zamanı testi yapılmadı) | Statik (T=1), risk yok |
| **Processed veri eksikliği** | **Yeni risk: processed dizin diskte yok, pipeline yeniden çalıştırılmalı** | **Yeni risk: processed dizin diskte yok, pipeline yeniden çalıştırılmalı** |
| Test split'te hand pose eksikliği | **Yeni risk: `test` split'te `mano_hand_pose_trajectory.jsonl` boş, yalnızca `train` kullanılabilir** | Geçerli değil (split yapısı farklı) |

## En Kritik Güncel Bulgu

1. **Kod seviyesinde** hem HOT3D hem OakInk için frame dönüşüm zinciri (`canonical/object-local → world → wrist`) doğru tasarlanmış ve uygulanmış durumda. Önceki raporda OakInk için iddia edilen "obj_anno hesaplanıp kaydedilmiyor" bug'ı **güncel kodda yok** — düzeltildi.
2. **Asıl mevcut risk artık veri varlığı, doğruluk mantığı değil**: `data/processed/hot3d_canonical/` ve `data/processed/oakink_canonical/` şu anda diskte yok. İki dataset için de pipeline'lar (`build_hot3d_canonical_full.py`, `build_oakink_canonical.py`) yeniden çalıştırılmadan ne eğitim ne de görsel audit yapılabilir.
3. HOT3D'de `test` split'inin hand pose verisi içermediği doğrulandı — bu split contact/penetration ile ilgili hiçbir işte kullanılmamalı.
4. OakInk'in kullandığı `OakBase` mesh kaynağı zaten metre ölçeğinde; `align/model_scale.json` dosyaları farklı bir asset kütüphanesine ait ve mevcut pipeline'da kullanılmıyor — bu nedenle "align/scale uygulanmalı" adımı OakInk için gereksiz.
5. `src/data/` dizini `.gitignore` kuralı yüzünden yanlışlıkla izlenmiyor; bu rapor kapsamının dışında ama ayrı olarak düzeltilmesi önerilir.

## Önerilen Sonraki Adımlar

1. `python src/preprocessing/build_hot3d_canonical_full.py` ve `python src/preprocessing/build_oakink_canonical.py` çalıştırılarak processed veriler yeniden üretilmeli (yalnızca HOT3D `train` split kullanılarak).
2. Üretilen `dataset.npz`/`obj_pts/*.npy` dosyalarında `obj_anno` (OakInk) ve `obj_world_t/q`, `wrist_world_t/q` (HOT3D) alanlarının gerçekten dolu geldiği doğrulanmalı (örnek script ile `np.load` + shape/range kontrolü).
3. Bir örnek seçilip görsel audit yapılmalı: `obj_pts_contact` (wrist frame) ile FK fingertip noktaları aynı 3D sahnede çizilip en yakın mesafe mm cinsinden raporlanmalı — hem HOT3D hem OakInk için.
4. `.gitignore:23`'teki `data/` satırı `/data/` olarak düzeltilip `src/data/` altındaki dosyaların commit'lendiğinden emin olunmalı.
5. Contact/penetration loss/metrikleri yalnızca `obj_pts_contact` üzerinden hesaplanmalı; ham `obj_pts` (PointNet girdisi) ile karıştırılmamalı.

## Kısa Sonuç

Önceki iki raporun ortak vardığı "OakInk processed verisinde sorun var" sonucu **doğru** ama **gerekçesi yanlış aktarılmıştı**: sorun kodun `obj_anno`'yu atlaması değil, processed verinin **şu anda hiç var olmaması**. Kod (hem HOT3D hem OakInk için) ham veriden doğru frame'e taşıma mantığını zaten doğru uyguluyor; eksik olan tek şey bu pipeline'ların çalıştırılıp çıktısının üretilmesi ve görsel/sayısal olarak doğrulanmasıdır.
