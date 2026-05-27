# HOT3D Veri Seti — Tam Açıklama

## 1. HOT3D Nedir?

**HOT3D**, **Hand and Object Tracking in 3D** (El ve Nesne 3D Takibi) anlamına gelir. **Meta Reality Labs** tarafından yayımlanan, insan ellerinin gerçek dünya nesneleriyle 3B uzayda nasıl etkileştiğini anlayan modelleri eğitmek ve değerlendirmek için tasarlanmış büyük ölçekli bir veri setidir.

Temel amaç: Birinci şahıs (egosantrik) video akışı verildiğinde, bir model her karede *ellerin 3B'de nerede olduğunu* ve *nesnelerin 3B'de nerede olduğunu* tahmin edebilir mi?

Bu, derinliği, tıkanmayı (occlusion) ve fiziksel teması akıl yürütmesi gerekeceğinden 2B poz tahminine kıyasla çok daha zordur.

---

## 2. HOT3D Neden Önemlidir?

| Problem | Neden Zor |
|---|---|
| Eller çok eklemlidir | El başına 21 eklem, sürekli öz-tıkanma |
| Nesneler şekil ve doku açısından çok çeşitlidir | Bardaktan kitaba, zımba makinesine kadar |
| Egosantrik görüş alışılmadıktır | Kamera başla hareket eder, sabit değil |
| 3B etiketler toplamak pahalıdır | 2B sınırlayıcı kutu çizmek gibi değil |

Önceki veri setleri; sadece 2B etiket, çok az nesne, yalnızca laboratuvar ortamı veya el-nesne teması içermiyordu. HOT3D ölçekli gerçek dünya egosantrik çekimiyle bu boşluğu dolduruyor.

---

## 3. Çekim Cihazları ve Kurulum

HOT3D **iki farklı egosantrik çekim platformu** kullanır:

### Meta Quest 3 (Başlık)
- Kafaya takılan VR/AR başlığı
- **Stereo RGB kameralar** (dışa bakan)
- **Derinlik sensörleri**
- **IMU** (atalet ölçüm birimi — jiroskop + ivmeölçer)
- Çözünürlük: Göz kamerası başına ~1280×1024

### Project Aria (Gözlük)
- Meta'nın hafif araştırma gözlükleri
- **RGB kamera**, **göz izleme kameraları**, **SLAM kameraları**
- **IMU**
- Normal gözlük gibi görünür — denekler için daha doğal
- Derinlik sensörü yok (Quest 3'ten farklı)

**Neden iki cihaz?** Genellemeyi test etmek için. Yalnızca Quest 3 verisi üzerinde eğitilmiş bir model o kameranın optiğine aşırı uyum sağlayabilir. İkisine sahip olmak karşılaştırmayı daha gerçekçi kılar.

Her kayıt oturumu = bir **sekans**. Denek Quest 3 ya da Aria takar ve masa üzerindeki nesnelerle etkileşime geçer.

---

## 4. Veri Seti Ölçeği ve İstatistikleri

| Özellik | Değer |
|---|---|
| Toplam sekans | ~800+ |
| Toplam kare | ~1,5 milyon |
| Benzersiz denek | 19 kişi |
| Benzersiz nesne | 33 ev eşyası |
| Cihazlar | Meta Quest 3 + Project Aria |
| Anotasyon türü | 3B el pozu + 6DoF nesne pozu |
| El modeli formatları | MANO ve UmeTrack |

33 nesne günlük kullanım eşyalarıdır: bardak, şişe, makas, zımba makinesi, telefon vb. Her birinin kontrollü ortamda yakalanan hassas bir **3B mesh'i** vardır.

---

## 5. Veri Modaliteleri

Her sekans için birden fazla senkronize veri akışı alırsınız:

### Görüntüler / Video
- Ego kamerasından **RGB kareler** (kişinin gördüğü)
- `.vrs` formatında saklanır (Meta'nın video kayıt sistemi)
- Cihaz başına birden fazla kamera akışı (Quest 3 için sol/sağ göz, Aria için birden fazla kamera)

### Derinlik (Yalnızca Quest 3)
- RGB ile senkronize piksel başına derinlik haritaları
- Stereo tahmini olmadan doğrudan 3B yeniden yapılandırmaya olanak tanır

### IMU
- Yüksek frekanslı hareket verisi (ivmeölçer + jiroskop)
- Kareler arası baş hareketini anlamak için kullanılır

### El Pozu Anotasyonları
- Her karede her iki el için 3B eklem konumları
- İki formatta sağlanır: **MANO** ve **UmeTrack** (bkz. Bölüm 7)

### Nesne Pozu Anotasyonları
- **6DoF poz** = 3B konum (x, y, z) + 3B yönelim (döndürme matrisi veya kuaterniyon)
- Nesne görünür/tutulduğunda kare başına bir poz

### Kamera Kalibrasyonu
- İçsel parametreler (odak uzaklığı, temel nokta, bozulma)
- Dışsal parametreler (her kameranın cihaza göre konum ve dönüşü)
- 3B noktaları 2B görüntü uzayına yansıtmak için gereklidir

---

## 6. Klasör ve Dosya Yapısı

```
hot3d/
│
├── sequences/
│   ├── <sequence_uid>/               # örn. "P0001_Q3_0001"
│   │   ├── video.vrs                 # ham çok akışlı video (RGB, derinlik, IMU)
│   │   ├── hand_poses.json           # kare başına 3B el anotasyonları
│   │   ├── object_poses.json         # kare başına 6DoF nesne pozları
│   │   ├── camera_calibration.json   # içsel + dışsal parametreler
│   │   └── metadata.json             # denek ID, cihaz türü, nesne listesi vb.
│   │
│   └── <sequence_uid>/               # başka bir sekans (Aria cihazı olabilir)
│       └── ...
│
├── objects/
│   ├── <object_id>/                  # örn. "cup_01"
│   │   ├── mesh.obj                  # nesnenin 3B üçgen mesh'i
│   │   ├── mesh.mtl                  # malzeme/doku referansı
│   │   └── texture.png              # mesh için doku haritası
│   └── ...
│
├── splits/
│   ├── train.txt                     # eğitim için sekans UID listesi
│   ├── val.txt                       # doğrulama sekansları
│   └── test.txt                      # test sekansları (etiketler yayınlanmadı)
│
└── object_library.json               # tüm 33 nesne için meta veri (isim, kategori vb.)
```

### Sekans UID Kuralı

`P0003_Q3_0017` gibi bir sekans ID'si şunları kodlar:
- `P0003` → Denek (katılımcı) #3
- `Q3` → Cihaz Quest 3 (Aria için `AR`)
- `0017` → O denek+cihaz için sekans numarası

---

## 7. Anotasyon Formatları

### 7a. El Pozu — MANO Formatı

**MANO**, *"MANO: Hand Model with Articulated and Non-rigid Deformations"* makalesinden endüstri standardı parametrik bir el modelidir.

Ham 3B eklem koordinatlarını saklamak yerine, MANO elin şeklini tanımlayan **parametreleri** saklar:

```
Kare başına el başına MANO parametreleri:
  - pose: 48 değer  (eksen-açı olarak 15 eklem dönüşü + global dönüş)
  - shape: 10 değer (kişiye özgü el şekli, denek başına sabit)
  - translation: 3 değer (kamera uzayında global 3B konum)
```

Bu ~61 sayıdan, MANO model kodu kullanılarak tam bir el mesh'i (778 köşe) ve 21 eklem konumu yeniden yapılandırılabilir. Bu kompakt ve türevlenebilirdir — sinir ağı çıktıları için idealdir.

**Neden düz eklem XYZ saklanmıyor?** MANO anatomik olarak olası pozları garanti eder (imkânsız parmak bükülmesi yok) ve yalnızca iskelet eklemlerini değil tam el yüzeyini verir.

### 7b. El Pozu — UmeTrack Formatı

**UmeTrack**, Xtended Hand veri setiyle birlikte geliştirilen Meta'nın kendi el takip formatıdır. Şunları saklar:

```
Kare başına el başına UmeTrack parametreleri:
  - joint_positions_3d: 21 × 3 = 63 değer (her eklem için ham XYZ)
  - joint_angles: kompakt açı temsili
  - wrist_transform: 4×4 matris (bileğin konum + yönelimi)
```

Bu MANO'dan daha doğrudandır — MANO model koduna ihtiyaç duymadan eklem konumlarıyla çalışırsınız. HOT3D **her iki formatı** sağlar, böylece araştırmacılar iş akışlarına uyan formatı kullanabilir.

### 7c. Nesne Pozu — 6DoF Formatı

Her nesnenin bilinen bir 3B mesh'i vardır. Anotasyon şunları saklar:

```json
{
  "frame_id": 1042,
  "object_id": "cup_01",
  "translation": [0.123, -0.045, 0.812],   // kamera uzayında x, y, z (metre)
  "rotation": [[r00, r01, r02],              // 3×3 döndürme matrisi
               [r10, r11, r12],
               [r20, r21, r22]],
  "visibility": 0.87                         // görünür nesne oranı (0–1)
}
```

Bu poz ile 3B mesh'i sahnede tam olarak bulunduğu yere yerleştirebilirsiniz. Ayrıca mesh'i görüntüye yansıtarak RGB kare üzerine bindirme yapabilirsiniz.

---

## 8. Veri Bölümleri

| Bölüm | Amaç | Etiketler Mevcut mu? |
|---|---|---|
| `train` | Modeli eğit | Evet — tam anotasyonlar |
| `val` | Hiperparametre ayarla, metrikleri kontrol et | Evet — tam anotasyonlar |
| `test` | Son karşılaştırmalı değerlendirme | Hayır — sunucuya gönderilir |

Test seti etiketleri Meta'nın değerlendirme sunucusunda tutulmaktadır. Tahminleri gönderirsiniz ve skorlar alırsınız. Bu, test etiketlerine aşırı uyumu önler.

Bölüm dosyaları satır başına bir sekans UID'si içeren sade metin dosyalarıdır:
```
P0001_Q3_0001
P0001_Q3_0002
P0002_AR_0001
...
```

---

## 9. Anlaşılması Gereken Temel Kavramlar

### Kamera Uzayı vs Dünya Uzayı

HOT3D'deki tüm pozlar **kamera uzayında** ifade edilir (sabit bir dünya noktasına değil, kamera başlangıç noktasına göre). Bu önemlidir:
- `[0, 0, 0.5]` konumu "kameranın tam 0,5 metre önünde" anlamına gelir
- Kamera hareket ettikçe (kişi başını çevirir), aynı nesnenin bir sonraki karede farklı koordinatları olacaktır

### VRS Formatı

`.vrs`, çok akışlı kayıtlar için Meta'nın dahili kapsayıcısıdır. Bir `.mkv` veya `.bag` dosyası gibi — birden fazla zaman senkronize veri kanalı tutar. Kareleri ve sensör verilerini çıkarmak için **PyVRS** kütüphanesi gereklidir. **PyHOT3D** kütüphanesi bunu sarar ve size temiz Python nesneleri verir.

### Senkronizasyon

Tüm veri akışları zaman damgalıdır. Bir kare yüklediğinizde şunları alırsınız:
- T zaman damgasında RGB görüntüsü
- (Yaklaşık) aynı T'de derinlik haritası
- T'de el pozu anotasyonu
- T'de nesne pozu anotasyonu

Küçük zamanlama sapmaları mevcuttur — PyHOT3D yükleyici interpolasyonu yönetir.

### Koordinat Sistemi

HOT3D **sağ el kuralı** koordinat sistemini kullanır:
- X → sağ
- Y → aşağı
- Z → öne (sahneye doğru)

Bu standart kamera koordinat kuralıdır.

---

## 10. Verilerin Uçtan Uca Akışı

```
                    ┌─────────────────────────┐
                    │     sequence_uid dizini  │
                    └────────────┬────────────┘
                                 │
              ┌──────────────────┼──────────────────┐
              │                  │                   │
         video.vrs        hand_poses.json    object_poses.json
              │                  │                   │
    ┌─────────▼────────┐   ┌─────▼──────┐   ┌───────▼────────┐
    │  T zaman damgasında  │   │ MANO param.│   │  6DoF poz yükle│
    │  RGB kare çıkar  │   │  yükle → 21│   │  + objects/ den│
    │                  │   │  3B eklem  │   │  nesne mesh'i  │
    └─────────┬────────┘   └─────┬──────┘   └───────┬────────┘
              │                  │                   │
              └──────────────────▼───────────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │  camera_calibration.json │
                    │  Tüm 3B noktaları        │
                    │  2B görüntü koordinatlarına│
                    │  yansıt                  │
                    └────────────┬────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │  Son kare:              │
                    │  - RGB görüntü          │
                    │  - 2B el işaret noktaları│
                    │  - 3B el eklemleri      │
                    │  - Nesne bindirmesi     │
                    └─────────────────────────┘
```

Bu iş akışı, ham HOT3D dosyalarını okuyup eğitim için tensörlere dönüştüren bu repodaki `build_dataset.py` gibi araçların uyguladığı şeydir.

---

## Özet

| Kavram | Tek Cümlelik Cevap |
|---|---|
| HOT3D nedir? | Meta'nın egosantrik 3B el+nesne takip veri seti |
| Nasıl çekildi? | Quest 3 başlığı + Project Aria gözlükleri |
| Sekans içinde ne var? | RGB video, derinlik, IMU, 3B el pozları, 6DoF nesne pozları |
| Eller nasıl temsil edilir? | MANO parametreleri veya UmeTrack eklem konumları |
| Nesneler nasıl temsil edilir? | 6DoF poz (öteleme + dönme) + 3B mesh |
| Video dosya formatı? | `.vrs` (Meta'nın çok akışlı kapsayıcısı) |
| Kaç nesne var? | 3B mesh'i olan 33 benzersiz ev eşyası |
| Kaç denek var? | Eğitim/doğrulama/test genelinde 19 kişi |
