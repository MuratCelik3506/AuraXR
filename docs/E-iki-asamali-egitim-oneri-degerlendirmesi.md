# E. İki Aşamalı Eğitim İçin Öneri Değerlendirmesi

Bu doküman, iki ayrı analiz metnindeki önerileri birleştirir ve her önerinin gerçekten fayda sağlayıp sağlamayacağını teknik olarak değerlendirir.

Hedef şudur: 2 aşamalı eğitim sonunda modelin yalnızca düşük reconstruction loss üretmesi değil, farklı eval metriklerinde tutarlı biçimde iyi, fiziksel olarak anlamlı ve Unity'ye taşınabilir grasp üretmesi.

Mevcut tabloya göre model parmak pozu dağılımını öğrenmeye başlamış görünüyor; ancak grasp kalitesi henüz güvenilir değildir. Başka bir deyişle model "elin şekli neye benzemeli?" sorusuna yaklaşmış, fakat "bu el objeyi gerçekten tutar mı?" sorusunda hâlâ zayıftır. Bu nedenle önerilerin çoğu reconstruction loss'u daha da düşürmekten çok temas, penetrasyon, temporal stabilite, kalite seçimi ve eval güvenilirliği üzerine yoğunlaşmalıdır.

## Kısa Sonuç

Bu öneriler genel olarak fayda sağlar; fakat hepsinin etkisi aynı değildir.

> **Güncel kontrol (2026-06-30):** Raw ve processed veri, loader kodu ve mevcut eğitim/eval çıktıları tekrar kontrol edildi. Öncelik 1'deki ana koordinat uyuşmazlığı giderilmiş durumda: `obj_pts_contact` wrist frame + metre ölçeğinde üretiliyor, `contact_penetration_loss` ve eval metrikleri bu alanı kullanıyor. HOT3D tarafında yeni MANO FK ile stored `fk_joints` arası ortalama fingertip farkı ~0.77 cm ölçüldü. Buna karşılık Phase 2 velocity/acceleration loss hâlâ pratikte aktif değil (`prev_pred_pose=None`), HOT3D manifest'te val/test split yok, eval metadata standardı eksik ve Unity success label pipeline'ı henüz kurulmuş değil.

En yüksek fayda beklenen maddeler:

| Öncelik | Öneri | Durum | Fayda beklentisi | Neden |
|---:|---|---|---|---|
| 1 | ~~Koordinat sistemi audit'i ve contact/penetration düzeltmesi~~ | **Tamamlandı** | Çok yüksek | Loss ve metrikler artık wrist-frame `obj_pts_contact` ile çalışıyor; MANO FK eski 18–20 cm hatadan HOT3D'de ~0.77 cm seviyesine indi. |
| 2 | Phase 2'de velocity/acceleration loss'u gerçekten aktif etmek | **Tamamlandı** | Çok yüksek | `prev_frame_feat`/`prev2_frame_feat` dataset'e eklendi; `no_grad` forward pass ile `pred_{t-1}`/`pred_{t-2}` üretilip `grasp_loss`'a geçiliyor. Phase 2'de `vel=0.010`, `acc=0.040` aktif, backward temiz. |
| 3 | Fingertip/FK tabanlı pozisyon loss'u eklemek | **Tamamlandı** | Yüksek | `grasp_loss`'a `L_tip = MSE(FK(pred), FK(gt))` eklendi (`tip_weight=0.5`, `--tip_weight` argümanı). Her iki fazda aktif, backward temiz. |
| 4 | Eval protokolünü temizlemek | **Tamamlandı** | Yüksek | `evaluate.py` tamamen yeniden yazıldı: `meta` bloğu (checkpoint, epoch, git commit, timestamp, phase, split, k, device), timestamped çıktı dosyaları, NaN filtreleme, `failed_run` flag, HOT3D için sequence-bazlı jitter/velocity/acceleration metrikleri, `obj_pts_contact` eksikse açık uyarı. |
| 5 | Quality/success head için anlamlı label üretmek | **Kısmi** | Yüksek | Heuristic `quality_label` düzeldi; Unity kaynaklı `success_label` yok. |
| 6 | Unity physics label pipeline'ı kurmak | **Açık** | Yüksek ama daha geç aşama | Unity contract var, fakat success label export/training pipeline'ı yok. |
| 7 | Mixed Phase 2 training yapmak | **Tamamlandı** | Orta-yüksek | `MixedDataLoader` eklendi: Phase 2'de HOT3D %70 + OakInk replay %30. `--oakink_replay_ratio` argümanıyla ayarlanabilir (0=devre dışı). OakInk batch'lerde vel/acc loss otomatik skip, HOT3D batch'lerde aktif. |
| 8 | Clamp ve joint-limit stratejisini düzeltmek | **Tamamlandı** | Orta | Hard clamp kaldırıldı (`grasp_model.py`); `joint_limit_loss` gradient ile öğretir. `joint_limit_saturation_rate` metriği `eval_metrics.py` + `evaluate.py`'ye eklendi. |
| 9 | LR scheduler ve early stopping eklemek | **Tamamlandı** | Orta | `ReduceLROnPlateau(factor=0.5, patience=5)` ve `early_stopping=10` eklendi. `--lr_patience` ve `--early_stopping` argümanlarıyla ayarlanabilir. Her epoch log'una `lr` alanı yazılıyor. |
| 10 | CVAE diversity / KL dengesini iyileştirmek | **Kısmi** | Orta | K sampling ve diversity metriği var; oracle selection / scorer ablation yok. |
| 11 | Runtime latency benchmark'ı eklemek | **Açık** | Orta | Unity contract'ta latency alanı var; ölçüm benchmark'ı yok. |

En kritik nokta şudur: Contact/penetration hesabındaki ana koordinat uyuşmazlığı şu an giderilmiş görünüyor. Bundan sonra contact loss ağırlığını artırmadan önce kalan riskler, yani OakInk'teki birkaç cm FK/template residual farkı, centroid-proxy penetration ve eval split/metadata eksikleri ayrı ayrı ele alınmalıdır.

---

## 1. ~~Koordinat Sistemi Uyuşmazlığı~~

**Güncel durum:** Tamamlandı. Raw/processed veri ve loader kontrolünde `obj_pts_contact` alanının object/canonical → world → wrist zinciriyle metre ölçeğinde üretildiği görüldü. Eğitimde `obj_pts_contact` kullanılıyor; normalized canonical `obj_pts` yalnız PointNet girdisi olarak kalıyor. HOT3D current FK ile stored `fk_joints` arası ortalama fingertip farkı ~0.77 cm. OakInk tarafında fingertip sırası düzeltilince current FK ile `fingertips_world` arası ortalama fark ~3.3 cm; bu eski 18–20 cm frame bug'ı değil, template/shape ve 16-joint proxy farkı olarak ele alınmalı.

### Problem

İkinci analizde en kritik iddia şudur: `contact_penetration_loss`, parmak ucu pozisyonlarını wrist/el bileği frame'inde üretirken, `obj_pts` normalize edilmiş global veya canonical frame'den geliyor olabilir. Eğer bu doğruysa `cdist(fingertip, object_point)` gerçek parmak-yüzey mesafesini değil, iki farklı koordinat sistemindeki noktaların anlamsız uzaklığını ölçer.

Bu durumda:

- Contact loss gerçek temasa doğru gradient üretmez.
- Penetration loss objenin içine girme davranışını doğru cezalandırmaz.
- OakInk tarafında çok yüksek penetration değerleri fiziksel model hatasından çok metrik hatası olabilir.
- Contact ratio'nun sıfıra yakın kalması modelin kötü olmasından değil, ölçümün yanlış olmasından kaynaklanabilir.

### Gerçekten fayda sağlar mı?

Evet, bu madde en yüksek fayda potansiyeline sahipti ve ana kısmı tamamlandı. Loss yanlış koordinat sisteminde hesaplanıyorsa modelin temas öğrenmesi beklenemezdi; mevcut kod/veri durumunda bu risk büyük ölçüde kapanmış görünüyor.

### Uygulama önerisi

Önce kod değişikliği değil, koordinat audit'i yapılmalıdır:

```text
hand FK points hangi frame'de?
obj_pts hangi frame'de?
wrist_tsl hangi frame'de?
target_pose hangi el köküne göre tanımlı?
OakInk obj_pts normalize mi, metre cinsinden mi?
HOT3D obj_pts object frame'de mi, world frame'de mi?
Unity collider frame'i bunlarla nasıl eşleşiyor?
```

Ardından iki tutarlı seçenekten biri seçilmelidir:

```text
Seçenek A:
FK fingertip noktalarını wrist/world/object frame'e taşı ve obj_pts ile aynı frame'de karşılaştır.

Seçenek B:
Obj_pts'i wrist frame'e taşı ve FK fingertip noktalarıyla orada karşılaştır.
```

Önemli olan hangi frame'in seçildiği değil, tüm loss ve metriklerin aynı frame'de hesaplanmasıdır.

### Beklenen etki

Bu düzeltme başarılı olursa contact ratio ve penetration metriklerinde keskin iyileşme beklenir. Ancak analizdeki "0.03'ten %40-60 bandına çıkar" gibi sayısal tahminler garanti değildir; bunlar makul beklenti olarak görülmeli, gerçek etki kontrollü ablation ile ölçülmelidir.

### Doğrulama

Minimum doğrulama:

```text
1. Tek bir sample seç.
2. Objeyi ve FK fingertip noktalarını aynı 3D sahnede çiz.
3. En yakın obje noktası mesafesini mm cinsinden raporla.
4. Aynı sample için contact/penetration loss'un görsel durumla tutarlı olup olmadığını kontrol et.
```

Bu doğrulama yapılmadan contact/penetration metrikleri tez sonucunda güçlü iddia olarak kullanılmamalıdır.

---

## 2. Phase 2 Temporal Loss'un Aktif Olmaması

**Güncel durum:** Tamamlandı. `dataset_hot3d.py`'ye `prev_frame_feat` (t-1 penceresi) ve `prev2_frame_feat` (t-2 penceresi) eklendi. `train_grasp.py`'de Phase 2 aktifken bu pencereler üzerinden `no_grad` forward pass yapılıp `pred_{t-1}`/`pred_{t-2}` üretiliyor; `grasp_loss`'a `prev_pred_pose`/`prev2_pred_pose` olarak geçiliyor. Gradient yalnızca ana `pred_t` üzerinden akıyor. Ek olarak `mano_fk.py`'deki inplace tensor atamaları out-of-place operasyonlarla değiştirilerek `backward()` hatası giderildi. Doğrulama: Phase 2'de `vel=0.010`, `acc=0.040` aktif, backward temiz.

### Problem

İki analiz de aynı soruna işaret ediyor: Phase 2'nin amacı HOT3D ile temporal kapanış ve stabilite öğrenmek olsa da `prev_pred_pose=None` verildiği için velocity ve acceleration loss pratikte çalışmıyor olabilir.

Bu durumda model GRU ile geçmiş frame bilgisini görüyor olsa bile açıkça şu davranışa zorlanmaz:

```text
ani sıçrama yapma
gerçek el hareketinin hızını takip et
parmak kapanışını frame'ler arasında yumuşak sürdür
```

### Gerçekten fayda sağlar mı?

Evet. Phase 2'nin ana gerekçesi temporal veri kullanmaktır. Temporal loss aktif değilse Phase 2, büyük ölçüde "HOT3D üzerinde anlık pose fine-tune" haline gelir. Bu da offline pose error'ı iyileştirebilir ama runtime jitter problemini çözmeyebilir.

### Uygulama önerisi

Batch içinde ardışık frame tahminleri üretilmeli ve prediction-space velocity/acceleration loss hesaplanmalıdır:

```text
L_vel = ||(pred_t - pred_t-1) - (gt_t - gt_t-1)||
L_acc = ||(pred_t - 2pred_t-1 + pred_t-2) - (gt_t - 2gt_t-1 + gt_t-2)||
```

Burada dikkat edilmesi gereken nokta şudur: `pred_t-1` için `torch.no_grad()` kullanılırsa sadece `pred_t` tarafına gradient akar. Bu başlangıç için kabul edilebilir olabilir; ancak daha doğru çözüm, kısa sequence unroll ederek tüm zaman adımlarına gradient akıtmaktır.

### Beklenen etki

Beklenen fayda reconstruction loss'tan çok temporal metriklerde görülmelidir:

```text
geodesic_velocity_error
geodesic_acceleration_error
jitter_score
frame-to-frame fingertip displacement variance
```

Bu metrikler eklenmeden "temporal training fayda sağladı" iddiası eksik kalır.

---

## 3. Fingertip/FK Position Loss

**Güncel durum:** Kısmi. `mano_fk.py` düzeltilmiş, `fingertip_position_error` ve `mpjpe` eval metrikleri var. Ancak eğitim loss'unda `FK(pred_pose)` ile `FK(target_pose)` arasında ayrı bir `L_tip` terimi yok; yalnızca rotation reconstruction, contact hinge ve penetration proxy var.

### Problem

Rotation error'ın düşmesi, grasp başarısını garanti etmez. 45 boyutlu axis-angle çıktısı eklem rotasyonlarını yaklaştırabilir; fakat küçük açısal hatalar kinematik zincirin ucunda büyük fingertip pozisyon hatasına dönüşebilir.

Özellikle küçük objelerde 20-40 mm fingertip hatası grasp'i tamamen bozabilir. Bu yüzden yalnızca geodesic rotation error'a bakmak yanıltıcıdır.

### Gerçekten fayda sağlar mı?

Evet, fayda potansiyeli yüksektir. Çünkü grasp kalitesi için parmak uçlarının objeye göre konumu, eklem açılarına göre daha doğrudan bir sinyaldir.

### Uygulama önerisi

Loss'a FK tabanlı parmak ucu pozisyon kaybı eklenmelidir:

```text
L_tip = ||FK(pred_pose)_tips - FK(gt_pose)_tips||
```

Eğer koordinat sistemi düzeltildiyse bu loss object-relative veya world frame'de de hesaplanabilir. Burada kritik karar şudur:

- Eğer amaç sadece insan demonstrasyonuna benzemekse `pred_tip` ile `gt_tip` karşılaştırılır.
- Eğer amaç objeye temas etmekse `pred_tip` ile obje yüzeyi arasındaki mesafe de ayrıca optimize edilir.

İkisi aynı şey değildir ve birlikte kullanılmaları daha sağlıklıdır.

### Beklenen etki

Fingertip error, contact ratio ve Unity success rate üzerinde dolaylı ama güçlü etki beklenir. Bu loss, özellikle rotation error'ın iyi görünüp contact'ın zayıf kaldığı durumlarda faydalıdır.

---

## 4. Contact Ratio Düşüklüğü

**Güncel durum:** Kısmi. Contact metriğinin frame'i düzeldi, fakat mevcut eval sonuçlarında contact ratio hâlâ düşük kalıyor. Bu artık öncelikle koordinat bug'ı değil; model kalitesi, fingertip loss eksikliği, Phase 2 temporal loss'un pasif olması ve yalnız fingertip tabanlı contact tanımıyla ilişkili.

### Problem

Contact ratio'nun çok düşük olması iki farklı anlama gelebilir:

```text
1. Model gerçekten objeye temas eden parmak pozları üretmiyor.
2. Contact metriği koordinat/ölçek/frame hatası nedeniyle yanlış ölçülüyor.
```

İlk analiz contact loss ağırlığını artırmayı ve sadece fingertip değil parmak segmentlerini de hesaba katmayı öneriyor. Bu fikir doğru yöndedir, fakat ikinci analizdeki koordinat sorunu çözülmeden uygulanırsa etkisi sınırlı kalır.

### Gerçekten fayda sağlar mı?

Koordinat sistemi düzeltildikten sonra evet. Öncesinde contact loss ağırlığını artırmak risklidir; çünkü yanlış loss daha güçlü hale getirilmiş olur.

### Uygulama önerisi

Contact hesabı kademeli genişletilmelidir:

```text
1. Fingertip-object minimum distance
2. Distal phalanx segment-object distance
3. Middle phalanx segment-object distance
4. Parmak başına contact coverage
5. Tüm el için contact ratio
```

Sadece parmak uçlarıyla contact ölçmek bazı grasp türlerini eksik değerlendirir. Güçlü kavrama, yan kavrama veya objeyi parmak segmentleriyle destekleme durumlarında distal ve middle phalanx temasları da önemlidir.

### Beklenen etki

Contact ratio yükselmeden Unity success rate'in anlamlı biçimde yükselmesi zordur. Ancak contact'ı tek başına maksimize etmek de doğru değildir; penetration ile birlikte optimize edilmelidir. Aksi halde model temas etmek için objenin içine girmeyi öğrenebilir.

---

## 5. Penetration Hesabı ve SDF İhtiyacı

**Güncel durum:** Açık. Penetration hâlâ centroid/point-cloud proxy. Kod yorumları bunun gerçek fizik ölçümü değil eğitim yönlendiricisi olduğunu doğru şekilde belirtiyor; gerçek penetration için Unity PhysX veya mesh/SDF tabanlı eval hâlâ gerekli.

### Problem

Penetration metrikleri point cloud veya centroid tabanlı proxy ile hesaplanıyorsa gerçek mesh penetrasyonunu doğru yansıtmayabilir. Point cloud yüzey örnekleri, objenin iç/dış ayrımını doğrudan vermez. Bu yüzden "temas et ama içine girme" ayrımı zayıf kalabilir.

### Gerçekten fayda sağlar mı?

Evet, fakat maliyeti contact düzeltmesine göre daha yüksektir. Gerçek mesh SDF veya Unity collider tabanlı penetration ölçümü, fiziksel anlamlılık açısından ciddi fayda sağlar.

### Uygulama önerisi

Kademeli yaklaşım:

```text
1. Eğitim sırasında ucuz approximate SDF veya signed distance proxy kullan.
2. Eval sırasında mesh SDF ya da Unity collider ile daha doğru penetration ölç.
3. Tez raporunda proxy training loss ile gerçek eval metric'i ayrı isimlerle ver.
```

Eğer mesh SDF pahalıysa her obje için offline precompute yapılabilir.

### Beklenen etki

Görsel kalite ve Unity physics stabilitesi iyileşir. Ancak SDF düzeltmesi tek başına grasp başarısını garanti etmez; contact ve fingertip loss ile birlikte çalışmalıdır.

---

## 6. Quality Head ve Success Head

**Güncel durum:** Kısmi. Heuristic `quality_label` artık gerçek fingertip kaynaklarıyla üretiliyor ve [0,1] aralığında anlamlı dağılıma sahip. Ancak `success_label` henüz datasetlerde yok; Phase 3 success head bu yüzden fiilen eğitilemiyor. Mevcut eval JSON'larında `quality_auc = NaN` görülebiliyor; bunun nedeni label threshold sonrası pozitif/negatif dağılım veya scorer kalibrasyonunun hâlâ zayıf olması olabilir.

### Problem

Quality score tarafında `quality_auc = NaN` veya çok düşük korelasyon görülüyorsa label dağılımı bozuk, sabit veya model çıktısıyla ilişkisiz olabilir. Success head için de Unity binary success label yoksa bu head henüz fiziksel başarıyı öğrenmiş sayılmaz.

Bu durumda CVAE ile K farklı aday üretmek fayda getirmeyebilir. Çünkü adaylar arasından hangisinin iyi olduğunu seçecek güvenilir skor yoktur.

### Gerçekten fayda sağlar mı?

Evet, ama sıralama önemlidir. Quality/success head'i düzeltmek, CVAE aday seçiminin çalışması için şarttır. Ancak bu head'lere verilecek label'lar contact/penetration metrikleri bozukken üretilirse yine yanlış öğrenme olur.

### Uygulama önerisi

Önce proxy quality label üretilebilir:

```text
quality = f(contact_ratio, penetration_depth, avg_finger_distance)
```

Ancak bu bileşenler tek skora sıkıştırılmadan önce ayrı ayrı loglanmalıdır:

```text
contact_ratio
penetration_depth
avg_finger_distance
fingertip_error
```

Sonra Unity label aşamasına geçilmelidir:

```text
success = object_not_dropped
          and displacement < threshold
          and rotation_change < threshold
```

Bu label geldikten sonra backbone dondurulup success head ayrı bir Phase 3 olarak kalibre edilebilir.

### Beklenen etki

Quality/success head güvenilir hale gelirse K adaylı CVAE gerçekten anlam kazanır. Aksi halde K=5 üretim sadece latency artıran ama başarıyı artırmayan bir mekanizma olabilir.

---

## 7. CVAE Diversity ve Aday Seçimi

**Güncel durum:** Kısmi. Model K aday üretebiliyor, `diversity_score` metriği mevcut ve success head aday başına skor üretiyor. Ancak scorer Unity `success_label` ile kalibre edilmediği için `argmax(success_prob)` henüz fiziksel başarı seçicisi olarak güvenilir değil. K=1/K=3/K=5 ve oracle selection ablation'ı da tamamlanmış değil.

### Problem

CVAE'nin amacı aynı obje için birden fazla geçerli grasp stratejisi üretmektir. Ancak KL çok hızlı düşüyor veya latent değişken kullanılmıyorsa decoder deterministik davranmaya başlar. Bu durumda K=3 veya K=5 sampling gerçek çeşitlilik sağlamaz.

Diğer yandan diversity tek başına iyi değildir. Model çok çeşitli ama fiziksel olarak kötü pozlar üretebilir.

### Gerçekten fayda sağlar mı?

Orta düzeyde fayda sağlar, fakat quality/success seçimi düzelmeden öncelik verilmemelidir. CVAE diversity ancak iyi aday seçilebiliyorsa nihai başarıya dönüşür.

### Uygulama önerisi

Şu ablation zorunlu hale getirilmelidir:

```text
K=1 deterministic
K=3 CVAE + quality selection
K=5 CVAE + quality selection
K=5 oracle selection
```

Oracle selection önemli bir teşhistir:

- Oracle iyi, quality selection kötü ise sorun generator'da değil scorer'dadır.
- Oracle da kötü ise sorun CVAE generator veya loss tasarımındadır.
- K arttıkça kalite artmıyor ama latency artıyorsa runtime'da K=1 veya K=3 tercih edilmelidir.

### Beklenen etki

Doğru seçiciyle CVAE grasp başarısını artırabilir. Ancak bu, tek başına KL ayarıyla değil, scorer ve eval protokolüyle birlikte kanıtlanmalıdır.

---

## 8. Hard Clamp ve Joint Limit Stratejisi

**Güncel durum:** Açık. Decoder hâlâ `torch.clamp(out, _LOWER, _UPPER)` ile hard clamp uyguluyor. Soft `joint_limit_loss` da mevcut olduğu için violation metriği yapay iyi görünebilir; `limit_saturation_rate` metriği yok.

### Problem

Decoder çıkışı hard clamp ile anatomik limitlere zorlanıyorsa joint limit violation metriği yapay olarak iyi görünebilir. Ayrıca clamp, sınır dışına çıkan değerlerde gradient'i keserek öğrenmeyi zorlaştırabilir.

İkinci analiz hard clamp ile soft joint-limit loss'un aynı anda kullanılmasını çelişkili buluyor. Bu değerlendirme teknik olarak makuldür.

### Gerçekten fayda sağlar mı?

Orta düzeyde fayda sağlar. Bu madde contact/temporal sorunlar kadar kritik değildir; fakat anatomik doğallık ve gradient akışı açısından önemlidir.

### Uygulama önerisi

İki stratejiden biri seçilmelidir:

```text
Strateji A:
Clamp kaldırılır, soft joint_limit_loss kullanılır.

Strateji B:
Clamp korunur, joint_limit_loss ana metrik gibi yorumlanmaz.
```

Eğer clamp korunacaksa ek metrik eklenmelidir:

```text
limit_saturation_rate
```

Bu metrik, çıktının ne kadarının alt/üst sınıra yapıştığını gösterir. Violation rate sıfır olsa bile saturation yüksekse pozlar doğal olmayabilir.

### Beklenen etki

Soft limit yaklaşımı daha doğal poz dağılımı sağlayabilir. Ancak tamamen clamp'i kaldırmak ilk etapta violation artışına yol açabilir; bu yüzden ablation ile denenmelidir.

---

## 9. Phase 1 ve Phase 2 Domain Farkı

**Güncel durum:** Açık. Phase 2 loader şu an yalnız `Hot3DTemporalDataset` kullanıyor; OakInk replay veya 70/30 mixed batch yok. HOT3D manifest'te ayrıca tüm satırlar `train`, yani Phase 2 val/test ayrımı da pratikte temiz değil.

### Problem

OakInk statik ama obje çeşitliliği yüksek bir veri setidir. HOT3D temporal ama obje sayısı daha sınırlıdır. Phase 2 sadece HOT3D ile yapılırsa model OakInk'te öğrendiği geniş obje bilgisini kısmen unutabilir.

### Gerçekten fayda sağlar mı?

Evet, mixed training mantıklı bir öneridir. Özellikle tez hedefi görülmemiş objelerde genelleme ise OakInk replay faydalı olur.

### Uygulama önerisi

Phase 2 içinde karışık eğitim yapılabilir:

```text
%70 HOT3D temporal + %30 OakInk static replay
```

OakInk örneklerinde temporal loss kapalı tutulur veya T=1 üzerinden sadece statik loss hesaplanır. HOT3D örneklerinde temporal loss aktif olur.

### Beklenen etki

HOT3D temporal kapanış öğretirken OakInk'in obje çeşitliliği korunur. Beklenen fayda özellikle held-out object split üzerinde görülmelidir.

---

## 10. Phase 2 Overfitting, LR Scheduler ve Early Stopping

**Güncel durum:** Açık. Eğitim döngüsünde AdamW ve best-val checkpoint var; `ReduceLROnPlateau`, early stopping veya skor kartı tabanlı checkpoint seçimi yok.

### Problem

Eğer validation reconstruction loss belirli epoch'lardan sonra artarken train loss düşmeye devam ediyorsa Phase 2 overfitting yapıyor olabilir. Sabit learning rate ve early stopping olmaması bunu güçlendirir.

### Gerçekten fayda sağlar mı?

Evet, fakat bu daha çok eğitim verimliliği ve generalization koruma önerisidir. Contact/temporal loss çalışmıyorsa scheduler tek başına temel problemi çözmez.

### Uygulama önerisi

```text
ReduceLROnPlateau:
  mode=min
  factor=0.5
  patience=5
  min_lr=1e-5

Early stopping:
  monitor=val_total veya seçilmiş skor kartı
  patience=10
```

Checkpoint seçimi yalnızca `val_total` ile yapılmamalıdır. Çünkü `val_total`, bozuk veya sabit contact/penetration loss içeriyorsa doğru modeli seçmeyebilir.

### Beklenen etki

Gereksiz epoch sayısı azalır ve daha erken bir checkpoint'in genellemesi korunabilir. Ancak "kaç saat tasarruf" gibi sonuçlar donanım ve dataset boyutuna bağlıdır.

---

## 11. Eval Protokolünün Temizlenmesi

**Güncel durum:** Kısmi. Eval script geodesic, MPJPE, fingertip error, contact, penetration, quality calibration ve diversity metriklerini yazıyor. Ancak JSON çıktılarında checkpoint path/epoch, training phase, eval timestamp, git commit, failed_run gibi metadata yok; HOT3D val/test split'i mevcut processed manifest'te boş olduğu için `n_batches=0` ve NaN raporlar üretilebiliyor.

### Problem

Failed run, NaN metric, farklı split veya farklı checkpoint'ten gelen raporlar aynı tabloda karışırsa yanlış model "en iyi" sanılabilir.

### Gerçekten fayda sağlar mı?

Evet. Bu madde doğrudan model kalitesini artırmaz, fakat doğru karar vermeyi sağlar. Yanlış eval protokolüyle yapılan tüm iyileştirmeler güvenilmez hale gelir.

### Uygulama önerisi

Her eval JSON içine en az şu metadata yazılmalıdır:

```text
checkpoint_path
checkpoint_epoch
training_phase
dataset
split
source
window
k
selection_mode
eval_timestamp
git_commit
failed_run
```

Standart eval matrisi:

```text
phase1_best -> OakInk val/test
phase2_best -> HOT3D val/test
final_best  -> HOT3D held-out test + OakInk held-out object test
```

NaN sonuçlar sıralamada otomatik olarak üst sıraya çıkmamalıdır. Failed run'lar final tabloya dahil edilmemelidir.

### Beklenen etki

Model geliştirme kararları daha güvenilir olur. Bu özellikle tez yazımında önemlidir; çünkü hangi checkpoint'in hangi split'te iyi olduğu açıkça izlenebilir.

---

## 12. Runtime Latency Benchmark

**Güncel durum:** Açık. Unity contract içinde `latency_ms` alanı var, fakat CPU/MPS/CUDA/ONNX ve K=1/3/5 benchmark matrisi ölçülmüş değil.

### Problem

Model PC GPU hedefli olsa bile XR runtime'da gecikme önemlidir. CVAE K aday, PointNet, GRU ve self-attention bileşenlerinin batch=1 inference maliyeti bilinmeden final sistem kararı verilemez.

### Gerçekten fayda sağlar mı?

Evet, final entegrasyon için gereklidir. Ancak model henüz fiziksel grasp kalitesi üretmiyorsa latency optimizasyonu erken öncelik olmamalıdır.

### Uygulama önerisi

Benchmark matrisi:

```text
K=1
K=3
K=5
batch=1
CPU
MPS
CUDA
ONNX export sonrası
Unity çağrı overhead'i dahil / hariç
```

### Beklenen etki

K aday sayısı ve runtime seçim stratejisi gerçek ölçüme dayanır. Örneğin K=5 kaliteyi artırsa bile 20 ms üzerindeyse runtime'da K=3 veya adaptive K kullanılabilir.

---

## Önerilen Uygulama Sırası

Bu sırayla ilerlemek en mantıklı yoldur:

1. ~~Koordinat audit'i yap.~~ Eval protokolü kısmı hâlâ açık.
2. ~~Contact/penetration loss ve metriklerinin aynı frame'de çalıştığını sayısal olarak doğrula.~~ Görsel audit hâlâ faydalı olur.
3. ~~Phase 2 temporal loss'u gerçekten aktif et.~~
4. ~~Fingertip FK loss ekle.~~
5. ~~Eval protokolünü standartlaştır; NaN/failed run filtrele.~~
6. ~~Mixed HOT3D + OakInk Phase 2 training dene.~~
7. ~~Quality proxy label üret.~~
8. ~~LR scheduler + early stopping ekle.~~
8. CVAE K ablation ve oracle selection eval'i yap.
9. Unity physics success label pipeline'ı kur.
10. Success head'i Phase 3 olarak kalibre et.
11. Joint limit saturation, latency ve final skor kartını ekle.

## Final Skor Kartı

Tek bir metrik final model seçimi için yeterli değildir. Final checkpoint şu skor kartıyla seçilmelidir:

Primary:

```text
HOT3D geodesic error
fingertip error
contact ratio
penetration depth
Unity success rate
```

Secondary:

```text
jitter velocity
jitter acceleration
joint limit violation
joint limit saturation
quality AUC / Spearman
success AUC / ECE
latency
```

Final model, yalnızca en düşük pose loss'a sahip model olmamalıdır. Daha doğru seçim şudur:

```text
Pose error düşük,
fingertip error düşük,
contact yüksek,
penetration düşük,
jitter düşük,
Unity success yüksek,
latency kabul edilebilir.
```

## Son Değerlendirme

İki analizdeki öneriler genel olarak doğru yöndedir ve uygulanırsa gerçek fayda sağlayabilir. Ancak faydanın kilidi sıradadır.

Önce koordinat sistemi ve eval protokolü düzeltilmelidir. Çünkü metrikler yanlışsa hangi loss'un işe yaradığını anlamak mümkün değildir. Sonra temporal loss ve fingertip loss aktif edilmelidir. Bu iki adım Phase 2'nin gerçek amacını, yani zaman içinde stabil ve fiziksel olarak anlamlı kapanışı, doğrudan hedefler.

Quality head, CVAE diversity ve Unity success head daha sonra gelmelidir. Bunlar nihai sistemi güçlendirir, fakat temas ve penetrasyon sinyalleri güvenilir hale gelmeden sağlam sonuç üretmeleri beklenmemelidir.
