# AuraXR — Doğruluk İyileştirme: Faz Sonuçları & Derin Analiz

Tüm metrikler **free-running MPJPE (mm)** — gerçek dağıtım koşulu, joint-space, FK ile (temsil-bağımsız).
Bu çalışmadaki tüm modeller: **çıktı = 45-dim axis-angle (aa45)**, **4 kategori** (hook/power/wide/pinch),
**birleşik veri** (HOT3D + DexYCB), **denek-held-out** split.

## Veri (Faz 1–3)
| Kaynak | Frame (kullanılabilir) | Not |
|---|---|---|
| HOT3D (4-kat, pinch dahil) | 316,743 | pinch +53,404 frame eklendi |
| DexYCB (sağ el) | 29,043 (480 session, 576 segment) | 10 yeni denek, YCB nesneleri |
| **Birleşik split** | train 263,303 / val 30,650 / test 51,833 | 53 nesne |

## Faz 5 — Ana 5 varyant (TEST, 8000 frame)
| Varyant | MPJPE | mod | not |
|---|---|---|---|
| **baseline (noprev)** | **7.65** | feedforward | 5'in en iyisi |
| C: obj_size | 8.19 | AR | |
| *cat_mean (statik)* | 8.21 | — | referans |
| A+C: fk+obj_size | 8.48 | AR | |
| B: AR+sched_samp | 8.55 | AR | ss_max=0.25 |
| A: fk_loss | 10.40 | AR | en kötü |

→ **Tüm autoregressive (AR) varyantlar feedforward baseline'ın altında**, çoğu statik cat_mean'i ya zar zor geçiyor ya da geçemiyor.

## Faz 6 — Mimari (hepsi feedforward + obj_size + fk; fark = backbone)
| Mimari | TEST MPJPE | val | params | yorum |
|---|---|---|---|---|
| **LSTM (feedforward)** | **7.21** | 8.51 | 300K | en iyi doğruluk/param |
| TCN (causal) | 7.32 | 8.45 | 627K | |
| Transformer (causal) | 7.17 | 8.16 | 2.13M | marjinal iyi, 7× param |

→ **En iyi model: feedforward LSTM + obj_size + FK = 7.21 mm.** obj_size+FK feedforward'a eklenince noprev'i 7.65→7.21 (−%6) iyileştirir; Transformer'ın 0.04mm avantajı 7× parametreye değmez (VR realtime).

## En iyi modelin kaynak/kategori kırılımı (arch_lstm, TEST)
| Kaynak | MPJPE | hook | power | wide | pinch |
|---|---|---|---|---|---|
| HOT3D (held-out denek) | 7.22 | 7.6 | 7.0 | 7.0 | 7.7 |
| DexYCB (yeni denek+nesne) | 7.07 | 6.6 | 6.2 | 7.9 | 7.8 |

→ **Çapraz-veri genellemesi başarılı**: hiç görülmemiş DexYCB denek/nesneleri (7.07) HOT3D held-out kadar (7.22) iyi. Yeni `pinch` kategorisi sağlam (~7.7).

## Neden başarılı / başarısız — analiz
1. **Feedforward ≫ Autoregressive (en kritik bulgu).** Birleşik/çeşitlenmiş veride AR free-running'de hata birikir (drift); feedforward bundan muaf. Orijinal (HOT3D-only, pca15) tabloda A+C (AR) kazanıyordu (6.99) — büyük + zor veride bu tersine döndü.
2. **obj_size + FK loss sinerjisi yalnız feedforward'da görünür.** AR modda drift faydayı maskeliyor (A+C=8.48); feedforward'da net kazanç (arch_lstm=7.21). Yani "özellik/loss faydasız" değil — "AR ile birlikte faydasız".
3. **DexYCB değeri = çeşitlilik, miktar değil.** Sadece +29k frame (%11) ama 10 yeni denek + YCB nesneleri; sonuç çapraz-veri genellemenin doğrulanması.
4. **aa45 takası.** Doğrudan 45-dim öğrenme erken epoch'ta daha yavaş yakınsıyor; PCA prior'ı kaybolduğu için en zor sınıf (hook) hafif geriliyor ama Unity'de PCA decode kalkıyor (daha basit runtime) ve ifade tavanı yükseliyor.
5. **Scheduled sampling (B) yine zayıf** (8.55) — ss_max düşürmek (0.25) felaketi önledi ama AR'ın temel sorununu çözmedi.

## Faz 7 — Dağıtım
- ONNX export (`onnx/c2h_step.onnx`): stateful per-frame, **çıktı aa45 (45-dim)**, feat_dim=16 (obj_size), 4 kategori. torch↔onnxruntime parite **3.6e-7 (OK)**.
- `MANODecoder.DecodeAxisAngle()` eklendi (PCA decode'suz, basis-free aa45→quaternion).
- `c2h_meta.json` 4 kategori + obj_extent feature + aa45 çıktı.
- **Kalan (Unity-in-the-loop):** InferenceManager'ın EMA-smoothing & mirror-mask tamponları hâlâ 15-dim PCA; aa45 için 45-dim'e genişletilmeli (test Unity'de yapılmalı).

## Çapraz-veri (zero-shot) — gerçek genelleme testi
`split_hot3d_only.json` ile: train=HOT3D, test=TÜM DexYCB (hiç görülmemiş). Runner: `scripts/run_cross_dataset.sh`.
| Koşul | DexYCB MPJPE | static'e karşı kazanç |
|---|---|---|
| Birleşik eğitim → DexYCB (DexYCB train'de) | 7.07 | + |
| **HOT3D-only → DexYCB (zero-shot)** | **9.16** | **−0.56** (statiğin altında!) |

→ Sıfır-atışta domain kayması ~2mm; model statik cat_mean'in bile altına düşüyor → DexYCB verisini eğitime katmak **gerçekten gerekli** (kanıt). Parmak-ucu sıfır-atışta 18.5mm.

## Obje-bazlı skorlama (52 obje, en iyi model)
En iyi (rijit/stereotipik): 021_bleach_cleanser 4.43 · 004_sugar_box 6.40 · spatula_red 6.46 · keyboard 6.87.
En kötü (derin/içbükey/ince-belirsiz): **024_bowl 11.63** · cellphone 10.07 · vase 9.61 · birdhouse_toy 9.22.
Desen: `bowl` HOT3D 7.03 vs `024_bowl` DexYCB 11.63 — aynı kategori, farklı set; kategori-ortalaması gizlerken obje-kırılımı ortaya çıkarır. Tam tablo: `results/by_object_<tag>.csv`.

## Ek metrikler
- **Parmak-ucu (distal) MPJPE: 14.5mm** — all-joint'in ~2 katı (kinematik zincir hata birikimi); görsel/precision açısından zayıf nokta.
- **Statik baseline'a kazanç: +1.0mm (%12)** — bilek controller'dan geldiği, parmaklar stereotipik olduğu için mütevazı. En çok `wide` (+1.4), en az `pinch`/`power` (~+0.6).

## Değerlendirme altyapısı (otomatik, tekrar-üretilebilir)
- **`src/eval_metrics.py`** = tek doğruluk kaynağı. `full_eval()` tüm metrikleri üretir → `results/eval_<tag>.json` + `results/by_object_<tag>.csv`.
- **`train.py` her eğitim sonunda** en iyi checkpoint'i TEST'te otomatik değerlendirir (yeni model = otomatik metrikler).
- `compare_variants.py` aynı modülü kullanır (cap=8000 hızlı tablo, 8 varyant + 3 mimari).
- `eval_metrics.py --tag <t> --split <s>` ile herhangi bir model/split standalone değerlendirilir.

## Özet karar
**Üretim modeli: `checkpoints/c2h_arch_lstm.pt`** (feedforward LSTM + obj_size + FK, aa45, 4-kat) → **7.21 mm**, çapraz-veri genelleyen, VR-realtime uyumlu (300K param, stateful ONNX).
