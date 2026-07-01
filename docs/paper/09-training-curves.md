# 9. Training Curves ve Loss Analizi

Checkpoint: `git commit 52f3ce8`. Tüm eğitimler Apple Silicon MPS üzerinde yapıldı.

---

## 9.1. Phase 1 — OakInk Statik Pre-training (50 Epoch)

**Konfigürasyon:**
- Optimizer: Adam, lr=3e-4
- KL weight: 0.0 → 0.01 (warm-up)
- Loss weights: limit=1.0, contact=0.3, penetration=0.1, tip=0.5, quality=0.1
- Checkpoint: `aura_phase1_best.pt` (epoch 46)
- Ortalama epoch süresi: ~11 saniye

**Reconstruction Loss (train / val):**

| Epoch | Train rec | Val rec | Val total |
|---|---|---|---|
| 0 | 0.04013 | 0.02417 | 0.09727 |
| 5 | — | — | — |
| 10 | 0.01973 | 0.01786 | — |
| 20 | 0.01612 | 0.01511 | — |
| 30 | 0.01447 | 0.01418 | — |
| 40 | — | — | — |
| 46 (best) | 0.01264 | **0.01205** | **0.05684** |
| 47 | 0.01261 | 0.01211 | 0.05689 |
| 48 | 0.01251 | 0.01236 | 0.05695 |
| 49 | 0.01245 | 0.01227 | 0.05706 |

**Loss bileşenleri (epoch 46–49, son 4 epoch):**

| Epoch | val_kl | val_limit | val_contact | val_penetration | val_quality |
|---|---|---|---|---|---|
| 46 | 0.07456 | 0.009670 | 0.10987 | 0.000222 | 0.01103 |
| 47 | 0.08008 | 0.009794 | 0.11001 | 0.000218 | 0.01142 |
| 48 | 0.07645 | 0.009626 | 0.10978 | 0.000228 | 0.01230 |
| 49 | 0.07415 | 0.009618 | 0.10980 | 0.000225 | 0.01441 |

**Gözlemler:**
- val_contact ~0.109–0.110: sabit kaldı, hiç iyileşmedi — contact loss yönlendirme yapmıyor
- val_penetration çok küçük (0.00022): centroid-proxy penetration penalty neredeyse sıfır — anlamlı ceza vermiyor
- val_quality: 0.011–0.014 arasında dalgalanıyor — heuristic label var ama öğrenme gürültülü
- val_kl: 0.074–0.080, yüksek varyans — latent space yeterince regularize değil mi?
- best epoch 46: 50 epoch içinde erken convergence, son 4 epoch stabil

---

## 9.2. Phase 2 — HOT3D Temporal Fine-tuning (14 Epoch: 47–60)

**Konfigürasyon:**
- Optimizer: AdamW, lr=1e-4 → 5e-5 (epoch 57'de decay)
- Mixed batch: %70 HOT3D + %30 OakInk replay
- vel_weight=0.01, acc_weight=0.04 (yalnızca HOT3D batch)
- Grad clip: max_norm=1.0
- Ortalama epoch süresi: ~322 saniye (~5.4 dakika)

**Tüm Phase 2 Epoch Logları:**

| Epoch | Train rec | Val rec | Train vel | Train acc | Val vel | Val acc | LR | Val total |
|---|---|---|---|---|---|---|---|---|
| 47 | 0.01813 | 0.02230 | 0.000430 | 0.02775 | 0.000286 | 0.06181 | 1e-4 | — |
| 48 | 0.01768 | 0.02240 | 0.000510 | 0.02986 | 0.000282 | 0.06368 | 1e-4 | — |
| 49 | 0.01763 | 0.02248 | 0.000450 | 0.03114 | 0.000277 | 0.06545 | 1e-4 | — |
| 50 (best) | 0.01759 | **0.02200** | 0.000400 | 0.03180 | 0.000283 | 0.06561 | 1e-4 | **0.06988** |
| 51 | 0.01755 | 0.02186 | 0.000380 | 0.03356 | 0.000280 | 0.06610 | 1e-4 | — |
| 52 | 0.01747 | 0.02159 | 0.000360 | 0.03656 | 0.000281 | 0.06714 | 1e-4 | — |
| 53 | 0.01742 | 0.02197 | 0.000350 | 0.03736 | 0.000282 | 0.06840 | 1e-4 | — |
| 54 | 0.01734 | 0.02163 | 0.000340 | 0.03971 | 0.000279 | 0.06907 | 1e-4 | — |
| 55 | 0.01726 | 0.02142 | 0.000330 | 0.04225 | 0.000286 | 0.07006 | 1e-4 | — |
| 56 | 0.01723 | 0.02198 | 0.000310 | 0.04219 | 0.000285 | 0.07038 | 1e-4 | — |
| 57 | 0.01711 | 0.02144 | 0.000290 | 0.04400 | 0.000286 | 0.07084 | 5e-5 | — |
| 58 | 0.01707 | 0.02154 | 0.000297 | 0.04615 | 0.000286 | 0.07176 | 5e-5 | — |
| 59 | 0.01703 | 0.02122 | 0.000286 | 0.04728 | 0.000283 | 0.06492 | 5e-5 | 0.07015 |
| 60 | 0.01700 | 0.02159 | 0.000281 | 0.04854 | 0.000281 | 0.06517 | 5e-5 | 0.07001 |

**Gözlemler:**
- val_rec Phase 2 başında +%80 arttı (0.012 → 0.022): HOT3D temporal distribüsyonu OakInk'ten farklı — beklenen
- train_vel monoton azalıyor (0.00043 → 0.00028): hız sinyali öğreniliyor ama çok küçük değerler
- train_acc monoton artıyor (0.028 → 0.049): ivme loss artıyor — model GT ivmeyi daha fazla öğrenmeye çalışıyor
- val_acc çok daha büyük değerler (0.061–0.071): val seti ivme dinamiğinde daha büyük hatalar
- best epoch 50 (val_total 0.06988): Phase 2 erken convergence, 14 epoch yeterli mi tartışmalı
- LR decay epoch 57: 5e-5'e düşünce train_rec biraz daha azalıyor ama val tutarsız

---

## 9.3. Loss Bileşeni Yorumları

### L_contact (val ~0.06–0.11)
- Phase 1'de 0.110 civarında sabit kaldı. Phase 2'de 0.060'a indi — temporal fine-tuning contact loss'u biraz azalttı
- Ama contact_ratio metriğinde (0.13–0.23) yansımıyor: model contact loss'u minimize ediyor ama gerçek temas üretmiyor
- **Sonuç**: Contact loss sinyali guidance için yetersiz. Hinge threshold (15mm) çok büyük olabilir, centroid-proxy penetration ise gerçek yüzeyi temsil etmiyor

### L_penetration (val ~0.0002)
- Tüm Phase 1 boyunca ~0.0002: neredeyse sıfır ceza. Centroid-proxy parmakların obje içine girmediğini gösteriyor (çünkü parmaklar zaten objeye ulaşmıyor)

### L_quality (val 0.011–0.035)
- Phase 2'de artıyor (0.011 → 0.033): HOT3D temporal sekansda quality prediction daha zor
- HOT3D Spearman 0.157 ile uyumlu — quality head HOT3D için anlamlı sinyal öğrenemiyor

### L_vel / L_acc
- Phase 2'de aktif, küçük değerler. Temporal smoothness sinyali mevcut ama baskın değil
- val_acc (0.06–0.07) >> val_rec (0.02): ivme residual büyük — model temporal ivme dinamiğini tam öğrenemiyor

---

## 9.4. Önerilen İyileştirmeler (Gözlemlerden)

1. **Contact loss redesign**: Centroid-proxy yerine gerçek mesh SDF kullan; hinge threshold'u 5–10mm'ye indir
2. **Contact weight artışı**: Mevcut 0.3'ten 1.0'a çıkar; contact loss dominant hale gelsin
3. **Joint limit loss weight**: Artır; temporal fine-tuning sonrası %54 violation kabul edilemez
4. **KL weight**: 0.01 çok düşük — diversity 0.051 ile tutarlı. 0.1'e çıkar ve diversity_score'u izle
5. **Phase 2 epoch sayısı**: 14 epoch yetersiz. 50+ epoch ile daha uzun fine-tuning
6. **Temporal quality label**: HOT3D için frame-level heuristic yerine sequence-level quality tasarla
