# Paper Draft — Dosya Dizini

## Paper Bölümleri

| Dosya | Bölüm | İçerik | Durum |
|---|---|---|---|
| [01-introduction.md](01-introduction.md) | 1. Introduction | Problem, katkılar, paper yapısı | Taslak |
| [02-related-work.md](02-related-work.md) | 2. Related Work | MANO, GrabNet, temporal modeller, XR | Taslak |
| [03-method.md](03-method.md) | 3. Method | Mimari: PointNet, GRU, self-attn, CVAE, heads, loss | Koddan güncellendi |
| [04-training.md](04-training.md) | 4. Training | Dataset formatları, split, 3 aşamalı training + gerçek log'lar | Dolduruldu |
| [05-evaluation.md](05-evaluation.md) | 5. Evaluation Protocol | Metrikler, Unity eval protokolü, latency | Taslak |
| [06-experiments.md](06-experiments.md) | 6. Experiments and Results | Tüm eval JSON sonuçları, per-object tablo, açık bekleyenler | Dolduruldu |
| [07-discussion.md](07-discussion.md) | 7. Discussion | Contact sorunu, jitter, confidence, per-object analiz | Gerçek bulgularla güncellendi |
| [08-conclusion.md](08-conclusion.md) | 8. Conclusion | 5 ana katkı, gelecek çalışmalar | Bulgular bekliyor |

## Ek / Referans Bölümleri

| Dosya | Bölüm | İçerik |
|---|---|---|
| [09-training-curves.md](09-training-curves.md) | 9. Training Curves | Phase 1 (50 ep) + Phase 2 (14 ep) tüm loss tabloları, loss bileşeni analizi |
| [10-model-architecture-detail.md](10-model-architecture-detail.md) | 10. Mimari Detay | Tüm katman boyutları, parametre sayıları, loss ağırlıkları, MANO eklem sırası |
| [11-data-pipeline.md](11-data-pipeline.md) | 11. Data Pipeline | Canonical format, HOT3D segmentasyon parametreleri, split, quality label, normalizasyon |
| [12-implementation-notes.md](12-implementation-notes.md) | 12. Implementation Notes | 7 kritik bug fix, ONNX export kısıtlaması, processed veri durumu |
| [13-unity-integration.md](13-unity-integration.md) | 13. Unity Entegrasyonu | XR retarget, blending, ONNX versiyonları, demo scene durumu, fizik eval protokolü |

---

## Mevcut Sonuç Özeti (commit 52f3ce8)

### Phase 1 — OakInk (aura_phase1_best.pt, epoch 46, T=16)

| Metrik | Val | Test |
|---|---|---|
| Geodesic error | 9.54° | 9.72° |
| MPJPE | 5.64mm | 5.73mm |
| Fingertip err | 11.78mm | 12.02mm |
| Contact ratio | 0.140 | 0.131 |
| Penetration | 0.46mm | 0.48mm |
| Jt. limit viol. | 38.6% | 38.2% |
| quality AUC | 0.944 | 0.967 |
| quality Spearman | 0.724 | 0.712 |

### Phase 2 — HOT3D (aura_phase2_best.pt, epoch 50, T=16)

| Metrik | HOT3D val (K=1) | HOT3D test (K=1) | OakInk val |
|---|---|---|---|
| Geodesic error | 11.70° | 12.58° | 9.16° |
| MPJPE | 6.06mm | 6.62mm | 5.48mm |
| Contact ratio | 0.146 | 0.230 | 0.139 |
| Penetration | 4.02mm | 1.36mm | 0.46mm |
| Jt. limit viol. | 53.9% | 55.1% | 38.3% |
| Jitter score | 6.15 | 4.91 | — |
| Geodesic vel | 3.36°/frame | 3.59°/frame | — |
| quality AUC | 0.336 | 0.704 | 0.949 |
| quality Spearman | 0.157 | 0.500 | 0.727 |
| Diversity (K=5) | 0.051 | — | — |

---

## Kritik Açık Problemler (Öncelik Sırasıyla)

| Problem | Ciddiyet | İlgili Bölüm |
|---|---|---|
| Contact ratio <<0.70 hedefi (mevcut: 0.13–0.23) | Yüksek | §6.0, §7.6, §9.3, §12.1 |
| Joint limit violation %38–54 | Yüksek | §6.0, §7.6 |
| HOT3D quality_score Spearman 0.157 (val) | Orta | §6.6, §7.3 |
| CVAE diversity 0.051 (KL weight çok düşük) | Orta | §6.5 |
| Unity physics eval yapılmadı | Orta (daha geç) | §6.10, §13.8 |
| success_prob eğitilmedi (Unity label yok) | Orta (daha geç) | §6.6 |
| Ablation deneyleri çalıştırılmadı | Düşük | §6.8 |
| Phase 2 yalnızca 14 epoch | Düşük | §9.2 |
| Unity GPUCompute backend test edilmedi | Düşük | §13.7 |

## Tamamlanacaklar (Öncelik Sırasıyla)

- [ ] Contact loss redesign (gerçek mesh SDF) veya hinge threshold optimizasyonu
- [ ] Joint limit weight artışı  
- [ ] SingleFrame baseline → temporal katkı kanıtı
- [ ] BBox baseline → geometri katkısı kanıtı
- [ ] Phase 2 daha uzun eğitim (50+ epoch)
- [ ] KL weight artışı → diversity iyileştirme
- [ ] Unity physics eval pipeline → success_label
- [ ] Ablation deneyleri
- [ ] Runtime latency benchmark (Python + Unity GPU)
