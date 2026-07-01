# 8. Conclusion

## 8.1. Ana Katkıların Özeti

1. **Birleşik veri pipeline'ı:** OakInk ve HOT3D farklı formatlarını `frame_feat (B,T,13)` standardına indirgeyen preprocessing; iki veri setinin aynı modelde kullanılmasını teknik olarak mümkün kıldı. Koordinat dönüşüm ve FK hatalarının düzeltilmesi bu sürecin büyük bölümünü oluşturdu.

2. **Temporal Geometry-Conditioned Grasp Model mimarisi:** Mini PointNet + FiLM + GRU + 15-eklem self-attention + CVAE bileşenlerinden oluşan model uygulandı ve eğitildi. OakInk üzerinde geodesic error ~9.7°, MPJPE ~5.7mm elde edildi. HOT3D temporal üzerinde ~12.6° geodesic error üretildi; ancak contact ratio hedefi (%70) karşılanamadı (mevcut: %13–23).

3. **İki-aşamalı eğitim:** OakInk pre-training + HOT3D fine-tuning pipeline'ı çalışır hale getirildi. L_vel ve L_acc temporal loss terimleri aktif edildi. Tek başına jitter azaltımına etkisi baseline karşılaştırması olmadan ölçülemedi.

4. **Confidence head'ler:** `quality_score` heuristic label OakInk'te Spearman 0.72 korelasyon gösterdi; HOT3D temporal'de 0.16'ya düştü. `success_prob` head Unity physics label olmadığından eğitilmedi — mevcut durumda bu head'in çıktıları bilgi taşımıyor.

5. **Unity demo entegrasyonu:** MANO → XR Hands retarget, ONNX export (Unity InferenceEngine GRU kısıtı nedeniyle GRU unroll gerektirdi) ve scene bileşenleri uygulandı. Fizik eval pipeline'ı henüz kurulmadı.

## 8.2. Temel Bulgular

- Contact ratio mevcut kayıp fonksiyonu ve centroid-proxy penalty ile hedef değerin (~%13–23 vs hedef %70) çok altında kaldı. Bu sistemin en kritik açık sorunudur.
- Joint limit violation oranı %38–55 — soft loss yeterli anatomik kısıtlama sağlamadı.
- CVAE K>1 mevcut KL ağırlığı (0.01) ile çeşitlilik üretemedi (diversity 0.051); K artışı metriklerde anlamlı fark yaratmadı.
- HOT3D quality_score kalibrasyonu başarısız; temporal bağlamda heuristic label kalite sinyali olarak işe yaramıyor.
- HOT3D temporal fine-tuning'in single-frame baseline'a göre katkısı ablation deneyleri yapılmadan ölçülemedi.

## 8.3. Açık Problemler ve Gelecek Çalışmalar

**Kritik (mevcut sonuçları geçersiz kılan):**
- Contact loss redesign: centroid-proxy yerine gerçek mesh SDF; hinge threshold optimizasyonu
- Joint limit regularizasyonu güçlendirilmesi
- Unity physics eval ve success_label pipeline'ı

**Metodoloji tamamlama:**
- SingleFrame ve BBox baseline deneyleri — temporal ve geometri katkılarını ölçmek için zorunlu
- Phase 2 daha uzun eğitim (14 epoch muhtemelen yetersiz)
- KL weight artışı ile diversity iyileştirme

**Gelecek araştırma:**
- Görülmemiş obje kategorilerine genelleme (obje bazlı split ile test)
- PointNet++ veya lokal geometri yakalayan encoder
- Kullanıcı çalışması (canned animasyon baseline karşılaştırması)
- Standalone Quest deployment (model küçültme)
