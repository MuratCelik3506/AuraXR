# 7. Discussion

## 7.1. Geometri Koşullamanın Katkısı
- BBox baseline deneyi yapılmadığından point cloud'un somut katkısı sayısal olarak gösterilemiyor
- Per-object geodesic hata farkı (cup 4.53° – mouse 11.68°) obje şekliyle ilişkili görünüyor; ancak bu farkın geometri encodingından mı yoksa örneklem büyüklüğünden mi (mouse: 80, cup: 3 örnek) kaynaklandığı belirsiz
- FiLM conditioning ve PointNet uygulandı; bunların BBox baseline'dan üstün olduğu henüz ölçülmedi
- PointNet global pooling: lokal yüzey bölgeleri (kulp vs. gövde) ayrıştırılamıyor — K adayların "farklı kavrama stratejisi" değil "farklı eklem konfigürasyonu" olduğu anlamına geliyor
- Diversity 0.051: CVAE anlamlı çeşitlilik üretemiyor; KL weight 0.01 çok düşük

## 7.2. Temporal Bağlamın Etkisi
- HOT3D test jitter_score 4.91, geodesic_velocity 3.59°/frame — bu değerlerin iyi mi kötü mü olduğu SingleFrame baseline olmadan değerlendirilemez
- val jitter_score (6.15) > test jitter_score (4.91) farkı: held-out objeler veya obje sayısının azlığından kaynaklanan varyans olabilir; yorumlanmamalı
- L_vel ve L_acc loss değerleri (train_vel ~0.0003, train_acc ~0.047) çok küçük kaldı; temporal regularizasyonun etkisi belirsiz
- Phase 2 fine-tuning joint limit violation'ı %38'den %54'e çıkardı: temporal eğitim anatomik kısıtlamayı bozdu
- K=1/3/5 arasında jitter neredeyse aynı (6.15 → 6.14): aday seçiminin jitter'a etkisi yok

## 7.3. Confidence Head Performansı
- OakInk quality_score Spearman 0.712–0.727 — heuristic label statik bağlamda kabul edilebilir korelasyon gösteriyor; ancak bu score'un gerçek fizik başarısıyla ilişkisi Unity eval olmadan bilinmiyor
- HOT3D quality_score Spearman 0.157 (val): temporal bağlamda heuristic label kalite sinyali olarak işe yaramıyor
- HOT3D val AUC 0.336: rastgele tahminden bile düşük — val set için quality_score'un gücü yok
- success_prob head Unity label olmadan eğitilemedi; mevcut çıktıları bilgi taşımıyor
- Sonuç: mevcut confidence head'lerin temporal grasp kalitesini öngörme kapasitesi çok sınırlı

## 7.4. Veri Seti Sınırlılıkları
- HOT3D thumb DOF hatası: başparmak açıları yüksek gürültülü; bu etki loss ve metrikler üzerinde ölçülmedi
- HOT3D 33 obje: val/test bölümlerinde 3'er obje var — bu kadar az örnekle metrik güvenilirliği sınırlı, sonuçlar obje-level varyansa çok duyarlı
- OakInk split sample-bazlı yapıldı, obje-bazlı değil: aynı objenin farklı graspları train ve test'te bir arada olabilir — gerçek görülmemiş obje genellemesi test edilmedi
- HOT3D segmentlerinin ~%79'unda quality_label = 0 (yaklaşım fazı): model büyük oranda "düşük kaliteli" örneklerle eğitildi

## 7.5. Sistem Tasarım Kararları

Kararlar ve mevcut sonuçlara göre değerlendirmeleri:

- Kural tabanlı faz geçişi: ayrı Approach Model karmaşıklığını elimine etti; ancak geçiş penceresinin blend davranışı Unity demo'da henüz test edilmedi
- GRU: basit, causal; katkısı Transformer/TCN varyantlarıyla karşılaştırılmadı
- CVAE: çok-modluluk amaçlandı; mevcut diversity 0.051 ile bu amaç karşılanamadı
- Hard clamp kaldırılıp soft joint limit loss kullanıldı: violation oranı hâlâ %38–54; bu kararın etkisi belirsiz

## 7.6. Başarısızlık Analizi

**Contact ratio (en kritik sorun):**
- Tüm koşullarda hedefin (~%70) çok altında kaldı: %13–23
- 24 OakInk val objesinin 8 tanesinde contact_ratio = 0.000 — model bu objeler için hiç temas üretmiyor
- Centroid-proxy L_contact ve 15mm hinge toleransı temas sinyali olarak yetersiz kaldı; yüzey temasını yönlendiremiyor

**Joint limit violation:**
- Phase 2 sonrası %54: model çıktılarının yarısından fazlası anatomik sınır dışında
- Saturation rate ~0.6%: ihlaller var ama soft loss bunları etkin biçimde düzeltemedi

**Per-object pattern:**
- Büyük/ağır objeler (mouse, cameras) → yüksek geodesic hata; küçük/silindirik objeler → düşük
- Bu örüntünün geometri encoding kalitesiyle mi yoksa örneklem boyutuyla mı ilgili olduğu kontrol edilmedi
- can_17, toothbrush_25, wineglass_14, screwdriver_21 gibi küçük/uzun objeler: contact_ratio = 0.000 ve düşük geodesic hata — model şekle yakın ama temas etmiyor

**Unity eval eksikliği:**
- Fizik başarısı ölçülmedi; tüm iddialar offline metriklere dayalı; gerçek XR kullanımındaki davranış bilinmiyor
