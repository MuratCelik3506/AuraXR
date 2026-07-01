# 1. Introduction

## 1.1. Problem Tanımı
- VR/XR ortamında gerçek zamanlı el kavraması problemi
- Kullanıcının controller hareketini korurken parmak kapanışını otomatik üretme

## 1.2. Mevcut Yaklaşımların Sınırlılıkları
- Önceden tanımlanmış (canned) animasyonlar: obje geometrisine uyum sağlayamaz
- Statik grasp modelleri: temporal geçişi modelleyemez, jitter üretir
- Tek-frame modeller: bağlamsal el hareketini görmezden gelir

## 1.3. Tezin Katkıları

Aşağıdaki katkılar tasarım ve uygulama düzeyinde gerçekleştirilmiştir; ancak bazılarının etkinliği (özellikle 4–6) henüz karşılaştırmalı deneylerle kanıtlanmamıştır:

1. OakInk (statik) ve HOT3D (temporal) veri setlerini tek modelde birleştiren eğitim pipeline'ı ve `frame_feat (B, T, 13)` ortak arayüzü
2. Mini PointNet + FiLM conditioning ile obje geometrisine koşullu grasp üretimi
3. GRU + 15-eklem self-attention ile temporal bağlam kodlama
4. CVAE ile K aday üretimi ve `success_prob` tabanlı aday seçim mekanizması — ancak mevcut sonuçlarda K>1'in anlamlı katkısı gözlemlenmedi
5. Heuristic `quality_score` head — OakInk'te orta düzey korelasyon (Spearman ~0.72), HOT3D temporal'de yetersiz (Spearman ~0.16)
6. Unity physics eval ve confidence kalibrasyon protokolü tasarımı — Unity label pipeline'ı henüz tamamlanmadı

## 1.4. Paper Yapısı
- Section 2: Related Work
- Section 3: Method
- Section 4: Training
- Section 5: Evaluation Protocol
- Section 6: Experiments and Results
- Section 7: Discussion
- Section 8: Conclusion
