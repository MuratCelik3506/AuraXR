# 2. Related Work

## 2.1. Hand Pose Estimation and Representation
- MANO parametrik el modeli (shape β, pose θ)
- UmeTrack (HOT3D'nin eklem temsili): bilek + 15 eklem rotasyonu
- Axis-angle vs quaternion vs 6D sürekli rotasyon temsili
- HOT3D'nin bilinen thumb DOF sınırlılığı

## 2.2. Object-Conditioned Grasp Generation
- GrabNet: CoarseNet + RefineNet, CVAE tabanlı çok-modlu grasp
- ContactOpt: temas haritasından geriye doğru optimizasyon
- GraspTTA: test-time adaptation ile grasp iyileştirme
- PointNet/PointNet++ tabanlı obje encoding

## 2.3. Temporal Hand Motion Models
- Sekans tabanlı el hareketi modelleri (LSTM/GRU/Transformer)
- Causal temporal encoding: gelecek bilgisi kullanmadan gerçek zamanlı çalışma
- Jitter azaltım yöntemleri: velocity/acceleration loss, EMA smoothing

## 2.4. Variational Approaches for Multi-Modal Grasp
- CVAE ile çok-modlu grasp dağılımı modelleme
- β-VAE: KL ağırlığı warm-up ile latent space düzenleme
- Diffusion tabanlı grasp generation (karşılaştırma / future work)

## 2.5. XR/VR Grasp Systems and Real-Time Constraints
- Unity XR Hands altyapısı ve OpenXR eklem temsili
- PC-based XR: latency bütçesi ve inference kısıtları
- Gerçek zamanlı grasp sistemleri: latency-quality tradeoff
