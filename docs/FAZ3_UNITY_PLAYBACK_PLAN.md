# Faz 3A — Unity Non-VR Playback Testi (Plan)

Amaç: Eğitilen modeli **gerçek Unity'de** çalıştırıp pipeline'ı de-risk etmek
(canlı VR değil; kaydedilmiş bir test trajektorisini Unity'de oynatıp doğrulamak).
Hedef "öğrenilebilirlik" değil, **entegrasyon doğruluğu**.

## Nerede kaldık (durum)
- **Model:** `c2h_noprev.pt` (non-autoregressive, test free-running 7.4mm; baseline'ları geçiyor).
- **ONNX:** `onnx/c2h_step.onnx` — tek-adım stateful (`feat(13)+category+h,c → pca15+h,c`),
  gömülü ağırlık (legacy exporter), **onnxruntime paritesi 2.4e-7 OK**.
- **Unity projesi:** `/Users/muratcelik/Desktop/Thesis/Unity/AURAXR/` (Unity 6.4,
  **Unity.InferenceEngine kurulu**, sahne `AuraXR_GamifiedModelTest`, `MANODecoder` +
  `AuraXRInferenceManager` scriptleri mevcut — ama eski SDF-tabanlı mimariden).
- **Unity'ye kopyalandı:** `Assets/AuraXR/Models/c2h_step.onnx` (import OK, ModelAsset),
  `Assets/StreamingAssets/c2h_playback.json` (47-frame mug_white test segmenti:
  her frame `feat[13], category, pca_expected, pca_gt, wrist_t/q, obj_t/q, betas`).
- **Açık sorun:** Inference Engine'de frame-frame inference denerken NullReference —
  büyük ihtimalle **çıktı ismi "pca15" değil** (importer yeniden adlandırmış) →
  `PeekOutput("pca15")` null. Çözüm: gerçek input/output isimlerini sorgulayıp kullanmak.

## Plan (adımlar)

### Adım 1 — Inference paritesi (de-risk #1: model Unity'de doğru koşuyor mu?)
1. Modelin gerçek input/output isimlerini sorgula (`model.inputs/outputs`); importer
   yeniden adlandırdıysa ona göre `PeekOutput(name)` veya index-bazlı oku.
2. `c2h_step.onnx`'i playback JSON üzerinde frame-frame koştur; LSTM state'i (h,c) geri besle.
3. Unity `pca15` ↔ torch `pca_expected` karşılaştır → **max|diff| < 1e-3** assert.
- **Çıktı:** parite PASS logu. (Inference Engine API ayrıntıları — tensor readback,
  isimler — burada iterasyonla netleşir.)

### Adım 2 — Decode + el rig'ini sürme (de-risk #2: PCA→poz→rig)
1. MANO bileşenlerini Unity'ye taşı: `data/models/mano_right_components.npz` →
   Unity-okunur asset (JSON/bytes: `hands_components[:15]`, `hands_mean`, J_regressor,
   v_template, kintree, shapedirs).
2. Mevcut `MANODecoder` aynı PCA bazını kullanıyorsa onu kullan; değilse kendi decode'umuzu
   (pca15→aa45) + joint-only FK / **rig retarget**'i C#'a port et.
3. Tahmin edilen parmak rotasyonlarını hand rig kemiklerine uygula
   (`HandRigController` / `BoneMappingValidator`). Bilek = playback `wrist_t/q`.

### Adım 3 — Handedness / koordinat dönüşümü (de-risk #3 — işaretli ana risk)
- Veri **sağ-elli** (HOT3D), Unity **sol-elli**. Pozisyon, quaternion ve göreli-poz
  feature'ları için dönüşümü tanımla + uygula.
- **Doğrulama:** Unity'deki el+obje pozunu Python görseliyle (`results/viz/reach_grasp_mug_white.png`)
  aynı segmentte karşılaştır. En riskli kısım; görsel pariteyle doğrulanır.

### Adım 4 — Playback görselleştirme + Python paritesi
1. 47-frame segmenti Unity'de oynat (obje = mug, `obj_t/q`'dan yerleştir).
2. Scene view capture; el kavramasını Python GT viz'iyle karşılaştır.
- **Çıktı:** Unity ekran görüntüsü — el kupayı Python'daki gibi kavrıyor.

## Done criteria (Faz 3A)
1. Unity Inference Engine paritesi: max|unity−torch pca| < 1e-3.
2. Tahmin edilen poz hand rig'i sürüyor (parmaklar kapanıyor).
3. Handedness dönüşümü doğru: Unity el+obje ≈ Python viz.
4. 47-frame playback Unity'de akıcı oynuyor + screenshot.

## Kararlar / riskler
- **Eski C# yeniden mi, sıfırdan mı:** Model I/O'su eskiden farklı (SDF). Öneri: model
  sürüş için **sıfırdan minimal playback driver** yaz; ama `MANODecoder` + hand rig +
  obje assetlerini uyumluysa **yeniden kullan**.
- **MANODecoder uyumu:** PCA bazı tutmazsa kendi decode'umuzu port ederiz.
- **Handedness:** ana risk; zaman ayır, görsel pariteyle çöz.
- **Inference Engine API:** çıktı isimleri/tensor readback — küçük, iterasyonla.

## Scope dışı (bu adımda yok)
Canlı VR (controller→el), proximity trigger/blend (Faz 0.5/4), model iyileştirme (Faz 2B),
DexYCB augment (1C).
