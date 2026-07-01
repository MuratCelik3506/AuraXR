# 12. Implementation Notes — Kritik Bug Fixes ve Teknik Kararlar

Kaynak: `docs/A-veri-hazirlama.md §A8`, `docs/E-iki-asamali-egitim-oneri-degerlendirmesi.md`, `src/model/`.

Bu bölüm paper'a doğrudan "limitations" veya "implementation" olarak girmeyebilir ama tez savunmasında sorulabilecek teknik kararların kaydıdır.

---

## 12.1. mano_fk.py FK Hatası ve Düzeltmesi

**Sorun:** Eski `mano_fk.py` simplified FK, Y eksenini parmak yönü olarak varsayıyordu. Gerçek MANO'da parmaklar −X yönünde uzanır.

**Etki (iki katmanlı):**
- `quality_label`: fingertip pozisyonları GT'den **18–20 cm** sapıyordu → tüm quality_label = 0.0
- `L_contact / L_penetration`: aynı FK kullanıldığından contact loss tüm eğitim boyunca geometrik olarak kör çalıştı

**Düzeltme:**
- `mano_fk.py` gerçek MANO kinematic tree + zero-beta template joints ile yeniden yazıldı
- `parents = [-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 0, 10, 11, 0, 13, 14]`
- 4×4 matris zinciri (out-of-place, torch.cat) — autograd güvenli
- FK hatası: **18–20 cm → ~0.9 cm** (HOT3D stored fk_joints ile doğrulandı, 430 frame)
- Hinge threshold da güncellendi: **40mm → 15mm** (eski 40mm simplified FK hatasını maskeliyordu)

| Metrik | Öncesi | Sonrası |
|---|---|---|
| FK fingertip hatası | 18–20 cm | ~0.9 cm |
| HOT3D quality_label mean | 0.000 | 0.065 |
| HOT3D quality_label >0 | %0 | %18 |
| OakInk quality_label mean | 0.000 | 0.101 |
| OakInk quality_label >0 | %0 | %26 |

---

## 12.2. Contact Eşiği Uyumsuzluğu

**Sorun:** `CONTACT_THRESHOLD_M = 0.005` (5mm) — HOT3D build'in kullandığı 3cm AABB eşiğiyle uyumsuzdu. Gerçek temas olan frame'lerde quality_label = 0 çıkıyordu.

**Düzeltme:** `CONTACT_THRESHOLD_M = 0.030` (30mm).

---

## 12.3. OakInk obj_anno Eksikliği

**Sorun:** Eski `build_oakink_canonical.py` `obj_anno` alanını `savez()` çağrısına eklemiyordu. Dataset loader fallback olarak zero array döndürüyordu → contact/penetration loss anlamsız.

**Düzeltme:** `obj_anno` `savez()` içine eklendi, dataset yeniden üretildi. `dataset.npz`: `obj_anno (11151, 12)` kayıtlı.

---

## 12.4. Autograd Inplace Op Hatası

**Sorun:** `mano_fk.py` FK zincirinde inplace tensor atamaları (`G[..., i, :, :] = ...`) autograd hesaplama grafiğini bozuyordu. Phase 2 `loss.backward()` sırasında:
```
RuntimeError: one of the variables needed for gradient computation has been modified
by an inplace operation
```

**Düzeltme:** `_make_T` yardımcı fonksiyonu — `torch.cat` ile out-of-place 4×4 transform. G zinciri Python listesi → `torch.stack`.

---

## 12.5. Fingertip Position Loss Eklenmesi

**Sorun:** `grasp_loss` yalnızca axis-angle MSE (L_recon) kullanıyordu. Küçük açı hataları kinematik zincirde birikip parmak ucunda büyük pozisyon hatasına dönüşebilir.

**Düzeltme:** `L_tip = MSE(FK(pred), FK(gt))` eklendi. `fingertip_positions(pred)` ve `fingertip_positions(target)` wrist frame'de (B,5,3) üretir. `tip_weight=0.5` ile loss'a eklendi. `--tip_weight` argümanı.

**Doğrulama:** Phase 1 `val_tip ≈ 0.00005`, Phase 2 `val_tip ≈ 0.00009`, backward temiz.

---

## 12.6. Phase 2 Temporal Loss Aktif Değildi

**Sorun:** `train_grasp.py` içinde `grasp_loss(..., prev_pred_pose=None)` hardcoded geçiliyordu. `vel_weight` ve `acc_weight` set edilse de L_vel ve L_acc hesaplanmıyordu → Phase 2 pratikte Phase 1 tekrarı.

**Düzeltme:**
1. `dataset_hot3d.py`: `prev_frame_feat` (t-1) ve `prev2_frame_feat` (t-2) batch'e eklendi
2. `train_grasp.py`: `no_grad` içinde önceki frame'ler üzerinden forward pass → `pred_{t-1}`, `pred_{t-2}` `.detach()` ile `prev_pred_pose`/`prev2_pred_pose` olarak geçiliyor
3. Gradient yalnızca `pred_t` üzerinden akıyor

**Doğrulama:** Phase 2: `train_vel=0.0003, train_acc=0.040`, backward temiz.

---

## 12.7. HOT3D obj_pts Eksikliği

**Sorun:** `build_hot3d_canonical_full.py` seq_*.npz üretiyordu ama `obj_pts/*.npy` üretmiyordu.

**Düzeltme:** 27 HOT3D objesinin `.glb` mesh'lerinden `trimesh` ile 1024 nokta örneklendi, `data/processed/hot3d_canonical/obj_pts/` altına kaydedildi.

---

## 12.8. ONNX Export — Unity InferenceEngine Uyumsuzluğu

**Sorun:** `export_onnx.py` genel ONNX export'u Unity `InferenceEngine 2.6.1`'de import edilemedi. Nedenler:
1. ONNX graph içinde `GRU` operator'ı desteklenmiyor
2. CVAE sampling (`torch.randn`) ve K adaylı selection dinamik şekil içeriyor

**Çözüm:** `src/export/export_unity_onnx.py` eklendi. Bu exporter:
- GRU 16 frame için manuel unroll edildi (ONNX'te `GRU` op yok)
- CVAE randomness kaldırıldı: `z = zeros(B, latent_dim)` (deterministik, K=1)
- Candidate selection kaldırıldı
- Sabit shape'li, statik graph

**Çıktı:** `checkpoints/grasp_model_unity.onnx`
**Unity smoke test:** `selected_pose (1,45)`, `quality_score (1,1)`, `success_prob (1,1)` — CPU backend üzerinde geçti.

**Önemli:**
- `grasp_model_unity.onnx` K=1, z=0 (mean pose, varyasyon yok)
- Araştırma amaçlı K>1 için `grasp_model.onnx` Python ONNX Runtime ile kullanılır
- Unity demo mevcut durumda deterministik tek aday ile çalışıyor

---

## 12.9. HOT3D val Penetration Anomalisi

**Gözlem:** HOT3D val penetration 4.02mm, test penetration 1.36mm — neredeyse 3× fark.

**Olası açıklama:** Centroid-proxy penetration ölçümü obje şekline çok bağlı. Val objeleri (keyboard, spatula, vase) convex dışı veya düzlemsel yapılar — centroid proxy bu objeler için yanıltıcı sonuç üretiyor. Test objeleri (coffee_pot, dumbbell, whiteboard_eraser) daha küresel/düzgün — proxy daha tutarlı.

**Sonuç:** Bu fark gerçek penetration farkı değil, metric artifact. "Penetration test'te daha düşük" iddiası doğrudan savunulamaz.

---

## 12.10. Mevcut Processed Veri Durumu

| Veri | Konum | İçerik | Durum |
|---|---|---|---|
| HOT3D seq'ler | `data/processed/hot3d_canonical/seq_*.npz` | 297k frame, 4113 segment | ✓ |
| HOT3D obj_pts | `data/processed/hot3d_canonical/obj_pts/` | 27 obje × 1024 nokta | ✓ |
| HOT3D istatistikler | `data/processed/hot3d_canonical/stats.json` | Normalizasyon mean/std | ✓ |
| HOT3D obj_split | `data/processed/hot3d_canonical/obj_split.json` | train/val/test obje mapping | ✓ |
| OakInk dataset | `data/processed/oakink_canonical/dataset.npz` | 11151 sample, obj_anno + fingertips_world | ✓ |
| OakInk obj_pts | `data/processed/oakink_canonical/obj_pts/` | 25 kategori × 1024 nokta | ✓ |
| OakInk split/stats | `data/processed/oakink_canonical/split.json`, `stats.json` | 80/10/10 | ✓ |

---

## 12.11. .gitignore Sorunu

`src/data/` loader kodları `.gitignore:23`'teki `data/` satırı yüzünden izlenmiyor. Satır `/data/` (kök-göreli) olarak düzeltilmeli; aksi halde `dataset_hot3d.py`, `dataset_oakink.py` commit edilemiyor.
