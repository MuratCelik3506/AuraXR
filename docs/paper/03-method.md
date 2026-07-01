# 3. Method

## 3.1. System Overview
- Tek model mimarisi: Temporal Geometry-Conditioned Grasp Model
- Kural tabanlı faz geçişi (approach controller) + AI grasp decoder ayrımı
- Runtime akışı: controller → proximity trigger → GRU enc → self-attn → CVAE decoder → best-of-K

```
Object Point Cloud
        │
   Mini PointNet
        │
 Object Feature (256)
        │
 frame_feat (B,T,13) ──► GRU ──► Temporal Feature
                                        │
                          ┌─────────────┤
                          │             │
                   Previous Pose   Object Feature
                    (B,45)          (256)
                          │             │
                          └──── Fusion ─┘
                                  │
                      Self-Attention (15 eklem)
                                  │
                             CVAE Decoder
                                  │
                    ┌─────────────┼─────────────┐
              Finger Pose (45)  Quality       Success
                              Score (1)      Prob (1)
```

## 3.2. Object Representation
- Mini PointNet encoder: N=1024 nokta → MLP per point (3→64→128→256, ReLU) → mean pooling + max pooling → concat(512) → Linear(512,256)+ReLU → Linear(256,256) → 256-dim global feature (`OBJ_EMB_DIM=256`)
- Canonical frame: obje rotasyonu çıkarılır; model orientasyon-invariant girdi alır, orientasyon ayrıca `rel_rot6d` ile verilir
- FiLM conditioning (`FiLM` sınıfı): `gamma, beta = Linear(256, 2*H)(obj_feat).chunk(2)` → `h_out = h * (1 + gamma) + beta`; backbone içindeki aktivasyonları ölçekler ve kaydırır
- Ablation varyantları: MLP-BBox / MLP-BBox+Pose / PointNet (mevcut) / PointNet+Normals (in_dim=6)

## 3.3. Temporal Wrist Encoding
- Object-relative frame sözleşmesi:
  - `T_wrist_in_object = T_object_in_world⁻¹ @ T_wrist_in_world`
  - `rel_pos(3)`, `rel_rot6d(6)`, `rel_vel(3)`, `dist(1)` → `frame_feat (B, T, 13)` (`HOT3D_FRAME_DIM=13`)
- `contact_flag (B, T, 1)` ile concat → `gru_input (B, T, 14)` → `Linear(14, 256)` → `GRU(256, 256, 1 layer, batch_first=True)`
- Son hidden state: temporal feature `(B, 256)`
- OakInk: T=1, rel_vel=0, contact_flag=0 (statik)
- HOT3D/Unity: T>1 sliding window (eval'de T=16 kullanıldı; docs'ta T=8 hedeflendi)
- `ContextEncoder`: `temporal_proj → FiLM(obj_emb) → Linear(256,256)+GELU → Linear(256,256)+GELU` → context `(B,256)`

## 3.4. Finger Joint Decoder
- `context_to_joint`: Linear(256, 128) → joint context `(B, 128)`
- `JointSelfAttention` (15 eklem, 4 head, 128-dim, 1 katman):
  - `learned_joint_emb`: (15, 128) trainable parametre (per-joint kimlik)
  - `prev_pose_proj`: Linear(3, 128) — `prev_pose(B,45).view(B,15,3)` → per-joint kinematik geçmiş
  - `tokens = context[:, None, :].expand(B,15,128) + learned_joint_emb + prev_pose_emb`
  - MultiheadAttention(128, 4 head, batch_first=True) + LayerNorm + FFN(128→256→128, GELU) + LayerNorm
  - Çıktı: `(B, 15, 128)` attended joint tokens
- `GraspCVAE`:
  - Encoder: `Linear(15*128+45, 15*64) → GELU → Linear(15*64, 128)` → mu, logvar (`LATENT_DIM=64`)
  - Decoder (per-joint): `Linear(128+64, 128) → GELU → Linear(128, 3)` → `(B, 15, 3)` → flatten → `(B, 45)`
  - K aday paralel üretim: z`(B,K,64)` → tokens genişletilir → tek batched decoder geçişi → `(B, K, 45)`

## 3.5. Multi-Task Output Heads
- **Parmak açısı:** CVAE decoder çıktısı, `(B, 45)` MANO axis-angle (index, middle, ring, pinky, thumb; MCP→PIP→DIP sırası)
- **quality_head** (heuristic): `flat_joint_tokens(B, 15*128) → Linear(1920, 128) → GELU → Linear(128, 1) → Sigmoid` → `quality_score ∈ [0,1]`
  - Label: `clip(w1*contact_ratio - w2*clip(pen/10mm,0,1) - w3*clip(dist/5cm,0,1), 0, 1)`, weights: `w1=1.0, w2=0.3, w3=0.2`
  - Loss: MSE
- **success_head** (Unity binary): `concat(flat_joint_tokens, candidate_pose(45)) → Linear(1920+45, 128) → GELU → Linear(128, 1) → Sigmoid`
  - Her K aday için ayrı çalışır; `flat_joint_tokens[:, None, :].expand(B,K,...)` + candidates reshape
  - Loss: BCE (yalnızca Aşama 3, Unity label gerektirir)
- Aday seçimi: `best_pose = candidates[argmax(success_probs, dim=1)]`

## 3.6. Loss Functions

**Aşama 1 (OakInk, T=1):**
```
L = L_recon + β*L_KL + λ_c*L_contact + λ_p*L_penetration + λ_q*L_quality + λ_tip*L_tip
```

**Aşama 2 (HOT3D fine-tuning, T>1):**
```
L = Aşama_1_loss
  + λ_vel * ||(p̂_t - p̂_{t-1}) - (p_t - p_{t-1})||
  + λ_acc * ||(p̂_t - 2p̂_{t-1} + p̂_{t-2}) - (p_t - 2p_{t-1} + p_{t-2})||
```

**Aşama 3 (Confidence calibration, backbone frozen):**
```
L = BCE(success_prob, unity_label)
```

| Loss | Açıklama | Aktif |
|---|---|---|
| `L_recon + β*L_KL` | CVAE pose reconstruction | Her zaman |
| `L_contact` | Fingertip → yüzey SDF → 0 | Her zaman |
| `L_penetration` | `mean(relu(-SDF(fingertip)))` | Her zaman |
| `L_tip` | `MSE(FK(pred), FK(gt))` fingertip position | Her zaman |
| `L_quality` | MSE(quality_score, heuristic_label) | Her zaman |
| `L_vel` | GT hız farkı eşleştirme | Yalnızca T>1 |
| `L_acc` | GT ivme farkı eşleştirme | Yalnızca T>1 |
| `L_success` | BCE(success_prob, unity_label) | Yalnızca Aşama 3 |
