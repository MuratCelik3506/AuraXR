# 10. Model Mimarisi — Detaylı Referans

Kaynak: `src/model/grasp_model.py`, `src/model/model_io.py` (commit 52f3ce8).

---

## 10.1. Sabit Değerler (model_io.py)

| Sabit | Değer | Açıklama |
|---|---|---|
| `HOT3D_FRAME_DIM` | 13 | frame_feat boyutu: rel_pos(3)+rel_rot6d(6)+rel_vel(3)+dist(1) |
| `FINGER_POSE_DIM` | 45 | MANO finger axis-angle: 15 eklem × 3 |
| `OBJ_EMB_DIM` | 256 | PointNet çıktı boyutu |
| `LATENT_DIM` | 64 | CVAE latent boyutu |
| `NUM_FINGER_JOINTS` | 15 | Eklem sayısı |
| `NUM_FINGERTIPS` | 5 | Parmak ucu sayısı |
| `DEFAULT_N_POINTS` | 1024 | Point cloud nokta sayısı |
| `WRIST_DIM` | 6 | Legacy: tsl(3) + global_orient_aa(3) |
| `CONTACT_THRESHOLD_M` | 0.030 | 30mm temas eşiği (HOT3D segment) |
| `QUALITY_W_CONTACT` | 1.0 | quality_label contact ağırlığı |
| `QUALITY_W_PENETRATION` | 0.3 | quality_label penetration ağırlığı |
| `QUALITY_W_DIST` | 0.2 | quality_label distance ağırlığı |

---

## 10.2. PointNetEncoder

```python
class PointNetEncoder(nn.Module):
    # (B, N, 3) -> (B, 256)
    point_mlp: nn.Sequential [Linear(3,64)+ReLU, Linear(64,128)+ReLU, Linear(128,256)+ReLU]
    out: nn.Sequential [Linear(512,256)+ReLU, Linear(256,256)]

    def forward(obj_pts):
        x = point_mlp(obj_pts)           # (B, N, 256)
        pooled = cat([x.mean(1), x.max(1).values], dim=-1)  # (B, 512)
        return out(pooled)               # (B, 256)
```

**Parametre sayısı (tahmini):**
- point_mlp: 3×64 + 64×128 + 128×256 ≈ 41K param
- out: 512×256 + 256×256 ≈ 197K param
- Toplam PointNet: ~238K param

---

## 10.3. FiLM

```python
class FiLM(nn.Module):
    # cond: (B, 256), feat: (B, H) -> (B, H)
    to_gamma_beta: Linear(256, H*2)

    def forward(feat, cond):
        gamma, beta = to_gamma_beta(cond).chunk(2, dim=-1)  # (B, H), (B, H)
        return feat * (1.0 + gamma) + beta
```

Residual form: `1 + gamma` ile feat sıfıra gitmez; başlangıçta gamma≈0, beta≈0 → identity başlangıç.

---

## 10.4. TemporalEncoder

```python
class TemporalEncoder(nn.Module):
    # frame_feat: (B, T, 13), contact_flag: (B, T, 1) -> (B, 256)
    input_proj: Linear(14, 256)   # 13 + 1 = 14
    gru: GRU(256, 256, num_layers=1, batch_first=True)

    def forward(frame_feat, contact_flag=None):
        if contact_flag is None: contact_flag = zeros(B, T, 1)
        x = cat([frame_feat, contact_flag], dim=-1)  # (B, T, 14)
        x = input_proj(x)                           # (B, T, 256)
        _, h = gru(x)                               # h: (1, B, 256)
        return h[-1]                                # (B, 256)
```

**Parametre sayısı:**
- input_proj: 14×256 ≈ 4K
- GRU: 4 × (256×256 + 256×256) ≈ 524K
- Toplam: ~528K

---

## 10.5. ContextEncoder

```python
class ContextEncoder(nn.Module):
    # temporal_feat: (B, 256), obj_emb: (B, 256) -> (B, 256)
    temporal_proj: Linear(256, 256)
    film: FiLM(cond_dim=256, feat_dim=256)
    backbone: [Linear(256,256)+GELU, Linear(256,256)+GELU]

    def forward(temporal_feat, obj_emb):
        h = temporal_proj(temporal_feat)   # (B, 256)
        h = film(h, obj_emb)               # (B, 256)
        return backbone(h)                 # (B, 256)
```

---

## 10.6. JointSelfAttention

```python
class JointSelfAttention(nn.Module):
    # fusion_out: (B, 128), prev_pose: (B, 45) -> (B, 15, 128)
    learned_joint_emb: Parameter(15, 128)
    prev_pose_proj: Linear(3, 128)
    attn: MultiheadAttention(128, num_heads=4, batch_first=True)
    norm: LayerNorm(128)
    ffn: [Linear(128,256)+GELU, Linear(256,128)]
    norm2: LayerNorm(128)

    def forward(fusion_out, prev_pose):
        prev_pose_emb = prev_pose_proj(prev_pose.view(B, 15, 3))  # (B, 15, 128)
        tokens = (fusion_out[:, None, :].expand(B, 15, 128)
                + learned_joint_emb[None, :, :]
                + prev_pose_emb)                                   # (B, 15, 128)
        attn_out, _ = attn(tokens, tokens, tokens)
        tokens = norm(tokens + attn_out)
        tokens = norm2(tokens + ffn(tokens))
        return tokens                                              # (B, 15, 128)
```

**Not**: 1 katman transformer block (attention + FFN + 2× LayerNorm). 15 eklem küçük sequence → 1 katman yeterli.

---

## 10.7. GraspCVAE

```python
class GraspCVAE(nn.Module):
    z_dim = 64, num_joints = 15, joint_dim = 128

    # Encoder (eğitimde)
    encoder: [Linear(15*128 + 45, 15*64) + GELU, Linear(15*64, 128)]
    # -> mu(B,64), logvar(B,64)

    # Decoder (per-joint)
    decoder_joint_head: [Linear(128+64, 128)+GELU, Linear(128,3)]
    # (B, 15, 128) + z(B,64) -> (B, 15, 3) -> flatten -> (B, 45)

    def sample(joint_tokens, k=1):
        z = randn(B, k, 64)
        tokens_expanded = joint_tokens[:, None, :, :].expand(B, k, 15, 128)
        flat_tokens = tokens_expanded.reshape(B*k, 15, 128)
        flat_z = z.reshape(B*k, 64)
        flat_pred = decode(flat_tokens, flat_z)    # (B*k, 45)
        return flat_pred.reshape(B, k, 45)
```

**Encoder parametre:**
- Linear(1920+45, 960) + Linear(960, 128) ≈ 1.97M param

---

## 10.8. Output Heads

```python
# quality_head
Linear(1920, 128) + GELU + Linear(128, 1) + Sigmoid
# Input: flat_joint_tokens = (B, 15*128) = (B, 1920)

# success_head
Linear(1920+45, 128) + GELU + Linear(128, 1) + Sigmoid
# Input: cat(flat_joint_tokens, candidate_pose) = (B, 1965) per candidate
```

**Inference'da K aday için success_head:**
```python
flat = joint_tokens.reshape(B, -1)   # (B, 1920)
# K aday için broadcast:
flat_expanded = flat[:, None, :].expand(B, K, 1920)  # (B, K, 1920)
success_input = cat([flat_expanded, candidates], dim=-1)  # (B, K, 1965)
success = success_head(success_input.reshape(B*K, 1965)).reshape(B, K)
best_idx = success.argmax(dim=1)
```

---

## 10.9. Loss Fonksiyonu Parametreleri (Varsayılan)

| Parametre | Değer | Açıklama |
|---|---|---|
| `kl_weight` | 0.01 | β-KL ağırlığı |
| `limit_weight` | 1.0 | Joint limit soft loss |
| `contact_weight` | 0.3 | L_contact ağırlığı |
| `penetration_weight` | 0.1 | L_penetration ağırlığı (proxy) |
| `quality_weight` | 0.1 | MSE(quality_score, label) |
| `success_weight` | 0.1 | BCE(success_prob, unity_label) |
| `vel_weight` | 0.01 (Phase 2) | L_vel GT hız farkı |
| `acc_weight` | 0.04 (Phase 2) | L_acc GT ivme farkı |
| `tip_weight` | 0.5 | MSE(FK(pred), FK(gt)) fingertip |

**Contact loss detayı:**
```python
# Hinge at 15mm (MANO FK ~9mm residual için tolerans)
l_contact = relu(nearest_dist_to_obj - 0.015).mean()

# Centroid-proxy penetration:
penetration = relu(nearest_to_centroid - tip_to_centroid)
l_penetration = penetration.mean()
```

**Joint limit bounds (MANO anatomik sınırlar):**
- Index/Middle/Ring/Pinky MCP flexion: [−0.3, 1.6] rad
- Thumb CMC: [−0.5, 1.2] rad
- Soft loss: `relu(lower - pred)^2 + relu(pred - upper)^2`

---

## 10.10. MANO Eklem Sırası (45-dim output)

| Aralık | Eklem | Parmak |
|---|---|---|
| 0–2 | MCP | Index |
| 3–5 | PIP | Index |
| 6–8 | DIP | Index |
| 9–11 | MCP | Middle |
| 12–14 | PIP | Middle |
| 15–17 | DIP | Middle |
| 18–20 | MCP | Ring |
| 21–23 | PIP | Ring |
| 24–26 | DIP | Ring |
| 27–29 | MCP | Pinky |
| 30–32 | PIP | Pinky |
| 33–35 | DIP | Pinky |
| 36–38 | CMC | Thumb |
| 39–41 | MCP | Thumb |
| 42–44 | IP | Thumb |

`FINGERTIP_JOINT_INDICES = [2, 5, 8, 11, 14]` — her parmağın son eklemi (DIP/IP).
