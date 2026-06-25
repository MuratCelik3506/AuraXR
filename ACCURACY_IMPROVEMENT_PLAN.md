# AuraXR Accuracy Improvement Plan

**Model:** `SDFLSTMModel` — 2-layer LSTM, hidden size 256, current input 29 dim, wrist rotation output 6D  
**Training data:** HOT3D temporal windows (`T=16`) + optional ARCTIC/DexYCB contact augmentation  
**Main problem:** train/inference domain gap, wrist rotation loss geometry, approach-angle data bias, limited object-relative context  
**Decision status:** Phase 0 and most Phase 1 items are implemented; current blocker is contact recall + runtime wrist quality.

---

## Current Status — 2026-06-25

Legend:

- `[DONE]` implemented in code
- `[PARTIAL]` implemented, but results show remaining quality gap
- `[TODO]` not implemented yet

Implemented in `src/train_lstm.py`:

- `[DONE]` Rotation utilities and angular wrist metrics.
- `[DONE]` Geodesic wrist loss.
- `[DONE]` Teacher-forced and autoregressive validation metrics.
- `[DONE]` Scheduled sampling training path.
- `[DONE]` Contact-frame weighted wrist loss.
- `[DONE]` Multi-source HOT3D + ARCTIC + DexYCB training loop with per-source validation.

Implemented in dataset/build pipeline:

- `[DONE]` Object-relative wrist position support via `--add_wrist_obj_pos`.
- `[DONE]` Feature metadata/versioning fields in H5 (`feature_names`, `feature_dim`, `augmentation_flags`, etc.).
- `[DONE]` Yaw augmentation support in dataset builders.

Implemented for debugging/visualization:

- `[DONE]` Python MP4 visualization for prediction vs ground truth.
- `[DONE]` Unity-free synthetic 3D LSTM smoke test video.
- `[PARTIAL]` MANO FK visualization works only when `smplx` is installed; current local fallback uses a procedural skeleton, so visual finger shape is approximate.

New findings from MP4 + metric inspection:

| Checkpoint | Seq | GT contact frames | Pred contact >= 0.5 | Max pred contact | Mean wrist error | Contact wrist error |
|------------|-----|-------------------|---------------------|------------------|------------------|---------------------|
| `checkpoints/lstm_right/best.pt` epoch 9 | 0 | 1085 | 14 | 0.838 | 126.5 deg | 119.6 deg |
| `checkpoints/lstm_right_v2/best.pt` epoch 50 | 0 | 1085 | 0 | 0.338 | 105.9 deg | 105.6 deg |

Interpretation:

1. The issue is not primarily Unity. Python stateful inference already shows failure modes.
2. Wrist orientation is still too inaccurate for grasp alignment, especially in runtime/autoregressive mode.
3. Contact head is severely under-detecting contact; v2 improves wrist somewhat but becomes even more conservative on contact.
4. Default right checkpoint may be stale (`lstm_right/best.pt` is epoch 9); `lstm_right_v2/best.pt` should be used for comparisons, but it still does not solve contact.
5. Next work should prioritize contact recall, checkpoint selection criteria, and runtime wrist metrics before more Unity debugging.

---

## Executive Summary

Mevcut model wrist rotation ve contact tahmininde üç temel sebeple sınırlanıyor:

1. **Train/inference mismatch:** Training'de her frame input'undaki `wrist_rot_6d` ground-truth. Inference'ta ise bu alan ya controller'dan geliyor ya da modelin önceki tahmininden besleniyor. Model kendi hatasıyla eğitimde karşılaşmadığı için exposure bias oluşuyor.
2. **Runtime wrist quality:** Geodesic loss ve scheduled sampling uygulanmış olmasına rağmen Python stateful inference hâlâ 100 derece üzeri wrist hatası gösterebiliyor.
3. **Contact under-detection:** Contact head temas frame'lerini büyük oranda kaçırıyor. Seq0 örneğinde 1085 GT contact frame'e karşı v2 checkpoint `>=0.5` contact üretmiyor.

Kısa vadede en yüksek getirili ve en düşük riskli paket:

1. `[DONE]` `wrist_deg` ve `final_wrist_deg` metriklerini ekle.
2. `[DONE]` Wrist loss'u 6D MSE'den geodesic angular loss'a taşı.
3. `[DONE]` Scheduled sampling ile autoregressive training ekle.
4. `[DONE]` Validation'a ayrıca autoregressive evaluation modu ekle.
5. `[DONE]` Contact-frame weighted wrist loss'u kontrollü şekilde dene.
6. `[TODO]` Contact imbalance için `pos_weight` / focal loss / threshold calibration ekle.
7. `[TODO]` Best checkpoint seçimini `ar_loss` yerine runtime wrist + contact recall odaklı metrikle yap.

İlk beş madde dataset rebuild gerektirmez ve mevcut model mimarisini bozmaz. Contact loss/selection değişiklikleri de dataset rebuild gerektirmez.

---

## Expected Impact Ranking

| # | Status | Change | Expected Impact | Risk | Effort | Phase |
|---|--------|--------|-----------------|------|--------|-------|
| 1 | DONE | Wrist angular metrics | Required for measurement | Very low | Very low | 0 |
| 2 | DONE | Geodesic wrist rotation loss | High | Low | Low | 1 |
| 3 | DONE | Autoregressive validation | High for truthful evaluation | Low | Low | 1 |
| 4 | DONE | Scheduled sampling | Very high | Medium | Medium | 1 |
| 5 | DONE | Contact-frame weighted wrist loss | Medium-high | Medium | Low | 1 |
| 6 | TODO | Contact imbalance fix (`pos_weight` / focal loss) | Very high for contact recall | Medium | Low | 1 |
| 7 | TODO | Runtime-aware checkpoint selection | High | Low | Low | 1 |
| 8 | DONE | Object-relative wrist position | Medium-high | Low | Medium | 2 |
| 9 | DONE | Yaw rotation augmentation support | Medium | Medium | Medium | 2 |
| 10 | TODO | Approach-angle balanced sampling | Medium | Low-medium | Medium | 2 |
| 11 | PARTIAL | Orientation-aware SDF injection | Medium | Medium | Low-medium | 3 |
| 12 | TODO | Selective input noise augmentation | Low-medium | Low | Low | 3 |
| 13 | TODO | Wrist angular velocity input | Low-medium | Medium | Medium | 3 |

---

## Success Criteria

Bu değişikliklerde tek başına train loss'a bakmak yeterli değil. Her deney için aşağıdaki metrikler loglanmalı:

| Metric | Meaning | Target |
|--------|---------|--------|
| `val_loss_tf` | teacher-forced validation loss | regression kontrolü |
| `val_loss_ar` | autoregressive validation loss | inference'a daha yakın kalite |
| `wrist_deg` | average wrist angular error, degrees | düşmeli |
| `final_wrist_deg` | last frame wrist angular error | grasp alignment için düşmeli |
| `contact_wrist_deg` | contact frame wrist error | contact varsa düşmeli |
| `pose_l1_final` | final MANO pose L1 error | kötüleşmemeli |
| `jitter_deg` | frame-to-frame wrist angular change | aşırı artmamalı |
| `contact_bce` | contact prediction quality | kötüleşmemeli |

Minimum kabul kriteri:

- `final_wrist_deg` ve `val_loss_ar` iyileşmeli.
- `pose_l1_final` belirgin kötüleşmemeli.
- Scheduled sampling sonrası `jitter_deg` artarsa smoothness veya daha yavaş schedule gerekir.

---

## Current Code Facts

Repo'da mevcut durum:

- `src/model.py`
  - `SDFLSTMModel(feat_dim=29, embed_dim=32, proj_dim=64, hidden_size=256)`
  - `forward_sequence(feat_seq, obj_embed)` training/eval path
  - `forward(frame_feat, obj_embed, h_0, c_0)` ONNX/stateful single-frame path
  - `_prepare_step(frame_feat, obj_embed)` exists
  - `initial_state(batch_size, device)` exists
- `src/train_lstm.py`
  - `[DONE]` `temporal_loss()` uses geodesic angular wrist loss.
  - `[DONE]` `temporal_loss()` applies contact-frame wrist weighting.
  - `[DONE]` `evaluate_temporal()` reports TF/AR-compatible metrics including `wrist_deg`, `final_wrist_deg`, `contact_wrist_deg`, `jitter_deg`, and `contact_bce`.
  - `[DONE]` training loop can use `forward_sequence_scheduled_sampling(...)`.
  - `[TODO]` contact BCE still uses unweighted `binary_cross_entropy`; contact recall is poor in runtime inspection.
  - `[TODO]` best checkpoint currently follows primary AR loss, not an explicit contact-recall + wrist-runtime score.

Assumed feature layout from dataset plan:

| Dim | Feature |
|-----|---------|
| `0:3` | `dir_world` |
| `3:6` | `dir_obj_local` |
| `6` | distance |
| `7` | approach speed |
| `8:11` | object velocity |
| `11:17` | wrist rotation 6D input |
| `17` | hand confidence |
| `18:22` | grip one-hot |
| `22:25` | bbox/local shape features |
| `25:28` | optional `wrist_in_obj` when dataset was built with `--add_wrist_obj_pos` |
| final `+4` | local SDF feature `[sdf_value, sdf_gradient_xyz]` from `sdf_features` |

Confirmed in `build_dataset_temporal.py` / `build_dataset_mano.py`: current processed right HOT3D H5 has `features=(N, 28)` and `sdf_features=(N, 4)`, while the checkpoint input is `feat_dim=29`, so current training/eval truncates or pads to match model input.

---

## Phase 0 — Measurement First

Goal: accuracy changes must be measured in the same geometry used by the task.

### 0.1 Add Rotation Utilities — DONE

Add utility functions to `src/train_lstm.py` or a shared small module if reuse is needed.

```python
def rot6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """Convert 6D rotation representation to SO(3) matrix."""
    a1, a2 = d6[..., :3], d6[..., 3:6]
    b1 = torch.nn.functional.normalize(a1, dim=-1, eps=1e-6)
    b2 = torch.nn.functional.normalize(
        a2 - (b1 * a2).sum(-1, keepdim=True) * b1,
        dim=-1,
        eps=1e-6,
    )
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)


def rotation_angle_rad(pred_6d: torch.Tensor, tgt_6d: torch.Tensor) -> torch.Tensor:
    """Per-sample angular error in radians. Shape: pred/tgt (..., 6) -> (...)."""
    r_pred = rot6d_to_matrix(pred_6d)
    r_tgt = rot6d_to_matrix(tgt_6d)
    r_diff = r_pred.transpose(-2, -1) @ r_tgt
    trace = r_diff.diagonal(dim1=-2, dim2=-1).sum(-1)
    cos = ((trace - 1.0) / 2.0).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
    return torch.acos(cos)
```

### 0.2 Extend Evaluation Metrics — DONE

`evaluate_temporal()` should return a dictionary instead of only `(loss, pose_err)`.

Required metrics:

```python
metrics = {
    "loss": ...,
    "pose_l1_final": ...,
    "wrist_deg": ...,
    "final_wrist_deg": ...,
    "jitter_deg": ...,
    "contact_bce": ...,
}
```

`jitter_deg` should use angular delta, not raw 6D vector distance:

```python
jitter_rad = rotation_angle_rad(pw[:, 1:, :], pw[:, :-1, :]).mean()
```

### 0.3 Keep Teacher-Forced and Autoregressive Metrics Separate — DONE

Do not replace current validation immediately. Report both:

- `val_tf`: current `model.forward_sequence()` behavior.
- `val_ar`: frame-by-frame behavior where previous predicted wrist is fed into the next frame's `WRIST_DIMS`.

This matters because teacher-forced validation can look good while Unity/inference still drifts.

---

## Phase 1 — Training Pipeline Fixes Without Dataset Rebuild

Files expected to change:

- `src/train_lstm.py`
- optionally small helpers in `src/model.py` only if needed

No dataset rebuild required.

---

### 1. Geodesic Wrist Rotation Loss — DONE

Original problem:

```python
wrist_loss = F.mse_loss(pred_wrist, tgt_wrist)
```

This optimizes Euclidean distance in 6D representation, not angular distance on SO(3).

Replace with:

```python
def geodesic_loss(pred_6d: torch.Tensor, tgt_6d: torch.Tensor) -> torch.Tensor:
    return rotation_angle_rad(pred_6d, tgt_6d).mean()
```

Then in `temporal_loss()`:

```python
wrist_loss = geodesic_loss(pred_wrist, tgt_wrist)
```

Important notes:

- Unit changes from normalized MSE to radians.
- Keep `WRIST_LOSS_W = 0.3` for first run, then tune from logs.
- A useful reference: `0.1 rad ~= 5.7 degrees`, `0.2 rad ~= 11.5 degrees`.
- If pose loss becomes dominated by wrist loss, lower `WRIST_LOSS_W`; if wrist barely improves, raise it to `0.5`.

Expected result:

- `wrist_deg` should decrease.
- `final_wrist_deg` should decrease or stay stable.
- Train loss numeric value may not be comparable to old runs.

Current status:

- Implemented in `src/train_lstm.py` via `rotation_angle_rad()` and `geodesic_loss()`.
- Still not sufficient alone: v2 checkpoint improves wrist error versus epoch-9 checkpoint, but runtime wrist error remains too high for stable grasp alignment.

---

### 2. Autoregressive Evaluation — DONE

Add an evaluation forward path that mimics inference.

```python
WRIST_DIMS = slice(11, 17)


def forward_sequence_autoregressive(model, inp_seq, obj_emb, replace_from_t=1):
    """Run frame by frame and feed previous predicted wrist into future input."""
    b, t, _ = inp_seq.shape
    h, c = model.initial_state(b, device=inp_seq.device)
    poses, wrists, contacts = [], [], []
    prev_wrist = None

    for step_idx in range(t):
        frame = inp_seq[:, step_idx, :].clone()
        if prev_wrist is not None and step_idx >= replace_from_t:
            frame[:, WRIST_DIMS] = prev_wrist.detach()

        pose, wrist, contact, h, c = model(frame, obj_emb, h, c)
        poses.append(pose)
        wrists.append(wrist)
        contacts.append(contact)
        prev_wrist = wrist

    return (
        torch.stack(poses, dim=1),
        torch.stack(wrists, dim=1),
        torch.stack(contacts, dim=1),
    )
```

Validation should log:

```text
val_tf_loss
val_ar_loss
val_tf_wrist_deg
val_ar_wrist_deg
val_ar_final_wrist_deg
val_ar_jitter_deg
```

Expected result before scheduled sampling:

- `val_ar_*` is likely worse than `val_tf_*`.
- The gap size quantifies the train/inference mismatch.

Current status:

- Implemented via `evaluate_temporal(..., autoregressive=True)` and stateful frame-by-frame forwarding.
- Runtime inspection confirms this metric is necessary: Python stateful inference can fail even when global MPJPE metrics look acceptable.

---

### 3. Scheduled Sampling — DONE

Current problem:

Training currently does:

```python
pj, pw, pc = model.forward_sequence(inp_seq, obj_emb)
```

Every timestep gets ground-truth wrist input. Inference does not.

Training forward with scheduled sampling:

```python
def forward_sequence_scheduled_sampling(model, inp_seq, obj_emb, ss_prob: float):
    b, t, _ = inp_seq.shape
    h, c = model.initial_state(b, device=inp_seq.device)
    poses, wrists, contacts = [], [], []
    prev_wrist = None

    for step_idx in range(t):
        frame = inp_seq[:, step_idx, :].clone()

        if prev_wrist is not None and ss_prob > 0.0:
            mask = torch.rand(b, device=frame.device) < ss_prob
            frame[mask, WRIST_DIMS] = prev_wrist[mask].detach()

        pose, wrist, contact, h, c = model(frame, obj_emb, h, c)
        poses.append(pose)
        wrists.append(wrist)
        contacts.append(contact)
        prev_wrist = wrist

    return (
        torch.stack(poses, dim=1),
        torch.stack(wrists, dim=1),
        torch.stack(contacts, dim=1),
    )
```

Schedule:

```python
SS_START_EPOCH = 10
SS_END_EPOCH = 80
SS_MAX_PROB = 0.50


def get_ss_prob(epoch: int) -> float:
    if epoch <= SS_START_EPOCH:
        return 0.0
    progress = min(epoch - SS_START_EPOCH, SS_END_EPOCH - SS_START_EPOCH)
    return SS_MAX_PROB * progress / (SS_END_EPOCH - SS_START_EPOCH)
```

Training loop:

```python
ss_prob = get_ss_prob(epoch)
pj, pw, pc = forward_sequence_scheduled_sampling(model, inp_seq, obj_emb, ss_prob)
```

Critical details:

- Start with `ss_prob=0` for warmup.
- Do not jump directly to `1.0`; this can destabilize training.
- Use `.detach()` for previous predicted wrist to avoid long gradient chains through input replacement.
- Log `ss_prob` every epoch.
- Compare both teacher-forced and autoregressive validation.

Expected result:

- `val_ar_final_wrist_deg` improves.
- `val_tf_loss` may slightly worsen; that is acceptable if autoregressive metrics improve.
- `jitter_deg` should be monitored carefully.

Current status:

- Implemented via `forward_sequence_scheduled_sampling()` and `get_ss_prob()`.
- Training log shows scheduled sampling probability is logged.
- Remaining issue: checkpoint selection and loss balance still allow poor contact recall and high runtime wrist error.

---

### 4. Contact-Frame Weighted Wrist Loss — DONE / NEEDS RETUNING

Current problem:

All frames contribute equally to wrist loss, but grasp alignment is most important near contact.

Add per-frame angular loss:

```python
def weighted_geodesic_loss(pred_6d, tgt_6d, frame_weight):
    angle = rotation_angle_rad(pred_6d, tgt_6d)      # (B, T)
    weight = frame_weight / frame_weight.mean().clamp_min(1e-6)
    return (angle * weight).mean()
```

Inside `temporal_loss()`:

```python
contact_weight = 1.0 + 3.0 * contact_label          # (B, T)
wrist_loss = weighted_geodesic_loss(pred_wrist, tgt_wrist, contact_weight)
```

Risk:

- Contact labels from HOT3D, ARCTIC and DexYCB may not be semantically identical.
- If contact labels are noisy, this can overfit wrong frames.

Safer variant:

```python
contact_weight = 1.0 + 2.0 * contact_label
```

or use late-frame weighting if contact labels are unreliable:

```python
time_weight = torch.linspace(1.0, 2.0, steps=T, device=device)
```

Expected result:

- `final_wrist_deg` and `contact_wrist_deg` improve.
- Average `wrist_deg` may improve less dramatically.

Current status:

- Implemented with `CONTACT_FRAME_EXTRA_W`.
- Needs retuning because contact-frame wrist error remains high in runtime inspection (`~105 deg` for v2 on seq0 contact frames).

---

### 4.1 Contact Imbalance Fix — TODO

Current problem:

The contact head severely under-detects contact in Python stateful inference.

Observed right-hand seq0:

```text
GT contact frames: 1085
lstm_right/best.pt epoch 9:     pred_contact >= 0.5 on 14 frames
lstm_right_v2/best.pt epoch 50: pred_contact >= 0.5 on 0 frames
```

Likely causes:

- Contact positives are underweighted relative to pose/wrist losses.
- Auxiliary ARCTIC/DexYCB sources currently have `contact_loss_w=0.0`, so contact learning is anchored mostly on HOT3D.
- `BCE` with default class balance encourages conservative no-contact predictions when positives are sparse or noisy.

Recommended implementation:

1. Compute positive/negative contact ratio from HOT3D train split.
2. Replace contact BCE with weighted BCE:

```python
contact_loss = F.binary_cross_entropy(
    pred_contact.squeeze(-1),
    contact_label,
    weight=torch.where(contact_label > 0.5, pos_weight, neg_weight),
)
```

Alternative if sigmoid is moved out of `SDFLSTMModel.contact_head`:

```python
contact_loss = F.binary_cross_entropy_with_logits(
    contact_logits.squeeze(-1),
    contact_label,
    pos_weight=contact_pos_weight,
)
```

3. Track contact metrics beyond BCE:

```text
contact_precision
contact_recall
contact_f1
contact_auc_pr
best_contact_threshold
```

4. Select the runtime threshold from validation PR/F1 rather than assuming `0.5`.

Success target:

- Seq0-style validation should show materially higher contact recall without making every near-object frame positive.
- Primary checkpoint should not be accepted if contact recall is near zero.

---

### 4.2 Runtime-Aware Checkpoint Selection — TODO

Current problem:

`best.pt` is saved by primary AR loss. That can still select a model with unusable contact recall or high wrist error in contact frames.

Recommended checkpoint score:

```text
primary_score =
    val_hot3d_ar_final_wrist_deg
  + 0.5 * val_hot3d_ar_contact_wrist_deg
  + 20.0 * max(0, target_contact_recall - contact_recall)
  + pose_penalty_if_pose_l1_regresses
```

Simpler first version:

```text
primary_score = hot3d_ar_final_wrist_deg + hot3d_ar_contact_wrist_deg
```

Do not use ARCTIC/DexYCB as the primary checkpoint criterion for runtime deployment. They are useful augmenters, but HOT3D is the runtime-like temporal source.

Required log fields:

```text
best_metric_name
hot3d_ar_final_wrist_deg
hot3d_ar_contact_wrist_deg
hot3d_contact_recall
hot3d_contact_f1
```

Success target:

- The selected checkpoint should be the one that improves runtime wrist/contact behavior, not just average validation loss.

---

### 5. Optional Wrist Smoothness

Use only if scheduled sampling creates visible or measured jitter.

Prefer angular smoothness:

```python
smooth_rad = rotation_angle_rad(pred_wrist[:, 1:, :], pred_wrist[:, :-1, :]).mean()
total = total + SMOOTH_LOSS_W * smooth_rad
```

Suggested initial value:

```python
SMOOTH_LOSS_W = 0.02
```

Do not add this before measuring jitter. Too much smoothness can make wrist response lag behind fast motion.

---

## Phase 1 Ablation Order

Run these as separate experiments:

| Run | Changes | Purpose |
|-----|---------|---------|
| A0 | baseline + new metrics only | establish truthful metrics |
| A1 | A0 + geodesic loss | isolate loss geometry |
| A2 | A1 + autoregressive validation | quantify domain gap |
| A3 | A2 + scheduled sampling | reduce domain gap |
| A4 | A3 + contact-frame weighting | improve grasp/contact alignment |
| A5 | A4 + smoothness if needed | reduce jitter only if measured |

Do not combine all changes before A0/A1, otherwise it becomes unclear which change helped.

---

## Phase 2 — Dataset and Feature Improvements

Files expected to change:

- `src/build_dataset_temporal.py`
- `src/train_lstm.py`
- possibly dataset metadata/version naming

Dataset rebuild required.

Important: feature dimension changes invalidate old checkpoints.

---

### 6. Object-Relative Wrist Position — DONE

Current limitation:

The model knows approach direction and distance, but not explicitly where the wrist is relative to the object in object coordinates.

Add:

```python
wrist_in_obj = rotate_vec(q_obj_inv, wrist_pos - obj_pos)  # (3,)
```

Feature extension:

```python
best_core = np.concatenate([
    direction,          # 0:3   dir_world
    dir_obj_local,      # 3:6   direction in object frame
    [dist],             # 6
    [approach_spd],     # 7
    obj_vel,            # 8:11
    wrist_rot_in,       # 11:17
    [hand_conf],        # 17
    grip_oh,            # 18:22
    bbox,               # 22:25
    wrist_in_obj,       # new
]).astype(np.float32)
```

Expected dimension:

- Current processed HOT3D right H5: core `features` is 28 dims and `sdf_features` is 4 dims.
- Current checkpoint input remains `feat_dim=29`, so old checkpoints are not fully using all current feature channels.

Why useful:

- Helps model learn "from above", "from left", "near object handle" in object frame.
- Improves object pose invariance.

Validation:

- Compare seen-object and unseen-object splits if available.
- Track per-object `final_wrist_deg`.

Current status:

- Implemented in builders behind `--add_wrist_obj_pos`.
- Existing processed data includes `feature_dim=28`, indicating the object-relative wrist position was included.
- Remaining work is not the builder change; it is making sure model/checkpoint input dimension and exported ONNX/Unity feature assembly consume the same feature contract.

---

### 7. Yaw Rotation Augmentation — DONE / NEEDS EXPERIMENT

Goal:

Reduce HOT3D approach-angle bias without collecting new data.

Rule:

Apply yaw augmentation before normalization and before writing H5, not on already-normalized tensors.

World-frame values to rotate:

- `dir_world`
- wrist/world position if present
- object velocity
- object/world pose fields if present
- wrist rotation input
- wrist rotation target
- MANO/global wrist target if stored separately

Object-local values:

- `dir_obj_local` should usually remain unchanged if both object and wrist/world frames are rotated together.
- `wrist_in_obj` should remain unchanged for the same reason.

6D rotation target update:

```python
R_aug = yaw_matrix(yaw_rad)
R_wrist_new = R_aug @ R_wrist
wrist_6d_new = matrix_to_rot6d(R_wrist_new)
```

Recommended angles:

```text
first experiment: +/-45, +/-90
later experiment: 180 if needed
```

Avoid immediately using too many augmented copies. A 5x dataset may slow iteration and can overweight synthetic views.

Validation rule:

- Augmented samples should be train-only.
- Validation/test should remain real HOT3D unless specifically measuring synthetic robustness.

Expected result:

- Better performance on held-out approach-angle bins.
- Average validation may improve less than angle-specific validation.

Current status:

- Builder support exists.
- Need a controlled experiment and per-angle-bin validation before claiming accuracy improvement.

---

### 8. Approach-Angle Balanced Sampling — TODO

Goal:

Prevent dominant approach directions from controlling training.

Compute yaw bins from `dir_world`:

```python
def compute_sequence_yaw_bins(dataset, n_bins=8):
    bins = []
    for positions in dataset._seq_windows:
        dir_world = dataset.features[positions, 0:3]
        mean_dir = torch.as_tensor(dir_world).float().mean(0)
        yaw = torch.atan2(mean_dir[2], mean_dir[0])
        bin_idx = int((yaw + torch.pi) / (2 * torch.pi) * n_bins) % n_bins
        bins.append(bin_idx)
    return bins
```

Sampler:

```python
bins = compute_sequence_yaw_bins(hot3d_train)
counts = torch.bincount(torch.tensor(bins), minlength=8).float().clamp_min(1)
weights = (1.0 / counts)[torch.tensor(bins)]
sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
```

Risk:

- If rare bins are very noisy or tiny, oversampling can overfit.

Mitigation:

- Cap max sample weight.
- Log per-bin sample counts.
- Report per-bin validation metrics.

---

## Phase 3 — Model/Robustness Improvements

These are not first-priority because Phase 1 and Phase 2 are likely to provide clearer gains.

---

### 9. Orientation-Aware SDF Injection — PARTIAL

Current limitation:

The 32-dim object embedding is static per object. It encodes geometry but not "which side is being approached".

Simple model-side injection:

```python
self.obj_inj = nn.Sequential(
    nn.Linear(proj_dim + embed_dim + 3, proj_dim),
    nn.LayerNorm(proj_dim),
    nn.ReLU(),
)
```

```python
def _prepare_step(self, frame_feat, obj_embed):
    frame = self.feat_proj(frame_feat)
    dir_obj_local = frame_feat[..., 3:6]
    combined = torch.cat([frame, obj_embed, dir_obj_local], dim=-1)
    return self.obj_inj(combined)
```

Important caveat:

If `frame_feat[..., 3:6]` is normalized by dataset statistics, it may no longer be a unit direction. Better options:

1. Exclude direction dims from mean/std normalization.
2. Store raw `dir_obj_local` separately.
3. Re-normalize the extracted direction before injection:

```python
dir_obj_local = F.normalize(frame_feat[..., 3:6], dim=-1, eps=1e-6)
```

Expected result:

- Better per-object and per-approach-side generalization.
- Most useful after Phase 2 adds better object-relative context.

Current status:

- `SDFLSTMModel` has an `orientation_aware_sdf` option and checkpoint shape inference.
- Needs a controlled training run and ONNX/Unity compatibility check before marking complete.

---

### 10. Selective Input Noise Augmentation — TODO

Goal:

Make training robust to controller/hand tracking noise.

Do not add noise to the entire feature tensor. Avoid corrupting:

- one-hot grip labels
- binary flags
- contact labels
- object identity/embedding
- normalized categorical-like fields

Use explicit masks:

```python
continuous_dims = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
inp_seq[:, :, continuous_dims] += 0.01 * torch.randn_like(inp_seq[:, :, continuous_dims])
inp_seq[:, :, 11:17] += 0.003 * torch.randn_like(inp_seq[:, :, 11:17])
```

Only apply during training.

Expected result:

- Slightly better robustness.
- Not expected to be the largest accuracy gain.

---

### 11. Wrist Angular Velocity Input — TODO

Current idea:

Add wrist rotation velocity to the input.

Simple but imperfect version:

```python
wrist_rot_vel = (wrist_rot_6d_t - wrist_rot_6d_prev) / dt
```

Better geometric version:

```python
R_delta = R_t @ R_prev.T
angle_axis_or_log = so3_log(R_delta) / dt
```

Recommendation:

- Do not add this before Phase 1 and Phase 2.
- If added, prefer 3D angular velocity over raw 6D difference.
- Confirm inference can compute the same value from previous predicted wrist, otherwise this introduces another train/inference gap.

Expected dimension:

- Raw 6D velocity: `+6`
- SO(3) angular velocity: `+3`

Risk:

- If training uses ground-truth previous wrist velocity but inference uses predicted previous wrist velocity, exposure bias returns.

---

## Dataset Versioning

Any feature layout change should create a new dataset/checkpoint version.

Suggested naming:

```text
dataset_mano_v1.h5        current 29-dim baseline
dataset_mano_v2_objpos.h5 + wrist_in_obj
dataset_mano_v3_yaw.h5    + yaw augmentation
```

Save metadata into H5:

```text
feature_dim
feature_names
feature_version
augmentation_flags
normalization_policy
```

Why:

- Prevents accidentally training a `feat_dim=29` model on `feat_dim=32` data.
- Makes old checkpoints interpretable.

---

## Normalization Policy

Before implementing Phase 2, decide which features should be normalized.

Recommended:

| Feature type | Normalize? | Note |
|--------------|------------|------|
| positions / distances | yes | mean/std |
| velocities | yes | mean/std |
| bbox continuous values | yes | mean/std |
| direction unit vectors | preferably no, or re-normalize after load | preserves geometry |
| 6D rotations | preferably no, or re-orthogonalize in loss | preserves rotation representation |
| one-hot grip | no | categorical |
| binary flags | no | categorical |

If current pipeline normalizes all dims globally, Phase 1 still works because loss operates on target wrist output. But Phase 2 orientation-aware logic should avoid treating normalized direction as a true unit vector.

---

## Unity Visual Grasp Quality

Offline metrics and Unity visual quality are related but not identical. The current Unity failure mode is:

- wrist/controller reaches the object,
- hand mesh appears near the object,
- fingers do not wrap around the object cleanly,
- palm/fingers may intersect the object or float beside it,
- contact frame does not look like a stable grasp.

This usually means the model is producing a plausible average pose, but there is no final geometric correction that forces contact with the actual object surface.

Likely root causes, in priority order:

1. **Wrist rotation / coordinate alignment error**
   - A small wrist rotation error can make fingers point to the wrong side of a mug/cylinder.
   - Unity conversion bugs, handedness mistakes, or 6D-to-quaternion conversion issues can create large visual errors even when pose loss looks acceptable.

2. **Train/inference domain gap**
   - Training sees ground-truth `wrist_rot_6d` in the input.
   - Unity inference may use controller rotation or previous model prediction.
   - This mismatch can create drift and incorrect approach orientation.

3. **No contact constraint**
   - The LSTM predicts pose, wrist rotation and contact probability.
   - It does not explicitly force fingertips to touch the object SDF surface.
   - It does not prevent hand-object penetration.

4. **Object-relative grasp conditioning is weak**
   - Static SDF embedding tells the model what the object is.
   - It does not fully encode "which side of this object is being grasped right now".

5. **Contact metric is already low**
   - Current `Contact@5mm` is about `27-32%`.
   - This matches the Unity symptom: contact frames are often not geometrically precise enough.

### Unity Debug Checklist

Before adding a large model change, run these controlled Unity checks:

| Test | What to change | Interpretation |
|------|----------------|----------------|
| Controller wrist rotation only | Use controller rotation, ignore model wrist output | If visual grasp improves, model wrist/autoregressive gap is the main issue |
| Model wrist rotation only | Use model wrist output, ignore controller rotation | If this worsens badly, wrist output or conversion is suspect |
| Freeze pose, vary wrist | Keep MANO pose fixed and rotate wrist | Shows sensitivity to wrist frame alignment |
| Draw object-local axes | Visualize object frame and approach direction | Detects object/world frame mismatch |
| Draw fingertip points | Render predicted fingertips | Shows whether fingers miss, float, or penetrate |
| Disable smoothing | Temporarily remove Unity smoothing/filtering | Detects lag introduced after model output |
| Compare left/right transforms | Test both hands on same object | Detects handedness/mirror conversion issues |

If the hand points in a globally wrong direction, fix coordinate conversion before retraining. Training changes cannot compensate for a Unity transform bug.

### Contact-Oriented Metrics for Unity

Add these to `src/evaluate_lstm.py` before judging visual quality:

| Metric | Why |
|--------|-----|
| `Contact@5mm` | strict current metric |
| `Contact@10mm` | more realistic visual threshold |
| `Contact@15mm` | broad grasp plausibility threshold |
| `contact_mpjpe_mm` | average error only on contact frames |
| `binary_contact_accuracy` | whether contact head predicts contact correctly |
| `fingertip_object_dist_mm` | direct visual contact quality |
| `penetration_depth_mm` | detects fingers/palm going through object |

Target interpretation:

- `Contact@5mm = 80%` is unlikely with the current LSTM-only setup.
- `Contact@10mm = 75-85%` is a more realistic strong target.
- `Contact@5mm = 45-60%` would already be a substantial improvement.
- For Unity visuals, `fingertip_object_dist_mm` and penetration are more important than global MPJPE.

### Contact Refinement Path

If Phase 1 improves metrics but Unity still looks bad, add a second-stage contact correction.

Option A — lightweight Unity-side IK:

- When `contact_prob` is high and wrist-object distance is small:
  - keep wrist stable,
  - move fingertips toward nearest object SDF surface points,
  - clamp maximum correction per frame,
  - prevent palm/finger penetration,
  - blend correction in/out over time.

Pros:

- Immediate visual improvement.
- Does not require retraining.
- Easy to tune per object class.

Cons:

- Less academically clean unless documented as post-processing.
- Needs robust object colliders/SDF queries in Unity.

Option B — learned contact refiner:

- Use current LSTM output as initial pose.
- Add a small `ContactRefinerMLP` or reuse `GraspFlowModel`.
- Train only or mostly on contact/near-contact frames.
- Inputs:
  - predicted MANO pose,
  - predicted wrist rotation,
  - contact probability,
  - object embedding,
  - object-relative wrist position,
  - optional LSTM hidden state.
- Outputs:
  - refined MANO pose delta,
  - optional wrist rotation delta.

Loss:

```text
refined_pose_loss
+ contact_frame_weight * FK joint loss
+ fingertip_to_surface_loss
+ penetration_loss
+ temporal_smoothness_loss
```

Pros:

- Better thesis contribution.
- Can directly improve contact visuals.

Cons:

- Requires object surface/SDF distance targets.
- More training/evaluation work.

Recommendation:

- First implement Phase 1 and contact metrics.
- If `Contact@10mm` improves but Unity still looks bad, add Unity-side IK for fast visual correction.
- If thesis scope allows, then replace or complement it with learned contact refinement.

---

## Incomplete Auxiliary Datasets: OakInk and DexYCB

OakInk and DexYCB can help, but only if their usable labels match the current model contract. Incomplete data can also hurt if mixed blindly.

### Expected Benefit

| Dataset | Potential benefit | Main risk |
|---------|-------------------|-----------|
| HOT3D | best match to target runtime and temporal wrist/object interaction | approach-angle bias |
| DexYCB | more hand-object contact diversity, common YCB objects | may lack the same temporal/controller/wrist setup |
| OakInk | broad grasp pose/object diversity | often more grasp-pose oriented than runtime trajectory oriented |
| ARCTIC | bimanual/contact-rich object interaction | domain mismatch and preprocessing complexity |

### When Incomplete Data Helps

Incomplete OakInk/DexYCB data is useful if it has at least:

- object identity or object geometry mapping,
- MANO hand pose,
- wrist/global hand transform,
- object pose or enough information to build object-relative features,
- contact/near-contact frames or distance proxy,
- consistent handedness and coordinate frames.

It can help even without full temporal trajectories if used for:

- contact pose refinement,
- grasp prior learning,
- object-conditioned pose pretraining,
- contact-frame oversampling,
- fingertip/object proximity losses.

### When It Can Hurt

Do not mix incomplete auxiliary data into the main LSTM sequence training if it lacks:

- reliable frame order,
- wrist/controller-like input features,
- object pose in the same convention,
- contact labels or distance labels,
- consistent scale/unit conversion,
- reliable left/right hand convention.

If these are missing, the model may learn a different domain and Unity behavior can get worse.

### Recommended Use of Incomplete OakInk/DexYCB

Use a staged strategy:

1. **Do not use them in Phase 1.**
   - Phase 1 should isolate model/loss/inference mismatch on the existing HOT3D pipeline.

2. **Audit usable fields.**
   - Count sequences/frames with valid MANO pose.
   - Count valid object pose/object id.
   - Count contact or near-contact frames.
   - Verify units and coordinate frames.
   - Verify left/right hand conversion.

3. **Use them first for contact/refiner training, not main LSTM training.**
   - Contact refiner needs grasp pose diversity more than exact controller trajectory.
   - This is where incomplete OakInk/DexYCB can still provide value.

4. **Use source-weighted training if added to LSTM.**
   - Keep HOT3D as the anchor source.
   - Add DexYCB/OakInk with lower loss weight.
   - Track per-source validation separately.

Suggested source weighting:

```text
HOT3D: 1.00
DexYCB complete temporal subset: 0.30-0.50
OakInk contact-pose/refiner subset: 0.20-0.40
Incomplete/noisy samples: exclude or use only for pretraining
```

### Dataset Audit Checklist

Before using OakInk/DexYCB, create a short audit report:

| Field | Required for LSTM? | Required for refiner? | Status |
|-------|--------------------|------------------------|--------|
| MANO pose | yes | yes | TBD |
| wrist/global transform | yes | useful | TBD |
| object pose | yes | yes | TBD |
| object geometry/SDF id | yes | yes | TBD |
| temporal frame order | yes | no | TBD |
| contact labels | useful | very useful | TBD |
| distance-to-object | useful | yes | TBD |
| handedness metadata | yes | yes | TBD |
| scale/unit metadata | yes | yes | TBD |

Decision rule:

- If temporal/controller fields are incomplete, use the dataset for refiner/pretraining only.
- If object pose or MANO pose is unreliable, do not use it.
- If coordinate conversion is uncertain, do not train on it until visualized and verified.

---

## Proposed Implementation Timeline

### Week 1 — Measurement and Loss Geometry

- `[DONE]` Add `rot6d_to_matrix()`
- `[DONE]` Add `rotation_angle_rad()`
- `[DONE]` Add `wrist_deg`, `final_wrist_deg`, `jitter_deg`
- `[DONE]` Add autoregressive validation
- `[DONE]` Replace wrist MSE with geodesic loss
- `[DONE]` Run initial A0/A1-style experiments

### Week 2 — Train/Inference Gap

- `[DONE]` Add scheduled sampling
- `[DONE]` Log `ss_prob`
- `[DONE]` Compare teacher-forced vs autoregressive validation
- `[TODO]` Tune `SS_MAX_PROB` among `0.25`, `0.50`, `0.75`
- `[DONE]` Add contact-frame weighted loss
- `[TODO]` Add contact imbalance weighting and contact threshold calibration
- `[TODO]` Change checkpoint selection to runtime wrist/contact metrics

### Week 3 — Dataset Features

- `[DONE]` Add `wrist_in_obj`
- `[DONE]` Version dataset metadata
- `[DONE]` Rebuild dataset with current feature metadata
- `[PARTIAL]` Retrain from scratch / v2 checkpoint exists, but runtime contact/wrist still poor
- `[TODO]` Add per-angle/per-object validation tables

### Week 4 — Data Bias

- `[DONE]` Add yaw augmentation support
- `[TODO]` Add approach-angle balanced sampler
- `[TODO]` Run angle-held-out validation if possible

### Week 5 — Optional Model Improvements

- `[PARTIAL]` Orientation-aware SDF injection option exists; needs controlled training/export validation
- `[TODO]` Selective noise augmentation
- Wrist angular velocity only if there is evidence it is needed

### Week 6 — Unity Visual Contact

- Run Unity debug checklist
- Add fingertip visualization
- Add `Contact@10mm`, `Contact@15mm`, `contact_mpjpe_mm`
- Decide between Unity-side IK and learned contact refiner
- Prototype contact correction on 2-3 representative objects

### Week 7 — Auxiliary Dataset Audit

- Audit OakInk/DexYCB completeness
- Decide whether each dataset is usable for LSTM, refiner, or neither
- If usable, build a small contact-pose subset first
- Train/evaluate with source-weighted losses only after HOT3D baseline is stable

---

## Experiment Log Template

Use this table in training notes or report.

| Run ID | Changes | Dataset | Checkpoint | `val_ar_final_wrist_deg` | `val_ar_wrist_deg` | `jitter_deg` | `pose_l1_final` | Decision |
|--------|---------|---------|------------|--------------------------|--------------------|--------------|------------------|----------|
| A0 | baseline metrics | v1 | TBD | TBD | TBD | TBD | TBD | baseline |
| A1 | + geodesic | v1 | TBD | TBD | TBD | TBD | TBD | keep/drop |
| A3 | + scheduled sampling | v1 | TBD | TBD | TBD | TBD | TBD | keep/drop |
| A4 | + contact weighting | v1 | TBD | TBD | TBD | TBD | TBD | keep/drop |
| B1 | + wrist_in_obj | v2 | TBD | TBD | TBD | TBD | TBD | keep/drop |
| B2 | + yaw augmentation | v3 | TBD | TBD | TBD | TBD | TBD | keep/drop |
| C1 | + Unity contact metrics | v1/v2 | TBD | TBD | TBD | TBD | TBD | diagnose |
| C2 | + contact refiner/IK | v2/v3 | TBD | TBD | TBD | TBD | TBD | keep/drop |
| D1 | + DexYCB/OakInk refiner subset | aux | TBD | TBD | TBD | TBD | TBD | keep/drop |

---

## Final Recommended Scope for First Implementation

If this plan is approved, implement only this first:

1. Rotation utility functions.
2. Extended validation metrics.
3. Autoregressive validation path.
4. Geodesic wrist loss.
5. Scheduled sampling with conservative schedule.

Hold for later:

- Dataset rebuild
- Yaw augmentation
- Balanced sampler
- Orientation-aware SDF injection
- Noise augmentation
- Angular velocity
- Unity-side contact IK
- Learned contact refiner
- OakInk/DexYCB mixing

Reason:

This isolates the biggest known issue, avoids checkpoint/dataset compatibility problems, and gives truthful metrics before larger pipeline changes. Unity visual contact and incomplete auxiliary datasets should be handled after the baseline wrist/contact metrics are trustworthy.
