# AuraXR — Master Project Plan
## Single Source of Truth: Completed, In-Progress, and Open Tasks

> **Last Updated:** 2026-05-13  
> **Current Project Week:** ~10 of 18  
> **Next Action:** Launch full training AND complete Unity Inspector tasks (both still pending)
>
> **Read this file first.** Earlier docs (`thesis_plan.md`) have outdated architecture dims. This file reflects the actual implemented system.

---

## SECTION 1 — What Is Actually Done

### Data Access & Download
| Item | Status | Location |
|------|--------|----------|
| HOT3D license | ✅ Obtained | — |
| Quest3 train sequences (157) | ✅ Done | `data/quest3/` |
| Quest3 test sequences (68, **no GT**) | ✅ Done | `data/quest3/` |
| Aria train sequences (136) | ✅ Done | `data/aria/` |
| Aria test sequences (62, **no GT**) | ✅ Done | `data/aria/` |
| 3D object assets | ✅ Done | `data/assets/` |
| HOT3D test GT | ❌ Withheld by design | BOP eval server only — use val split |

### Data Pipeline (Python)
| Script | Purpose | Status |
|--------|---------|--------|
| `08_preprocess_annotations.py` | ZIPs → per-sequence .npz (293 files) | ✅ Done |
| `09_build_dataset.py` | .npz → HDF5 T=16 windows | ✅ Done |
| `hot3d_dataset.py` | PyTorch Dataset class | ✅ Done |

`data/hot3d_training.h5` — built and ready.

### Model & Training Code
| Script | Purpose | Status |
|--------|---------|--------|
| `10_intentformer.py` | IntentFormer architecture (5.4M params) | ✅ Done |
| `11_train.py` | Training loop, AdamW, checkpoints | ✅ Done |
| `12_evaluate.py` | MPJPE, PA-MPJPE, ablation evaluation | ✅ Done |
| `13_export_onnx.py` | ONNX opset 17 for Unity Sentis | ✅ Done |

### Export & Unity Integration
| Item | Status | Location |
|------|--------|----------|
| `intentformer.onnx` | ✅ Exported | `data/intentformer.onnx` |
| `intentformer_meta.json` | ✅ Exported | `data/intentformer_meta.json` |
| `AuraXRMetaLoader.cs` | ✅ Written | `UnityScripts/` |
| `AuraXRFeatureAssembler.cs` | ✅ Written | `UnityScripts/` |
| `AuraXRInferenceManager.cs` | ✅ Written | `UnityScripts/` |
| Quest 3 inference latency | ✅ **< 10 ms** | Confirmed on device |
| Wall/table penetration bug | ✅ Fixed 2026-05-13 | `ThumbstickLocomotion.ClampHeadPosition` |

---

## SECTION 2 — Architecture Ground Truth (Authoritative)

> **⚠ Earlier docs had wrong dims. This section is correct.**

### Input Feature Vector — 96 dims per frame

```
[0..8]    Left controller:  xyz(3) + wxyz(4) + grip(1) + trigger(1)   = 9 values
[9..17]   Right controller: same layout                                 = 9 values
[18..24]  Nearest object to LEFT hand:  centroid(3) + bbox(3) + cat(1) = 7 values
[25..31]  Nearest object to RIGHT hand: same layout                    = 7 values
[32..95]  Visual embedding: 64 floats  (currently ALL ZEROS — placeholder)

Total: 9 + 9 + 7 + 7 + 64 = 96 dims
Input tensor shape: [1, 16, 96]
```

**Notes vs. original plan:**
- Object category is `1 float` (raw ID 1–33), not a 16-dim learned embedding
- No closest surface point or surface normal in actual implementation
- Visual branch is disabled (zeros); activating it is a Phase A decision

### Output Vector — 78 dims per frame

```
Per hand (layout identical for hand 0 and hand 1):
  [0..14]   MANO pose θ:    15 floats  (1 DoF per joint)
  [15..24]  MANO shape β:   10 floats
  [25..27]  Wrist position:  3 floats  (x, y, z world metres)
  [28..31]  Wrist rotation:  4 floats  (w, x, y, z quaternion)
  [32..34]  ΔT position:     3 floats  (controller → wrist translation offset)
  [35..38]  ΔT rotation:     4 floats  (controller → wrist rotation offset)
            = 39 floats per hand

Total: 39 × 2 = 78 dims
Output tensor shape: [1, 78]
```

**Why MANO pose = 15, not 45:**
HOT3D ground truth provides 1 scalar per joint (curl angle), not the full 3D axis-angle MANO standard. The thesis Methodology chapter must document this representation choice.

### Virtual Wrist Placement Formula (Unity, every frame)

```csharp
anchor.position = controller.position + pose.DeltaPosition;
anchor.rotation = controller.rotation * pose.DeltaRotation;
```

### Controller Proxy Strategy (Q-A Resolved)

HOT3D has no controller tracking data. **Implemented: wrist transform used directly as controller proxy (Option C — zero offset).** At Quest 3 inference, real controller 6DoF is provided by hardware; the model's predicted ΔT then shifts from controller origin to the user's actual wrist.

### Evaluation: Val Split Only

HOT3D test GT is intentionally withheld (BOP benchmark server). **All quantitative metrics use the validation split** (held-out participants from train set).

---

## SECTION 3 — Resolved Questions

| ID | Question | Resolution | Date |
|----|----------|------------|------|
| Q-A | Controller poses in training | ✅ Option C: wrist xform directly as proxy | Week 8 |
| Q-D | Wrist 6DoF vs ΔT redundancy | ✅ Both kept in 78-dim; wrist provides aux supervision | Week 8 |
| Q-E | Hand shape β handling | ✅ Per-frame in model (10 dims); fixed average β in Unity (no per-user calibration in v1) | Week 8 |
| Q-F | Per-hand vs shared object context | ✅ Per-hand (implemented in feature assembler) | Week 8 |
| Q-J | HOT3D data access | ✅ 293 sequences preprocessed, HDF5 built | Week 6 |
| —   | HOT3D test GT availability | ✅ Not available locally — val split is the evaluation set | Week 7 |

---

## SECTION 4 — Open Critical Issues

### ✅ C-1: Visual Branch — RESOLVED (Week 9)

**Decision:** Keep 96-dim input unchanged. Visual dims [32..95] remain zeros — no HDF5 rebuild, no retraining on different dims.  
**Rationale:** Zero-effort option; model already trained and exported at 96-dim. Document as limitation ("visual grounding is future work").

| Option | Description | Effort | Decision |
|--------|-------------|--------|----------|
| A | Keep 96-dim with zeros as-is (current) | 0 | ✅ **Chosen for v1** |
| B | Remove visual dims → 32-dim retrain | 1 day | Rejected (unnecessary effort) |
| C | Train ResNet-18 on Aria RGB, inject embeddings | 2–3 days | For v2 / future work |

**Thesis note:** "Visual grounding is future work. v1 uses geometry-only input (18 controller + 14 object dims); visual dims [32..95] are placeholder zeros."

---

### 🔴 C-2: Frame Rate Mismatch (30 FPS training → 72 Hz inference)

**Status:** T=16 @30 FPS = 533 ms context at training. Quest 3 at 72 Hz → T=16 = 222 ms context.

| Option | Description | Pros | Cons |
|--------|-------------|------|------|
| A | Cap Quest 3 at 30 FPS | No retraining | 30 Hz feels choppy |
| B | Increase T to 38 @72 Hz, retrain | Same temporal coverage | Longer training, more memory |
| C | Infer every 2–3 frames, interpolate in between | Fast, smooth | Interpolation artifacts |

**Recommended: Option C short-term, Option B if training time allows.**  
**Implement in `AuraXRInferenceManager.cs`:** skip inference on frames 2–3, lerp positions + slerp rotations between outputs.  
**Deadline:** Week 9

---

### 🔴 C-3: No Baseline Models for Comparison

**Status:** No comparison implemented. Thesis RQ1 needs a reference point.

| Baseline | Description | Effort |
|----------|-------------|--------|
| Static Pose | Per-category median pose, no motion | 1 hour |
| Per-Frame MLP | Single-frame input, no temporal window | 2 hours (already in `11_train.py` as `SingleFrameMLP`) |
| GRU Temporal | Replace Transformer with GRU, same input | 4 hours (already in `11_train.py` as `GRUBaseline`) |

**Action:** Run all three baselines in Week 10. Compare MPJPE: Static vs. MLP vs. GRU vs. IntentFormer.  
**Check `11_train.py`** — `GRUBaseline` and `SingleFrameMLP` may already be implemented.

---

### 🟠 H-1: Data Augmentation Not Implemented

**Status:** No augmentation in training loop. Risk: overfitting to 13 training participants.  
**Minimum needed:**
- β perturbation: ±0.5σ Gaussian noise on shape params
- Left/right hand mirror flip: 50% probability per window
- Controller position noise: ±1 cm Gaussian

**Add to `11_train.py` DataLoader transform.** Effort: 2 hours. Deadline: Week 9.

---

### 🟠 H-2: No Physics-Aware Loss (Penetration / Contact)

**Status:** Loss has no geometric terms. Hands may clip through objects.  
**Option:** Use object bbox SDF approximation — penalise when wrist is inside bbox + skin depth.  
**Effort:** 3–4 hours. Add to `11_train.py`.  
**Deadline:** Week 10 (can be skipped if MPJPE meets target without it).

---

### 🟠 H-3: User Study Protocol Not Finalized

**Status:** Partially designed. Three open items:
- [ ] IRB / ethics approval (or confirm advisor sign-off is sufficient)
- [ ] Final questionnaire wording (11 items, see Section 7)
- [ ] Participant recruitment (target n=20, start Week 14)

**Deadline for ethics:** Week 13. Do not wait until Week 15.

---

## SECTION 5 — Week-by-Week Route

### Current State: End of Week 8

Pipeline end-to-end working. ONNX on Quest 3. Full dataset preprocessed. Training loop ready. **Start Phase A now.**

---

### PHASE A — Full Training & Evaluation (Weeks 9–11)

#### Week 9
- [x] **Decide:** visual branch → ✅ **Option A: keep 96-dim with zeros** (no HDF5 rebuild needed)
- [ ] **Decide:** frame rate strategy — implement lerp/slerp interpolation in `AuraXRInferenceManager.cs`
- [x] **Add augmentation** to `11_train.py` and `hot3d_dataset.py` — ✅ **Done** (β noise ±0.5σ, mirror flip 50%, position noise ±1 cm)
- [x] **Fix MPS autograd crash** in `10_intentformer.py` — ✅ **Done** (removed in-place quaternion normalization)
- [ ] **Launch full training** (all 293 sequences, 100 epochs) — ❌ **Still pending**:
  ```bash
  cd hot3d_exploration && source .venv/bin/activate
  python3 11_train.py --epochs 100 --batch 64
  ```
- [ ] Implement **Static Pose baseline** (1 hour)

#### Week 10
- [ ] **Evaluate on val split:**
  ```bash
  python3 12_evaluate.py
  ```
  Target: MPJPE < 50 mm, latency < 5 ms
- [ ] **Run GRUBaseline and SingleFrameMLP** — collect comparison metrics
- [ ] **Create results table:** Static vs. MLP vs. GRU vs. IntentFormer
- [ ] **Export new ONNX:** `python3 13_export_onnx.py`
- [ ] **Deploy to Quest 3** — confirm latency still < 10 ms

#### Week 11
- [ ] **Preliminary user test (n=2–3):** 10 min each, kitchen task, collect qualitative feedback
- [ ] If MPJPE > 50 mm: add temporal smoothing (exponential moving average on wrist position/rotation)
- [ ] If latency > 5 ms: profile GPU, try INT8 quantization in ONNX
- [ ] Document findings in `logs/week11_preliminary_report.txt`

**Phase A Exit Criteria:**
- MPJPE < 50 mm on val split ✓
- Inference < 5 ms on Quest 3 ✓
- No NaN in outputs ✓
- Baseline comparison table ready ✓

---

### PHASE B — UX & Scenario (Weeks 12–14)

#### Week 12 — Interaction Mechanics
- [ ] **Kitchen scenario state machine** (`ScenarioKitchenTask.cs`): Bottle / Cup / Table / Shelf
- [ ] **Task flow:** Pick bottle → pour into cup → place → pick cup
- [ ] **Haptic feedback** (`HapticFeedbackManager.cs`): grip pulse on object contact
- [ ] **Visual feedback** (`GraspIndicator.cs`): yellow highlight when hand < 15 cm from object
- [ ] Wire ProximityDetector to objects with correct HOT3D category IDs
- [ ] Test task flow end-to-end in Unity Editor

#### Week 13 — Scene Polish
- [ ] **Hand mesh:** import or create PBR hand model (albedo, normal map, roughness)
- [ ] **Environment:** walls, floor, directional light, shadow planes
- [ ] **UI overlays:** task instruction text + timer (`UITaskDisplay.cs`)
- [ ] **Sound:** bottle pickup / pour / task complete (`SoundManager.cs`)
- [ ] 4× MSAA, ambient light, readable text at arm's distance
- [ ] Deploy to Quest 3, do 20-minute self-test

#### Week 14 — Scenario Testing
- [ ] **Informal test with 3–5 colleagues** (15 min each)
- [ ] **Iterate on critical issues:** hand offset, jitter, finger pose, clipping
- [ ] **Finalize:** scene must be playable in Condition A (virtual hands) and Condition B (controller)

**Phase B Exit Criteria:**
- Task completable in < 5 min ✓
- No motion sickness in 3 test participants ✓
- Task state machine reliable ✓

---

### PHASE C — Formal User Study (Weeks 15–17)

#### Week 15 — Preparation
- [ ] Submit ethics / get advisor approval
- [ ] Finalize questionnaire (see Section 7)
- [ ] Create consent form
- [ ] Recruit 20 participants (university email, social)
- [ ] Confirm 3-condition design + counterbalancing order (Latin square)
- [ ] Power analysis: n=18–20 for Cohen's d=1.0, α=0.05, power=0.80

#### Week 16 — Run Sessions (5 per day, Mon–Fri)
- [ ] Per session (~50 min): Consent → VR check → Condition A (10 min) → Break (5 min) → Condition B (10 min) → Condition C (10 min) → Questionnaire (5 min) → Optional interview
- [ ] Log CSV: hand positions, grip, task events, timestamps
- [ ] Screen recording per session
- [ ] Observer notes per session

#### Week 17 — Analysis
- [ ] Analyze presence scores (paired t-test: virtual vs. controller)
- [ ] Compute Cohen's d effect size
- [ ] Analyze objective metrics (jitter, task completion time, errors)
- [ ] Write preliminary results report

**Phase C Exit Criteria:**
- N ≥ 15 participants complete all conditions ✓
- Virtual Hands > Controller on presence, p < 0.05 ✓
- No adverse events ✓

---

### PHASE D — Thesis Writing (Weeks 16–18)

**Start writing in Week 16, not Week 18.**

| Chapter | Content | Target Week |
|---------|---------|-------------|
| 1: Introduction | Motivation, problem, RQ1–RQ3, contributions | 16 |
| 2: Background | VR presence, MANO, HOT3D, Transformer, related work | 16 |
| 3: System Design | Architecture (corrected dims), pipeline, decisions | 17 |
| 4: Dataset & Training | HOT3D preprocessing, training, augmentation, baselines | 17 |
| 5: Evaluation | Quantitative metrics, ablations, user study | 17 |
| 6: Discussion | Findings, limitations, future work | 18 |
| 7: Conclusion | Contributions, impact | 18 |
| Appendix | Code listings, questionnaire, consent form | 18 |

**Week 18 checklist:**
- [ ] Spell-check + read-aloud pass
- [ ] All references cited and formatted (IEEE/ACM)
- [ ] All figures have captions; axes labeled
- [ ] Abstract < 250 words, covers problem + approach + results
- [ ] Supplementary: demo video (30s), code, questionnaire CSV
- [ ] File named: `Celik_AuraXR_Thesis_2026.pdf`
- [ ] Sent to advisor for final approval

---

## SECTION 6 — Metrics & Success Criteria

| Metric | Target | Measurement |
|--------|--------|-------------|
| **MPJPE** | < 50 mm (acceptable) / < 35 mm (good) | `12_evaluate.py` on val split |
| **PA-MPJPE** | < 25 mm | `12_evaluate.py` |
| **Alignment Error** | < 10 mm | Controller origin vs. virtual wrist |
| **Inference Latency** | < 5 ms | Unity Profiler on Quest 3 |
| **Temporal Jitter** | < 10 mm | Joint velocity std-dev across frames |
| **Presence Score** | Virtual > Controller, p < 0.05 | Paired t-test |
| **Embodiment Score** | > 5.5 / 7 | Questionnaire mean |
| **Motion Sickness** | < 1 (SUS) | Post-session SUS scale |
| **Task Completion** | > 80% of attempts | Observer log |

---

## SECTION 7 — User Study Questionnaire (Final Draft)

**3-Condition Within-Subjects Design:** Virtual Hands (A) / Controller (B) / Static Hand Pose (C)  
**Order counterbalanced** with Latin square across 6 participant groups.

### Questions (1–7 Likert unless noted)

| # | Dimension | Question |
|---|-----------|---------|
| 1 | Presence | I had a strong sense of "being there" in the virtual environment. |
| 2 | Presence | The virtual environment felt real to me. |
| 3 | Presence | I felt like I was in the virtual space, not just looking at a screen. |
| 4 | Embodiment | I felt like the virtual hands were my own hands. |
| 5 | Embodiment | The hand motion matched my real hand motion well. |
| 6 | Embodiment | I could feel what I was interacting with through the hands. |
| 7 | Naturalness | Hand interactions felt natural and intuitive. |
| 8 | Naturalness | Grasping objects felt realistic. |
| 9 | Naturalness | I forgot I was not using my real hands. |
| 10 | Usability | I could easily control the virtual hand. |
| 11 | Usability | I understood what the hand was doing at all times. |
| SUS | Sickness | Simulator Sickness Questionnaire (0–6 per symptom) |

**Collect immediately after each condition** (not at end of session).

---

## SECTION 8 — Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| MPJPE > 50 mm after full training | Medium | High | Augmentation, longer training, reduce LR |
| Visual branch domain gap (Aria→Quest3) | High | Low | Keep zeros (Option A) — document as limitation |
| 72 Hz jitter after frame rate fix | Medium | Medium | Output interpolation (lerp/slerp) |
| Few participants (N < 12) | Medium | High | Recruit by Week 14, offer compensation |
| Training divergence (NaN loss) | Low | High | Gradient clip=1.0, AdamW + LR warmup |
| Writing crunch at Week 18 | High | High | **Start Chapter 1 in Week 16** |
| Quest 3 build breakage | Low | Medium | Test APK weekly during Phase B |

---

## SECTION 9 — Quick Reference

### Commands
```bash
# Train
cd /Users/muratcelik/Desktop/Thesis/Workspace/V3/hot3d_exploration
source .venv/bin/activate
python3 11_train.py

# Evaluate
python3 12_evaluate.py

# Export ONNX
python3 13_export_onnx.py
```

### Key Files
```
data/
├── intentformer.onnx          ← current POC model (in Unity now)
├── intentformer_meta.json     ← normalization stats (T=16, F=96, OUT=78)
├── hot3d_training.h5          ← full training dataset (293 sequences)
├── checkpoints/               ← training checkpoints
└── preprocessed/              ← 293 per-sequence .npz files

UnityScripts/
├── AuraXRMetaLoader.cs        ← loads meta JSON, normalizes features
├── AuraXRFeatureAssembler.cs  ← 96-dim feature builder, T=16 ring buffer
└── AuraXRInferenceManager.cs  ← ONNX runner, 78-dim decoder, ΔT formula
```

### Week Summary Table

| Week | Phase | Key Deliverable | Status |
|------|-------|-----------------|--------|
| 1–4 | Setup + POC | Python env, HOT3D download, POC model, Unity smoke test | ✅ Done |
| 5–8 | Full Pipeline | 293 sequences preprocessed, IntentFormer trained, ONNX exported, Quest 3 working | ✅ Done |
| 9 | A | Full retraining with augmentation + baselines | ⚠️ Partial (aug ✅, MPS fix ✅, wall fix ✅ — training NOT launched) |
| **10** | **A** | **Launch training + evaluate + baselines** | 🔄 **Current** |
| 10 | A | Evaluate (MPJPE, latency) + new ONNX export | ⏳ |
| 11 | A | Preliminary user test, iterate on model | ⏳ |
| 12 | B | Kitchen scenario Unity scene | ⏳ |
| 13 | B | Scene polish, hand mesh, audio, UI | ⏳ |
| 14 | B | Scenario testing with colleagues | ⏳ |
| 15 | C | Ethics + questionnaire + recruit participants | ⏳ |
| 16 | C+D | Run 5 study sessions + start thesis writing | ⏳ |
| 17 | C+D | Data analysis + methods/results chapters | ⏳ |
| 18 | D | Polish + final submission | ⏳ |

---

*This plan supersedes all earlier status information in other MD files. For architecture dims, trust this document over `thesis_plan.md`.*
