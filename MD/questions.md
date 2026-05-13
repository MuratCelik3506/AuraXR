# AuraXR — Questions & Decisions Log
> Last Updated: 2026-05-07
> Status: All critical questions resolved. See `plan.md` for implementation tasks.

---

## CRITICAL ARCHITECTURE DECISIONS

---

### Q-A ✅ Controller Poses in Training
**Decision: Wrist transform directly as controller proxy (zero offset — Option C).**

HOT3D has no controller tracking. MANO wrist position + quaternion is used as the controller input during training. The model's predicted ΔT absorbs the true wrist-to-controller offset at runtime.

- Implemented in: `09_build_dataset.py`
- At runtime: Quest 3 provides real controller 6DoF; model's ΔT positions the virtual hand correctly.
- Thesis note: Document as limitation — proxy offset is zero during training but non-zero at runtime.

---

### Q-B ✅ Visual Branch — Resolved (Week 9)
**Decision: Keep 96-dim input unchanged. Visual dims [32..95] = zeros.**

HOT3D Quest 3 sequences are monochrome (no RGB). Rather than rebuilding the HDF5 and retraining on 32-dim, the simpler path is keeping the current 96-dim format with visual dims as placeholder zeros. No code changes needed.

- No HDF5 rebuild, no dim change in `10_intentformer.py` or `AuraXRFeatureAssembler.cs`
- Thesis note: "Visual grounding is future work. v1 uses geometry-only input (18 controller + 14 object active dims); visual embedding dims [32..95] are placeholder zeros."

---

### Q-C ✅ Frame Rate Mismatch (30 FPS Training → 72 Hz Inference)
**Decision: Temporal interpolation — train at 30 FPS, render at 72 Hz via lerp/slerp.**

Model runs inference at ~30 FPS cadence (~every 2.4 Quest 3 frames). Between inference calls, Unity interpolates positions (lerp) and rotations (slerp).

- Implementation pending in `AuraXRInferenceManager.cs` (overdue — was Week 9, now Week 10):
  - Track last two inference outputs + timestamp
  - In `Update()`: alpha = `(Time.time - lastInferenceTime) / (1f/30f)`
  - lerp on DeltaPosition, slerp on DeltaRotation + WristRotation, lerp on joint angles
- Thesis note: Model predicts at 30 FPS; Unity renders at 72 Hz via interpolation.

---

### Q-D ✅ Output: Wrist 6DoF vs ΔT Redundancy
**Decision: Keep both. Wrist provides auxiliary supervision; ΔT is used at runtime.**

Loss weights in `11_train.py`: `wrist_t λ=1.0`, `wrist_q λ=1.0`, `delta_t λ=0.4`, `delta_q λ=0.4`.

At runtime only ΔT is used: `anchor.position = controller.position + pose.DeltaPosition`

---

### Q-E ✅ Hand Shape β — Per-Frame vs. Fixed
**Decision: Per-frame prediction in model (10 dims). Average β as Unity fallback if jitter visible.**

Model outputs β every frame. In `AuraXRInferenceManager.cs`, if β jitter is noticeable: apply EMA (`smoothedBeta = α * newBeta + (1-α) * smoothedBeta`). No per-user calibration in v1.

---

### Q-F ✅ Per-Hand Object Context
**Decision: Already correctly implemented — per-hand object slots.**

Feature layout: `[18..24]` = nearest object to left hand, `[25..31]` = nearest object to right hand. `ProximityDetector.cs` sets these independently.

---

## ARCHITECTURE DECISIONS

---

### Q-G ✅ Baseline Models
**Decision: Three baselines — MLP and GRU already in `11_train.py`; StaticPose needs 1 hr of work.**

| Baseline | Status | Command |
|----------|--------|---------|
| Per-Frame MLP | ✅ In `11_train.py` | `python3 11_train.py --model mlp` |
| GRU Temporal | ✅ In `11_train.py` | `python3 11_train.py --model gru` |
| Static Pose | 🔧 1 hour to add | Predict per-category median θ |

Run all three in Week 10, create comparison table: Static vs. MLP vs. GRU vs. IntentFormer.

---

### Q-H ✅ Cross-Hand Coordination
**Decision: 2 separate query tokens sharing a common encoder (already implemented).**

From `10_intentformer.py`: `self.query_tokens = nn.Parameter(torch.randn(1, 2, d_model) * 0.02)` — both tokens attend to the same encoder output. Implicit bimanual coordination via shared encoder.

---

### Q-I ✅ Novel Object Category at Inference
**Decision: Category 0 = "no nearby object" fallback. Manual HOT3D mapping for Unity scene objects.**

| Unity Object | HOT3D Category | ID |
|-------------|----------------|-----|
| Bottle | water_bottle | Check `data/assets/` metadata |
| Cup / Mug | cup / mug | Check `data/assets/` metadata |
| Spoon | utensil | Check `data/assets/` metadata |

Wire category IDs to `InteractableObject.categoryId` in Unity Inspector (Week 12).

---

## DATA PIPELINE DECISIONS

---

### Q-J ✅ HOT3D Access
**Resolved (Week 6).** 293 sequences downloaded and preprocessed. `data/hot3d_training.h5` built.  
HOT3D test GT is withheld (BOP server only) — use validation split for all quantitative metrics.

---

### Q-K ✅ Contact Annotations
**Decision: No contact loss. Distance-proxy metric in evaluation.**

HOT3D has no frame-level contact labels. Report **contact ratio** = fraction of frames where predicted wrist is within 8 cm of nearest object centroid (proxy metric). Add to `12_evaluate.py` (Week 10).

---

### Q-L ✅ Data Augmentation
**Decision: 3 augmentations implemented in `hot3d_dataset.py` + `11_train.py`. Done Week 9.**

| Augmentation | Strength | Status |
|--------------|----------|--------|
| Controller position noise | ±1 cm uniform | ✅ Done |
| Beta perturbation on θ | ±0.5σ Gaussian | ✅ Done |
| Mirror flip (swap h0 ↔ h1) | 50% probability | ✅ Done |

Enabled by default for IntentFormer; pass `--no_aug` to disable for ablation.

---

## SCENE & UX DECISIONS

---

### Q1 ✅ Unity Scene
Kitchen counter scenario: Bottle (cat 1), Cup (cat 3), Spoon, Table, Shelf.

Task flow: `Ready → PickBottle → Pour → PlaceBottle → PickCup → Complete` — build Week 12.

---

### Q2 ✅ Transition Trigger
**Decision: Always-on virtual hands.** `HandVisibilityController.cs` enables renderer unconditionally. `TransitionBlender` smooth-fades controller out over 0.3 s at app start.

---

### Q3 ✅ Virtual Hand Appearance
**Decision: PBR skin material + normal map (Standard shader, Built-in RP).**

- Albedo: skin tone RGB ~230/180/140, Roughness 0.6, No metallic, Normal map with wrinkle detail.
- Build target: Week 13.

---

### Q4 ✅ Per-User β Calibration
**Decision: Fixed average β in v1.** Per-user calibration via hand silhouette is future work.

---

## USER STUDY DECISIONS

---

### Q5 ✅ Study Conditions
**Decision: 3-condition within-subjects design.**

| Condition | Description |
|-----------|-------------|
| A — Virtual Hands | AuraXR predicted virtual hands |
| B — Controller | Default Quest 3 controller model |
| C — Static Hand | Fixed average MANO pose, no motion |

Latin square counterbalancing across 6 orderings. Session: ~50 min per participant.

---

### Q5a ✅ Sample Size
**Decision: N = 20 participants (within-subjects).**

Power analysis: Cohen's d = 1.0, α = 0.05, power = 0.80 → N = 17 minimum. Target N = 20 with 10% dropout buffer. Recruitment starts Week 14.

---

### Q5b ✅ Questionnaire
**Decision: 11-item Likert (1–7) + SSQ. Collected immediately after each condition.**

| # | Dimension | Item |
|---|-----------|------|
| 1–3 | Presence | Sense of being there / felt real / in the space |
| 4–6 | Embodiment | Hands feel mine / motion matched / feel contact |
| 7–9 | Naturalness | Natural/intuitive / grasping realistic / forgot not real hands |
| 10–11 | Usability | Could control / understood what hand was doing |
| SSQ | Sickness | Simulator Sickness Questionnaire (9 symptoms, 0–3 each) |

Primary outcome: Presence score (items 1–3 mean). Comparison: Condition A vs. B, paired t-test.

---

### Q6 ⚠️ Thesis Submission Deadline
**Assumed: End of Week 18 (late August 2026). Confirm with advisor.**

---

## RESOURCE DECISIONS

---

### Q7 ✅ GPU Hardware
**Decision: Training runs locally. Colab Pro fallback if >48 hr per run.**

Monitor device: `python3 -c "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'MPS/CPU')"`

### Q7.1 ✅ Inference Latency
**< 10 ms confirmed on Quest 3 (POC model).** Profile again after full retrain.

If > 5 ms: (1) switch to `BackendType.GPUCompute` in Sentis, (2) quantize to INT8, (3) reduce decoder layers 4→2.

### Q8.1 ✅ Acceptance Criteria
Defined in `plan.md` Section 6. Key: MPJPE < 50 mm, latency < 5 ms, presence p < 0.05.

---

*All questions resolved. For implementation tasks and week-by-week schedule, see `plan.md`.*
