# AuraXR — Next Steps
**As of:** 2026-05-13 | **Current Week:** 10 of 18

---

## RIGHT NOW — Two Parallel Tracks

### Track 1: Launch Full Training (Terminal)

Nothing can proceed on the ML side until this finishes (~7 hours).

```bash
cd /Users/muratcelik/Desktop/Thesis/Workspace/V3/hot3d_exploration
source .venv/bin/activate
python3 11_train.py --epochs 100 --batch 256
```

Monitor in a second tab:
```bash
tail -f /Users/muratcelik/Desktop/Thesis/Workspace/V3/data/logs/intentformer_training_log.jsonl
```

Target: `val_mpjpe` should fall from ~800 mm toward **< 50 mm**.  
If stopped before: add `--resume ../data/checkpoints/best.pt`

---

### Track 2: Unity Inspector Tasks (Unity Editor — do while training runs)

| # | Task | Impact if skipped |
|---|------|------------------|
| 1 | Add `InteractableObject` component to **Bottle** (categoryId=1) and **Cup** (categoryId=3) | Cannot grab; hands stay invisible |
| 2 | Set `HandSkinMaterial` Rendering Mode → **Fade** | Hand fade-in has zero effect |
| 3 | Enable **Emission checkbox** on BottleMaterial and CupMaterial | Yellow highlight does nothing |
| 4 | Tick **autoStart** on `ScenarioKitchenTask` | Task stays in Idle; no instructions shown |
| 5 | `ThumbstickLocomotion → Head Collision Layers` → `Environment` | Wall penetration may still occur |
| 6 | Assign `GraspIndicator.leftHandRig` / `rightHandRig` on Bottle and Cup | Yellow highlight doesn't track hand |
| 7 | `AuraXRInferenceManager → Virtual Hand Left` → `LeftHandRig` | Model output not applied to hand rig |
| 8 | `AuraXRInferenceManager → Virtual Hand Right` → `RightHandRig` | Same as above |
| 9 | Verify Layer Collision Matrix: Player↔Environment ON, HandRig↔Environment OFF | Physics conflicts |
| 10 | Disable `AuraXRObjectTracker` if present in scene | Overwrites ProximityDetector every frame |

---

## AFTER TRAINING FINISHES (Week 10)

### Step 1 — Evaluate on val split
```bash
cd /Users/muratcelik/Desktop/Thesis/Workspace/V3/hot3d_exploration
source .venv/bin/activate
python3 12_evaluate.py
```
**Target:** MPJPE < 50 mm, PA-MPJPE < 25 mm  
If MPJPE > 50 mm: add temporal smoothing (EMA on wrist position/rotation) or run more epochs.

### Step 2 — Run baseline models (for thesis comparison table)
```bash
# GRU baseline (~2–3 hours each)
python3 11_train.py --epochs 100 --batch 256 --model gru --no_aug --resume ""

# Single-frame MLP baseline
python3 11_train.py --epochs 100 --batch 256 --model mlp --no_aug --resume ""
```
Run `12_evaluate.py` after each. Goal: table comparing Static vs. MLP vs. GRU vs. IntentFormer.

### Step 3 — Export new ONNX and deploy to Unity
```bash
python3 15_export_onnx_unity.py
```
Drop new `intentformer.onnx` into `Assets/AuraXR/Models/` in Unity. Rebuild APK.

### Step 4 — Implement lerp/slerp interpolation in Unity (overdue from Week 9)
File: `UnityScripts/AuraXRInferenceManager.cs`  
Model infers at 30 FPS cadence (every ~2.4 frames at 72 Hz). Between inference calls:
- Track last two outputs + timestamp
- In `Update()`: `alpha = (Time.time - lastInferenceTime) / (1f / 30f)`
- `lerp` on DeltaPosition, `slerp` on DeltaRotation + WristRotation, `lerp` on joint angles

---

## WEEK 11 — Preliminary User Test

- Deploy to Quest 3, run 2–3 colleagues through the kitchen task (~10 min each)
- Collect qualitative feedback: hand offset, jitter, finger pose, clipping
- If MPJPE > 50 mm after feedback: try INT8 quantization or reduce transformer layers 4→2

---

## WEEK 12–14 — Phase B (Unity Scene)

| Week | Task |
|------|------|
| 12 | Kitchen scenario state machine end-to-end test; wire all haptic + audio cues |
| 13 | PBR hand mesh import; environment polish (4× MSAA, readable UI text); deploy to Quest 3 self-test (20 min) |
| 14 | Colleague testing (3–5 people, 15 min each); iterate on critical issues |

---

## WEEK 15–17 — Phase C (User Study)

- **Week 15:** Submit ethics / get advisor approval; finalize consent form; recruit 20 participants
- **Week 16:** Run 5 sessions/day (Mon–Fri); log CSV per session
- **Week 17:** Analyze data (paired t-test presence A vs. B; Cohen's d); write preliminary results

---

## WEEK 16–18 — Thesis Writing (start Week 16, NOT Week 18)

| Chapter | Target week |
|---------|------------|
| 1: Introduction + 2: Background | 16 |
| 3: System Design + 4: Dataset & Training | 17 |
| 5: Evaluation + 6: Discussion + 7: Conclusion | 17–18 |
| Polish, references, figures, abstract | 18 |

Final file: `Celik_AuraXR_Thesis_2026.pdf`

---

## Success Criteria (from plan.md)

| Metric | Target |
|--------|--------|
| MPJPE on val split | < 50 mm |
| PA-MPJPE | < 25 mm |
| Inference latency on Quest 3 | < 5 ms |
| Presence score (Condition A vs B) | p < 0.05 |
| Embodiment score | > 5.5 / 7 |
| Task completion rate | > 80% |
