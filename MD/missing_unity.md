# AuraXR — Unity Debug Guide
**Updated:** 2026-05-13

---

## CONFIRMED WORKING — Do not redo these

| Item | Evidence |
|------|----------|
| Console: `[AuraXR] Meta loaded` + `Model loaded` | Screenshot confirmed |
| `InteractableObject` on Bottle and Cup | Screenshot confirmed (Cup: categoryId=3) |
| `HandSkinMaterial` Rendering Mode = Fade | Screenshot confirmed |
| All Inspector references (per setup_unity.md) | User confirmed |

---

## PROBLEM 1 — Virtual hand does not appear near object

### Confirmed NOT the cause
- InteractableObject ✅ → `HandProximityVisibility` can find objects
- Material Fade ✅ → alpha writes work
- References assigned ✅ → script does not disable itself

### Remaining cause A — LeftHandRig has no SkinnedMeshRenderer children

`HandProximityVisibility.Start()` runs:
```csharp
_leftHandRenderers = leftHandRig.GetComponentsInChildren<SkinnedMeshRenderer>();
if (_leftHandRenderers.Length == 0)
    Debug.LogWarning("[HandProximityVisibility] leftHandRig has no SkinnedMeshRenderer children — hand will never appear.");
```
If `OVRCustomHandPrefab_L` is not nested **inside** `LeftHandRig` as a child, this returns empty and no alpha is ever set. The hand has no mesh to show.

**Check:** Select `LeftHandRig` in Hierarchy → expand it → you must see `OVRCustomHandPrefab_L` (or equivalent) as a child with a `SkinnedMeshRenderer`. If it is not there, drag it in as a child now.

**Check Console for this warning** — if it appears, this is the cause.

### Remaining cause B — HandSkinMaterial not on the hand mesh

`SetHandAlpha` modifies `r.material.color.a` on the SkinnedMeshRenderer it finds. If `HandSkinMaterial` was created but never dragged onto the hand mesh's SkinnedMeshRenderer, it modifies the default Unity material instead — no visible effect.

**Check:** Select the SkinnedMeshRenderer inside `LeftHandRig` → Inspector → Materials → slot 0 must show `HandSkinMaterial`. If it shows `Default-Material` or anything else, drag `HandSkinMaterial` onto it.

### Remaining cause C — Hand IS visible but in the wrong place (most likely)

The POC model (3 epochs, smoke test only) outputs a near-random `DeltaPosition`. `ApplyToAnchor` places `LeftHandRig` at:
```csharp
anchor.position = controller.position + pose.DeltaPosition;
```
`DeltaPosition` from the untrained model could be `(1.5, -2.3, 0.8)` — putting the hand metres away from the controller. **The hand may actually be fading in somewhere you are not looking.**

**Fix: add debug bypass mode to InferenceManager** (see Code Fix section below). This forces `DeltaPosition = Vector3.zero` so the hand tracks directly at the controller.

---

## PROBLEM 2 — Cannot grab object

### Confirmed NOT the cause
- `InteractableObject` on Bottle and Cup ✅ → `_allInteractables` is not empty
- `leftHandWrist` / `rightHandWrist` assigned ✅ → grip path runs

### Remaining cause — LeftHandRig is not near the object

`VirtualHandGrab.FindNearest()` searches within `grabRadius = 0.15 m` from `LeftHandRig.position`. Since the POC model puts `LeftHandRig` at the wrong position (Cause C above), `FindNearest` cannot find anything within 15 cm even when the controller is touching the bottle.

This is the same root cause as Problem 1 Cause C. **Fixing the bypass mode (below) fixes this too.**

Two-line workaround if you want to test grab without the code fix:
- Select **GameManager** → `VirtualHandGrab` → set `Grab Radius` to `0.5`
- This widens the search sphere enough that the wrist (wherever it is) may still reach the object

---

## PROBLEM 3 — Cannot confirm inference is running per-frame

### Confirmed NOT the cause
- Startup logs appear ✅ → model and meta loaded, inference path is reachable

### Remaining cause — RunInference() has zero debug logs

`RunInference()` is called every 2 frames silently. There is no way to know if it ran, threw an exception, or produced sensible output without adding a temporary log.

**Add this one line at the end of `RunInference()` in `AuraXRInferenceManager.cs`:**

```csharp
Debug.Log($"[AuraXR] Infer — L.DeltaPos={_rawLeftHand.DeltaPosition:F2}  R.DeltaPos={_rawRightHand.DeltaPosition:F2}");
```

In Play mode you should see these lines every ~2 frames. If `DeltaPos` values are large (e.g. `(1.5, -2.3, 0.8)`) that confirms Cause C above — POC model is outputting garbage offsets.

Remove this log after confirming.

---

## CODE FIX — Debug bypass mode in AuraXRInferenceManager

This is the key fix for both Problem 1 and Problem 2. It adds a single checkbox `Debug Bypass Model` that when enabled, skips the ONNX output and places the hand directly at the controller with zero offset. This lets you verify the full grab + visibility pipeline works, independently of model quality.

Open `UnityScripts/AuraXRInferenceManager.cs` and add the field and bypass block:

**Add field after the `[Header("Frame Rate Handling")]` block (around line 36):**
```csharp
[Header("Debug")]
[Tooltip("Skip ONNX output — hand tracks controller directly. Use to test grab/visibility with untrained model.")]
public bool debugBypassModel = false;
```

**Replace `RunInference()` with this version:**
```csharp
private void RunInference()
{
    if (debugBypassModel)
    {
        _rawLeftHand  = new HandPose(); // DeltaPosition = Vector3.zero by default
        _rawRightHand = new HandPose();
        Debug.Log("[AuraXR] BYPASS MODE — hand tracking controller directly.");
        return;
    }

    // 1. Collect normalised feature window
    featureAssembler.CopyWindowFlat(_flatWindow);

    int T = AuraXRFeatureAssembler.WindowFrames;
    int F = AuraXRFeatureAssembler.FeatureDim;
    for (int t = 0; t < T; t++)
        for (int f = 0; f < F; f++)
        {
            int idx = t * F + f;
            _flatWindow[idx] = (_flatWindow[idx] - metaLoader.FeatureMean[f])
                               / metaLoader.FeatureStd[f];
        }

    // 2. Build Sentis input tensor [1, 16, 96]
    _inputTensor?.Dispose();
    _inputTensor = new Tensor<float>(new TensorShape(1, T, F), _flatWindow);

    // 3. Run inference
    _worker.Schedule(_inputTensor);
    var outputTensor = _worker.PeekOutput("pose") as Tensor<float>;
    using var cpuTensor = outputTensor.ReadbackAndClone();

    // 4. Copy raw output to float[]
    float[] raw = new float[78];
    for (int i = 0; i < 78; i++)
        raw[i] = cpuTensor[0, i];

    // 5. De-normalise
    metaLoader.DenormaliseTarget(raw);

    // 6. Decode into raw HandPose structs
    _rawLeftHand  = DecodeHand(raw, offset: 0);
    _rawRightHand = DecodeHand(raw, offset: 39);

    Debug.Log($"[AuraXR] Infer — L.DeltaPos={_rawLeftHand.DeltaPosition:F2}  R.DeltaPos={_rawRightHand.DeltaPosition:F2}");
}
```

**In Unity Inspector after saving:**
- Select **GameManager** → `AuraXRInferenceManager` → check **Debug Bypass Model** ✓
- Press **Play**

With bypass enabled:
- `LeftHandRig` position = `LeftControllerAnchor.position + Vector3.zero` = exactly at controller
- Hand should appear where the controller is
- Grab should work when controller is within `grabRadius` of the bottle

**After full training completes (100 epochs):**
- Uncheck `Debug Bypass Model` — real model output takes over
- Hand will track to actual wrist position relative to controller
- Restore `Grab Radius` to `0.15` if you widened it

---

## DIAGNOSIS FLOWCHART

```
Press Play
   │
   ├─ No "Meta loaded" or "Model loaded" in Console?
   │     → metaJson or modelAsset not assigned on GameManager
   │
   ├─ "[HandProximityVisibility] leftHandRig has no SkinnedMeshRenderer" warning?
   │     → OVRCustomHandPrefab_L not nested inside LeftHandRig
   │
   ├─ Both startup logs appear, but hand never visible anywhere?
   │     → Enable debugBypassModel → if hand now appears at controller: POC model DeltaPos garbage (expected)
   │     → If STILL invisible: HandSkinMaterial not on SkinnedMeshRenderer
   │
   └─ Hand visible at controller position but grab doesn't fire?
         → Check grabRadius (set to 0.25 or 0.5 as workaround)
         → Check Console for "[VirtualHandGrab] Grabbed" — if missing, wrist not within radius
```

---

## STATUS AFTER FULL TRAINING

Once `python3 11_train.py --epochs 100 --batch 256` completes and new ONNX is exported:
- Uncheck `debugBypassModel`
- Restore `Grab Radius` to `0.15`
- `DeltaPosition` will be meaningful — hand appears near actual wrist, not at controller origin
- Remove the `Debug.Log` line from `RunInference()`
