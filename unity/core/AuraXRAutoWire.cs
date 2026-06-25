using UnityEngine;

namespace AuraXR
{
    /// <summary>
    /// Runs before every other AuraXR script (ExecutionOrder -200) and fills in
    /// Inspector references automatically — no manual drag-and-drop needed.
    ///
    /// Wiring order:
    ///   1. HandProximityVisibility  — controller + hand rig transforms for fade logic
    ///   2. VirtualHandGrab          — wrist transforms + controller anchors for grab
    ///   3. AuraXRInferenceManager   — featureAssembler reference
    ///   4. HandSkeletonAnchor       — added to hand rigs when OVRSkeleton is present
    ///   5. ProximityDetector        — featureAssembler reference (nearest-object context)
    ///   6. AuraXRFeatureAssembler   — controller transforms (critical: hand anchor placement)
    ///   7. Hand materials           — set to Transparent so HandProximityVisibility can fade alpha
    ///
    /// Attach this component to any GameObject in the scene (e.g. GameManager).
    /// It self-destructs after Awake() leaving zero runtime overhead.
    /// </summary>
    [DefaultExecutionOrder(-200)]
    public class AuraXRAutoWire : MonoBehaviour
    {
        [Header("Auto-Wire Options")]
        [Tooltip("Wire all references on HandProximityVisibility (controllers + hand rigs).")]
        public bool wireHandProximityVisibility = true;

        [Tooltip("Wire LeftHandRig/RightHandRig wrists and controller anchors on VirtualHandGrab.")]
        public bool wireVirtualHandGrab = true;

        [Tooltip("Wire featureAssembler on AuraXRInferenceManager if not already assigned.")]
        public bool wireInferenceManager = true;

        [Tooltip("Add HandSkeletonAnchor to LeftHandRig / RightHandRig when OVRSkeleton is present.")]
        public bool wireHandSkeletonAnchors = true;

        [Tooltip("Set hand rig SkinnedMesh materials to Transparent (URP or Standard shader).")]
        public bool setHandMaterialsTransparent = true;

        // ── Awake ─────────────────────────────────────────────────────────────

        void Awake()
        {
            if (wireHandProximityVisibility) WireHandProximityVisibility();
            if (wireVirtualHandGrab)         WireVirtualHandGrab();
            if (wireInferenceManager)        WireInferenceManagerAnchors();
            if (wireHandSkeletonAnchors)     WireHandSkeletonAnchors();
            WireProximityDetector();
            WireFeatureAssemblerControllers();
            WireHandRigControllers();
            WireInferenceManagerVirtualHands();
            WireIKRefinement();
            EnsureObjectColliders();
            if (setHandMaterialsTransparent) SetHandMaterialsTransparent();

            Debug.Log("[AutoWire] Wiring complete. Self-destroying.");
            Destroy(this);
        }

        // ── HandProximityVisibility ───────────────────────────────────────────

        /// <summary>
        /// Fills leftController/rightController and leftHandRig/rightHandRig on
        /// HandProximityVisibility so the hand/controller cross-fade works at runtime.
        /// </summary>
        private void WireHandProximityVisibility()
        {
            var hpv = FindAny<HandProximityVisibility>();
            if (hpv == null) { Log("HandProximityVisibility not found — skipped."); return; }

            var leftRig  = FindGoByNameAny("LeftHandRig",  "OVRCustomHandPrefab_L");
            var rightRig = FindGoByNameAny("RightHandRig", "OVRCustomHandPrefab_R");

            if (hpv.leftHandRig  == null && leftRig  != null) { hpv.leftHandRig  = leftRig;  Log("HPV: leftHandRig  → " + leftRig.name);  }
            if (hpv.rightHandRig == null && rightRig != null) { hpv.rightHandRig = rightRig; Log("HPV: rightHandRig → " + rightRig.name); }

            if (hpv.leftController == null)
            {
                var lc = FindTrackingAnchor(true);
                if (lc != null) { hpv.leftController = lc.transform; Log("HPV: leftController → " + lc.name); }
                else Debug.LogWarning("[AutoWire] HPV: left controller anchor not found — try 'LeftHandAnchor'.");
            }
            else if (IsProbablyControllerVisual(hpv.leftController))
                WarnVisualAssigned("HPV.leftController", hpv.leftController);

            if (hpv.rightController == null)
            {
                var rc = FindTrackingAnchor(false);
                if (rc != null) { hpv.rightController = rc.transform; Log("HPV: rightController → " + rc.name); }
                else Debug.LogWarning("[AutoWire] HPV: right controller anchor not found — try 'RightHandAnchor'.");
            }
            else if (IsProbablyControllerVisual(hpv.rightController))
                WarnVisualAssigned("HPV.rightController", hpv.rightController);
        }

        // ── VirtualHandGrab ───────────────────────────────────────────────────

        /// <summary>
        /// Wires VirtualHandGrab's wrist transforms (hand rigs) and the physical
        /// controller anchors used for grab proximity detection.
        /// </summary>
        private void WireVirtualHandGrab()
        {
            var grab = FindAny<VirtualHandGrab>();
            if (grab == null) { Log("VirtualHandGrab not found — skipped."); return; }

            var leftRig  = FindGoByNameAny("LeftHandRig",  "OVRCustomHandPrefab_L");
            var rightRig = FindGoByNameAny("RightHandRig", "OVRCustomHandPrefab_R");

            if (grab.leftHandWrist  == null && leftRig  != null) { grab.leftHandWrist  = leftRig.transform;  Log("VHGrab: leftHandWrist  → " + leftRig.name);  }
            if (grab.rightHandWrist == null && rightRig != null) { grab.rightHandWrist = rightRig.transform; Log("VHGrab: rightHandWrist → " + rightRig.name); }

            if (grab.leftController == null)
            {
                var lc = FindTrackingAnchor(true);
                if (lc != null) { grab.leftController = lc.transform; Log("VHGrab: leftController → " + lc.name); }
            }
            else if (IsProbablyControllerVisual(grab.leftController))
                WarnVisualAssigned("VirtualHandGrab.leftController", grab.leftController);

            if (grab.rightController == null)
            {
                var rc = FindTrackingAnchor(false);
                if (rc != null) { grab.rightController = rc.transform; Log("VHGrab: rightController → " + rc.name); }
            }
            else if (IsProbablyControllerVisual(grab.rightController))
                WarnVisualAssigned("VirtualHandGrab.rightController", grab.rightController);
        }

        // ── AuraXRInferenceManager ────────────────────────────────────────────

        /// <summary>
        /// Wires AuraXRInferenceManager.featureAssembler so inference has access to
        /// controller transforms and nearest-object context every frame.
        /// </summary>
        private void WireInferenceManagerAnchors()
        {
            var inf = FindAny<AuraXRInferenceManager>();
            if (inf == null) { Log("AuraXRInferenceManager not found — skipped."); return; }

            if (inf.featureAssembler == null)
            {
                inf.featureAssembler = FindAny<AuraXRFeatureAssembler>();
                if (inf.featureAssembler != null)
                    Log($"InfMgr: featureAssembler → '{inf.featureAssembler.gameObject.name}'.");
                else
                    Debug.LogWarning("[AutoWire] InfMgr: AuraXRFeatureAssembler not found — assign manually.");
            }
            else
                Log("InfMgr: featureAssembler already assigned — skipped.");

            // Wire ObjectSDFDatabase (PCA embeddings) for LSTM v4 mode
            if (inf.sdfDatabase == null)
            {
                inf.sdfDatabase = FindAny<ObjectSDFDatabase>();
                if (inf.sdfDatabase != null)
                    Log($"InfMgr: sdfDatabase → '{inf.sdfDatabase.gameObject.name}'.");
            }

            // Wire SDFGridDatabase (exact trilinear grids) for LSTM v4 mode
            if (inf.sdfGridDatabase == null)
            {
                inf.sdfGridDatabase = FindAny<SDFGridDatabase>();
                if (inf.sdfGridDatabase != null)
                    Log($"InfMgr: sdfGridDatabase → '{inf.sdfGridDatabase.gameObject.name}'.");
                else
                    Debug.LogWarning("[AutoWire] SDFGridDatabase not found — SDF local features will use bbox fallback.");
            }
        }

        // ── HandSkeletonAnchor ────────────────────────────────────────────────

        /// <summary>
        /// Adds HandSkeletonAnchor to LeftHandRig and RightHandRig when OVRSkeleton is
        /// present. HandSkeletonAnchor re-roots bones each LateUpdate so the wrist
        /// stays at the position driven by AuraXRInferenceManager.
        /// </summary>
        private void WireHandSkeletonAnchors()
        {
            AddHandSkeletonAnchorIfNeeded("LeftHandRig",  "OVRCustomHandPrefab_L");
            AddHandSkeletonAnchorIfNeeded("RightHandRig", "OVRCustomHandPrefab_R");
        }

        private void AddHandSkeletonAnchorIfNeeded(params string[] names)
        {
            var go = FindGoByNameAny(names);
            if (go == null)
            {
                Debug.LogWarning($"[AutoWire] HandSkeletonAnchor: '{names[0]}' not found — skipped.");
                return;
            }

            if (go.GetComponent<HandSkeletonAnchor>() != null) { Log($"HandSkeletonAnchor already on '{go.name}' — skipped."); return; }
            if (go.GetComponent<OVRSkeleton>()        == null) { Log($"No OVRSkeleton on '{go.name}' — HandSkeletonAnchor not needed."); return; }

            go.AddComponent<HandSkeletonAnchor>();
            Log($"Added HandSkeletonAnchor to '{go.name}'.");
        }

        // ── ProximityDetector ─────────────────────────────────────────────────

        /// <summary>
        /// Wires ProximityDetector.featureAssembler so the nearest-object category
        /// flows into the ONNX model's feature vector every frame.
        /// Without this, object-context features are always zero.
        /// </summary>
        private void WireProximityDetector()
        {
            var detector = FindAny<ProximityDetector>();
            if (detector == null) { Log("ProximityDetector not found — skipped."); return; }

            if (detector.featureAssembler == null)
            {
                detector.featureAssembler = FindAny<AuraXRFeatureAssembler>();
                if (detector.featureAssembler != null)
                    Log($"ProximityDetector.featureAssembler → '{detector.featureAssembler.gameObject.name}'.");
                else
                    Log("ProximityDetector: AuraXRFeatureAssembler not found — nearest-object features will be zero.");
            }
            else
                Log("ProximityDetector.featureAssembler already assigned — skipped.");
        }

        // ── AuraXRFeatureAssembler ────────────────────────────────────────────

        /// <summary>
        /// Wires AuraXRFeatureAssembler.leftControllerTransform and rightControllerTransform
        /// to the OVR hand anchors. Without these references:
        ///   • Feature vector slots [0-2] and [9-11] are always (0,0,0).
        ///   • AuraXRInferenceManager.ApplyToAnchor returns early → virtual hand never moves.
        /// This is the most common cause of "hand floating at a fixed point in the scene."
        /// </summary>
        private void WireFeatureAssemblerControllers()
        {
            var fa = FindAny<AuraXRFeatureAssembler>();
            if (fa == null) { Log("AuraXRFeatureAssembler not found — controller wiring skipped."); return; }

            if (fa.leftControllerTransform == null)
            {
                var lc = FindTrackingAnchor(true);
                if (lc != null) { fa.leftControllerTransform = lc.transform; Log($"FeatureAssembler: leftControllerTransform  → '{lc.name}'"); }
                else Debug.LogWarning("[AutoWire] FeatureAssembler: left controller anchor not found — assign leftControllerTransform manually.");
            }
            else if (IsProbablyControllerVisual(fa.leftControllerTransform))
            {
                var lc = FindTrackingAnchor(true);
                if (lc != null && lc.transform != fa.leftControllerTransform)
                {
                    WarnVisualAssigned("FeatureAssembler.leftControllerTransform", fa.leftControllerTransform);
                    fa.leftControllerTransform = lc.transform;
                    Log($"FeatureAssembler: leftControllerTransform corrected → '{lc.name}'");
                }
            }
            else
                Log("FeatureAssembler: leftControllerTransform already assigned — skipped.");

            if (fa.rightControllerTransform == null)
            {
                var rc = FindTrackingAnchor(false);
                if (rc != null) { fa.rightControllerTransform = rc.transform; Log($"FeatureAssembler: rightControllerTransform → '{rc.name}'"); }
                else Debug.LogWarning("[AutoWire] FeatureAssembler: right controller anchor not found — assign rightControllerTransform manually.");
            }
            else if (IsProbablyControllerVisual(fa.rightControllerTransform))
            {
                var rc = FindTrackingAnchor(false);
                if (rc != null && rc.transform != fa.rightControllerTransform)
                {
                    WarnVisualAssigned("FeatureAssembler.rightControllerTransform", fa.rightControllerTransform);
                    fa.rightControllerTransform = rc.transform;
                    Log($"FeatureAssembler: rightControllerTransform corrected → '{rc.name}'");
                }
            }
            else
                Log("FeatureAssembler: rightControllerTransform already assigned — skipped.");
        }

        // ── Hand Material Transparency ────────────────────────────────────────

        /// <summary>
        /// Sets every SkinnedMeshRenderer material on the hand rigs to a Transparent
        /// rendering mode so HandProximityVisibility can fade alpha at runtime.
        /// Supports URP Lit (_Surface) and Standard shader (_Mode).
        /// </summary>
        private void SetHandMaterialsTransparent()
        {
            string[] rigNames = { "LeftHandRig", "OVRCustomHandPrefab_L", "RightHandRig", "OVRCustomHandPrefab_R" };

            foreach (var rigName in rigNames)
            {
                var go = FindGoByNameAny(rigName);
                if (go == null) continue;

                foreach (var smr in go.GetComponentsInChildren<SkinnedMeshRenderer>(includeInactive: true))
                    SetMaterialTransparent(smr.material, go.name);
            }
        }

        private static void SetMaterialTransparent(Material mat, string ownerName)
        {
            if (mat == null) return;

            // URP Lit / URP Unlit
            if (mat.HasProperty("_Surface"))
            {
                if (Mathf.Approximately(mat.GetFloat("_Surface"), 1f)) return;
                mat.SetFloat("_Surface", 1f);
                mat.SetFloat("_Blend",   0f);
                mat.renderQueue = (int)UnityEngine.Rendering.RenderQueue.Transparent;
                mat.SetOverrideTag("RenderType", "Transparent");
                mat.EnableKeyword("_SURFACE_TYPE_TRANSPARENT");
                mat.DisableKeyword("_ALPHAPREMULTIPLY_ON");
                Debug.Log($"[AutoWire] URP Transparent set on '{mat.name}' (rig: {ownerName})");
                return;
            }

            // Standard shader (Built-in RP)
            if (mat.HasProperty("_Mode"))
            {
                if (Mathf.Approximately(mat.GetFloat("_Mode"), 2f)) return;
                mat.SetFloat("_Mode", 2f);
                mat.SetInt("_SrcBlend", (int)UnityEngine.Rendering.BlendMode.SrcAlpha);
                mat.SetInt("_DstBlend", (int)UnityEngine.Rendering.BlendMode.OneMinusSrcAlpha);
                mat.SetInt("_ZWrite",   0);
                mat.renderQueue = (int)UnityEngine.Rendering.RenderQueue.Transparent;
                mat.SetOverrideTag("RenderType", "Transparent");
                mat.DisableKeyword("_ALPHATEST_ON");
                mat.EnableKeyword("_ALPHABLEND_ON");
                mat.DisableKeyword("_ALPHAPREMULTIPLY_ON");
                Debug.Log($"[AutoWire] Standard Fade set on '{mat.name}' (rig: {ownerName})");
            }
        }

        // ── HandRigController ─────────────────────────────────────────────────

        /// <summary>
        /// Wires HandRigController.inferenceManager on both hand rigs so finger
        /// angles are driven every frame. Without this the fingers never move.
        /// </summary>
        private void WireHandRigControllers()
        {
            var inf = FindAny<AuraXRInferenceManager>();
            if (inf == null) { Log("HandRigControllers: AuraXRInferenceManager not found — skipped."); return; }

            foreach (var rig in Object.FindObjectsByType<HandRigController>(FindObjectsInactive.Exclude))
            {
                if (rig.inferenceManager != null) { Log($"HandRigController on '{rig.gameObject.name}' already wired — skipped."); continue; }
                rig.inferenceManager = inf;
                Log($"HandRigController '{rig.gameObject.name}' → inferenceManager wired.");
            }
        }

        // ── AuraXRInferenceManager virtual hand anchors ───────────────────────

        /// <summary>
        /// Wires AuraXRInferenceManager.virtualHandLeft and virtualHandRight to the
        /// hand rig root transforms. Without these the wrist position is never updated.
        /// </summary>
        private void WireInferenceManagerVirtualHands()
        {
            var inf = FindAny<AuraXRInferenceManager>();
            if (inf == null) { Log("VirtualHands: AuraXRInferenceManager not found — skipped."); return; }

            if (inf.virtualHandLeft == null)
            {
                var go = FindGoByNameAny("LeftHandRig", "OVRCustomHandPrefab_L");
                if (go != null) { inf.virtualHandLeft = go.transform; Log($"InfMgr: virtualHandLeft → '{go.name}'"); }
                else Debug.LogWarning("[AutoWire] VirtualHands: LeftHandRig not found — assign virtualHandLeft manually.");
            }
            else Log("InfMgr: virtualHandLeft already assigned — skipped.");

            if (inf.virtualHandRight == null)
            {
                var go = FindGoByNameAny("RightHandRig", "OVRCustomHandPrefab_R");
                if (go != null) { inf.virtualHandRight = go.transform; Log($"InfMgr: virtualHandRight → '{go.name}'"); }
                else Debug.LogWarning("[AutoWire] VirtualHands: RightHandRig not found — assign virtualHandRight manually.");
            }
            else Log("InfMgr: virtualHandRight already assigned — skipped.");
        }

        // ── IKRefinement ──────────────────────────────────────────────────────

        /// <summary>
        /// Adds IKRefinement to each HandRigController GameObject (so it can run AFTER
        /// HandRigController in LateUpdate). Wires the rigController + featureAssembler refs.
        /// </summary>
        private void WireIKRefinement()
        {
            var fa = FindAny<AuraXRFeatureAssembler>();
            foreach (var rig in Object.FindObjectsByType<HandRigController>(FindObjectsInactive.Exclude))
            {
                var ik = rig.GetComponent<IKRefinement>();
                if (ik == null)
                {
                    ik = rig.gameObject.AddComponent<IKRefinement>();
                    Log($"Added IKRefinement to '{rig.gameObject.name}'.");
                }
                if (ik.rigController == null)    ik.rigController    = rig;
                if (ik.featureAssembler == null) ik.featureAssembler = fa;
                Log($"IKRefinement on '{rig.gameObject.name}' wired (enableIK={ik.enableIK}).");
            }
        }

        // ── Object Colliders ──────────────────────────────────────────────────

        /// <summary>
        /// Walks every InteractableObject in the scene and ensures it has at least one
        /// Collider so ProximityDetector + IKRefinement can detect it. If a MeshRenderer
        /// exists but no Collider, adds a MeshCollider (convex) for low cost.
        /// </summary>
        private void EnsureObjectColliders()
        {
            int added = 0;
            foreach (var io in Object.FindObjectsByType<InteractableObject>(FindObjectsInactive.Exclude))
            {
                if (io.GetComponentInChildren<Collider>() != null) continue;
                var mf = io.GetComponentInChildren<MeshFilter>();
                if (mf == null) continue;
                var mc = mf.gameObject.AddComponent<MeshCollider>();
                mc.convex = true;
                mc.isTrigger = false;
                added++;
                Log($"Added convex MeshCollider to '{io.name}'.");
            }
            if (added > 0) Log($"EnsureObjectColliders: added {added} colliders.");
        }

        // ── Helpers ───────────────────────────────────────────────────────────

        private static T FindAny<T>() where T : Component =>
            Object.FindAnyObjectByType<T>(FindObjectsInactive.Exclude);

        private static GameObject FindGoByNameAny(params string[] names)
        {
            foreach (var n in names)
            {
                var go = GameObject.Find(n);
                if (go != null) return go;
            }
            return null;
        }

        private static GameObject FindTrackingAnchor(bool left)
        {
            return left
                ? FindGoByNameAny("LeftHandAnchor", "LeftControllerAnchor", "LeftHandTransform")
                : FindGoByNameAny("RightHandAnchor", "RightControllerAnchor", "RightHandTransform");
        }

        private static bool IsProbablyControllerVisual(Transform t)
        {
            if (t == null) return false;
            string n = t.name.ToLowerInvariant();
            return n.Contains("visual") || n.Contains("model") || n.Contains("mesh");
        }

        private static void WarnVisualAssigned(string field, Transform t)
        {
            Debug.LogWarning($"[AutoWire] {field} is assigned to '{t.name}', which looks like a controller visual. " +
                             "For 1:1 virtual-hand alignment, assign the tracking anchor instead (LeftHandAnchor / RightHandAnchor).");
        }

        private static void Log(string msg) =>
            Debug.Log($"[AutoWire] {msg}");
    }
}
