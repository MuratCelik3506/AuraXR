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
                var lc = FindGoByNameAny("LeftHandAnchor", "LeftControllerAnchor",
                                         "OVRLeftControllerVisual", "LeftHandTransform",
                                         "LeftControllerVisual");
                if (lc != null) { hpv.leftController = lc.transform; Log("HPV: leftController → " + lc.name); }
                else Debug.LogWarning("[AutoWire] HPV: left controller anchor not found — try 'LeftHandAnchor'.");
            }

            if (hpv.rightController == null)
            {
                var rc = FindGoByNameAny("RightHandAnchor", "RightControllerAnchor",
                                          "OVRRightControllerVisual", "RightHandTransform",
                                          "RightControllerVisual");
                if (rc != null) { hpv.rightController = rc.transform; Log("HPV: rightController → " + rc.name); }
                else Debug.LogWarning("[AutoWire] HPV: right controller anchor not found — try 'RightHandAnchor'.");
            }
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
                var lc = FindGoByNameAny("LeftHandAnchor", "LeftControllerAnchor", "OVRLeftControllerVisual");
                if (lc != null) { grab.leftController = lc.transform; Log("VHGrab: leftController → " + lc.name); }
            }

            if (grab.rightController == null)
            {
                var rc = FindGoByNameAny("RightHandAnchor", "RightControllerAnchor", "OVRRightControllerVisual");
                if (rc != null) { grab.rightController = rc.transform; Log("VHGrab: rightController → " + rc.name); }
            }
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
                var lc = FindGoByNameAny("LeftHandAnchor", "LeftControllerAnchor",
                                         "OVRLeftControllerVisual", "LeftHandTransform",
                                         "LeftControllerVisual");
                if (lc != null) { fa.leftControllerTransform = lc.transform; Log($"FeatureAssembler: leftControllerTransform  → '{lc.name}'"); }
                else Debug.LogWarning("[AutoWire] FeatureAssembler: left controller anchor not found — assign leftControllerTransform manually.");
            }
            else
                Log("FeatureAssembler: leftControllerTransform already assigned — skipped.");

            if (fa.rightControllerTransform == null)
            {
                var rc = FindGoByNameAny("RightHandAnchor", "RightControllerAnchor",
                                          "OVRRightControllerVisual", "RightHandTransform",
                                          "RightControllerVisual");
                if (rc != null) { fa.rightControllerTransform = rc.transform; Log($"FeatureAssembler: rightControllerTransform → '{rc.name}'"); }
                else Debug.LogWarning("[AutoWire] FeatureAssembler: right controller anchor not found — assign rightControllerTransform manually.");
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

        private static void Log(string msg) =>
            Debug.Log($"[AutoWire] {msg}");
    }
}
