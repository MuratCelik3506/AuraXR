using UnityEngine;
using TMPro;

namespace AuraXR
{
    /// <summary>
    /// Runs before every other AuraXR script (ExecutionOrder -200) and fills in
    /// every Inspector reference that would otherwise need manual drag-and-drop.
    ///
    /// What it does:
    ///   1.  Adds PhysicsObjectSetup to GameManager if missing.
    ///   2.  Creates a SnapZone on the table for the Bottle and wires it to ScenarioKitchenTask.
    ///   3.  Creates a TaskScoreUI panel on the TaskCanvas and wires it back to the task.
    ///   4.  Adds InteractableObject to Bottle/Cup if missing (required for grab + proximity).
    ///   5.  Assigns LeftHandRig/RightHandRig on every GraspIndicator.
    ///   6.  Wires all references on HandProximityVisibility (controllers + hand rigs).
    ///   7.  Wires leftHandWrist/rightHandWrist on VirtualHandGrab.
    ///   8.  Wires virtualHandLeft/Right on AuraXRInferenceManager.
    ///   9.  Disables AuraXRObjectTracker (conflicts with ProximityDetector).
    ///   10. Sets ScenarioKitchenTask.autoStart = true so task begins on scene load.
    ///   11. Sets hand rig materials to Transparent rendering mode (URP or Standard).
    ///   12. Wires AuraXRFeatureAssembler left/rightControllerTransform (critical for anchor placement).
    ///
    /// Add this component to any GameObject in the scene (e.g. GameManager).
    /// It destroys itself after Awake() to leave no runtime overhead.
    /// </summary>
    [DefaultExecutionOrder(-200)]
    public class AuraXRAutoWire : MonoBehaviour
    {
        [Header("Snap Zone Positioning")]
        [Tooltip("Height of the snap zone above the TableTop surface (metres).")]
        public float snapZoneHeightAboveTable = 0.06f;

        [Tooltip("X offset of the bottle snap zone from the table centre.")]
        public float bottleSnapZoneX = -0.25f;

        [Header("Auto-Wire Options")]
        [Tooltip("Set ScenarioKitchenTask.autoStart = true so the task begins immediately on scene load.")]
        public bool autoStartTask = true;

        [Tooltip("Add InteractableObject to Bottle/Cup if missing (required for grab + proximity).")]
        public bool wireInteractableObjects = true;

        [Tooltip("Assign LeftHandRig/RightHandRig on every GraspIndicator component in the scene.")]
        public bool wireGraspIndicators = true;

        [Tooltip("Wire all references on HandProximityVisibility (controllers + hand rigs).")]
        public bool wireHandProximityVisibility = true;

        [Tooltip("Wire LeftHandRig/RightHandRig (wrists) and controller anchors on VirtualHandGrab.")]
        public bool wireVirtualHandGrab = true;

        [Tooltip("Wire virtualHandLeft/Right on AuraXRInferenceManager if not already assigned.")]
        public bool wireInferenceManager = true;

        [Tooltip("Disable AuraXRObjectTracker (it conflicts with ProximityDetector for object tracking).")]
        public bool disableObjectTracker = true;

        [Tooltip("Attempt to set hand rig Skinned Mesh materials to Transparent (URP or Standard shader).")]
        public bool setHandMaterialsTransparent = true;

        [Tooltip("Pre-enable the _EMISSION keyword on Bottle/Cup renderer materials so GraspIndicator yellow highlights work reliably at runtime.")]
        public bool prepareObjectEmission = true;

        [Tooltip("Add HandSkeletonAnchor to LeftHandRig / RightHandRig (requires OVRSkeleton on those objects). Keeps virtual-hand bones anchored to our controller-driven position.")]
        public bool wireHandSkeletonAnchors = true;

        // ── Awake ─────────────────────────────────────────────────────────────

        void Awake()
        {
            var task        = FindAny<ScenarioKitchenTask>();
            var canvas      = FindGoByName("TaskCanvas");
            var tableTop    = FindGoByName("TableTop");
            var gameManager = FindGoByName("GameManager");

            if (task == null)
            {
                Debug.LogError("[AutoWire] ScenarioKitchenTask not found — aborting.");
                return;
            }

            // ── Original wiring ───────────────────────────────────────────────
            WirePhysicsSetup(gameManager ?? gameObject, task);
            WireSnapZone(task, tableTop);
            WireScoreUI(task, canvas);

            // ── New wiring ────────────────────────────────────────────────────
            if (wireInteractableObjects)     WireInteractableObjects(task);
            if (prepareObjectEmission)       PrepareObjectEmission(task);
            if (wireGraspIndicators)         WireGraspIndicators();
            if (wireHandProximityVisibility) WireHandProximityVisibility();
            if (wireVirtualHandGrab)         WireVirtualHandGrab();
            if (wireInferenceManager)        WireInferenceManagerAnchors();
            if (wireHandSkeletonAnchors)     WireHandSkeletonAnchors();
            if (disableObjectTracker)        DisableObjectTracker();
            WireProximityDetector();
            WireFeatureAssemblerControllers();
            if (autoStartTask)               EnsureAutoStart(task);
            if (setHandMaterialsTransparent) SetHandMaterialsTransparent();

            Debug.Log("[AutoWire] Wiring complete. Self-destroying.");
            Destroy(this);   // zero runtime cost after startup
        }

        // ── PhysicsObjectSetup ────────────────────────────────────────────────

        private void WirePhysicsSetup(GameObject host, ScenarioKitchenTask task)
        {
            var existing = FindAny<PhysicsObjectSetup>();
            if (existing != null) { Log("PhysicsObjectSetup already present — skipped."); return; }

            var setup = host.AddComponent<PhysicsObjectSetup>();

            // HOT3D-calibrated masses:
            //   bottle_mustard (ID 13) ~0.5 kg  — sauce bottle, full
            //   mug_patterned  (ID 8)  ~0.35 kg — ceramic mug
            if (task.bottle != null && task.cup != null)
            {
                setup.massOverrides = new[]
                {
                    new PhysicsObjectSetup.MassOverride { obj = task.bottle, mass = 0.5f  },
                    new PhysicsObjectSetup.MassOverride { obj = task.cup,    mass = 0.35f }
                };
            }

            Log("Added PhysicsObjectSetup to " + host.name);
        }

        // ── SnapZone ──────────────────────────────────────────────────────────

        private void WireSnapZone(ScenarioKitchenTask task, GameObject tableTop)
        {
            if (task.bottleSnapZone != null) { Log("bottleSnapZone already assigned — skipped."); return; }
            if (task.bottle == null)         { Debug.LogWarning("[AutoWire] task.bottle is null — cannot create snap zone."); return; }

            float surfaceY = DetermineSurfaceY(tableTop, task.bottle.transform.position.y);

            var zoneGO = new GameObject("SnapZone_Bottle");

            if (tableTop != null)
                zoneGO.transform.SetParent(tableTop.transform, worldPositionStays: true);

            zoneGO.transform.position = new Vector3(
                task.bottle.transform.position.x + bottleSnapZoneX,
                surfaceY,
                task.bottle.transform.position.z);

            var col  = zoneGO.AddComponent<BoxCollider>();
            col.size      = new Vector3(0.18f, 0.14f, 0.18f);
            col.isTrigger = true;

            var zone = zoneGO.AddComponent<SnapZone>();
            zone.acceptedObject = task.bottle;
            zone.snapOffset     = new Vector3(0f, snapZoneHeightAboveTable, 0f);

            task.bottleSnapZone = zone;

            Log($"Created SnapZone_Bottle at {zoneGO.transform.position} and wired to ScenarioKitchenTask.");
        }

        private static float DetermineSurfaceY(GameObject tableTop, float fallback)
        {
            if (tableTop == null) return fallback;

            var r = tableTop.GetComponent<Renderer>();
            if (r != null) return r.bounds.max.y;

            var c = tableTop.GetComponent<Collider>();
            if (c != null) return c.bounds.max.y;

            return tableTop.transform.position.y + 0.02f;
        }

        // ── TaskScoreUI ───────────────────────────────────────────────────────

        private void WireScoreUI(ScenarioKitchenTask task, GameObject canvasGO)
        {
            if (task.scoreUI != null) { Log("scoreUI already assigned — skipped."); return; }

            var existing = FindAny<TaskScoreUI>();
            if (existing != null)
            {
                task.scoreUI     = existing;
                existing.task    = task;
                Log("Found existing TaskScoreUI and wired to task.");
                return;
            }

            if (canvasGO == null)
            {
                Debug.LogWarning("[AutoWire] TaskCanvas not found — TaskScoreUI will not be created.");
                return;
            }

            var panel   = BuildScorePanel(canvasGO);
            var scoreUI = panel.GetComponent<TaskScoreUI>();  // already added inside BuildScorePanel
            scoreUI.task = task;
            task.scoreUI = scoreUI;

            Log("Created TaskScoreUI panel inside TaskCanvas and wired to ScenarioKitchenTask.");
        }

        private static GameObject BuildScorePanel(GameObject canvasGO)
        {
            var panel = new GameObject("ScorePanel");
            panel.transform.SetParent(canvasGO.transform, worldPositionStays: false);

            var rect = panel.AddComponent<RectTransform>();
            rect.anchoredPosition = new Vector2(160f, 0f);
            rect.sizeDelta        = new Vector2(300f, 360f);

            string[] names = { "Step1", "Step2", "Step3", "Step4", "Timer", "Stars", "Result" };
            float[]  yPos  = {  130f,    90f,    50f,   10f,  -50f, -100f,  -140f };

            TextMeshProUGUI[] texts = new TextMeshProUGUI[names.Length];
            for (int i = 0; i < names.Length; i++)
                texts[i] = MakeTMP(names[i], panel.transform, yPos[i]);

            var ui = panel.AddComponent<TaskScoreUI>();
            ui.step1Text  = texts[0];
            ui.step2Text  = texts[1];
            ui.step3Text  = texts[2];
            ui.step4Text  = texts[3];
            ui.timerText  = texts[4];
            ui.starsText  = texts[5];
            ui.resultText = texts[6];

            return panel;
        }

        private static TextMeshProUGUI MakeTMP(string name, Transform parent, float yPos)
        {
            var go   = new GameObject(name);
            go.transform.SetParent(parent, worldPositionStays: false);

            var rect = go.AddComponent<RectTransform>();
            rect.anchoredPosition = new Vector2(0f, yPos);
            rect.sizeDelta        = new Vector2(280f, 36f);

            var tmp  = go.AddComponent<TextMeshProUGUI>();
            tmp.fontSize  = 18f;
            tmp.color     = Color.white;
            tmp.alignment = TextAlignmentOptions.Left;
            tmp.text      = name;
            return tmp;
        }

        // ── InteractableObject ────────────────────────────────────────────────

        /// <summary>
        /// Adds InteractableObject to Bottle/Cup GameObjects if missing, and ensures
        /// correct categoryIds (bottle=1, cup=3 per project convention).
        /// Also backfills task.bottle / task.cup references if they are null.
        /// </summary>
        private void WireInteractableObjects(ScenarioKitchenTask task)
        {
            // ── Bottle ──
            if (task.bottle == null)
            {
                var go = FindGoByNameAny("Bottle", "bottle", "Bottle_Mustard", "MustardBottle", "bottle_mustard");
                if (go != null)
                {
                    var io = go.GetComponent<InteractableObject>() ?? go.AddComponent<InteractableObject>();
                    if (io.categoryId == 0) { io.categoryId = 1; io.categoryName = "bottle_mustard"; }
                    task.bottle = io;
                    Log($"InteractableObject(id={io.categoryId}) wired to task.bottle → '{go.name}'");
                }
                else
                    Debug.LogWarning("[AutoWire] Bottle GameObject not found by name. Add InteractableObject manually.");
            }
            else if (task.bottle.categoryId == 0)
            {
                task.bottle.categoryId   = 1;
                task.bottle.categoryName = "bottle_mustard";
                Log("Fixed Bottle categoryId → 1");
            }

            // ── Cup ──
            if (task.cup == null)
            {
                var go = FindGoByNameAny("Cup", "cup", "Mug", "mug", "Cup_Mug", "mug_patterned");
                if (go != null)
                {
                    var io = go.GetComponent<InteractableObject>() ?? go.AddComponent<InteractableObject>();
                    if (io.categoryId == 0) { io.categoryId = 3; io.categoryName = "mug_patterned"; }
                    task.cup = io;
                    Log($"InteractableObject(id={io.categoryId}) wired to task.cup → '{go.name}'");
                }
                else
                    Debug.LogWarning("[AutoWire] Cup/Mug GameObject not found by name. Add InteractableObject manually.");
            }
            else if (task.cup.categoryId == 0)
            {
                task.cup.categoryId   = 3;
                task.cup.categoryName = "mug_patterned";
                Log("Fixed Cup categoryId → 3");
            }
        }

        // ── GraspIndicator ────────────────────────────────────────────────────

        /// <summary>
        /// Finds all GraspIndicator components in the scene and assigns LeftHandRig /
        /// RightHandRig transforms so the yellow highlight follows the virtual hand.
        /// </summary>
        private void WireGraspIndicators()
        {
            var leftRig  = FindGoByNameAny("LeftHandRig",  "OVRCustomHandPrefab_L");
            var rightRig = FindGoByNameAny("RightHandRig", "OVRCustomHandPrefab_R");

            if (leftRig  == null) { Debug.LogWarning("[AutoWire] LeftHandRig not found  — GraspIndicator wiring skipped."); return; }
            if (rightRig == null) { Debug.LogWarning("[AutoWire] RightHandRig not found — GraspIndicator wiring skipped."); return; }

            var indicators = Object.FindObjectsByType<GraspIndicator>(FindObjectsInactive.Exclude);
            if (indicators.Length == 0) { Log("No GraspIndicators found in scene."); return; }

            foreach (var gi in indicators)
            {
                bool changed = false;
                if (gi.leftHandRig  == null) { gi.leftHandRig  = leftRig.transform;  changed = true; }
                if (gi.rightHandRig == null) { gi.rightHandRig = rightRig.transform; changed = true; }
                if (changed) Log($"GraspIndicator on '{gi.gameObject.name}': wired hand rigs.");
            }
        }

        // ── HandProximityVisibility ───────────────────────────────────────────

        /// <summary>
        /// Fills in all four references on HandProximityVisibility:
        ///   leftController / rightController — OVR controller anchor transforms
        ///   leftHandRig    / rightHandRig    — virtual hand rig GameObjects
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
        /// Wires VirtualHandGrab's wrist transforms (hand rigs) and optional
        /// physical-controller transforms used for grab proximity detection.
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
        /// controller transforms and nearest-object context.
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
        /// Adds HandSkeletonAnchor to LeftHandRig and RightHandRig if they have an
        /// OVRSkeleton component but no HandSkeletonAnchor yet.
        ///
        /// HandSkeletonAnchor re-roots all OVR skeleton bones each LateUpdate so the
        /// wrist sits exactly at the Transform's position (set by AuraXRInferenceManager),
        /// overriding OVRSkeleton's own world-space bone placement.
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
                Debug.LogWarning($"[AutoWire] HandSkeletonAnchor: GameObject '{names[0]}' not found — skipped.");
                return;
            }

            if (go.GetComponent<HandSkeletonAnchor>() != null)
            {
                Log($"HandSkeletonAnchor already on '{go.name}' — skipped.");
                return;
            }

            // Only useful when OVRSkeleton is present
            if (go.GetComponent<OVRSkeleton>() == null)
            {
                Log($"No OVRSkeleton on '{go.name}' — HandSkeletonAnchor not needed.");
                return;
            }

            go.AddComponent<HandSkeletonAnchor>();
            Log($"Added HandSkeletonAnchor to '{go.name}'.");
        }

        // ── AuraXRObjectTracker ───────────────────────────────────────────────

        /// <summary>
        /// Disables AuraXRObjectTracker to prevent conflicts with ProximityDetector.
        /// Both systems write to the feature assembler's "nearest object" slot; only
        /// ProximityDetector should be active.
        /// </summary>
        private void DisableObjectTracker()
        {
            var tracker = FindAny<AuraXRObjectTracker>();
            if (tracker == null) { Log("AuraXRObjectTracker not found — nothing to disable."); return; }
            if (!tracker.enabled)  { Log("AuraXRObjectTracker already disabled — skipped."); return; }
            tracker.enabled = false;
            Log("Disabled AuraXRObjectTracker (ProximityDetector takes priority).");
        }

        // ── ProximityDetector wiring ──────────────────────────────────────────

        /// <summary>
        /// Wires ProximityDetector.featureAssembler so nearest-object context
        /// flows into the model's feature vector every frame.
        /// Without this, indices [18-31] are always zero and the model
        /// cannot react to object proximity.
        /// </summary>
        private void WireProximityDetector()
        {
            var detector  = FindAny<ProximityDetector>();
            if (detector == null) { Log("ProximityDetector not found in scene — skipped."); return; }

            if (detector.featureAssembler == null)
            {
                detector.featureAssembler = FindAny<AuraXRFeatureAssembler>();
                if (detector.featureAssembler != null)
                    Log($"ProximityDetector.featureAssembler → '{detector.featureAssembler.gameObject.name}'.");
                else
                    Log("ProximityDetector: AuraXRFeatureAssembler not found — nearest-object features will be zero.");
            }
            else
            {
                Log("ProximityDetector.featureAssembler already assigned — skipped.");
            }
        }

        // ── AuraXRFeatureAssembler controller transforms ──────────────────────

        /// <summary>
        /// Wires AuraXRFeatureAssembler.leftControllerTransform and rightControllerTransform
        /// to the OVR hand anchors.  Without these two references:
        ///   • Feature vector slots [0-2] and [9-11] are always (0,0,0) — wrong controller position.
        ///   • AuraXRInferenceManager.ApplyToAnchor returns early (controller == null),
        ///     so the virtual hand anchor NEVER moves from its default scene position.
        /// This is the most common cause of "hand floating at a fixed point" in the scene.
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
                if (lc != null)
                {
                    fa.leftControllerTransform = lc.transform;
                    Log($"FeatureAssembler: leftControllerTransform  → '{lc.name}'");
                }
                else
                    Debug.LogWarning("[AutoWire] FeatureAssembler: left controller anchor not found. " +
                                     "Assign leftControllerTransform manually (e.g. LeftHandAnchor).");
            }
            else
                Log("FeatureAssembler: leftControllerTransform already assigned — skipped.");

            if (fa.rightControllerTransform == null)
            {
                var rc = FindGoByNameAny("RightHandAnchor", "RightControllerAnchor",
                                          "OVRRightControllerVisual", "RightHandTransform",
                                          "RightControllerVisual");
                if (rc != null)
                {
                    fa.rightControllerTransform = rc.transform;
                    Log($"FeatureAssembler: rightControllerTransform → '{rc.name}'");
                }
                else
                    Debug.LogWarning("[AutoWire] FeatureAssembler: right controller anchor not found. " +
                                     "Assign rightControllerTransform manually (e.g. RightHandAnchor).");
            }
            else
                Log("FeatureAssembler: rightControllerTransform already assigned — skipped.");
        }

        // ── AutoStart ─────────────────────────────────────────────────────────

        /// <summary>
        /// Marks the task as auto-starting so ScenarioKitchenTask.Start() immediately
        /// calls StartTask() — no button press needed for the user study.
        /// </summary>
        private void EnsureAutoStart(ScenarioKitchenTask task)
        {
            if (task.autoStart) { Log("autoStart already true — skipped."); return; }
            task.autoStart = true;
            Log("Set ScenarioKitchenTask.autoStart = true");
        }

        // ── Object Emission ───────────────────────────────────────────────────

        /// <summary>
        /// Pre-enables the _EMISSION keyword on Bottle and Cup renderer materials
        /// (creates a per-instance material copy via .material so the shared asset
        /// is never modified). GraspIndicator will then be able to toggle emission
        /// at runtime without needing the keyword to be set in the Editor.
        ///
        /// Note: GraspIndicator.Start() calls _mat.DisableKeyword("_EMISSION") to
        /// ensure objects start un-highlighted, so this is safe to call first.
        /// </summary>
        private void PrepareObjectEmission(ScenarioKitchenTask task)
        {
            EnableEmissionOnObject(task.bottle?.gameObject, "Bottle");
            EnableEmissionOnObject(task.cup?.gameObject,    "Cup");
        }

        private static void EnableEmissionOnObject(GameObject go, string label)
        {
            if (go == null) { Debug.LogWarning($"[AutoWire] PrepareEmission: {label} GameObject is null — skipped."); return; }

            var renderer = go.GetComponent<Renderer>();
            if (renderer == null) { Debug.LogWarning($"[AutoWire] PrepareEmission: No Renderer on '{go.name}' — skipped."); return; }

            // .material creates a per-instance copy — shared asset is NOT modified
            var mat = renderer.material;

            // Enable keyword so GraspIndicator's EnableKeyword("_EMISSION") call works
            mat.EnableKeyword("_EMISSION");

            // Start with black (invisible) emission; GraspIndicator.LateUpdate sets the real colour
            mat.SetColor(Shader.PropertyToID("_EmissionColor"), Color.black);

            Log($"Emission keyword enabled (black/off) on '{go.name}' → ready for GraspIndicator.");
        }

        // ── Hand Material Transparency ────────────────────────────────────────

        /// <summary>
        /// Sets every SkinnedMeshRenderer material on the hand rigs to a Transparent
        /// rendering mode so HandProximityVisibility can fade alpha correctly.
        /// Supports URP Lit (_Surface) and Standard shader (_Mode).
        /// </summary>
        private void SetHandMaterialsTransparent()
        {
            string[] rigNames =
            {
                "LeftHandRig",  "OVRCustomHandPrefab_L",
                "RightHandRig", "OVRCustomHandPrefab_R"
            };

            foreach (var rigName in rigNames)
            {
                var go = FindGoByNameAny(rigName);
                if (go == null) continue;

                foreach (var smr in go.GetComponentsInChildren<SkinnedMeshRenderer>(includeInactive: true))
                    SetMaterialTransparent(smr.material, go.name);
            }
        }

        /// <summary>
        /// Makes a single material transparent at runtime.
        /// URP: sets _Surface=1 (Transparent) + keywords.
        /// Standard BiRP: sets _Mode=2 (Fade) + blend state.
        /// </summary>
        private static void SetMaterialTransparent(Material mat, string ownerName)
        {
            if (mat == null) return;

            // ── URP Lit / URP Unlit ───────────────────────────────────────────
            if (mat.HasProperty("_Surface"))
            {
                if (Mathf.Approximately(mat.GetFloat("_Surface"), 1f)) return; // already transparent
                mat.SetFloat("_Surface", 1f);   // 0 = Opaque, 1 = Transparent
                mat.SetFloat("_Blend",   0f);   // 0 = Alpha blend
                mat.renderQueue = (int)UnityEngine.Rendering.RenderQueue.Transparent;
                mat.SetOverrideTag("RenderType", "Transparent");
                mat.EnableKeyword("_SURFACE_TYPE_TRANSPARENT");
                mat.DisableKeyword("_ALPHAPREMULTIPLY_ON");
                Debug.Log($"[AutoWire] URP Transparent set on '{mat.name}' (rig: {ownerName})");
                return;
            }

            // ── Standard shader (Built-in RP) ─────────────────────────────────
            if (mat.HasProperty("_Mode"))
            {
                if (Mathf.Approximately(mat.GetFloat("_Mode"), 2f)) return; // already Fade
                mat.SetFloat("_Mode", 2f);  // 0=Opaque 1=Cutout 2=Fade 3=Transparent
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

        /// <summary>Finds a GameObject by exact name. Logs a warning if not found.</summary>
        private static GameObject FindGoByName(string n)
        {
            var go = GameObject.Find(n);
            if (go == null) Debug.LogWarning($"[AutoWire] '{n}' not found in scene.");
            return go;
        }

        /// <summary>
        /// Tries a list of names in order and returns the first match, or null.
        /// No warning is logged — caller decides whether to warn.
        /// </summary>
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
