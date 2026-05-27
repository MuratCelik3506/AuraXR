using UnityEngine;

namespace AuraXR
{
    /// <summary>
    /// Applies predicted hand pose (from AuraXRInferenceManager) to MANO skeleton.
    /// Attach this to LeftHandRig or RightHandRig.
    /// Uses fixed neutral beta (all zeros) — no per-prediction bone scaling needed.
    /// </summary>
    [DefaultExecutionOrder(500)]   // Run AFTER OVRSkeleton (order -80) and any OVR LateUpdate, but BEFORE HandSkeletonAnchor (1000)
    public class HandRigController : MonoBehaviour
    {
        [Header("References")]
        [Tooltip("The AuraXRInferenceManager instance (usually on GameManager)")]
        public AuraXRInferenceManager inferenceManager;

        [Tooltip("True if this is the left hand, false for right hand")]
        public bool isLeftHand = true;

        [Header("Hand Skeleton Bones")]
        [Tooltip("All 15 finger joint transforms in MANO order: Thumb(0-2), Index(3-5), Middle(6-8), Ring(9-11), Pinky(12-14)")]
        public Transform[] fingerJoints;

        [Header("Grab Pose Override")]
        [Tooltip("Drag the VirtualHandGrab component here (usually on GameManager)")]
        public VirtualHandGrab grabSystem;

        [Tooltip("Closed-fist angles in radians — MANO order: Thumb(MCP,PIP,DIP) Index(MCP,PIP,DIP) Middle Ring Pinky")]
        public float[] grabPoseAnglesRad = new float[15]
        {
            // Thumb
            0.30f, 0.40f, 0.30f,
            // Index
            1.30f, 1.20f, 0.90f,
            // Middle
            1.40f, 1.30f, 0.90f,
            // Ring
            1.40f, 1.30f, 0.90f,
            // Pinky
            1.30f, 1.20f, 0.80f,
        };

        [Tooltip("How fast the hand blends into/out of the grab pose (units/sec)")]
        [Range(1f, 20f)]
        public float grabBlendSpeed = 10f;

        [Header("HOT3D ↔ Rig Sign Convention")]
        [Tooltip("Per-joint sign multiplier (+1 or -1). With geometric-IK KP model, all angles are positive for " +
                 "flexion — set all to +1. Only needed if switching back to MANO model where some joints encode " +
                 "flexion as a negative angle. Order: Thumb(0-2), Index(3-5), Middle(6-8), Ring(9-11), Pinky(12-14).")]
        public float[] jointSignMultipliers = new float[15]
        {
            // Thumb MCP, PIP, DIP
            1f, 1f, 1f,
            // Index MCP, PIP, DIP
            1f, 1f, 1f,
            // Middle MCP, PIP, DIP
            1f, 1f, 1f,
            // Ring MCP, PIP, DIP
            1f, 1f, 1f,
            // Pinky MCP, PIP, DIP
            1f, 1f, 1f,
        };

        [Header("Smoothing")]
        [Tooltip("Per-frame smoothing applied to inference angles. 0 = raw, 0.8 = heavy smoothing. Keep low — AuraXRHandRenderer adds more smoothing if also active.")]
        [Range(0f, 0.95f)]
        public float smoothing = 0.0f;

        [Header("Angle Clamping")]
        [Tooltip("Minimum angle for all joints after sign correction (degrees). Allows slight hyperextension. Default −15.")]
        [Range(-90f, 0f)]
        public float jointMinAngleDeg = -15f;

        [Tooltip("Maximum flexion for all joints after sign correction (degrees). " +
                 "130° allows full thumb curl at grip=1 (Th.PIP peaks at ~123°). " +
                 "With in-distribution Quest3 inputs, no other joint exceeds ~85°.")]
        [Range(0f, 180f)]
        public float jointMaxAngleDeg = 130f;

        [Header("Bone Flexion Axis")]
        [Tooltip("Local-space axis around which a positive angle produces FLEXION (curling toward palm). " +
                 "OVR hand bones: long axis = local +X, dorsal = local +Y → flexion axis = local -Z = Vector3.back = (0,0,-1). " +
                 "If fingers twist instead of curl, try (0,0,1). If they spread, try (0,1,0).")]
        public Vector3 bendAxis = new Vector3(0f, 0f, -1f);   // Vector3.back

        [Header("Debug")]
        [Tooltip("Force all finger joints to 0.5 rad (~28°) regardless of inference. " +
                 "Enable temporarily to confirm bones respond to angle changes.")]
        public bool debugForceTestPose = false;

        [Tooltip("Multiply all finger angles — crank this up (e.g. 5-10) in Play mode to confirm bones are actually moving")]
        [Range(0.1f, 20f)]
        public float debugAngleMultiplier = 1f;

        [Header("Verbose Logging (works on Quest device too)")]
        [Tooltip("Enable per-joint angle logs every 60 frames. Works in Editor AND on device (adb logcat).")]
        public bool verboseLog = false;

        [Header("Visual")]
        [Tooltip("Optional: Renderer to toggle visibility")]
        public Renderer handRenderer;

        [Tooltip("If true, force-enables all SkinnedMeshRenderers on this GameObject in LateUpdate. " +
                 "Use in Editor / debug to make the OVR hand visible even without Quest tracking.")]
        public bool forceVisible = false;

        // Rest-pose rotations captured before OVR drives the bones
        private Quaternion[] _restPose;
        private float[]      _smoothedAngles;
        private float        _grabBlend = 0f;

        // Set to true until fingerJoints are populated (either manually or from OVRSkeleton)
        private bool _needsBoneInit = true;

        private System.IO.StreamWriter _rigLog;
        private string Hand => isLeftHand ? "L" : "R";

        private void RLog(string msg)
        {
            string line = $"[{Time.frameCount:D6}|{Hand}] {msg}";
            _rigLog?.WriteLine(line);
            Debug.Log($"[HandRig {Hand}] {msg}");
        }

        private void Awake()
        {
            string logFolder = System.IO.Path.Combine(Application.persistentDataPath, "Logs");
            System.IO.Directory.CreateDirectory(logFolder);
            string path = System.IO.Path.Combine(logFolder,
                $"handrig_{(isLeftHand ? "L" : "R")}_{System.DateTime.Now:yyyy_MM_dd_HH_mm_ss}.txt");
            _rigLog = new System.IO.StreamWriter(path, append: false) { AutoFlush = true };
            RLog($"=== HandRigController started. debugForceTestPose={debugForceTestPose} ===");
            Debug.Log($"[HandRig {Hand}] Rig log → {path}");

            if (inferenceManager == null)
                inferenceManager = FindAnyObjectByType<AuraXRInferenceManager>();
            RLog($"inferenceManager={(inferenceManager != null ? inferenceManager.name : "NULL")}");

            if (grabSystem == null)
                grabSystem = FindAnyObjectByType<VirtualHandGrab>();
        }

        private void OnDestroy() { _rigLog?.Close(); }

        private void Start()
        {
            // If joints were assigned manually in the Inspector, capture rest-pose immediately.
            int preAssigned = 0;
            if (fingerJoints != null)
                foreach (var j in fingerJoints) if (j != null) preAssigned++;

            RLog($"Start: fingerJoints.Length={(fingerJoints?.Length ?? -1)}  preAssigned={preAssigned}");

            if (preAssigned > 0)
            {
                CaptureRestPose();
                _needsBoneInit = false;
                // Log first 3 joint names and their rest poses
                for (int i = 0; i < Mathf.Min(3, fingerJoints.Length); i++)
                    if (fingerJoints[i] != null)
                        RLog($"  joint[{i}] name={fingerJoints[i].name}  parent={fingerJoints[i].parent?.name ?? "NULL"}  restEuler={fingerJoints[i].localEulerAngles:F1}");
            }
            // Otherwise _needsBoneInit stays true and TryInitBonesFromOVRSkeleton() is called
            // each LateUpdate until OVRSkeleton is ready.
        }

        private void LateUpdate()
        {
            // Try to auto-populate fingerJoints from OVRSkeleton each frame until found
            if (_needsBoneInit) TryInitBonesFromOVRSkeleton();

            if (inferenceManager == null) return;

            var pose = isLeftHand ? inferenceManager.LeftHand : inferenceManager.RightHand;

            // ── Debug: force a half-curl test pose to confirm bones actually move ──
            if (debugForceTestPose)
            {
                pose = new HandPose();
                for (int i = 0; i < 15; i++) pose.ManoJointAngles[i] = 0.5f;
                if (Time.frameCount % 120 == 0)
                    RLog($"debugForceTestPose ACTIVE (28°). inferenceManager={inferenceManager != null}  fingerJoints={(fingerJoints?.Length ?? -1)}  needsBoneInit={_needsBoneInit}");
            }

            // NOTE: wrist position/rotation is set by AuraXRInferenceManager.ApplyToAnchor.
            // Do NOT use pose.WristPosition here — it is in HOT3D world-space, not Quest space.

            // ── Grab-blend target ─────────────────────────────────────────────────
            float grabTarget = 0f;
            if (grabSystem != null)
            {
                bool isGrabbing = isLeftHand ? grabSystem.IsGrabbingLeft : grabSystem.IsGrabbingRight;
                if (isGrabbing)
                {
                    // Button-pressed grab → full close
                    grabTarget = 1f;
                }
                else if (grabSystem.proximityAutoClose)
                {
                    // Proximity-driven smooth close — no button needed
                    grabTarget = isLeftHand ? grabSystem.ProximityBlendLeft : grabSystem.ProximityBlendRight;
                }
            }
            _grabBlend = Mathf.MoveTowards(_grabBlend, grabTarget,
                                           grabBlendSpeed * Time.deltaTime);

            if (pose.ManoJointAngles != null && fingerJoints != null)
            {
                int applied = 0;
                int count   = Mathf.Min(pose.ManoJointAngles.Length, fingerJoints.Length);
                if (_smoothedAngles == null || _smoothedAngles.Length < count)
                    _smoothedAngles = new float[count];
                for (int i = 0; i < count; i++)
                {
                    if (fingerJoints[i] == null) continue;

                    // Apply per-joint sign convention (HOT3D vs. rig axis mismatch)
                    float sign = (jointSignMultipliers != null && i < jointSignMultipliers.Length)
                                  ? jointSignMultipliers[i] : 1f;

                    // Blend inference angle with grab-pose angle
                    float inferenceRad = pose.ManoJointAngles[i] * sign;
                    float grabRad      = (grabPoseAnglesRad != null && i < grabPoseAnglesRad.Length)
                                         ? grabPoseAnglesRad[i] : inferenceRad;
                    float targetRad    = Mathf.Lerp(inferenceRad, grabRad, _grabBlend);

                    // Per-frame smoothing
                    _smoothedAngles[i] = Mathf.Lerp(targetRad, _smoothedAngles[i], smoothing);

                    float deg = Mathf.Clamp(
                        _smoothedAngles[i] * Mathf.Rad2Deg * debugAngleMultiplier,
                        jointMinAngleDeg, jointMaxAngleDeg);
                    Quaternion rest = (_restPose != null && i < _restPose.Length)
                                       ? _restPose[i] : Quaternion.identity;
                    // Correct formula: flex around the bone's OWN local axis.
                    // rest * bendAxis transforms bendAxis from bone-local → parent frame.
                    // AngleAxis(...) * rest then adds the flexion on top of the rest pose.
                    Vector3 axis = rest * bendAxis;
                    fingerJoints[i].localRotation = Quaternion.AngleAxis(deg, axis) * rest;
                    applied++;
                }

                // ── FILE LOG: bone state after applying rotations ─────────────────
                if (Time.frameCount % 60 == 0)
                {
                    string[] jnames = { "Th.MCP","Th.PIP","Th.DIP","Idx.MCP","Idx.PIP","Idx.DIP","Mid.MCP","Mid.PIP","Mid.DIP","Rng.MCP","Rng.PIP","Rng.DIP","Pnk.MCP","Pnk.PIP","Pnk.DIP" };
                    var sb = new System.Text.StringBuilder();
                    sb.AppendLine($"applied={applied}/15  grabBlend={_grabBlend:F2}  needsBoneInit={_needsBoneInit}  forceTestPose={debugForceTestPose}");
                    for (int i = 0; i < count; i++)
                    {
                        bool  hasJ       = fingerJoints[i] != null;
                        string bname     = hasJ ? fingerJoints[i].name : "NULL";
                        string parent    = hasJ ? (fingerJoints[i].parent?.name ?? "NO_PARENT") : "N/A";
                        string euler     = hasJ ? fingerJoints[i].localEulerAngles.ToString("F1") : "N/A";
                        float targetDeg  = Mathf.Clamp(pose.ManoJointAngles[i] * Mathf.Rad2Deg * debugAngleMultiplier, jointMinAngleDeg, jointMaxAngleDeg);
                        sb.Append($"  [{i}]{jnames[i]} bone={bname} parent={parent} targetDeg={targetDeg:F1} afterEuler={euler}\n");
                    }
                    RLog(sb.ToString());

                    // ── BONE WORLD POSITIONS: model açısı vs eklem dünya konumu ───
                    var wpos = new System.Text.StringBuilder();
                    wpos.AppendLine($"[BONE_WORLD] hand={(isLeftHand?"L":"R")} frame={Time.frameCount}");
                    wpos.AppendLine("  Joint      modelDeg    worldX    worldY    worldZ  localEulerX  localEulerY  localEulerZ");
                    for (int i = 0; i < count; i++)
                    {
                        float modelDeg = Mathf.Clamp(
                            pose.ManoJointAngles[i] * Mathf.Rad2Deg * debugAngleMultiplier,
                            jointMinAngleDeg, jointMaxAngleDeg);
                        if (fingerJoints[i] == null)
                        {
                            wpos.AppendLine($"  {jnames[i],-10} {modelDeg,9:F2} NULL");
                            continue;
                        }
                        Vector3    wp = fingerJoints[i].position;
                        Vector3    le = fingerJoints[i].localEulerAngles;
                        wpos.AppendLine($"  {jnames[i],-10} {modelDeg,9:F2} {wp.x,9:F4} {wp.y,9:F4} {wp.z,9:F4} {le.x,12:F2} {le.y,12:F2} {le.z,12:F2}");
                    }
                    RLog(wpos.ToString());
                }

                // ── VERBOSE LOG (works on Quest device too) ──────────────────────
                if (verboseLog && Time.frameCount % 60 == 0)
                {
                    string hand = isLeftHand ? "L" : "R";
                    int nullCount = 0;
                    for (int i = 0; i < fingerJoints.Length; i++)
                        if (fingerJoints[i] == null) nullCount++;

                    var sb = new System.Text.StringBuilder();
                    sb.AppendLine($"[HandRig|VERBOSE {hand}] frame={Time.frameCount}  joints={fingerJoints.Length}  nullJoints={nullCount}  applied={applied}  grabBlend={_grabBlend:F2}  needsBoneInit={_needsBoneInit}");
                    string[] jnames2 = { "Th.MCP","Th.PIP","Th.DIP","Idx.MCP","Idx.PIP","Idx.DIP","Mid.MCP","Mid.PIP","Mid.DIP","Rng.MCP","Rng.PIP","Rng.DIP","Pnk.MCP","Pnk.PIP","Pnk.DIP" };
                    for (int i = 0; i < count; i++)
                    {
                        float s    = (jointSignMultipliers != null && i < jointSignMultipliers.Length) ? jointSignMultipliers[i] : 1f;
                        float degV = pose.ManoJointAngles[i] * s * Mathf.Rad2Deg;
                        bool  hasJ = fingerJoints[i] != null;
                        string bname      = hasJ ? fingerJoints[i].name : "NULL";
                        string euler      = hasJ ? fingerJoints[i].localEulerAngles.ToString("F1") : "N/A";
                        string parentName = (hasJ && fingerJoints[i].parent != null) ? fingerJoints[i].parent.name : "NO_PARENT";
                        sb.AppendLine($"  [{i}]{jnames2[i]}  bone={bname}  parent={parentName}  signed_deg={degV:F1}  localEuler={euler}");
                    }
                    sb.AppendLine("  KEY: if all parents are same root → FLAT hierarchy (finger curling won't work via localRotation!)");
                    Debug.Log(sb.ToString());
                }

#if UNITY_EDITOR
                if (Time.frameCount % 30 == 0)
                {
                    string hand = isLeftHand ? "L" : "R";
                    int nullCount = 0;
                    for (int i = 0; i < fingerJoints.Length; i++)
                        if (fingerJoints[i] == null) nullCount++;

                    // Log effective angles (after sign correction) for easier visual debugging
                    var effectiveDeg = new float[pose.ManoJointAngles.Length];
                    for (int j = 0; j < effectiveDeg.Length; j++)
                    {
                        float s = (jointSignMultipliers != null && j < jointSignMultipliers.Length) ? jointSignMultipliers[j] : 1f;
                        effectiveDeg[j] = pose.ManoJointAngles[j] * s * Mathf.Rad2Deg;
                    }
                    string allAngles = string.Join(" ", System.Array.ConvertAll(effectiveDeg, a => a.ToString("F1")));
                    Debug.Log($"[HandRig {hand}] applied={applied}/15  nullJoints={nullCount}  grabBlend={_grabBlend:F2}  angles(signed): {allAngles}");

                    for (int i = 0; i < Mathf.Min(3, fingerJoints.Length); i++)
                        if (fingerJoints[i] != null)
                            Debug.Log($"[HandRig {hand}] joint[{i}] localEuler={fingerJoints[i].localEulerAngles:F1}  name={fingerJoints[i].name}");

                    Debug.Log($"[HandRig {hand}] wrist worldPos={transform.position:F3}  localPos={transform.localPosition:F3}");
                }
#endif
            }
            else if (Time.frameCount % 60 == 0)
            {
                string hand = isLeftHand ? "L" : "R";
                // Always log this — it means nothing is being driven (bones null or no pose data)
                Debug.LogWarning($"[HandRig {hand}] SKIP — ManoJointAngles null={pose.ManoJointAngles == null}  fingerJoints null={fingerJoints == null}  fingerJoints len={(fingerJoints?.Length ?? -1)}  needsBoneInit={_needsBoneInit}");
            }

            if (handRenderer != null)
                handRenderer.enabled = true;

            // Force all SkinnedMeshRenderers visible (useful in Editor without Quest tracking,
            // where OVRHand.IsTracked=false would otherwise hide the hand mesh)
            if (forceVisible)
            {
                foreach (var smr in GetComponentsInChildren<SkinnedMeshRenderer>(includeInactive: true))
                    smr.enabled = true;
            }
        }

        public void SetupFingerJoints(Transform[] bones)
        {
            fingerJoints   = bones;
            _needsBoneInit = false;
            CaptureRestPose();
        }

        // ── Lazy OVRSkeleton bone initialisation ──────────────────────────────

        /// <summary>
        /// Attempts to auto-populate fingerJoints from the OVRSkeleton on this GameObject.
        /// Called every LateUpdate until bones are found or OVRSkeleton is absent.
        /// OVRSkeleton initialises asynchronously when Quest tracking starts, so we must
        /// poll rather than doing this once in Start().
        /// </summary>
        private void TryInitBonesFromOVRSkeleton()
        {
            // If joints were assigned externally between frames, just capture rest-pose and stop.
            if (fingerJoints != null)
            {
                int assigned = 0;
                foreach (var j in fingerJoints) if (j != null) assigned++;
                if (assigned > 0) { _needsBoneInit = false; CaptureRestPose(); return; }
            }

            var skel = GetComponent<OVRSkeleton>();
            if (skel == null)
            {
                // No OVRSkeleton — nothing to auto-populate; disable further polling.
                _needsBoneInit = false;
                Debug.LogWarning($"[HandRig {(isLeftHand?"L":"R")}] No OVRSkeleton on this GameObject — bones must be assigned manually in Inspector. fingerJoints.Length={(fingerJoints?.Length ?? 0)}");
                return;
            }

            // OVRSkeleton hasn't finished initialising yet — try again next frame.
            if (skel.Bones == null || skel.Bones.Count == 0) return;

            // MANO order: Thumb(0-2), Index(3-5), Middle(6-8), Ring(9-11), Pinky(12-14)
            var boneMap = new[]
            {
                OVRSkeleton.BoneId.Hand_Thumb1,  OVRSkeleton.BoneId.Hand_Thumb2,  OVRSkeleton.BoneId.Hand_Thumb3,
                OVRSkeleton.BoneId.Hand_Index1,  OVRSkeleton.BoneId.Hand_Index2,  OVRSkeleton.BoneId.Hand_Index3,
                OVRSkeleton.BoneId.Hand_Middle1, OVRSkeleton.BoneId.Hand_Middle2, OVRSkeleton.BoneId.Hand_Middle3,
                OVRSkeleton.BoneId.Hand_Ring1,   OVRSkeleton.BoneId.Hand_Ring2,   OVRSkeleton.BoneId.Hand_Ring3,
                OVRSkeleton.BoneId.Hand_Pinky1,  OVRSkeleton.BoneId.Hand_Pinky2,  OVRSkeleton.BoneId.Hand_Pinky3,
            };

            var joints = new Transform[boneMap.Length];
            foreach (var bone in skel.Bones)
                for (int i = 0; i < boneMap.Length; i++)
                    if (bone.Id == boneMap[i]) { joints[i] = bone.Transform; break; }

            int found = 0;
            foreach (var j in joints) if (j != null) found++;

            if (found == 0) return;  // bones exist but IDs don't match yet — retry

            fingerJoints   = joints;
            _needsBoneInit = false;
            CaptureRestPose();

            string hand = isLeftHand ? "L" : "R";
            Debug.Log($"[HandRig {hand}] Auto-populated {found}/15 bones from OVRSkeleton.");
        }

        /// <summary>Captures the current local rotations of all assigned joints as rest-pose.</summary>
        private void CaptureRestPose()
        {
            if (fingerJoints == null || fingerJoints.Length == 0) return;
            _restPose       = new Quaternion[fingerJoints.Length];
            _smoothedAngles = new float[fingerJoints.Length];
            for (int i = 0; i < fingerJoints.Length; i++)
            {
                _restPose[i]       = fingerJoints[i] != null ? fingerJoints[i].localRotation : Quaternion.identity;
                _smoothedAngles[i] = 0f;
            }
        }
    }
}
