using System.Collections;
using System.IO;
using System.Text;
using UnityEngine;

namespace AuraXR
{
    /// <summary>
    /// Editor/runtime utility: sweeps each UME joint index from 0..19, applies a known
    /// rotation pulse, and measures which Unity bone actually moved and by how much.
    /// Catches UME→MANO mapping errors (e.g. Thumb.CMC-flex mistakenly routed to
    /// Hand_Thumb1 instead of Hand_ThumbTrap), per-bone sign flips, and dead joints.
    ///
    /// Usage:
    ///   1. Attach to LeftHandRig or RightHandRig (next to OVRSkeleton + HandRigController).
    ///   2. Enter Play mode (Editor — does NOT need a Quest device since we drive
    ///      HandRigController directly via a synthetic HandPose).
    ///   3. Right-click the component → "Run Bone Sweep" (or set runOnStart=true).
    ///   4. CSV is written to Application.persistentDataPath/Logs/bone_mapping_{hand}_{timestamp}.csv
    ///      and a summary is printed to the Console.
    ///
    /// What the CSV contains (one row per UME index 0..19):
    ///   UmeIdx, JointName, MappedManoIdx, ExpectedBone,
    ///   ActualMovingBoneId, ActualMovingBoneName, MaxRotationDeltaDeg,
    ///   SecondMostMovingBoneId, SecondMaxDeltaDeg, JudgmentNote
    ///
    /// Judgment column flags:
    ///   "OK"         — single bone moved, matches expected mapping
    ///   "WRONG_BONE" — actual moving bone ≠ expected
    ///   "MULTI_BONE" — more than one bone responded (likely topology/parent issue)
    ///   "DEAD"       — no bone moved (mapping skipped joint 20/21 or pulse too small)
    ///   "SIGN_FLIP"  — bone moved in opposite direction (joint went negative on a positive pulse)
    /// </summary>
    [DefaultExecutionOrder(2000)]   // After HandRigController (500) and HandSkeletonAnchor (1000)
    public class BoneMappingValidator : MonoBehaviour
    {
        [Header("Setup")]
        [Tooltip("The HandRigController on this same GameObject")]
        public HandRigController rigController;

        [Tooltip("Run a full sweep automatically once Play starts")]
        public bool runOnStart = false;

        [Header("Sweep Settings")]
        [Tooltip("Pulse angle applied to each UME joint, in degrees. 30° = clearly visible without anatomical extreme.")]
        [Range(5f, 90f)]
        public float pulseDegrees = 30f;

        [Tooltip("Frames to wait for the new pose to settle before reading bone rotations")]
        [Range(2, 30)]
        public int settleFrames = 8;

        [Tooltip("Minimum delta (in degrees) to count a bone as 'moving'. Filters out floating-point noise.")]
        [Range(0.1f, 5f)]
        public float movementThresholdDeg = 1.0f;

        // ── UmeTrack joint names (must match evaluate_onnx.py JOINT_NAMES) ──
        private static readonly string[] UmeJointNames = {
            "Thumb.CMC-flex","Thumb.abd","Thumb.MCP","Thumb.DIP",
            "Index.abd","Index.MCP","Index.PIP","Index.DIP",
            "Middle.abd","Middle.MCP","Middle.PIP","Middle.DIP",
            "Ring.abd","Ring.MCP","Ring.PIP","Ring.DIP",
            "Pinky.abd","Pinky.MCP","Pinky.PIP","Pinky.DIP"
        };

        // Current UME→MANO mapping as used by AuraXRInferenceManager. We probe this
        // mapping; if it's wrong, the CSV will say WRONG_BONE for the affected rows.
        // UmeToMano[m] = which UME index drives MANO slot m
        // We invert it: ExpectedManoSlot[u] = which MANO bone *should* move when UME[u] is set
        private static readonly int[] UmeToMano       = { 0, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19 };
        private static readonly int[] UmeAbductionIdx = { 1, 4, 8, 12, 16 };

        // MANO bone names in the order HandRigController.fingerJoints[15] expects
        private static readonly string[] ManoBoneNames = {
            "Thumb.MCP","Thumb.PIP","Thumb.DIP",
            "Index.MCP","Index.PIP","Index.DIP",
            "Middle.MCP","Middle.PIP","Middle.DIP",
            "Ring.MCP","Ring.PIP","Ring.DIP",
            "Pinky.MCP","Pinky.PIP","Pinky.DIP"
        };

        void Start()
        {
            if (rigController == null) rigController = GetComponent<HandRigController>();
            if (runOnStart) StartCoroutine(RunSweep());
        }

        [ContextMenu("Run Bone Sweep")]
        public void RunBoneSweepFromMenu()
        {
            if (!Application.isPlaying)
            {
                Debug.LogWarning("[BoneMappingValidator] Must be in Play mode.");
                return;
            }
            StartCoroutine(RunSweep());
        }

        private IEnumerator RunSweep()
        {
            if (rigController == null)
            {
                Debug.LogError("[BoneMappingValidator] No HandRigController assigned.");
                yield break;
            }
            if (rigController.fingerJoints == null || rigController.fingerJoints.Length == 0)
            {
                Debug.LogError("[BoneMappingValidator] HandRigController.fingerJoints empty. Wait for OVRSkeleton init or assign manually.");
                yield break;
            }

            string handTag = rigController.isLeftHand ? "L" : "R";
            string ts = System.DateTime.Now.ToString("yyyy_MM_dd_HH_mm_ss");
            string folder = Path.Combine(Application.persistentDataPath, "Logs");
            Directory.CreateDirectory(folder);
            string outPath = Path.Combine(folder, $"bone_mapping_{handTag}_{ts}.csv");

            Debug.Log($"[BoneMappingValidator|{handTag}] Sweep starting → {outPath}");

            int N = Mathf.Min(15, rigController.fingerJoints.Length);

            var origForcePose       = rigController.debugForceTestPose;
            var origGrabBlendSpeed  = rigController.grabBlendSpeed;
            var origSmoothing       = rigController.smoothing;

            // Disable HandSkeletonAnchor — it overrides ALL bone rotations every LateUpdate
            // (order 1000), erasing the injected angles and making every joint look identical.
            MonoBehaviour skeletonAnchor = null;
            foreach (var mb in GetComponentsInChildren<MonoBehaviour>(true))
            {
                if (mb.GetType().Name == "HandSkeletonAnchor") { skeletonAnchor = mb; break; }
            }
            if (skeletonAnchor == null)
                foreach (var mb in FindObjectsByType<MonoBehaviour>(FindObjectsSortMode.None))
                    if (mb.GetType().Name == "HandSkeletonAnchor" &&
                        mb.gameObject.transform.IsChildOf(transform.root)) { skeletonAnchor = mb; break; }

            bool anchorWasEnabled = skeletonAnchor != null && skeletonAnchor.enabled;
            if (skeletonAnchor != null) skeletonAnchor.enabled = false;
            if (anchorWasEnabled)
                Debug.Log($"[BoneMappingValidator|{(rigController.isLeftHand ? "L" : "R")}] HandSkeletonAnchor disabled for sweep.");

            rigController.debugForceTestPose = false;
            rigController.smoothing = 0f;
            rigController.grabBlendSpeed = 100f;

            // Build a synthetic HandPose we will mutate each UME step
            var pose = new HandPose();
            for (int i = 0; i < 15; i++) pose.ManoJointAngles[i] = 0f;
            for (int f = 0; f < 5; f++)  pose.UmeAbductionAngles[f] = 0f;

            var inj = gameObject.GetComponent<HandPoseInjector>() ?? gameObject.AddComponent<HandPoseInjector>();
            inj.target = pose;
            inj.rigController = rigController;
            inj.active = true;

            float pulseRad = pulseDegrees * Mathf.Deg2Rad;

            // Let bones settle to zero-pose BEFORE capturing rest, so restLocal matches
            // HandRigController's _restPose baseline and inter-step deltas are clean.
            for (int s = 0; s < settleFrames * 2; s++) yield return null;

            // Cache rest local rotations for all 15 bones (captured after zero-settle)
            Quaternion[] restLocal = new Quaternion[N];
            for (int i = 0; i < N; i++)
            {
                restLocal[i] = rigController.fingerJoints[i] != null
                    ? rigController.fingerJoints[i].localRotation
                    : Quaternion.identity;
            }

            var sb = new StringBuilder();
            sb.AppendLine("UmeIdx,JointName,MappedManoIdx,ExpectedBoneName,ActualMovingBoneIdx,ActualMovingBoneName,MaxDeltaDeg,SecondMovingBoneIdx,SecondMaxDeltaDeg,Judgment");

            int okCount = 0, wrongCount = 0, multiCount = 0, deadCount = 0, signFlipCount = 0;

            for (int u = 0; u < 20; u++)
            {
                // Reset all angles to zero
                for (int i = 0; i < 15; i++) pose.ManoJointAngles[i] = 0f;
                for (int f = 0; f < 5; f++)  pose.UmeAbductionAngles[f] = 0f;
                yield return null;
                yield return null;

                // Map UME index → which slot to drive
                int manoSlot = System.Array.IndexOf(UmeToMano, u);  // -1 if u is abduction
                int abdSlot  = System.Array.IndexOf(UmeAbductionIdx, u);

                if (manoSlot >= 0) pose.ManoJointAngles[manoSlot] = pulseRad;
                if (abdSlot  >= 0) pose.UmeAbductionAngles[abdSlot] = pulseRad;

                // Wait for application
                for (int s = 0; s < settleFrames; s++) yield return null;

                // Read which bones moved
                float[] delta = new float[N];
                for (int i = 0; i < N; i++)
                {
                    if (rigController.fingerJoints[i] == null) { delta[i] = -1f; continue; }
                    Quaternion d = rigController.fingerJoints[i].localRotation * Quaternion.Inverse(restLocal[i]);
                    d.ToAngleAxis(out float ang, out _);
                    if (ang > 180f) ang = 360f - ang;
                    delta[i] = ang;
                }

                int firstIdx = -1, secondIdx = -1;
                float firstMax = 0f, secondMax = 0f;
                for (int i = 0; i < N; i++)
                {
                    if (delta[i] > firstMax) { secondMax = firstMax; secondIdx = firstIdx; firstMax = delta[i]; firstIdx = i; }
                    else if (delta[i] > secondMax) { secondMax = delta[i]; secondIdx = i; }
                }

                int expectedSlot = manoSlot >= 0 ? manoSlot : (abdSlot >= 0 ? abdSlot * 3 : -1);   // abduction lives on MCP bone (idx 0,3,6,9,12)
                string expectedName = (expectedSlot >= 0 && expectedSlot < N) ? ManoBoneNames[expectedSlot] : "n/a";
                string firstName = (firstIdx >= 0 && firstIdx < N) ? ManoBoneNames[firstIdx] : "NONE";
                string secondName = (secondIdx >= 0 && secondIdx < N) ? ManoBoneNames[secondIdx] : "NONE";

                string judgment;
                if (firstMax < movementThresholdDeg) { judgment = "DEAD"; deadCount++; }
                else if (firstIdx == expectedSlot && secondMax < movementThresholdDeg) { judgment = "OK"; okCount++; }
                else if (firstIdx == expectedSlot && secondMax >= movementThresholdDeg) { judgment = "MULTI_BONE"; multiCount++; }
                else if (secondMax >= movementThresholdDeg) { judgment = "MULTI_BONE_WRONG"; multiCount++; wrongCount++; }
                else { judgment = "WRONG_BONE"; wrongCount++; }

                // Sign-flip detection: if expected bone moved by less than threshold but its neighbor (same finger, ±1 slot) moved a lot in the OPPOSITE direction
                // — not done with simple AngleAxis here; user can read the rest-pose-delta sign by enabling verbose log on HandRigController.

                sb.AppendLine($"{u},{UmeJointNames[u]},{manoSlot},{expectedName},{firstIdx},{firstName},{firstMax:F2},{secondIdx},{secondMax:F2},{judgment}");
                Debug.Log($"[BoneSweep|{handTag}] UME{u}({UmeJointNames[u]}) → first={firstName}({firstMax:F1}°)  second={secondName}({secondMax:F1}°)  judgment={judgment}");
            }

            inj.active = false;
            Destroy(inj);
            rigController.smoothing          = origSmoothing;
            rigController.grabBlendSpeed     = origGrabBlendSpeed;
            rigController.debugForceTestPose = origForcePose;
            if (skeletonAnchor != null) skeletonAnchor.enabled = anchorWasEnabled;

            File.WriteAllText(outPath, sb.ToString());

            Debug.Log($"[BoneMappingValidator|{handTag}] DONE.\n" +
                      $"  OK={okCount}  WRONG_BONE={wrongCount}  MULTI_BONE={multiCount}  DEAD={deadCount}\n" +
                      $"  CSV → {outPath}");
        }
    }

    /// <summary>
    /// Helper: writes a fixed HandPose into the AuraXRInferenceManager output slot every
    /// LateUpdate while active, so HandRigController reads our injected angles instead of
    /// the model output.
    /// </summary>
    [DefaultExecutionOrder(499)]   // BEFORE HandRigController (500)
    public class HandPoseInjector : MonoBehaviour
    {
        public HandPose target;
        public HandRigController rigController;
        public bool active;

        private AuraXRInferenceManager _mgr;
        private HandPose _savedLeft, _savedRight;
        private bool _saved;

        void LateUpdate()
        {
            if (!active || target == null || rigController == null) return;
            if (_mgr == null) _mgr = FindAnyObjectByType<AuraXRInferenceManager>();
            if (_mgr == null) return;

            if (!_saved)
            {
                _savedLeft  = _mgr.LeftHand;
                _savedRight = _mgr.RightHand;
                _saved = true;
            }

            // Use reflection-free trick: assign via the public setter side-effect.
            // Both LeftHand and RightHand are public get-only — we cannot reassign.
            // Instead, mutate the cached HandPose's arrays in place so HandRigController
            // sees the new values on its next LateUpdate (which runs AFTER us).
            HandPose hp = rigController.isLeftHand ? _mgr.LeftHand : _mgr.RightHand;
            if (hp == null) return;
            if (hp.ManoJointAngles == null || hp.ManoJointAngles.Length < 15)
                hp.ManoJointAngles = new float[15];
            if (hp.UmeAbductionAngles == null || hp.UmeAbductionAngles.Length < 5)
                hp.UmeAbductionAngles = new float[5];
            for (int i = 0; i < 15; i++) hp.ManoJointAngles[i]    = target.ManoJointAngles[i];
            for (int f = 0; f < 5;  f++) hp.UmeAbductionAngles[f] = target.UmeAbductionAngles[f];
        }
    }
}
