using UnityEngine;

namespace AuraXR
{
    /// <summary>
    /// Applies predicted hand pose (from AuraXRInferenceManager) to MANO skeleton.
    /// Attach this to LeftHandRig or RightHandRig.
    /// Uses fixed neutral beta (all zeros) — no per-prediction bone scaling needed.
    /// </summary>
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

        [Header("Debug")]
        [Tooltip("Multiply all finger angles — crank this up (e.g. 5-10) in Play mode to confirm bones are actually moving")]
        [Range(0.1f, 20f)]
        public float debugAngleMultiplier = 1f;

        [Header("Visual")]
        [Tooltip("Optional: Renderer to toggle visibility")]
        public Renderer handRenderer;

        // Rest-pose rotations captured before OVR drives the bones
        private Quaternion[] _restPose;

        private void Awake()
        {
            if (inferenceManager == null)
                inferenceManager = FindAnyObjectByType<AuraXRInferenceManager>();
            if (inferenceManager == null)
                Debug.LogWarning($"[HandRig {(isLeftHand ? "L" : "R")}] inferenceManager not found — assign it in the Inspector.");
        }

        private void Start()
        {
            if (fingerJoints == null) return;
            _restPose = new Quaternion[fingerJoints.Length];
            for (int i = 0; i < fingerJoints.Length; i++)
                _restPose[i] = fingerJoints[i] != null ? fingerJoints[i].localRotation : Quaternion.identity;
        }

        private void LateUpdate()
        {
            if (inferenceManager == null) return;

            var pose = isLeftHand ? inferenceManager.LeftHand : inferenceManager.RightHand;

            // NOTE: wrist position/rotation is set by AuraXRInferenceManager.ApplyToAnchor
            // (controller.position + pose.DeltaPosition). Do NOT override it here with
            // pose.WristPosition — that value is in HOT3D world-space, not Quest space.

            if (pose.ManoJointAngles != null && fingerJoints != null)
            {
                int applied = 0;
                for (int i = 0; i < Mathf.Min(pose.ManoJointAngles.Length, fingerJoints.Length); i++)
                {
                    if (fingerJoints[i] != null)
                    {
                        float deg = pose.ManoJointAngles[i] * Mathf.Rad2Deg * debugAngleMultiplier;
                        Quaternion rest = (_restPose != null && i < _restPose.Length) ? _restPose[i] : Quaternion.identity;
                        fingerJoints[i].localRotation = rest * Quaternion.AngleAxis(deg, Vector3.right);
                        applied++;
                    }
                }

#if UNITY_EDITOR
                if (Time.frameCount % 30 == 0)
                {
                    string hand = isLeftHand ? "L" : "R";
                    // How many joints are null (wiring issue)?
                    int nullCount = 0;
                    for (int i = 0; i < fingerJoints.Length; i++)
                        if (fingerJoints[i] == null) nullCount++;

                    // Sample angles across all 15 joints
                    string allAngles = string.Join(" ", System.Array.ConvertAll(pose.ManoJointAngles,
                        a => (a * Mathf.Rad2Deg).ToString("F1")));

                    Debug.Log($"[HandRig {hand}] applied={applied}/15  nullJoints={nullCount}  angles: {allAngles}");

                    // Report actual transform localEulerAngles for first 3 joints to confirm write
                    for (int i = 0; i < Mathf.Min(3, fingerJoints.Length); i++)
                    {
                        if (fingerJoints[i] != null)
                            Debug.Log($"[HandRig {hand}] joint[{i}] localEuler={fingerJoints[i].localEulerAngles:F1}  name={fingerJoints[i].name}");
                    }

                    // Wrist position from anchor
                    Debug.Log($"[HandRig {hand}] wrist worldPos={transform.position:F3}  localPos={transform.localPosition:F3}");
                }
#endif
            }
#if UNITY_EDITOR
            else if (Time.frameCount % 30 == 0)
            {
                string hand = isLeftHand ? "L" : "R";
                Debug.Log($"[HandRig {hand}] SKIP — ManoJointAngles null={pose.ManoJointAngles == null}  fingerJoints null={fingerJoints == null}  fingerJoints len={(fingerJoints?.Length ?? -1)}");
            }
#endif

            if (handRenderer != null)
                handRenderer.enabled = true;
        }

        public void SetupFingerJoints(Transform[] bones) => fingerJoints = bones;
    }
}
