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

        [Header("Visual")]
        [Tooltip("Optional: Renderer to toggle visibility")]
        public Renderer handRenderer;

        private void LateUpdate()
        {
            if (inferenceManager == null) return;

            var pose = isLeftHand ? inferenceManager.LeftHand : inferenceManager.RightHand;

            // NOTE: wrist position/rotation is set by AuraXRInferenceManager.ApplyToAnchor
            // (controller.position + pose.DeltaPosition). Do NOT override it here with
            // pose.WristPosition — that value is in HOT3D world-space, not Quest space.

            if (pose.ManoJointAngles != null && fingerJoints != null)
            {
                for (int i = 0; i < Mathf.Min(pose.ManoJointAngles.Length, fingerJoints.Length); i++)
                {
                    if (fingerJoints[i] != null)
                        fingerJoints[i].localEulerAngles = Vector3.right * pose.ManoJointAngles[i] * Mathf.Rad2Deg;
                }
            }

            if (handRenderer != null)
                handRenderer.enabled = true;
        }

        public void SetupFingerJoints(Transform[] bones) => fingerJoints = bones;
    }
}
