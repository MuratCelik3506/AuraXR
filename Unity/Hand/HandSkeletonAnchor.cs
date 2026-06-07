using UnityEngine;

namespace AuraXR
{
    /// <summary>
    /// Runs after OVRSkeleton and re-roots all skeleton bones so the wrist
    /// sits at this transform's position/rotation. This overrides OVRSkeleton's
    /// world-space bone placement with our controller-driven anchor.
    ///
    /// Add this component to LeftHandRig and RightHandRig alongside OVRSkeleton.
    /// Script Execution Order must be higher (later) than OVRSkeleton.
    /// </summary>
    [DefaultExecutionOrder(1000)]
    public class HandSkeletonAnchor : MonoBehaviour
    {
        private OVRSkeleton _skeleton;

        void Awake()
        {
            _skeleton = GetComponent<OVRSkeleton>();
            if (_skeleton == null)
                Debug.LogWarning("[HandSkeletonAnchor] No OVRSkeleton found on this GameObject.");
        }

        void LateUpdate()
        {
            if (_skeleton == null) return;
            var bones = _skeleton.Bones;
            if (bones == null || bones.Count == 0) return;

            Transform wrist = bones[0].Transform;
            if (wrist == null) return;

            Vector3    targetPos = transform.position;
            Quaternion targetRot = transform.rotation;

            // Compute the change in wrist pose
            Vector3    posOffset = targetPos - wrist.position;
            Quaternion rotDelta  = targetRot * Quaternion.Inverse(wrist.rotation);

            // Re-root all bones: translate + rotate around the new wrist origin
            foreach (var bone in bones)
            {
                if (bone.Transform == null) continue;
                // Rotate position around the old wrist, then translate
                Vector3 relPos = bone.Transform.position - wrist.position;
                bone.Transform.position = targetPos + rotDelta * relPos;
                bone.Transform.rotation = rotDelta * bone.Transform.rotation;
            }
        }
    }
}
