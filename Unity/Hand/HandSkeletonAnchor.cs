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

            // Snapshot every bone's CURRENT world pose BEFORE touching wrist.
            // Setting wrist.position/rotation propagates to children (because they're
            // parented to wrist), so reading bone.position inside the apply-loop would
            // see post-wrist-move values. That leads to a double-rotation on children.
            Vector3    wristPosOrig = wrist.position;
            Quaternion wristRotOrig = wrist.rotation;
            Quaternion rotDelta     = targetRot * Quaternion.Inverse(wristRotOrig);

            int n = bones.Count;
            var origPos = new Vector3[n];
            var origRot = new Quaternion[n];
            for (int i = 0; i < n; i++)
            {
                if (bones[i].Transform == null) continue;
                origPos[i] = bones[i].Transform.position;
                origRot[i] = bones[i].Transform.rotation;
            }

            // Re-root: rotate around old wrist origin, then translate to target wrist.
            for (int i = 0; i < n; i++)
            {
                var t = bones[i].Transform;
                if (t == null) continue;
                Vector3 relPos = origPos[i] - wristPosOrig;
                t.position = targetPos + rotDelta * relPos;
                t.rotation = rotDelta * origRot[i];
            }
        }
    }
}
