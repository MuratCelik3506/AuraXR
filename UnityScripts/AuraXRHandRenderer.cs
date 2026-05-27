using UnityEngine;

namespace AuraXR
{
    /// <summary>
    /// Maps IntentFormer's 15 MANO joint angles onto a hand skeleton.
    ///
    /// MANO 1-DoF layout (HOT3D convention, radians, flexion positive):
    ///   [0..2]   Thumb  MCP, PIP, DIP
    ///   [3..5]   Index  MCP, PIP, DIP
    ///   [6..8]   Middle MCP, PIP, DIP
    ///   [9..11]  Ring   MCP, PIP, DIP
    ///   [12..14] Pinky  MCP, PIP, DIP
    ///
    /// Setup in Inspector:
    ///   1. Assign WristAnchor → the wrist bone Transform of your hand rig.
    ///   2. Expand JointBones and assign each of the 15 bone Transforms in order above.
    ///   3. Assign BendAxis per bone (default: Vector3.right works for most rigs
    ///      where flexion rotates around local X).
    ///   4. Set IsLeftHand to mirror axis signs for the left hand if needed.
    /// </summary>
    public class AuraXRHandRenderer : MonoBehaviour
    {
        [Header("Source")]
        public AuraXRInferenceManager inferenceManager;
        public bool isLeftHand = false;

        [Header("Wrist anchor (root bone)")]
        public Transform wristAnchor;

        [Header("15 joint bones — order: Thumb0-2, Index0-2, Middle0-2, Ring0-2, Pinky0-2")]
        public Transform[] jointBones = new Transform[15];

        [Header("Bend axis in each bone's local space (default: Vector3.right)")]
        public Vector3 bendAxis = Vector3.right;

        [Header("Scale factor applied to raw MANO angle (radians→degrees built-in)")]
        [Tooltip("1.0 = use raw MANO angle. Reduce if hand looks over-flexed.")]
        [Range(0.1f, 2.0f)]
        public float angleScale = 1.0f;

        [Header("Smoothing (0 = no smoothing, 1 = frozen)")]
        [Range(0f, 0.95f)]
        public float smoothing = 0.5f;

        // Last-frame angles for smoothing
        private float[]          _smoothed = new float[15];
        private bool             _initialized;
        // If HandRigController is present on the same object, it takes ownership of bone writes.
        private HandRigController _rigController;

        void Awake()
        {
            // HandRigController and AuraXRHandRenderer both write to finger bones.
            // If HandRigController is active on the same GameObject, defer to it to avoid
            // each overwriting the other every LateUpdate.
            _rigController = GetComponent<HandRigController>();
            if (_rigController == null)
                _rigController = GetComponentInParent<HandRigController>();
        }

        void LateUpdate()
        {
            // Yield to HandRigController when it is present and enabled
            if (_rigController != null && _rigController.enabled) return;

            if (inferenceManager == null) return;

            HandPose pose = isLeftHand ? inferenceManager.LeftHand : inferenceManager.RightHand;
            if (pose?.ManoJointAngles == null || pose.ManoJointAngles.Length < 15) return;

            // Drive joint bones
            for (int i = 0; i < 15; i++)
            {
                if (jointBones[i] == null) continue;

                float raw = pose.ManoJointAngles[i]; // radians

                // Smooth
                if (!_initialized) _smoothed[i] = raw;
                _smoothed[i] = Mathf.Lerp(raw, _smoothed[i], smoothing);

                float degrees = _smoothed[i] * Mathf.Rad2Deg * angleScale;
                jointBones[i].localRotation = Quaternion.AngleAxis(degrees, bendAxis);
            }

            _initialized = true;
        }

#if USING_OVR_SDK
        public void AutoPopulateFromOVRSkeleton(OVRSkeleton skeleton)
        {
            if (skeleton == null) return;
            var bones = skeleton.Bones;
            var map = new[]
            {
                OVRSkeleton.BoneId.Hand_Thumb1,
                OVRSkeleton.BoneId.Hand_Thumb2,
                OVRSkeleton.BoneId.Hand_Thumb3,
                OVRSkeleton.BoneId.Hand_Index1,
                OVRSkeleton.BoneId.Hand_Index2,
                OVRSkeleton.BoneId.Hand_Index3,
                OVRSkeleton.BoneId.Hand_Middle1,
                OVRSkeleton.BoneId.Hand_Middle2,
                OVRSkeleton.BoneId.Hand_Middle3,
                OVRSkeleton.BoneId.Hand_Ring1,
                OVRSkeleton.BoneId.Hand_Ring2,
                OVRSkeleton.BoneId.Hand_Ring3,
                OVRSkeleton.BoneId.Hand_Pinky1,
                OVRSkeleton.BoneId.Hand_Pinky2,
                OVRSkeleton.BoneId.Hand_Pinky3,
            };

            for (int i = 0; i < map.Length; i++)
            {
                foreach (var b in bones)
                {
                    if (b.Id == map[i])
                    {
                        jointBones[i] = b.Transform;
                        break;
                    }
                }
            }
            Debug.Log("[AuraXR] AutoPopulateFromOVRSkeleton done.");
        }
#endif
    }
}
