using UnityEngine;

namespace AuraXR
{
    /// <summary>
    /// Builds the 96-dim feature vector each frame and maintains a T=16 ring buffer.
    /// Feature layout (matches 09_build_dataset.py):
    ///   [0..2]    ctrl_position_h0    (x,y,z  world metres)
    ///   [3..6]    ctrl_rotation_h0    (w,x,y,z quaternion)
    ///   [7]       grip_h0             (0–1)
    ///   [8]       trigger_h0          (0–1)
    ///   [9..17]   same for hand 1
    ///   [18..20]  nearest_obj_centroid_h0  (x,y,z metres)
    ///   [21..23]  nearest_obj_bbox_h0      (half-extents x,y,z metres)
    ///   [24]      nearest_obj_category_h0  (1–33, 0 = unknown)
    ///   [25..31]  same for hand 1
    ///   [32..95]  visual_embedding         (64 floats, currently all 0)
    /// </summary>
    public class AuraXRFeatureAssembler : MonoBehaviour
    {
        public const int FeatureDim   = 96;
        public const int WindowFrames = 16;

        [Header("XR Node references (auto-detected if left null)")]
        public Transform leftControllerTransform;
        public Transform rightControllerTransform;

        [Header("Optional: nearest interactable objects (set by game logic)")]
        public Transform nearestObjectLeft;
        public Transform nearestObjectRight;
        [Tooltip("Category IDs matching HOT3D (1–33, 0=unknown)")]
        public int nearestObjectCategoryLeft  = 0;
        public int nearestObjectCategoryRight = 0;

        // Last frame grip/trigger values — read by ScenarioKitchenTask
        public float LastLeftGrip    { get; private set; }
        public float LastRightGrip   { get; private set; }
        public float LastLeftTrigger { get; private set; }
        public float LastRightTrigger{ get; private set; }

        // Ring buffer: [WindowFrames, FeatureDim]
        private float[][] _buffer;
        private int       _head;
        private bool      _full;

        void Awake()
        {
            _buffer = new float[WindowFrames][];
            for (int i = 0; i < WindowFrames; i++)
                _buffer[i] = new float[FeatureDim];
        }

        void LateUpdate()
        {
            BuildFrame(_buffer[_head]);
            _head = (_head + 1) % WindowFrames;
            if (_head == 0) _full = true;
        }

        public bool IsReady => _full;

        public void CopyWindowTo(float[,] outWindow)
        {
            for (int t = 0; t < WindowFrames; t++)
            {
                int srcIdx = (_head + t) % WindowFrames;
                for (int f = 0; f < FeatureDim; f++)
                    outWindow[t, f] = _buffer[srcIdx][f];
            }
        }

        public void CopyWindowFlat(float[] outFlat)
        {
            for (int t = 0; t < WindowFrames; t++)
            {
                int srcIdx = (_head + t) % WindowFrames;
                System.Array.Copy(_buffer[srcIdx], 0, outFlat, t * FeatureDim, FeatureDim);
            }
        }

        private void BuildFrame(float[] f)
        {
            // --- Hand 0 (left controller) ---
            Vector3    posL = leftControllerTransform  != null ? leftControllerTransform.position  : Vector3.zero;
            Quaternion rotL = leftControllerTransform  != null ? leftControllerTransform.rotation  : Quaternion.identity;
            float gripL    = OVRInput.Get(OVRInput.Axis1D.PrimaryHandTrigger,  OVRInput.Controller.LTouch);
            float triggerL = OVRInput.Get(OVRInput.Axis1D.PrimaryIndexTrigger, OVRInput.Controller.LTouch);

            f[0] = posL.x; f[1] = posL.y; f[2] = posL.z;
            f[3] = rotL.w; f[4] = rotL.x; f[5] = rotL.y; f[6] = rotL.z;
            f[7] = gripL;  f[8] = triggerL;
            LastLeftGrip    = gripL;
            LastLeftTrigger = triggerL;

            // --- Hand 1 (right controller) ---
            Vector3    posR = rightControllerTransform != null ? rightControllerTransform.position : Vector3.zero;
            Quaternion rotR = rightControllerTransform != null ? rightControllerTransform.rotation : Quaternion.identity;
            float gripR    = OVRInput.Get(OVRInput.Axis1D.PrimaryHandTrigger,  OVRInput.Controller.RTouch);
            float triggerR = OVRInput.Get(OVRInput.Axis1D.PrimaryIndexTrigger, OVRInput.Controller.RTouch);

            f[9]  = posR.x; f[10] = posR.y; f[11] = posR.z;
            f[12] = rotR.w; f[13] = rotR.x; f[14] = rotR.y; f[15] = rotR.z;
            f[16] = gripR;  f[17] = triggerR;
            LastRightGrip    = gripR;
            LastRightTrigger = triggerR;

            // --- Nearest object (left hand) ---
            if (nearestObjectLeft != null)
            {
                Bounds b = GetBounds(nearestObjectLeft);
                f[18] = b.center.x; f[19] = b.center.y; f[20] = b.center.z;
                f[21] = b.extents.x; f[22] = b.extents.y; f[23] = b.extents.z;
            }
            else
            {
                for (int i = 18; i < 24; i++) f[i] = 0f;
            }
            f[24] = nearestObjectCategoryLeft;

            // --- Nearest object (right hand) ---
            if (nearestObjectRight != null)
            {
                Bounds b = GetBounds(nearestObjectRight);
                f[25] = b.center.x; f[26] = b.center.y; f[27] = b.center.z;
                f[28] = b.extents.x; f[29] = b.extents.y; f[30] = b.extents.z;
            }
            else
            {
                for (int i = 25; i < 31; i++) f[i] = 0f;
            }
            f[31] = nearestObjectCategoryRight;

            // --- Visual embedding (placeholder: all zeros) ---
            for (int i = 32; i < 96; i++) f[i] = 0f;
        }

        private static Bounds GetBounds(Transform t)
        {
            var renderer = t.GetComponentInChildren<Renderer>();
            if (renderer != null) return renderer.bounds;
            return new Bounds(t.position, Vector3.one * 0.05f);
        }
    }
}
