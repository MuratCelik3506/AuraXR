using UnityEngine;

namespace AuraXR
{
    /// <summary>
    /// Builds the 96-dim feature vector each frame and maintains a T=16 ring buffer.
    /// Subsamples to ~30 fps (frameSampleRate=2 at 72 Hz) to match HOT3D training data rate.
    ///
    /// Feature layout (matches 09_build_dataset.py) — all positions in HOT3D frame (Z negated):
    ///   [0..2]    ctrl_position_h0    (x, y, -z  world metres — Z negated for HOT3D frame)
    ///   [3..6]    ctrl_rotation_h0    (w, x, y, -z quaternion — Z negated)
    ///   [7]       grip_h0             (0–1)
    ///   [8]       trigger_h0          (0–1)
    ///   [9..17]   same for hand 1
    ///   [18..20]  nearest_obj_centroid_h0  (absolute world x, y, -z — NOT controller-relative)
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

        [Header("Frame Sampling (match training FPS)")]
        [Tooltip("Add a frame to the ring buffer only every N Unity frames. " +
                 "Training data was 30 fps; Quest 3 renders at 72 fps → use 2 (≈36 fps). " +
                 "Keeps the 16-frame window covering ~0.44 s, close to the 0.53 s training window.")]
        public int frameSampleRate = 2;

        [Header("Editor / Simulator Override")]
        [Tooltip("Force HOT3D mean controller pose (tabletop, tilted ~60° inward).\n" +
                 "Enable in Meta XR Simulator when default arm pose makes model output closed/wrong hands.\n" +
                 "Disable to use real OVR input (Meta Link or standalone Quest 3).")]
        public bool forceHot3dSimulation = false;

        // Ring buffer: [WindowFrames, FeatureDim]
        private float[][] _buffer;
        private int       _head;
        private bool      _full;
        private int       _sampleCounter;

        void Awake() => EnsureBufferInit();

        private void EnsureBufferInit()
        {
            if (_buffer != null) return;
            _buffer = new float[WindowFrames][];
            for (int i = 0; i < WindowFrames; i++)
                _buffer[i] = new float[FeatureDim];
        }

        void LateUpdate()
        {
            if (_buffer == null) EnsureBufferInit();

            // Subsample to ~30 fps to match training temporal distribution.
            if (++_sampleCounter < frameSampleRate) return;
            _sampleCounter = 0;

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

#if UNITY_EDITOR
            // _edSim=true when: no OVR controllers connected, OR forceHot3dSimulation is checked.
            // forceHot3dSimulation is useful when using the Meta XR Simulator, whose default
            // arm pose doesn't match the HOT3D training distribution (tilted toward table).
            bool  _edSim = forceHot3dSimulation ||
                           (OVRInput.GetConnectedControllers() & OVRInput.Controller.LTouch) == 0;
            float _edT   = Time.time;
            if (_edSim)
            {
                // HOT3D mean controller orientation for left hand (tabletop manipulation).
                // Unity left-handed frame: x=-0.260, y=-0.642, z=+0.076, w=0.717.
                gripL    = Mathf.Abs(Mathf.Sin(_edT * 0.5f));
                triggerL = Mathf.Abs(Mathf.Sin(_edT * 0.3f));
                posL = new Vector3( 0.30f + Mathf.Sin(_edT * 0.40f) * 0.08f,
                                    0.70f + Mathf.Sin(_edT * 0.25f) * 0.05f,
                                    0.10f + Mathf.Sin(_edT * 0.15f) * 0.04f);
                rotL = Quaternion.Normalize(new Quaternion(-0.260f, -0.642f, 0.076f, 0.717f))
                     * Quaternion.Euler(Mathf.Sin(_edT * 0.30f) * 8f, Mathf.Sin(_edT * 0.20f) * 8f,
                                        Mathf.Sin(_edT * 0.15f) * 4f);
            }
#endif

            // HOT3D training used right-handed Y-up (Z backward); Unity is left-handed Y-up (Z forward).
            // Convert positions: negate Z. Convert quaternion: negate Z imaginary component.
            f[0] = posL.x; f[1] = posL.y; f[2] = -posL.z;
            f[3] = rotL.w; f[4] = rotL.x; f[5] = rotL.y; f[6] = -rotL.z;
            f[7] = gripL;  f[8] = triggerL;
            LastLeftGrip    = gripL;
            LastLeftTrigger = triggerL;

            // --- Hand 1 (right controller) ---
            Vector3    posR = rightControllerTransform != null ? rightControllerTransform.position : Vector3.zero;
            Quaternion rotR = rightControllerTransform != null ? rightControllerTransform.rotation : Quaternion.identity;
            float gripR    = OVRInput.Get(OVRInput.Axis1D.PrimaryHandTrigger,  OVRInput.Controller.RTouch);
            float triggerR = OVRInput.Get(OVRInput.Axis1D.PrimaryIndexTrigger, OVRInput.Controller.RTouch);

#if UNITY_EDITOR
            if (_edSim)
            {
                // Mirror of left: y-component flipped. Unity: x=-0.260, y=+0.642, z=-0.076, w=0.717.
                gripR    = Mathf.Abs(Mathf.Sin(_edT * 0.70f));
                triggerR = Mathf.Abs(Mathf.Sin(_edT * 0.45f));
                posR = new Vector3(-0.30f + Mathf.Sin(_edT * 0.35f) * 0.08f,
                                    0.70f + Mathf.Sin(_edT * 0.20f) * 0.05f,
                                    0.10f + Mathf.Sin(_edT * 0.18f) * 0.04f);
                rotR = Quaternion.Normalize(new Quaternion(-0.260f, 0.642f, -0.076f, 0.717f))
                     * Quaternion.Euler(Mathf.Sin(_edT * 0.40f) * 8f, Mathf.Sin(_edT * 0.25f) * 8f,
                                        Mathf.Sin(_edT * 0.18f) * 4f);
            }
#endif

            f[9]  = posR.x; f[10] = posR.y; f[11] = -posR.z;
            f[12] = rotR.w; f[13] = rotR.x; f[14] = rotR.y; f[15] = -rotR.z;
            f[16] = gripR;  f[17] = triggerR;
            LastRightGrip    = gripR;
            LastRightTrigger = triggerR;

            // --- Nearest object (left hand) ---
            // HOT3D training stored ABSOLUTE world-space centroid positions (not controller-relative).
            // Send absolute world position with Z negated for HOT3D frame.
            if (nearestObjectLeft != null)
            {
                Bounds b = GetBounds(nearestObjectLeft);
                f[18] = b.center.x; f[19] = b.center.y; f[20] = -b.center.z;
                f[21] = b.extents.x; f[22] = b.extents.y; f[23] = b.extents.z;
                f[24] = nearestObjectCategoryLeft;
            }
            else
            {
#if UNITY_EDITOR
                // Simulate a mustard bottle near the left hand to stay in training distribution.
                // HOT3D training mean object centroid ≈ (0.28, 0.77, -0.05) — replicated here.
                f[18] = 0.25f + Mathf.Sin(_edT * 0.11f) * 0.03f;
                f[19] = 0.72f + Mathf.Sin(_edT * 0.09f) * 0.02f;
                f[20] = -0.15f;
                f[21] = 0.04f; f[22] = 0.04f; f[23] = 0.03f;
                f[24] = 13f;   // mustard bottle
#else
                for (int i = 18; i < 25; i++) f[i] = 0f;
#endif
            }

            // --- Nearest object (right hand) ---
            if (nearestObjectRight != null)
            {
                Bounds b = GetBounds(nearestObjectRight);
                f[25] = b.center.x; f[26] = b.center.y; f[27] = -b.center.z;
                f[28] = b.extents.x; f[29] = b.extents.y; f[30] = b.extents.z;
                f[31] = nearestObjectCategoryRight;
            }
            else
            {
#if UNITY_EDITOR
                // Simulate a mug near the right hand.
                f[25] = -0.25f + Mathf.Sin(_edT * 0.13f) * 0.03f;
                f[26] = 0.72f + Mathf.Sin(_edT * 0.08f) * 0.02f;
                f[27] = -0.15f;
                f[28] = 0.04f; f[29] = 0.04f; f[30] = 0.04f;
                f[31] = 8f;    // mug
#else
                for (int i = 25; i < 32; i++) f[i] = 0f;
#endif
            }

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
