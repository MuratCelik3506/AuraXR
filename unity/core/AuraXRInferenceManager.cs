using System;
using System.Collections.Generic;
using UnityEngine;
using Unity.InferenceEngine;

namespace AuraXR
{
    /// <summary>
    /// Runs AuraXR SDF-LSTM inference each frame via Unity Inference Engine.
    ///
    /// LSTM ONNX inputs:
    ///   "frame_feat" [1, 29]:
    ///       [0-2]   dir_world           (HOT3D frame)
    ///       [3-5]   dir_obj_local       (object-local frame)
    ///       [6]     dist
    ///       [7]     approach_speed
    ///       [8-10]  obj_vel_world       (nearest object's linear velocity)
    ///       [11-16] wrist_rot_6d_input  (canonical-relative wrist 6D, Unity frame)
    ///       [17]    hand_confidence     (defaulted to 1.0 at runtime — no tracker conf available)
    ///       [18-21] grip_oh
    ///       [22-24] bbox
    ///
    /// LSTM ONNX outputs:
    ///   "pose_pca" [1, 15]: 15 MANO PCA components (normalized) — denorm with target_mean/std
    ///   "wrist_rot_6d" [1,  6]: palm orientation as 6D rotation (normalized) — denorm, then
    ///                           Gram-Schmidt → rotation matrix → Quaternion in Unity frame
    ///   "contact_prob" [1, 1]
    ///   "h_n"          [2, 1, 256]
    ///   "c_n"          [2, 1, 256]
    ///
    /// Coordinate frame: HOT3D is right-handed Y-up. Unity is left-handed Y-up.
    ///   Conversion: negate Z of positions and quaternion Z component.
    ///   ToHOT3D(v)    = (x, y, -z)
    ///   ToHOT3DQuat(q)= (qx, qy, -qz, qw)
    ///
    /// Public output read by HandRigController each frame:
    ///   LeftHand.UmeJointAngles[22]  — radians, UMeTrack order
    ///   RightHand.UmeJointAngles[22] — radians, UMeTrack order
    /// </summary>
    public class AuraXRInferenceManager : MonoBehaviour
    {
        [Header("ONNX Models")]
        [Tooltip("Drag auraxr_lstm_right.onnx here")]
        public ModelAsset rightModelAsset;
        [Tooltip("Drag auraxr_lstm_left.onnx here")]
        public ModelAsset leftModelAsset;

        [Header("Model Meta (JSON TextAssets)")]
        [Tooltip("Drag model_meta_lstm_right.json here")]
        public TextAsset rightMetaJson;
        [Tooltip("Drag model_meta_lstm_left.json here")]
        public TextAsset leftMetaJson;

        [Header("Feature Assembler")]
        [Tooltip("Provides controller transforms and nearest object data")]
        public AuraXRFeatureAssembler featureAssembler;

        [Header("Virtual Hand Anchors")]
        [Tooltip("Root Transform of the left virtual hand")]
        public Transform virtualHandLeft;
        [Tooltip("Root Transform of the right virtual hand")]
        public Transform virtualHandRight;

        [Header("Hand Mesh Pivot Offset")]
        [Tooltip("Controller-local offset from controller transform to the virtual wrist. " +
                 "Default (0,0,0) = wrist locked AT controller position (recommended — no drift on rotation). " +
                 "Use the Calibrate Pivot button below to auto-measure if you want offset from real OVR wrist bone.")]
        public Vector3 handPivotOffset = Vector3.zero;

        [Tooltip("If true, wrist position is ALWAYS at controller.position (offset ignored). " +
                 "Recommended for stable 1:1 lock when controllers rotate.")]
        public bool lockWristToController = true;

        [Tooltip("If true, virtual hand root rotation follows the physical controller exactly. " +
                 "Keep enabled for 1:1 controller/hand alignment; disable only when evaluating " +
                 "model-predicted wrist orientation.")]
        public bool lockWristRotationToController = true;

        [Header("Controller → Hand Visual Calibration")]
        [Tooltip("Controller-local wrist offset for the left virtual hand. This positions the " +
                 "hand mesh anatomically around the physical controller while the controller " +
                 "anchor remains the tracking source.")]
        public Vector3 leftVisualWristOffset = new Vector3(0.1685f, 0f, 0.0351f);

        [Tooltip("Controller-local wrist offset for the right virtual hand.")]
        public Vector3 rightVisualWristOffset = new Vector3(0.1685f, 0f, 0.0351f);

        [Tooltip("Extra controller-local Euler rotation for the left hand mesh after tracking rotation.")]
        public Vector3 leftVisualEulerOffset = Vector3.zero;

        [Tooltip("Extra controller-local Euler rotation for the right hand mesh after tracking rotation.")]
        public Vector3 rightVisualEulerOffset = Vector3.zero;

        [Header("Inference Rate")]
        [Tooltip("Run inference every N frames to match ~30 FPS training rate at 72 Hz")]
        public int inferenceEveryNFrames = 2;

        [Header("Temporal Smoothing")]
        [Range(0f, 1f)]
        [Tooltip("EMA alpha for joint angles (0=frozen, 1=no smoothing)")]
        public float emaAlpha = 0.35f;
        [Range(0f, 1f)]
        [Tooltip("EMA alpha for wrist rotation — lower = smoother, less jerk near vertical approach")]
        public float rotEmaAlpha = 0.25f;

        // -----------------------------------------------------------------------
        // Input Sanitization (Madde 1.8) — domain-gap defense
        // -----------------------------------------------------------------------
        [Header("Input Sanitization — Domain Gap Defense")]
        [Tooltip("Clamp approach_speed feature to ±this value (m/s). HOT3D training " +
                 "distribution rarely exceeds ±2 m/s; Unity controller jitter at 72Hz can " +
                 "spike to ±10 m/s and push the model out of distribution.")]
        [Range(0.5f, 10f)]
        public float approachSpeedClampMs = 2.0f;

        [Range(0f, 1f)]
        [Tooltip("Per-frame EMA smoothing on the approach_speed input BEFORE feeding the model. " +
                 "0 = use raw; 0.3 = moderate smoothing; 1 = always zero (use only direction).")]
        public float approachSpeedInputEma = 0.3f;

        [Tooltip("If the nearest object's world rotation is within this tolerance of identity, " +
                 "fall back to dir_world for dir_obj_local (avoids garbage when scene meshes have " +
                 "no meaningful canonical orientation).")]
        public bool dirObjLocalIdentityGuard = true;

        [Range(0.01f, 0.2f)]
        [Tooltip("Width of the distance hysteresis band around the 40cm threshold. " +
                 "Inside this band the model output is lerp'd toward a neutral pose, " +
                 "preventing snap at the activation threshold. 0.05 = 35–45cm soft fade.")]
        public float distanceHysteresisM = 0.05f;

        [Range(0.30f, 0.50f)]
        [Tooltip("Outer distance at which model contribution = 0 (snap line).")]
        public float distanceOuterM = 0.40f;

        [Header("One-Euro Filter (optional)")]
        [Tooltip("Use One-Euro filter on joint angles instead of plain EMA. " +
                 "Better at preserving fast motion while smoothing slow drift. Disabled by default — toggle and tune.")]
        public bool useOneEuroFilter = false;
        [Range(0.1f, 5f)]
        public float oneEuroMinCutoffHz = 1.0f;
        [Range(0.0f, 1f)]
        public float oneEuroBeta = 0.007f;

        [Header("SDF-LSTM Model v4 (Temporal + Geometry)")]
        [Tooltip("Use SDF-LSTM model (requires auraxr_lstm_*.onnx + ObjectSDFDatabase).")]
        public bool useLSTMModel = true;
        [Tooltip("Use one canonical right-hand LSTM for both hands. Left hand features/outputs are mirrored at runtime.")]
        public bool useSharedCanonicalLSTM = false;
        [Tooltip("Drag auraxr_lstm_shared.onnx here. Used for both hands when useSharedCanonicalLSTM is enabled.")]
        public ModelAsset lstmSharedModelAsset;
        [Tooltip("Drag model_meta_lstm_shared.json here. Used for both hands when useSharedCanonicalLSTM is enabled.")]
        public TextAsset lstmSharedMetaJson;
        [Tooltip("Drag auraxr_lstm_right.onnx here")]
        public ModelAsset lstmRightModelAsset;
        [Tooltip("Drag auraxr_lstm_left.onnx here")]
        public ModelAsset lstmLeftModelAsset;
        [Tooltip("Drag model_meta_lstm_right.json here")]
        public TextAsset lstmRightMetaJson;
        [Tooltip("Drag model_meta_lstm_left.json here")]
        public TextAsset lstmLeftMetaJson;
        [Tooltip("SDF PCA embedding database — auto-found in scene if null")]
        public ObjectSDFDatabase sdfDatabase;
        [Tooltip("SDF grid database for exact trilinear SDF queries — auto-found if null")]
        public SDFGridDatabase sdfGridDatabase;

        [Header("Debug")]
        [Tooltip("E0 baseline: keep both hands in a static neutral/open pose while controller roots still move.")]
        public bool debugStaticNeutralPose = false;
        public bool debugBypassModel = false;

        // -----------------------------------------------------------------------
        // Public output — read by HandRigController each frame
        // -----------------------------------------------------------------------
        public HandPose LeftHand  { get; set; } = new HandPose();
        public HandPose RightHand { get; set; } = new HandPose();

        // -----------------------------------------------------------------------
        // Runtime state
        // -----------------------------------------------------------------------
        private Worker _workerRight, _workerLeft;

        // Normalization stats (15 feature dims, 22 target dims, 6 wrist rot dims)
        private float[] _featMeanRight, _featStdRight;
        private float[] _tgtMeanRight,  _tgtStdRight;
        private float[] _rotMeanRight,  _rotStdRight;

        private float[] _featMeanLeft,  _featStdLeft;
        private float[] _tgtMeanLeft,   _tgtStdLeft;
        private float[] _rotMeanLeft,   _rotStdLeft;

        // EMA smoothed MANO PCA components (15 per hand)
        private float[] _smoothRight = new float[15];
        private float[] _smoothLeft  = new float[15];
        private bool    _firstRight  = true, _firstLeft = true;

        private int _frameCounter;

        // Previous wrist positions and timestamps for approach_speed computation
        private Vector3 _prevWristRight = Vector3.zero;
        private Vector3 _prevWristLeft  = Vector3.zero;
        private float   _prevTimeRight  = -1f;
        private float   _prevTimeLeft   = -1f;

        // Last known approach direction per hand — used as fallback when no object is present
        // so the model receives a stable direction instead of always (0,0,1).
        private Vector3 _lastDirRight = new Vector3(0f, 0f, 1f);
        private Vector3 _lastDirLeft  = new Vector3(0f, 0f, 1f);

        // ── Input sanitization (1.8) buffers ────────────────────────────────
        // EMA-smoothed approach_speed input per hand
        private float _smoothApproachSpeedRight = 0f;
        private float _smoothApproachSpeedLeft  = 0f;
        private bool  _firstApproachR = true, _firstApproachL = true;

        // ── v3 object-velocity tracking ──────────────────────────────────────
        // Previous (HOT3D-frame) position and timestamp of the nearest object per hand.
        // Used to compute obj_vel_world (feature dims [8-10]). We track only the nearest
        // object; switching objects resets velocity to zero, matching build_dataset.py.
        private Transform _prevNearestRight = null, _prevNearestLeft = null;
        private Vector3   _prevObjPosRight  = Vector3.zero, _prevObjPosLeft = Vector3.zero;
        private float     _prevObjTimeRight = -1f,         _prevObjTimeLeft = -1f;

        // One-Euro filter state per hand (per joint)
        private OneEuroFilterArray _oneEuroR;
        private OneEuroFilterArray _oneEuroL;

        // ── LSTM v4 state ─────────────────────────────────────────────────────
        private Worker _lstmWorkerRight, _lstmWorkerLeft;
        private const int LSTM_LAYERS = 2, LSTM_HIDDEN = 256;
        private const int LSTM_STATE_TOTAL = LSTM_LAYERS * 1 * LSTM_HIDDEN;  // 512
        private static readonly float[] FeatureMirrorMask = {
            -1f, 1f, 1f, -1f, 1f, 1f, 1f, 1f, -1f, 1f, 1f,
            -1f, 1f, 1f, -1f, 1f, 1f, 1f, 1f, 1f, 1f, 1f, 1f, 1f, 1f
        };
        // MANO PCA mirror mask: negate odd-indexed components (abduction-like axes)
        // Matches Python build_dataset_mano._default_mirror_mask()
        private static readonly float[] ManoPcaMirrorMask = {
            1f, -1f, 1f, -1f, 1f, -1f, 1f, -1f, 1f, -1f, 1f, -1f, 1f, -1f, 1f
        };
        private static readonly float[] WristRotMirrorMask = { -1f, 1f, 1f, -1f, 1f, 1f };
        // h and c tensors for each hand: shape (2, 1, 256) stored flat
        private float[] _hRight = new float[LSTM_STATE_TOTAL];
        private float[] _cRight = new float[LSTM_STATE_TOTAL];
        private float[] _hLeft  = new float[LSTM_STATE_TOTAL];
        private float[] _cLeft  = new float[LSTM_STATE_TOTAL];
        // Track Approach phase to reset state on transition
        private bool _approachActiveRight = false, _approachActiveLeft = false;
        // SDF feature norm stats (4-dim: sdf_value + gradient_xyz)
        private float[] _sdfMeanRight = new float[4], _sdfStdRight = { 1f, 1f, 1f, 1f };
        private float[] _sdfMeanLeft  = new float[4], _sdfStdLeft  = { 1f, 1f, 1f, 1f };

        // Neutral pose for hysteresis fade: all-zero MANO PCA → open-hand
        private float[] _neutralPose15 = new float[15];   // all zeros → open-hand

        // ── Pivot Calibration ────────────────────────────────────────────────
        [Header("Pivot Calibration")]
        [Tooltip("Check to start a 5-second pivot calibration session. Uncheck to stop early.")]
        public bool calibratePivotNow = false;
        [Tooltip("How many frames to collect before averaging (at 60 Hz: 300 = 5 sec)")]
        public int calibrationFrames = 300;
        [Tooltip("Last calibration result (read-only, shown for reference)")]
        public Vector3 lastCalibratedOffset = Vector3.zero;

        private readonly System.Collections.Generic.List<Vector3> _pivotSamplesR = new();
        private readonly System.Collections.Generic.List<Vector3> _pivotSamplesL = new();
        private bool _calibrationActive = false;

        // Model-predicted wrist rotations in Unity frame — updated every N frames, EMA smoothed
        private Quaternion _predictedRotRight = Quaternion.identity;
        private Quaternion _predictedRotLeft  = Quaternion.identity;
        private bool       _firstRotRight     = true;
        private bool       _firstRotLeft      = true;

        // File-based debug log
        private string _debugLogPath;
        private System.IO.StreamWriter _debugWriter;

        // MANO PCA: model now outputs 15-dim PCA directly — no UMeTrack remapping needed
        // ManoJointAngles[0..14] = pose_pca[0..14] (denormalized)
        private static readonly int[] UmeFingerStart = { 0, 3, 6, 9, 12 };  // per-finger start in 15-dim

        // -----------------------------------------------------------------------
        // BOP object → grip category + bbox (grip_categories.py — must stay in sync)
        // -----------------------------------------------------------------------
        private static readonly Dictionary<int, int> BopToGrip = new Dictionary<int, int>
        {
            { 1,3},  // Pinch:     holder_black
            { 2,0},  // Power:     bowl
            { 3,2},  // Palmar:    plate_bamboo
            { 4,1},  // Precision: spoon_wooden
            { 5,1},  // Precision: potato_masher
            { 6,1},  // Precision: spatula_red
            { 7,0},  // Power:     coffee_pot
            { 8,0},  // Power:     mug_patterned
            { 9,0},  // Power:     mug_white
            {10,0},  // Power:     can_soup
            {11,0},  // Power:     can_parmesan
            {12,0},  // Power:     can_tomato_sauce
            {13,0},  // Power:     bottle_mustard
            {14,0},  // Power:     bottle_bbq
            {15,0},  // Power:     bottle_ranch
            {16,0},  // Power:     vase
            {17,0},  // Power:     carton_milk
            {18,0},  // Power:     carton_oj
            {19,0},  // Power:     flask
            {20,2},  // Palmar:    food_waffles
            {21,0},  // Power:     food_vegetables
            {22,0},  // Power:     dumbbell_5lb
            {23,3},  // Pinch:     aria_small
            {24,2},  // Palmar:    cellphone
            {25,3},  // Pinch:     holder_gray
            {26,0},  // Power:     birdhouse_toy
            {27,3},  // Pinch:     dino_toy
            {28,2},  // Palmar:    keyboard
            {29,2},  // Palmar:    whiteboard_eraser
            {30,3},  // Pinch:     puzzle_toy
            {31,3},  // Pinch:     mouse
            {32,1},  // Precision: whiteboard_marker
            {33,2},  // Palmar:    dvd_remote
        };

        // Half-extents (x,y,z) in metres
        private static readonly Dictionary<int, float[]> BopToBbox = new Dictionary<int, float[]>
        {
            { 1, new float[]{0.030f,0.075f,0.030f}},  // holder_black
            { 2, new float[]{0.100f,0.040f,0.100f}},  // bowl
            { 3, new float[]{0.130f,0.010f,0.130f}},  // plate_bamboo
            { 4, new float[]{0.015f,0.010f,0.150f}},  // spoon_wooden
            { 5, new float[]{0.045f,0.010f,0.125f}},  // potato_masher
            { 6, new float[]{0.035f,0.005f,0.140f}},  // spatula_red
            { 7, new float[]{0.075f,0.125f,0.060f}},  // coffee_pot
            { 8, new float[]{0.045f,0.050f,0.045f}},  // mug_patterned
            { 9, new float[]{0.045f,0.050f,0.045f}},  // mug_white
            {10, new float[]{0.038f,0.050f,0.038f}},  // can_soup
            {11, new float[]{0.045f,0.075f,0.045f}},  // can_parmesan
            {12, new float[]{0.038f,0.055f,0.038f}},  // can_tomato_sauce
            {13, new float[]{0.035f,0.100f,0.035f}},  // bottle_mustard
            {14, new float[]{0.035f,0.140f,0.035f}},  // bottle_bbq
            {15, new float[]{0.030f,0.140f,0.030f}},  // bottle_ranch
            {16, new float[]{0.060f,0.150f,0.060f}},  // vase
            {17, new float[]{0.045f,0.105f,0.045f}},  // carton_milk
            {18, new float[]{0.045f,0.100f,0.045f}},  // carton_oj
            {19, new float[]{0.035f,0.090f,0.035f}},  // flask
            {20, new float[]{0.100f,0.020f,0.075f}},  // food_waffles
            {21, new float[]{0.050f,0.060f,0.050f}},  // food_vegetables
            {22, new float[]{0.125f,0.050f,0.050f}},  // dumbbell_5lb
            {23, new float[]{0.075f,0.025f,0.040f}},  // aria_small
            {24, new float[]{0.040f,0.005f,0.080f}},  // cellphone
            {25, new float[]{0.030f,0.075f,0.030f}},  // holder_gray
            {26, new float[]{0.075f,0.090f,0.075f}},  // birdhouse_toy
            {27, new float[]{0.060f,0.050f,0.030f}},  // dino_toy
            {28, new float[]{0.175f,0.015f,0.075f}},  // keyboard
            {29, new float[]{0.060f,0.020f,0.030f}},  // whiteboard_eraser
            {30, new float[]{0.075f,0.020f,0.075f}},  // puzzle_toy
            {31, new float[]{0.030f,0.020f,0.050f}},  // mouse
            {32, new float[]{0.008f,0.008f,0.070f}},  // whiteboard_marker
            {33, new float[]{0.025f,0.008f,0.090f}},  // dvd_remote
        };

        private static readonly float[] DefaultBbox = new float[] { 0.03f, 0.03f, 0.08f };

        // -----------------------------------------------------------------------
        // Unity lifecycle
        // -----------------------------------------------------------------------
        void Start()
        {
            string logFolder = System.IO.Path.Combine(Application.persistentDataPath, "Logs");
            System.IO.Directory.CreateDirectory(logFolder);
            _debugLogPath = System.IO.Path.Combine(logFolder, $"auraxr_debug_{System.DateTime.Now:yyyy_MM_dd_HH_mm_ss}.txt");
            _debugWriter = new System.IO.StreamWriter(_debugLogPath, append: false) { AutoFlush = true };
            DLog($"=== AuraXRInferenceManager debug log started ===");
            DLog($"persistentDataPath: {Application.persistentDataPath}");
            Debug.Log($"[AuraXR] Debug log → {_debugLogPath}");

            // Restore previously calibrated pivot offset (if any)
            if (PlayerPrefs.HasKey("AuraXR_PivotX"))
            {
                handPivotOffset = new Vector3(
                    PlayerPrefs.GetFloat("AuraXR_PivotX"),
                    PlayerPrefs.GetFloat("AuraXR_PivotY"),
                    PlayerPrefs.GetFloat("AuraXR_PivotZ"));
                lastCalibratedOffset = handPivotOffset;
                DLog($"[CALIB] Loaded saved pivot offset = {handPivotOffset:F4}");
            }

            if (rightModelAsset == null) rightModelAsset = Resources.Load<ModelAsset>("auraxr_lstm_right");
            if (leftModelAsset  == null) leftModelAsset  = Resources.Load<ModelAsset>("auraxr_lstm_left");
            if (rightMetaJson   == null) rightMetaJson   = Resources.Load<TextAsset>("model_meta_lstm_right");
            if (leftMetaJson    == null) leftMetaJson    = Resources.Load<TextAsset>("model_meta_lstm_left");

            if (featureAssembler == null)
                featureAssembler = FindAnyObjectByType<AuraXRFeatureAssembler>();

            LoadModel(rightModelAsset, rightMetaJson,
                out _workerRight,
                out _featMeanRight, out _featStdRight,
                out _tgtMeanRight,  out _tgtStdRight,
                out _rotMeanRight,  out _rotStdRight,
                "right");

            LoadModel(leftModelAsset, leftMetaJson,
                out _workerLeft,
                out _featMeanLeft, out _featStdLeft,
                out _tgtMeanLeft,  out _tgtStdLeft,
                out _rotMeanLeft,  out _rotStdLeft,
                "left");

            if (useLSTMModel)
            {
                if (sdfDatabase == null)
                    sdfDatabase = FindAnyObjectByType<ObjectSDFDatabase>();
                if (sdfDatabase == null)
                    Debug.LogWarning("[AuraXR] LSTM mode: ObjectSDFDatabase not found — SDF embeddings will be zero.");

                if (sdfGridDatabase == null)
                    sdfGridDatabase = FindAnyObjectByType<SDFGridDatabase>();
                if (sdfGridDatabase == null)
                    Debug.LogWarning("[AuraXR] LSTM mode: SDFGridDatabase not found — falling back to bbox SDF approximation.");

                // Auto-load from Resources if not assigned in Inspector
                if (lstmRightModelAsset == null) lstmRightModelAsset = Resources.Load<ModelAsset>("auraxr_lstm_right");
                if (lstmLeftModelAsset  == null) lstmLeftModelAsset  = Resources.Load<ModelAsset>("auraxr_lstm_left");
                if (lstmRightMetaJson   == null) lstmRightMetaJson   = Resources.Load<TextAsset>("model_meta_lstm_right");
                if (lstmLeftMetaJson    == null) lstmLeftMetaJson    = Resources.Load<TextAsset>("model_meta_lstm_left");

                if (useSharedCanonicalLSTM)
                {
                    if (lstmSharedModelAsset == null) lstmSharedModelAsset = Resources.Load<ModelAsset>("auraxr_lstm_shared");
                    if (lstmSharedMetaJson   == null) lstmSharedMetaJson   = Resources.Load<TextAsset>("model_meta_lstm_shared");

                    LoadLSTMModel(lstmSharedModelAsset, lstmSharedMetaJson,
                        out _lstmWorkerRight,
                        ref _featMeanRight, ref _featStdRight,
                        ref _tgtMeanRight,  ref _tgtStdRight,
                        ref _rotMeanRight,  ref _rotStdRight,
                        ref _sdfMeanRight,  ref _sdfStdRight,
                        "lstm_shared");
                    _lstmWorkerLeft = _lstmWorkerRight;
                    _featMeanLeft = _featMeanRight; _featStdLeft = _featStdRight;
                    _tgtMeanLeft  = _tgtMeanRight;  _tgtStdLeft  = _tgtStdRight;
                    _rotMeanLeft  = _rotMeanRight;  _rotStdLeft  = _rotStdRight;
                    _sdfMeanLeft  = _sdfMeanRight;  _sdfStdLeft  = _sdfStdRight;
                }
                else
                {
                    LoadLSTMModel(lstmRightModelAsset, lstmRightMetaJson,
                        out _lstmWorkerRight,
                        ref _featMeanRight, ref _featStdRight,
                        ref _tgtMeanRight,  ref _tgtStdRight,
                        ref _rotMeanRight,  ref _rotStdRight,
                        ref _sdfMeanRight,  ref _sdfStdRight,
                        "lstm_right");
                    LoadLSTMModel(lstmLeftModelAsset, lstmLeftMetaJson,
                        out _lstmWorkerLeft,
                        ref _featMeanLeft, ref _featStdLeft,
                        ref _tgtMeanLeft,  ref _tgtStdLeft,
                        ref _rotMeanLeft,  ref _rotStdLeft,
                        ref _sdfMeanLeft,  ref _sdfStdLeft,
                        "lstm_left");
                }
            }
        }

        void Update()
        {
            Transform rightCtrl = featureAssembler?.rightControllerTransform;
            Transform leftCtrl  = featureAssembler?.leftControllerTransform;

            Transform nearestR  = featureAssembler?.nearestObjectRight;
            Transform nearestL  = featureAssembler?.nearestObjectLeft;
            int       categoryR = featureAssembler?.nearestObjectCategoryRight ?? 0;
            int       categoryL = featureAssembler?.nearestObjectCategoryLeft  ?? 0;

            // Rotation source:
            //   - Default: controller rotation for true 1:1 visual alignment.
            //   - Optional research mode: model-predicted wrist orientation near objects.
            //     The predicted path is useful for experiments, but it makes the hand root
            //     visibly diverge from the physical controller during approach.
            if (nearestR == null && rightCtrl != null) _predictedRotRight = rightCtrl.rotation;
            if (nearestL == null && leftCtrl  != null) _predictedRotLeft  = leftCtrl.rotation;

            Quaternion rootRotR = lockWristRotationToController && rightCtrl != null ? rightCtrl.rotation : _predictedRotRight;
            Quaternion rootRotL = lockWristRotationToController && leftCtrl  != null ? leftCtrl.rotation  : _predictedRotLeft;
            rootRotR *= Quaternion.Euler(rightVisualEulerOffset);
            rootRotL *= Quaternion.Euler(leftVisualEulerOffset);

            // Position: keep the actual visual wrist at controller.position.
            // The hand prefab wrist bone is offset under the root by roughly
            // -rightVisualWristOffset / -leftVisualWristOffset, so root=controller
            // leaves the visible wrist floating away. Apply the inverse offset in the
            // same rotation frame as the hand root, not the controller, so model-driven
            // palm rotation pivots around the locked controller/wrist point.
            Vector3 handPosR = rightCtrl != null
                ? (lockWristToController ? rightCtrl.position + rootRotR * rightVisualWristOffset
                                         : rightCtrl.position + rightCtrl.rotation * handPivotOffset)
                : Vector3.zero;
            Vector3 handPosL = leftCtrl != null
                ? (lockWristToController ? leftCtrl.position + rootRotL * leftVisualWristOffset
                                         : leftCtrl.position + leftCtrl.rotation * handPivotOffset)
                : Vector3.zero;
            if (virtualHandRight != null && rightCtrl != null)
                virtualHandRight.SetPositionAndRotation(handPosR, rootRotR);
            if (virtualHandLeft != null && leftCtrl != null)
                virtualHandLeft.SetPositionAndRotation(handPosL, rootRotL);

            // Start/stop calibration session when flag is toggled in Inspector
            if (calibratePivotNow && !_calibrationActive)
            {
                _calibrationActive = true;
                _pivotSamplesR.Clear();
                _pivotSamplesL.Clear();
                DLog($"[CALIB] Pivot calibration started. Collecting {calibrationFrames} frames…");
            }
            else if (!calibratePivotNow && _calibrationActive)
            {
                _calibrationActive = false;
                DLog("[CALIB] Calibration cancelled.");
            }

            if (_calibrationActive)
                CollectPivotSamples(rightCtrl, virtualHandRight, leftCtrl, virtualHandLeft);

            if (Time.frameCount % 90 == 0) LogWristOffset(rightCtrl, virtualHandRight, "R");
            if (Time.frameCount % 90 == 5) LogWristOffset(leftCtrl,  virtualHandLeft,  "L");

            _frameCounter++;
            if (_frameCounter % inferenceEveryNFrames != 0) return;

            if (debugStaticNeutralPose)
            {
                RightHand = new HandPose();
                LeftHand  = new HandPose();
                if (_frameCounter % 120 == 0)
                    DLog("[BASELINE] debugStaticNeutralPose=true — static neutral hands. Disable to use learned inference.");
                return;
            }

            if (debugBypassModel)
            {
                var testR = new HandPose();
                var testL = new HandPose();
                for (int i = 0; i < 15; i++) { testR.ManoJointAngles[i] = 0.5f; testL.ManoJointAngles[i] = 0.5f; }
                RightHand = testR;
                LeftHand  = testL;
                if (_frameCounter % 120 == 0)
                    DLog("[BYPASS] debugBypassModel=true — all joints 0.5 rad. Disable to use inference.");
                return;
            }

            if (_frameCounter % 60 == 0)
                DLog($"[WIRE] nearestR={(nearestR?.name ?? "NULL")} cat={categoryR}  " +
                     $"nearestL={(nearestL?.name ?? "NULL")} cat={categoryL}  " +
                     $"rightCtrl={(rightCtrl?.name ?? "NULL")}  leftCtrl={(leftCtrl?.name ?? "NULL")}");

            if (useLSTMModel)
            {
                RightHand = RunLSTMInference(
                    rightCtrl, nearestR, categoryR,
                    _lstmWorkerRight,
                    _featMeanRight, _featStdRight, _tgtMeanRight, _tgtStdRight,
                    _rotMeanRight, _rotStdRight, _sdfMeanRight, _sdfStdRight,
                    ref _hRight, ref _cRight, ref _approachActiveRight,
                    ref _smoothRight, ref _firstRight,
                    ref _prevWristRight, ref _prevTimeRight,
                    ref _predictedRotRight, ref _firstRotRight,
                    ref _smoothApproachSpeedRight, ref _firstApproachR,
                    ref _oneEuroR, ref _lastDirRight,
                    ref _prevNearestRight, ref _prevObjPosRight, ref _prevObjTimeRight,
                    false,
                    RightHand);
                LeftHand = RunLSTMInference(
                    leftCtrl, nearestL, categoryL,
                    _lstmWorkerLeft,
                    _featMeanLeft, _featStdLeft, _tgtMeanLeft, _tgtStdLeft,
                    _rotMeanLeft, _rotStdLeft, _sdfMeanLeft, _sdfStdLeft,
                    ref _hLeft, ref _cLeft, ref _approachActiveLeft,
                    ref _smoothLeft, ref _firstLeft,
                    ref _prevWristLeft, ref _prevTimeLeft,
                    ref _predictedRotLeft, ref _firstRotLeft,
                    ref _smoothApproachSpeedLeft, ref _firstApproachL,
                    ref _oneEuroL, ref _lastDirLeft,
                    ref _prevNearestLeft, ref _prevObjPosLeft, ref _prevObjTimeLeft,
                    useSharedCanonicalLSTM,
                    LeftHand);
            }
            else
            {
            RightHand = RunInference(
                rightCtrl, nearestR, categoryR,
                _workerRight,
                _featMeanRight, _featStdRight, _tgtMeanRight, _tgtStdRight,
                _rotMeanRight, _rotStdRight,
                ref _smoothRight, ref _firstRight,
                ref _prevWristRight, ref _prevTimeRight,
                ref _predictedRotRight, ref _firstRotRight,
                ref _smoothApproachSpeedRight, ref _firstApproachR,
                ref _oneEuroR, ref _lastDirRight,
                ref _prevNearestRight, ref _prevObjPosRight, ref _prevObjTimeRight,
                RightHand);

            LeftHand = RunInference(
                leftCtrl, nearestL, categoryL,
                _workerLeft,
                _featMeanLeft, _featStdLeft, _tgtMeanLeft, _tgtStdLeft,
                _rotMeanLeft, _rotStdLeft,
                ref _smoothLeft, ref _firstLeft,
                ref _prevWristLeft, ref _prevTimeLeft,
                ref _predictedRotLeft, ref _firstRotLeft,
                ref _smoothApproachSpeedLeft, ref _firstApproachL,
                ref _oneEuroL, ref _lastDirLeft,
                ref _prevNearestLeft, ref _prevObjPosLeft, ref _prevObjTimeLeft,
                LeftHand);
            }
        }

        void OnDestroy()
        {
            _workerRight?.Dispose();
            _workerLeft?.Dispose();
            _lstmWorkerRight?.Dispose();
            if (!ReferenceEquals(_lstmWorkerLeft, _lstmWorkerRight))
                _lstmWorkerLeft?.Dispose();
            _debugWriter?.Close();
        }

        private void LogWristOffset(Transform ctrl, Transform handRoot, string side)
        {
            if (ctrl == null || handRoot == null) return;
            Vector3 ctrlPos  = ctrl.position;
            Vector3 rootPos  = handRoot.position;
            Vector3 wristWorld = rootPos;
            string  wristSrc   = "handRoot";
            var ovrSkel = handRoot.GetComponentInChildren<OVRSkeleton>();
            if (ovrSkel != null && ovrSkel.Bones != null && ovrSkel.Bones.Count > 0 && ovrSkel.Bones[0].Transform != null)
            {
                wristWorld = ovrSkel.Bones[0].Transform.position;
                wristSrc   = $"OVRSkel.bones[0]({ovrSkel.Bones[0].Id})";
            }
            else if (handRoot.childCount > 0)
            {
                wristWorld = handRoot.GetChild(0).position;
                wristSrc   = $"child[0]({handRoot.GetChild(0).name})";
            }
            Vector3 deltaWorld = wristWorld - ctrlPos;
            Vector3 deltaLocal = Quaternion.Inverse(ctrl.rotation) * deltaWorld;
            DLog($"[PIVOT|{side}] ctrl={ctrlPos:F4}  handRoot={rootPos:F4}  " +
                 $"wrist({wristSrc})={wristWorld:F4}  " +
                 $"delta_world={deltaWorld:F4}  delta_local={deltaLocal:F4}  dist={deltaWorld.magnitude*100f:F1}cm  " +
                 $"currentOffset={handPivotOffset:F4}");
        }

        // ── Pivot calibration helpers ────────────────────────────────────────

        /// Returns the controller-local wrist delta for a hand, or Vector3.zero if unavailable.
        private static Vector3 MeasureDeltaLocal(Transform ctrl, Transform handRoot)
        {
            if (ctrl == null || handRoot == null) return Vector3.zero;
            Vector3 wristWorld = handRoot.position;
            var ovrSkel = handRoot.GetComponentInChildren<OVRSkeleton>();
            if (ovrSkel != null && ovrSkel.Bones != null && ovrSkel.Bones.Count > 0
                && ovrSkel.Bones[0].Transform != null)
                wristWorld = ovrSkel.Bones[0].Transform.position;
            else if (handRoot.childCount > 0)
                wristWorld = handRoot.GetChild(0).position;
            return Quaternion.Inverse(ctrl.rotation) * (wristWorld - ctrl.position);
        }

        private void CollectPivotSamples(Transform rCtrl, Transform rRoot,
                                         Transform lCtrl, Transform lRoot)
        {
            Vector3 dR = MeasureDeltaLocal(rCtrl, rRoot);
            Vector3 dL = MeasureDeltaLocal(lCtrl, lRoot);
            if (dR != Vector3.zero) _pivotSamplesR.Add(dR);
            if (dL != Vector3.zero) _pivotSamplesL.Add(dL);

            int target = calibrationFrames;
            int collected = Mathf.Min(_pivotSamplesR.Count, _pivotSamplesL.Count > 0 ? _pivotSamplesL.Count : _pivotSamplesR.Count);
            if (Time.frameCount % 60 == 0)
                DLog($"[CALIB] Collecting… {collected}/{target} samples");

            if (_pivotSamplesR.Count >= target || _pivotSamplesL.Count >= target)
                FinishCalibration();
        }

        private void FinishCalibration()
        {
            _calibrationActive = false;
            calibratePivotNow  = false;

            Vector3 avgR = Vector3.zero, avgL = Vector3.zero;
            if (_pivotSamplesR.Count > 0)
            {
                foreach (var s in _pivotSamplesR) avgR += s;
                avgR /= _pivotSamplesR.Count;
            }
            if (_pivotSamplesL.Count > 0)
            {
                foreach (var s in _pivotSamplesL) avgL += s;
                avgL /= _pivotSamplesL.Count;
            }

            // Use whichever hand had more samples; prefer average of both
            Vector3 newOffset = Vector3.zero;
            if (_pivotSamplesR.Count > 0 && _pivotSamplesL.Count > 0)
                newOffset = (avgR + avgL) * 0.5f;
            else if (_pivotSamplesR.Count > 0)
                newOffset = avgR;
            else if (_pivotSamplesL.Count > 0)
                newOffset = avgL;

            lastCalibratedOffset = newOffset;
            handPivotOffset      = newOffset;

            // Persist so the value survives Play/Stop cycles
            PlayerPrefs.SetFloat("AuraXR_PivotX", newOffset.x);
            PlayerPrefs.SetFloat("AuraXR_PivotY", newOffset.y);
            PlayerPrefs.SetFloat("AuraXR_PivotZ", newOffset.z);
            PlayerPrefs.Save();

            DLog($"[CALIB] ✓ Done. R={_pivotSamplesR.Count} samples avg={avgR:F4}  " +
                 $"L={_pivotSamplesL.Count} samples avg={avgL:F4}");
            DLog($"[CALIB] → new handPivotOffset = {newOffset:F4}  (saved to PlayerPrefs)");
            Debug.Log($"[AuraXR] Pivot calibration complete → handPivotOffset = {newOffset:F4}");
        }

        private void DLog(string msg)
        {
            string line = $"[{Time.frameCount:D6}] {msg}";
            _debugWriter?.WriteLine(line);
            Debug.Log($"[AuraXR] {msg}");
        }

        // -----------------------------------------------------------------------
        // Inference for one hand
        // -----------------------------------------------------------------------
        private HandPose RunInference(
            Transform ctrl, Transform nearestObj, int categoryId,
            Worker worker,
            float[] featMean, float[] featStd,
            float[] tgtMean,  float[] tgtStd,
            float[] rotMean,  float[] rotStd,
            ref float[]    smooth,     ref bool      firstFrame,
            ref Vector3    prevWristH, ref float     prevTime,
            ref Quaternion predictedRot,
            ref bool       firstRot,
            ref float      smoothApproachSpeed, ref bool firstApproach,
            ref OneEuroFilterArray oneEuro, ref Vector3 lastDir,
            ref Transform  prevNearest, ref Vector3 prevObjPos, ref float prevObjTime,
            HandPose current)
        {
            if (ctrl == null || worker == null)
            {
                if (Time.frameCount % 60 == 0)
                    DLog($"[SKIP] ctrl={(ctrl?.name ?? "NULL")}  worker={worker != null}");
                return current;
            }

            bool doLog = (Time.frameCount % 60 == 0);

            // 1. Positions and world-frame direction in HOT3D coordinate space
            Vector3 wristPosH = ToHOT3D(ctrl.position);

            // No-object fallback: run model with a synthetic far-away object so inference
            // is never skipped. At dist=2m the model reliably outputs an open-hand pose
            // (consistent with training distribution where far distance → no grip intent).
            Vector3 dirWorld;
            Vector3 dirObjLoc;
            float   dist;

            if (nearestObj == null)
            {
                // No object in range — run with synthetic far feature so the model still
                // produces an open-hand pose instead of freezing at the last inference.
                dist      = 2.0f;
                dirWorld  = lastDir;   // keep last known direction for smooth feature transition
                dirObjLoc = lastDir;
                categoryId = 0;

                if (doLog) DLog($"[NO_OBJ] fallback dist=2.0 dirWorld={lastDir:F2} → open-hand pose");
            }
            else
            {
                Vector3 objPosH  = ToHOT3D(nearestObj.position);
                Vector3 relWorld = objPosH - wristPosH;
                dist = relWorld.magnitude;

                if (dist < 1e-6f) return current;

                dirWorld = relWorld / dist;
                lastDir  = dirWorld;   // remember for no-object fallback

                // 2. Object-local direction
                // (1.8) Identity-guard: fall back to dir_world when object rotation is
                // near-identity (no meaningful canonical orientation).
                Quaternion objRotH = ToHOT3DQuat(nearestObj.rotation);
                if (dirObjLocalIdentityGuard && Quaternion.Angle(objRotH, Quaternion.identity) < 1f)
                    dirObjLoc = dirWorld;
                else
                    dirObjLoc = Quaternion.Inverse(objRotH) * dirWorld;
            }

            // 3. Approach speed (wrist velocity dot approach direction)
            //    (1.8) clamped to training-distribution band + per-input EMA, since
            //    Unity controller velocity at 72Hz can spike well beyond the HOT3D
            //    training ±2 m/s range and push the model out of distribution.
            float now = Time.realtimeSinceStartup;
            float approachSpeedRaw = 0f;
            if (prevTime >= 0f)
            {
                float dt = now - prevTime;
                if (dt > 1e-4f)
                {
                    Vector3 velWorld = (wristPosH - prevWristH) / dt;
                    approachSpeedRaw = Vector3.Dot(velWorld, dirWorld);
                }
            }
            prevWristH = wristPosH;
            prevTime   = now;
            // Clamp to ±approachSpeedClampMs
            float approachSpeedClamped = Mathf.Clamp(approachSpeedRaw,
                                                    -approachSpeedClampMs,
                                                    +approachSpeedClampMs);
            // Per-input EMA smoothing (1.8)
            float approachSpeed;
            if (firstApproach)
            {
                smoothApproachSpeed = approachSpeedClamped;
                firstApproach = false;
                approachSpeed = smoothApproachSpeed;
            }
            else
            {
                smoothApproachSpeed = approachSpeedInputEma * approachSpeedClamped
                                    + (1f - approachSpeedInputEma) * smoothApproachSpeed;
                approachSpeed = smoothApproachSpeed;
            }

            // 4. Object linear velocity in HOT3D world frame (v3 dims [8-10]).
            //    Switching to a new nearest object resets velocity to zero — matches
            //    build_dataset.py which tracks per-(nearest-bop-id) state.
            Vector3 objVelWorld = Vector3.zero;
            if (nearestObj != null)
            {
                Vector3 objPosH = ToHOT3D(nearestObj.position);
                float   tNow    = Time.realtimeSinceStartup;
                if (prevNearest == nearestObj && prevObjTime >= 0f)
                {
                    float dtObj = tNow - prevObjTime;
                    if (dtObj > 1e-4f)
                        objVelWorld = (objPosH - prevObjPos) / dtObj;
                }
                prevNearest  = nearestObj;
                prevObjPos   = objPosH;
                prevObjTime  = tNow;
            }
            else
            {
                prevNearest  = null;
                prevObjTime  = -1f;
            }

            // 5. Wrist 6D input (v3 dims [11-16]) — canonical-relative encoding,
            //    matches Python hot3d_utils.wrist_rot_to_6d (Unity frame).
            float[] wristRotInput = ComputeWristRot6dInput(ctrl.rotation, dirWorld);

            // 6. Hand confidence (v3 dim [17]). Quest controllers don't expose an analogue,
            //    so we default to 1.0 — matches training (no filtering applied either).
            float handConf = 1.0f;

            // 7. Grip category and bbox (v3 dims [18-21] and [22-24])
            int     grip = BopToGrip.TryGetValue(categoryId, out int g) ? g : 0;
            float[] bbox = BopToBbox.TryGetValue(categoryId, out float[] b) ? b : DefaultBbox;
            string[] gripNames = { "Power", "Precision", "Palmar", "Pinch" };

            // 8. Assemble raw 25-dim v3 feature.
            const int FEAT_DIM = 25;
            float[] feat = new float[FEAT_DIM];
            feat[0]  = dirWorld.x;   feat[1]  = dirWorld.y;   feat[2]  = dirWorld.z;
            feat[3]  = dirObjLoc.x;  feat[4]  = dirObjLoc.y;  feat[5]  = dirObjLoc.z;
            feat[6]  = dist;
            feat[7]  = approachSpeed;
            feat[8]  = objVelWorld.x; feat[9] = objVelWorld.y; feat[10] = objVelWorld.z;
            feat[11] = wristRotInput[0]; feat[12] = wristRotInput[1]; feat[13] = wristRotInput[2];
            feat[14] = wristRotInput[3]; feat[15] = wristRotInput[4]; feat[16] = wristRotInput[5];
            feat[17] = handConf;
            feat[18] = grip == 0 ? 1f : 0f;
            feat[19] = grip == 1 ? 1f : 0f;
            feat[20] = grip == 2 ? 1f : 0f;
            feat[21] = grip == 3 ? 1f : 0f;
            feat[22] = bbox[0]; feat[23] = bbox[1]; feat[24] = bbox[2];

            // 9. Normalize
            for (int i = 0; i < FEAT_DIM; i++)
                feat[i] = (feat[i] - featMean[i]) / featStd[i];

            // 10. Run ONNX model (25-dim input)
            using var xTensor = new Tensor<float>(new TensorShape(1, FEAT_DIM), feat);

            worker.SetInput("x", xTensor);
            worker.Schedule();

            var anglesTensor = worker.PeekOutput("pose_pca")    as Tensor<float>;
            var rotTensor    = worker.PeekOutput("wrist_rot_6d") as Tensor<float>;
            var confTensor   = worker.PeekOutput("confidence")   as Tensor<float>;  // optional legacy

            if (anglesTensor == null)
            {
                DLog("[ERROR] PeekOutput('pose_pca') returned null.");
                return current;
            }

            using var angleCpu = anglesTensor.ReadbackAndClone();

            // 9. Denormalize 15 MANO PCA components
            float[] angles = new float[15];
            for (int i = 0; i < 15; i++)
                angles[i] = angleCpu[0, i] * tgtStd[i] + tgtMean[i];

            // 10. (1.8) Distance hysteresis blend: fade model output toward neutral pose
            //     within [outer - hysteresis, outer]. Prevents snap at 40 cm threshold.
            float fadeAlpha = 1f;
            if (distanceHysteresisM > 1e-4f)
            {
                float innerBand = distanceOuterM - distanceHysteresisM;
                fadeAlpha = Mathf.Clamp01((distanceOuterM - dist) / (distanceOuterM - innerBand));
            }
            if (fadeAlpha < 0.999f)
            {
                for (int i = 0; i < 15; i++)
                    angles[i] = Mathf.Lerp(_neutralPose15[i], angles[i], fadeAlpha);
            }

            // 11. Optional per-joint confidence (legacy output — not present in MANO models)
            float[] modelConf = new float[15];
            if (confTensor != null)
            {
                using var confCpu = confTensor.ReadbackAndClone();
                for (int i = 0; i < Mathf.Min(15, confCpu.shape[1]); i++)
                    modelConf[i] = confCpu[0, i];
            }
            else
            {
                for (int i = 0; i < 15; i++) modelConf[i] = 1f;
            }

            // Temporal smoothing — confidence-weighted EMA or One-Euro.
            if (useOneEuroFilter)
            {
                if (oneEuro == null || oneEuro.Size != 15) oneEuro = new OneEuroFilterArray(15);
                oneEuro.MinCutoffHz = oneEuroMinCutoffHz;
                oneEuro.Beta        = oneEuroBeta;
                oneEuro.Filter(angles, Time.unscaledTime, smooth);
                firstFrame = false;
            }
            else if (firstFrame)
            {
                Array.Copy(angles, smooth, 15);
                firstFrame = false;
            }
            else
            {
                for (int i = 0; i < 15; i++)
                {
                    float effAlpha = emaAlpha * Mathf.Clamp01(modelConf[i]);
                    smooth[i] = effAlpha * angles[i] + (1f - effAlpha) * smooth[i];
                }
            }

            // 12. Model outputs MANO PCA 15-dim directly — copy to HandPose
            var pose = new HandPose();
            Array.Copy(smooth, pose.ManoJointAngles, 15);
            // UmeAbductionAngles unused with MANO PCA — zero them
            Array.Clear(pose.UmeAbductionAngles, 0, 5);

            // Per-finger confidence (3 PCA components per finger: 5×3 = 15)
            for (int f = 0; f < 5; f++)
            {
                int s = UmeFingerStart[f];
                pose.FingerConfidence[f] = (modelConf[s] + modelConf[s+1] + modelConf[s+2]) / 3f;
            }

            // 13. Decode wrist rotation from 6D output → Unity Quaternion
            if (rotTensor != null && rotMean != null && rotStd != null)
            {
                using var rotCpu = rotTensor.ReadbackAndClone();

                // Denormalize 6D
                float[] rot6d = new float[6];
                for (int i = 0; i < 6; i++)
                    rot6d[i] = rotCpu[0, i] * rotStd[i] + rotMean[i];

                // Gram-Schmidt: col0 = normalize(rot6d[0..2]), col1 = orthogonalize(rot6d[3..5])
                var col0 = new Vector3(rot6d[0], rot6d[1], rot6d[2]).normalized;
                var col1 = new Vector3(rot6d[3], rot6d[4], rot6d[5]);
                col1 = (col1 - Vector3.Dot(col1, col0) * col0).normalized;
                var col2 = Vector3.Cross(col0, col1);   // right-hand rule (same formula in Unity)

                // Rotation matrix → quaternion via Unity Matrix4x4
                var mat = new Matrix4x4();
                mat.SetColumn(0, new Vector4(col0.x, col0.y, col0.z, 0f));
                mat.SetColumn(1, new Vector4(col1.x, col1.y, col1.z, 0f));
                mat.SetColumn(2, new Vector4(col2.x, col2.y, col2.z, 0f));
                mat.SetColumn(3, new Vector4(0f, 0f, 0f, 1f));
                Quaternion qRel = mat.rotation;

                // Canonical rotation: local Z points toward object in Unity frame
                // dir_world is in HOT3D frame → convert to Unity frame by negating Z
                Vector3    dirWorldUnity = new Vector3(dirWorld.x, dirWorld.y, -dirWorld.z);
                // Smooth canonical to avoid jerk when direction is near-vertical
                Quaternion qCanonical    = SmoothLookRotation(dirWorldUnity);

                // Reconstruct wrist rotation: q_wrist = canonical * q_rel  (Unity frame)
                Quaternion rawPredRot = qCanonical * qRel;

                // EMA smoothing — prevents frame-to-frame jerk near vertical singularity
                if (firstRot)
                {
                    predictedRot = rawPredRot;
                    firstRot = false;
                }
                else
                {
                    predictedRot = Quaternion.Slerp(predictedRot, rawPredRot, rotEmaAlpha);
                }

            }

            // Fill context fields
            pose.WristPosition     = wristPosH;
            pose.WristRotation     = predictedRot;
            pose.ApproachDirection = dirWorld;
            pose.ApproachDistance  = dist;
            pose.GripCategory      = grip;

            if (doLog)
            {
                string side = (ctrl == featureAssembler?.rightControllerTransform) ? "R" : "L";
                string[] fnames = { "Th", "Ix", "Md", "Rg", "Pk" };
                var sb = new System.Text.StringBuilder();
                sb.AppendLine($"╭─[INF|{side}] frame={Time.frameCount} ──────────────────────────");
                sb.AppendLine($"│ ctx   dist={dist:F3}m  grip={gripNames[grip]}  cat={categoryId}  " +
                              $"obj={(nearestObj?.name ?? "NONE")}");
                sb.AppendLine($"│ wrist pos={wristPosH:F3}  euler={predictedRot.eulerAngles:F1}");
                sb.AppendLine($"│ dir   world=({dirWorld.x:F2},{dirWorld.y:F2},{dirWorld.z:F2})  " +
                              $"speed={approachSpeed:F2}m/s");
                sb.Append("│ fing ");
                // Per-finger MCP/PIP/DIP (degrees) one line per finger
                int[] off = { 0, 3, 6, 9, 12 };  // 3 MANO PCA components per finger group
                for (int f = 0; f < 5; f++)
                {
                    sb.Append($" {fnames[f]}[{pose.ManoJointAngles[off[f]]:F2}," +
                                            $"{pose.ManoJointAngles[off[f]+1]:F2}," +
                                            $"{pose.ManoJointAngles[off[f]+2]:F2}]");
                }
                sb.AppendLine();
                sb.Append("╰─");
                DLog(sb.ToString());
            }

            return pose;
        }

        // -----------------------------------------------------------------------
        // LSTM v4 helpers
        // -----------------------------------------------------------------------

        private static void LoadLSTMModel(
            ModelAsset asset, TextAsset metaJson,
            out Worker worker,
            ref float[] featMean, ref float[] featStd,
            ref float[] tgtMean,  ref float[] tgtStd,
            ref float[] rotMean,  ref float[] rotStd,
            ref float[] sdfMean, ref float[] sdfStd,
            string tag)
        {
            worker = null;
            if (asset == null)   { Debug.LogWarning($"[AuraXR] {tag}: model asset not assigned."); return; }
            if (metaJson == null){ Debug.LogWarning($"[AuraXR] {tag}: meta JSON not assigned.");   return; }

            var model = ModelLoader.Load(asset);
            worker = new Worker(model, BackendType.GPUCompute);

            var meta = JsonUtility.FromJson<LSTMModelMeta>(metaJson.text);

            if (meta.feature_mean != null && meta.feature_mean.Length == 25) featMean = meta.feature_mean;
            if (meta.feature_std  != null && meta.feature_std.Length  == 25) featStd  = meta.feature_std;
            if (meta.target_mean  != null && meta.target_mean.Length  == 15) tgtMean  = meta.target_mean;
            if (meta.target_std   != null && meta.target_std.Length   == 15) tgtStd   = meta.target_std;
            if (meta.wrist_rot_mean != null && meta.wrist_rot_mean.Length == 6) rotMean = meta.wrist_rot_mean;
            if (meta.wrist_rot_std  != null && meta.wrist_rot_std.Length  == 6) rotStd  = meta.wrist_rot_std;

            if (featStd != null) for (int i = 0; i < featStd.Length; i++) if (featStd[i] < 1e-6f) featStd[i] = 1f;
            if (tgtStd  != null) for (int i = 0; i < tgtStd.Length;  i++) if (tgtStd[i]  < 1e-6f) tgtStd[i]  = 1f;
            if (rotStd  != null) for (int i = 0; i < rotStd.Length;  i++) if (rotStd[i]  < 1e-6f) rotStd[i]  = 1f;

            if (meta.sdf_mean != null && meta.sdf_mean.Length == 4) sdfMean = meta.sdf_mean;
            if (meta.sdf_std  != null && meta.sdf_std.Length  == 4)
            {
                sdfStd = meta.sdf_std;
                for (int i = 0; i < sdfStd.Length; i++) if (sdfStd[i] < 1e-6f) sdfStd[i] = 1f;
            }
            Debug.Log($"[AuraXR] {tag} LSTM model loaded. feat={(featMean?.Length ?? 0)} tgt={(tgtMean?.Length ?? 0)} rot={(rotMean?.Length ?? 0)} sdf={(sdfMean?.Length ?? 0)}");
        }

        /// <summary>
        /// Runs one step of the SDF-LSTM model. Mirrors RunInference but:
        ///   - Adds 4-dim SDF local feature (approximated without full grid)
        ///   - Passes + updates LSTM hidden state (h, c)
        ///   - Resets state on Far→Approach transition
        ///   - No confidence output; uses fixed EMA
        /// </summary>
        private HandPose RunLSTMInference(
            Transform ctrl, Transform nearestObj, int categoryId,
            Worker lstmWorker,
            float[] featMean, float[] featStd,
            float[] tgtMean,  float[] tgtStd,
            float[] rotMean,  float[] rotStd,
            float[] sdfMean,  float[] sdfStd,
            ref float[] h, ref float[] c, ref bool approachActive,
            ref float[]    smooth,     ref bool      firstFrame,
            ref Vector3    prevWristH, ref float     prevTime,
            ref Quaternion predictedRot, ref bool    firstRot,
            ref float      smoothApproachSpeed, ref bool firstApproach,
            ref OneEuroFilterArray oneEuro, ref Vector3 lastDir,
            ref Transform  prevNearest, ref Vector3 prevObjPos, ref float prevObjTime,
            bool mirrorForCanonicalRight,
            HandPose current)
        {
            if (ctrl == null || lstmWorker == null) return current;

            // ── Feature assembly (identical to RunInference steps 1-8) ─────────
            Vector3 wristPosH = ToHOT3D(ctrl.position);
            Vector3 dirWorld, dirObjLoc, objPosH = Vector3.zero;
            Quaternion objRotH = Quaternion.identity;
            float dist;

            bool hasObj = (nearestObj != null);
            if (!hasObj)
            {
                dist = 2.0f; dirWorld = lastDir; dirObjLoc = lastDir; categoryId = 0;
            }
            else
            {
                objPosH = ToHOT3D(nearestObj.position);
                Vector3 relWorld = objPosH - wristPosH;
                dist = relWorld.magnitude;
                if (dist < 1e-6f) return current;
                dirWorld = relWorld / dist;
                lastDir  = dirWorld;
                objRotH  = ToHOT3DQuat(nearestObj.rotation);
                if (dirObjLocalIdentityGuard && Quaternion.Angle(objRotH, Quaternion.identity) < 1f)
                    dirObjLoc = dirWorld;
                else
                    dirObjLoc = Quaternion.Inverse(objRotH) * dirWorld;
            }

            // LSTM state reset on Far→Approach transition (within 40cm)
            bool inApproach = hasObj && dist < distanceOuterM;
            if (inApproach && !approachActive)
            {
                Array.Clear(h, 0, h.Length);
                Array.Clear(c, 0, c.Length);
                approachActive = true;
            }
            else if (!inApproach)
            {
                approachActive = false;
            }

            float now = Time.realtimeSinceStartup;
            float approachSpeedRaw = 0f;
            if (prevTime >= 0f)
            {
                float dt = now - prevTime;
                if (dt > 1e-4f)
                    approachSpeedRaw = Vector3.Dot((wristPosH - prevWristH) / dt, dirWorld);
            }
            prevWristH = wristPosH;
            prevTime   = now;
            float speedClamped = Mathf.Clamp(approachSpeedRaw, -approachSpeedClampMs, approachSpeedClampMs);
            float approachSpeed;
            if (firstApproach) { smoothApproachSpeed = speedClamped; firstApproach = false; approachSpeed = speedClamped; }
            else { smoothApproachSpeed = approachSpeedInputEma * speedClamped + (1f - approachSpeedInputEma) * smoothApproachSpeed; approachSpeed = smoothApproachSpeed; }

            Vector3 objVelWorld = Vector3.zero;
            if (hasObj)
            {
                float tNow = Time.realtimeSinceStartup;
                if (prevNearest == nearestObj && prevObjTime >= 0f)
                {
                    float dtObj = tNow - prevObjTime;
                    if (dtObj > 1e-4f) objVelWorld = (objPosH - prevObjPos) / dtObj;
                }
                prevNearest = nearestObj; prevObjPos = objPosH; prevObjTime = tNow;
            }
            else { prevNearest = null; prevObjTime = -1f; }

            float[] wristRotInput = ComputeWristRot6dInput(ctrl.rotation, dirWorld);
            int   grip = BopToGrip.TryGetValue(categoryId, out int g) ? g : 0;
            float[] bbox = BopToBbox.TryGetValue(categoryId, out float[] b) ? b : DefaultBbox;

            const int FEAT_DIM = 25;
            float[] feat25 = new float[FEAT_DIM];
            feat25[0]=dirWorld.x; feat25[1]=dirWorld.y; feat25[2]=dirWorld.z;
            feat25[3]=dirObjLoc.x; feat25[4]=dirObjLoc.y; feat25[5]=dirObjLoc.z;
            feat25[6]=dist; feat25[7]=approachSpeed;
            feat25[8]=objVelWorld.x; feat25[9]=objVelWorld.y; feat25[10]=objVelWorld.z;
            feat25[11]=wristRotInput[0]; feat25[12]=wristRotInput[1]; feat25[13]=wristRotInput[2];
            feat25[14]=wristRotInput[3]; feat25[15]=wristRotInput[4]; feat25[16]=wristRotInput[5];
            feat25[17]=1.0f;  // hand_confidence
            feat25[18]= grip==0?1f:0f; feat25[19]= grip==1?1f:0f;
            feat25[20]= grip==2?1f:0f; feat25[21]= grip==3?1f:0f;
            feat25[22]=bbox[0]; feat25[23]=bbox[1]; feat25[24]=bbox[2];
            if (mirrorForCanonicalRight)
            {
                for (int i = 0; i < FEAT_DIM; i++) feat25[i] *= FeatureMirrorMask[i];
            }
            if (featMean == null || featStd == null || featMean.Length < FEAT_DIM || featStd.Length < FEAT_DIM)
            {
                DLog($"[LSTM] Invalid feature stats: mean={(featMean?.Length ?? 0)} std={(featStd?.Length ?? 0)} expected={FEAT_DIM}. Check model_meta_lstm_*.json wiring.");
                return current;
            }
            for (int i = 0; i < FEAT_DIM; i++) feat25[i] = (feat25[i] - featMean[i]) / featStd[i];

            // ── SDF local feature (4-dim) ───────────────────────────────────────
            // Exact trilinear query from pre-baked 32³ grid (matches training exactly).
            // Falls back to bbox approximation if SDFGridDatabase is not available.
            float[] sdfLocal;
            if (sdfGridDatabase != null && hasObj)
            {
                Vector3 wristInObj = Quaternion.Inverse(objRotH) * (wristPosH - objPosH);
                sdfLocal = sdfGridDatabase.Query(categoryId, wristInObj);
            }
            else
            {
                // Fallback: bbox approximation (used when SDFGridDatabase not loaded)
                float bboxMax   = Mathf.Max(bbox[0], bbox[1], bbox[2]);
                float sdfApprox = Mathf.Max(dist - bboxMax, 0f);
                Vector3 gradLocal = Vector3.zero;
                if (hasObj)
                {
                    Vector3 wristInObj = Quaternion.Inverse(objRotH) * (wristPosH - objPosH);
                    gradLocal = wristInObj.magnitude > 1e-6f ? wristInObj.normalized : Vector3.up;
                }
                sdfLocal = new float[] { sdfApprox, gradLocal.x, gradLocal.y, gradLocal.z };
            }
            for (int i = 0; i < 4; i++) sdfLocal[i] = (sdfLocal[i] - sdfMean[i]) / sdfStd[i];

            // ── Concatenate 25+4 = 29-dim input ────────────────────────────────
            float[] feat29 = new float[29];
            Array.Copy(feat25, feat29, 25);
            Array.Copy(sdfLocal, 0, feat29, 25, 4);

            // ── SDF embedding lookup (32-dim) ───────────────────────────────────
            float[] objEmbed = sdfDatabase != null ? sdfDatabase.GetEmbedding(categoryId) : new float[32];

            // ── Run LSTM ONNX ────────────────────────────────────────────────────
            using var featTensor  = new Tensor<float>(new TensorShape(1, 29), feat29);
            using var embedTensor = new Tensor<float>(new TensorShape(1, 32), objEmbed);
            using var hTensor     = new Tensor<float>(new TensorShape(LSTM_LAYERS, 1, LSTM_HIDDEN), h);
            using var cTensor     = new Tensor<float>(new TensorShape(LSTM_LAYERS, 1, LSTM_HIDDEN), c);

            lstmWorker.SetInput("frame_feat",  featTensor);
            lstmWorker.SetInput("obj_embed",   embedTensor);
            lstmWorker.SetInput("h_0",         hTensor);
            lstmWorker.SetInput("c_0",         cTensor);
            lstmWorker.Schedule();

            var anglesTensor = lstmWorker.PeekOutput("pose_pca")    as Tensor<float>;
            var rotTensor    = lstmWorker.PeekOutput("wrist_rot") as Tensor<float>;
            var hnTensor     = lstmWorker.PeekOutput("h_n")          as Tensor<float>;
            var cnTensor     = lstmWorker.PeekOutput("c_n")          as Tensor<float>;

            if (anglesTensor == null) { DLog("[LSTM] PeekOutput pose_pca null"); return current; }

            using var angleCpu = anglesTensor.ReadbackAndClone();
            float[] angles = new float[15];
            if (tgtMean == null || tgtStd == null || tgtMean.Length < 15 || tgtStd.Length < 15)
            {
                DLog($"[LSTM] Invalid target stats: mean={(tgtMean?.Length ?? 0)} std={(tgtStd?.Length ?? 0)} expected=15. Check model_meta_lstm_*.json wiring.");
                return current;
            }
            for (int i = 0; i < 15; i++)
                angles[i] = angleCpu[0, i] * tgtStd[i] + tgtMean[i];
            if (mirrorForCanonicalRight)
            {
                for (int i = 0; i < 15; i++) angles[i] *= ManoPcaMirrorMask[i];
            }

            // Update LSTM state — flat index handles both [2,1,256] and [2,1,1,256] shapes
            if (hnTensor != null)
            {
                using var hnCpu = hnTensor.ReadbackAndClone();
                for (int i = 0; i < LSTM_STATE_TOTAL; i++) h[i] = hnCpu[i];
            }
            if (cnTensor != null)
            {
                using var cnCpu = cnTensor.ReadbackAndClone();
                for (int i = 0; i < LSTM_STATE_TOTAL; i++) c[i] = cnCpu[i];
            }

            // Distance hysteresis fade near activation threshold
            float fadeAlpha = 1f;
            if (distanceHysteresisM > 1e-4f)
            {
                float innerBand = distanceOuterM - distanceHysteresisM;
                fadeAlpha = Mathf.Clamp01((distanceOuterM - dist) / (distanceOuterM - innerBand));
            }
            if (fadeAlpha < 0.999f)
                for (int i = 0; i < 15; i++)
                    angles[i] = Mathf.Lerp(_neutralPose15[i], angles[i], fadeAlpha);

            // Temporal smoothing
            if (useOneEuroFilter)
            {
                if (oneEuro == null || oneEuro.Size != 15) oneEuro = new OneEuroFilterArray(15);
                oneEuro.MinCutoffHz = oneEuroMinCutoffHz;
                oneEuro.Beta        = oneEuroBeta;
                oneEuro.Filter(angles, Time.unscaledTime, smooth);
                firstFrame = false;
            }
            else if (firstFrame)
            { Array.Copy(angles, smooth, 15); firstFrame = false; }
            else
            { for (int i=0;i<15;i++) smooth[i] = emaAlpha*angles[i] + (1f-emaAlpha)*smooth[i]; }

            // Wrist rotation decode
            if (rotTensor != null && rotMean != null && rotStd != null && rotMean.Length >= 6 && rotStd.Length >= 6)
            {
                using var rotCpu = rotTensor.ReadbackAndClone();
                float[] rot6d = new float[6];
                for (int i = 0; i < 6; i++) rot6d[i] = rotCpu[0, i] * rotStd[i] + rotMean[i];
                if (mirrorForCanonicalRight)
                {
                    for (int i = 0; i < 6; i++) rot6d[i] *= WristRotMirrorMask[i];
                }
                var col0 = new Vector3(rot6d[0], rot6d[1], rot6d[2]).normalized;
                var col1 = new Vector3(rot6d[3], rot6d[4], rot6d[5]);
                col1 = (col1 - Vector3.Dot(col1, col0) * col0).normalized;
                var col2 = Vector3.Cross(col0, col1);
                var mat  = new Matrix4x4();
                mat.SetColumn(0, new Vector4(col0.x, col0.y, col0.z, 0f));
                mat.SetColumn(1, new Vector4(col1.x, col1.y, col1.z, 0f));
                mat.SetColumn(2, new Vector4(col2.x, col2.y, col2.z, 0f));
                mat.SetColumn(3, new Vector4(0f, 0f, 0f, 1f));
                Quaternion qRel      = mat.rotation;
                Vector3    dirUnity  = new Vector3(dirWorld.x, dirWorld.y, -dirWorld.z);
                Quaternion qCanon    = SmoothLookRotation(dirUnity);
                Quaternion rawRot    = qCanon * qRel;
                if (firstRot) { predictedRot = rawRot; firstRot = false; }
                else           predictedRot = Quaternion.Slerp(predictedRot, rawRot, rotEmaAlpha);
            }

            // Build HandPose — model outputs MANO PCA 15-dim directly
            var pose = new HandPose();
            Array.Copy(smooth, pose.ManoJointAngles, 15);
            // UmeAbductionAngles not used with MANO PCA — zero them
            Array.Clear(pose.UmeAbductionAngles, 0, 5);
            for (int f = 0; f < 5; f++) pose.FingerConfidence[f] = 1f;
            pose.WristPosition     = wristPosH;
            pose.WristRotation     = predictedRot;
            pose.ApproachDirection = dirWorld;
            pose.ApproachDistance  = dist;
            pose.GripCategory      = grip;
            return pose;
        }

        // -----------------------------------------------------------------------
        // Coordinate frame conversion: Unity (left-handed) ↔ HOT3D (right-handed)
        // -----------------------------------------------------------------------
        private static Vector3    ToHOT3D(Vector3 v)        => new Vector3(v.x, v.y, -v.z);
        private static Quaternion ToHOT3DQuat(Quaternion q) => new Quaternion(q.x, q.y, -q.z, q.w);

        /// Compute the input wrist 6D rotation feature (v3 dims [11-16]).
        /// Mirrors Python hot3d_utils.wrist_rot_to_6d:
        ///   q_rel = canonical(dirWorldUnity)^-1 * q_wristUnity
        ///   6D    = [R_rel.col0, R_rel.col1]
        /// The controller already exposes Unity-frame rotation, so no quaternion conversion
        /// is needed beyond reusing SmoothLookRotation on the Unity-frame approach direction.
        private static float[] ComputeWristRot6dInput(Quaternion wristUnity, Vector3 dirWorldHot3d)
        {
            Vector3    dirWorldUnity = new Vector3(dirWorldHot3d.x, dirWorldHot3d.y, -dirWorldHot3d.z);
            Quaternion qCanonical   = SmoothLookRotation(dirWorldUnity);
            Quaternion qRel         = Quaternion.Inverse(qCanonical) * wristUnity;
            Matrix4x4  R            = Matrix4x4.Rotate(qRel);
            return new float[] {
                R.m00, R.m10, R.m20,   // col0
                R.m01, R.m11, R.m21,   // col1
            };
        }

        /// Matches Python hot3d_utils.look_rotation exactly.
        /// Near-vertical singularity (forward ≈ up) switches up to Vector3.forward.
        /// Remaining runtime jerk is handled by the EMA Slerp in RunInference (rotEmaAlpha).
        private static Quaternion SmoothLookRotation(Vector3 forward)
        {
            forward = forward.normalized;
            Vector3 up = Vector3.up;
            if (Vector3.Cross(up, forward).sqrMagnitude < 1e-6f)
                up = Vector3.forward;
            return Quaternion.LookRotation(forward, up);
        }

        // -----------------------------------------------------------------------
        // Load model + parse meta JSON
        // -----------------------------------------------------------------------
        private static void LoadModel(
            ModelAsset asset, TextAsset metaJson,
            out Worker worker,
            out float[] featMean, out float[] featStd,
            out float[] tgtMean,  out float[] tgtStd,
            out float[] rotMean,  out float[] rotStd,
            string tag)
        {
            worker = null;
            featMean = featStd = tgtMean = tgtStd = rotMean = rotStd = null;

            if (asset == null)   { Debug.LogError($"[AuraXR] {tag} model asset not assigned."); return; }
            if (metaJson == null){ Debug.LogError($"[AuraXR] {tag} meta JSON not assigned.");   return; }

            var model = ModelLoader.Load(asset);
            worker = new Worker(model, BackendType.GPUCompute);

            var meta = JsonUtility.FromJson<ModelMeta>(metaJson.text);
            featMean = meta.feature_mean;
            featStd  = meta.feature_std;
            tgtMean  = meta.target_mean;
            tgtStd   = meta.target_std;
            rotMean  = meta.wrist_rot_mean  ?? new float[6];
            rotStd   = meta.wrist_rot_std   ?? new float[] { 1f, 1f, 1f, 1f, 1f, 1f };

            for (int i = 0; i < featStd.Length; i++) if (featStd[i] < 1e-6f) featStd[i] = 1f;
            for (int i = 0; i < tgtStd.Length;  i++) if (tgtStd[i]  < 1e-6f) tgtStd[i]  = 1f;
            for (int i = 0; i < rotStd.Length;  i++) if (rotStd[i]  < 1e-6f) rotStd[i]  = 1f;

            Debug.Log($"[AuraXR] {tag} loaded. feat={featMean.Length} tgt={tgtMean.Length} rot={rotMean.Length}");
        }

        // -----------------------------------------------------------------------
        // JSON deserialization helper
        // -----------------------------------------------------------------------
        [Serializable]
        private class ModelMeta
        {
            public float[] feature_mean;
            public float[] feature_std;
            public float[] target_mean;
            public float[] target_std;
            public float[] wrist_rot_mean;
            public float[] wrist_rot_std;
        }

        // LSTM meta adds sdf_mean / sdf_std on top of base fields
        [Serializable]
        private class LSTMModelMeta
        {
            public float[] feature_mean;
            public float[] feature_std;
            public float[] sdf_mean;
            public float[] sdf_std;
            public float[] target_mean;
            public float[] target_std;
            public float[] wrist_rot_mean;
            public float[] wrist_rot_std;
        }
    }

    // -----------------------------------------------------------------------
    // One-Euro Filter — vectorized over an array of values (one filter state per index)
    // Reference: Casiez et al., CHI 2012. Better than EMA for fast motion + slow drift.
    // -----------------------------------------------------------------------
    public class OneEuroFilterArray
    {
        public int   Size { get; }
        public float MinCutoffHz { get; set; } = 1.0f;
        public float Beta        { get; set; } = 0.007f;
        public float DerivCutoffHz { get; set; } = 1.0f;

        private readonly float[] _xPrev;
        private readonly float[] _dxPrev;
        private float            _tPrev = -1f;
        private bool             _initialized = false;

        public OneEuroFilterArray(int size)
        {
            Size = size;
            _xPrev  = new float[size];
            _dxPrev = new float[size];
        }

        private static float Alpha(float dt, float cutoff)
        {
            float tau = 1f / (2f * Mathf.PI * cutoff);
            return 1f / (1f + tau / dt);
        }

        public void Filter(float[] x, float t, float[] output)
        {
            if (!_initialized || _tPrev < 0f)
            {
                Array.Copy(x, _xPrev, Size);
                Array.Clear(_dxPrev, 0, Size);
                Array.Copy(x, output, Size);
                _tPrev = t;
                _initialized = true;
                return;
            }
            float dt = t - _tPrev;
            if (dt <= 1e-5f) dt = 1e-5f;
            float aDeriv = Alpha(dt, DerivCutoffHz);
            for (int i = 0; i < Size; i++)
            {
                float dx     = (x[i] - _xPrev[i]) / dt;
                float dxHat  = aDeriv * dx + (1f - aDeriv) * _dxPrev[i];
                float cutoff = MinCutoffHz + Beta * Mathf.Abs(dxHat);
                float aData  = Alpha(dt, cutoff);
                float xHat   = aData * x[i] + (1f - aData) * _xPrev[i];

                output[i]    = xHat;
                _xPrev[i]    = xHat;
                _dxPrev[i]   = dxHat;
            }
            _tPrev = t;
        }
    }

    // -----------------------------------------------------------------------
    // Shared output type — read by HandRigController each frame
    // -----------------------------------------------------------------------
    [Serializable]
    public class HandPose
    {
        // ── Finger angles ──────────────────────────────────────────────────
        public float[] ManoJointAngles;    // [15] radians — MANO flexion order
        public float[] UmeAbductionAngles; // [5]  radians — Thumb,Idx,Mid,Rng,Pnk spread

        // ── Wrist ──────────────────────────────────────────────────────────
        public Vector3    WristPosition;   // world pos in HOT3D frame (x, y, -z)
        public Quaternion WristRotation;   // palm orientation in Unity world frame (model-predicted)

        // ── Approach context ───────────────────────────────────────────────
        public Vector3 ApproachDirection; // unit vector wrist→object, HOT3D world frame
        public float   ApproachDistance;  // metres
        public int     GripCategory;      // 0=Power 1=Precision 2=Palmar 3=Pinch

        // ── Per-finger confidence ──────────────────────────────────────────
        public float[] FingerConfidence;  // [5] 0–1 — EMA stability proxy

        public HandPose()
        {
            ManoJointAngles    = new float[15];
            UmeAbductionAngles = new float[5];
            WristPosition      = Vector3.zero;
            WristRotation      = Quaternion.identity;
            ApproachDirection  = Vector3.forward;
            ApproachDistance   = 0f;
            GripCategory       = 0;
            FingerConfidence   = new float[] { 1f, 1f, 1f, 1f, 1f };
        }
    }
}
