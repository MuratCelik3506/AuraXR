using System;
using System.Collections.Generic;
using UnityEngine;
using Unity.InferenceEngine;

namespace AuraXR
{
    /// <summary>
    /// Runs AuraXRModel (15-dim MLP) inference each frame via Unity Sentis.
    ///
    /// Two ONNX inputs:
    ///   "spatial_input" [1, 8]: [dir_world(3), dir_obj_local(3), dist(1), approach_speed(1)]
    ///   "object_input"  [1, 7]: [grip_oh(4), bbox(3)]
    ///
    /// Two ONNX outputs:
    ///   "joint_angles"  [1, 22]: 22 UME angles (normalized) — denorm with target_mean/std
    ///   "wrist_rot_6d"  [1,  6]: palm orientation as 6D rotation (normalized) — denorm, then
    ///                            Gram-Schmidt → rotation matrix → Quaternion in Unity frame
    ///
    /// Wrist placement (every frame):
    ///   position = controller.position
    ///   rotation = _predictedRot{side}  (model output, updated every N frames)
    ///   fallback  = controller.rotation  (when no nearest object present)
    ///
    /// Coordinate frame: HOT3D is right-handed Y-up. Unity is left-handed Y-up.
    ///   Conversion: negate Z of positions and quaternion Z component.
    ///   ToHOT3D(v)    = (x, y, -z)
    ///   ToHOT3DQuat(q)= (qx, qy, -qz, qw)
    ///
    /// Public output read by HandRigController each frame:
    ///   LeftHand.ManoJointAngles[15]  — radians, MANO order
    ///   RightHand.ManoJointAngles[15] — radians, MANO order
    /// </summary>
    public class AuraXRInferenceManager : MonoBehaviour
    {
        [Header("ONNX Models")]
        [Tooltip("Drag auraxr_right.onnx here")]
        public ModelAsset rightModelAsset;
        [Tooltip("Drag auraxr_left.onnx here")]
        public ModelAsset leftModelAsset;

        [Header("Model Meta (JSON TextAssets)")]
        [Tooltip("Drag model_meta_right.json here")]
        public TextAsset rightMetaJson;
        [Tooltip("Drag model_meta_left.json here")]
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
        [Tooltip("Local-space offset applied to both hands when placing them at the controller anchor.")]
        public Vector3 handPivotOffset = new Vector3(0.1685f, 0f, 0.0351f);

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

        [Header("Debug")]
        public bool debugBypassModel = false;

        // -----------------------------------------------------------------------
        // Public output — read by HandRigController each frame
        // -----------------------------------------------------------------------
        public HandPose LeftHand  { get; private set; } = new HandPose();
        public HandPose RightHand { get; private set; } = new HandPose();

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

        // EMA smoothed UME joint angles (22 per hand)
        private float[] _smoothRight = new float[22];
        private float[] _smoothLeft  = new float[22];
        private bool    _firstRight  = true, _firstLeft = true;

        private int _frameCounter;

        // Previous wrist positions and timestamps for approach_speed computation
        private Vector3 _prevWristRight = Vector3.zero;
        private Vector3 _prevWristLeft  = Vector3.zero;
        private float   _prevTimeRight  = -1f;
        private float   _prevTimeLeft   = -1f;

        // Model-predicted wrist rotations in Unity frame — updated every N frames, EMA smoothed
        private Quaternion _predictedRotRight = Quaternion.identity;
        private Quaternion _predictedRotLeft  = Quaternion.identity;
        private bool       _firstRotRight     = true;
        private bool       _firstRotLeft      = true;

        // File-based debug log
        private string _debugLogPath;
        private System.IO.StreamWriter _debugWriter;

        // UME joint indices that map to MANO flexion angles (skip abduction joints)
        // UME per finger: [abduction, MCP, PIP, DIP] — Thumb[0-3], Index[4-7], Mid[8-11], Ring[12-15], Pinky[16-19]
        // Note: Thumb index 0 = CMC/MCP flex (not abduction)
        private static readonly int[] UmeToMano       = { 0, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19 };
        private static readonly int[] UmeAbductionIdx = { 1, 4, 8, 12, 16 };
        private static readonly int[] UmeFingerStart  = { 0, 4, 8, 12, 16 };

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

            if (rightModelAsset == null) rightModelAsset = Resources.Load<ModelAsset>("auraxr_right");
            if (leftModelAsset  == null) leftModelAsset  = Resources.Load<ModelAsset>("auraxr_left");
            if (rightMetaJson   == null) rightMetaJson   = Resources.Load<TextAsset>("model_meta_right");
            if (leftMetaJson    == null) leftMetaJson    = Resources.Load<TextAsset>("model_meta_left");

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
        }

        void Update()
        {
            Transform rightCtrl = featureAssembler?.rightControllerTransform;
            Transform leftCtrl  = featureAssembler?.leftControllerTransform;

            Transform nearestR  = featureAssembler?.nearestObjectRight;
            Transform nearestL  = featureAssembler?.nearestObjectLeft;
            int       categoryR = featureAssembler?.nearestObjectCategoryRight ?? 0;
            int       categoryL = featureAssembler?.nearestObjectCategoryLeft  ?? 0;

            // Fallback rotation when no object is tracked: mirror controller
            if (nearestR == null && rightCtrl != null) _predictedRotRight = rightCtrl.rotation;
            if (nearestL == null && leftCtrl  != null) _predictedRotLeft  = leftCtrl.rotation;

            // Position from controller, rotation from last model prediction
            if (virtualHandRight != null && rightCtrl != null)
                virtualHandRight.SetPositionAndRotation(
                    rightCtrl.position + _predictedRotRight * handPivotOffset,
                    _predictedRotRight);
            if (virtualHandLeft != null && leftCtrl != null)
                virtualHandLeft.SetPositionAndRotation(
                    leftCtrl.position + _predictedRotLeft * handPivotOffset,
                    _predictedRotLeft);

            if (Time.frameCount % 90 == 0) LogWristOffset(rightCtrl, virtualHandRight, "R");
            if (Time.frameCount % 90 == 5) LogWristOffset(leftCtrl,  virtualHandLeft,  "L");

            _frameCounter++;
            if (_frameCounter % inferenceEveryNFrames != 0) return;

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

            RightHand = RunInference(
                rightCtrl, nearestR, categoryR,
                _workerRight,
                _featMeanRight, _featStdRight, _tgtMeanRight, _tgtStdRight,
                _rotMeanRight, _rotStdRight,
                ref _smoothRight, ref _firstRight,
                ref _prevWristRight, ref _prevTimeRight,
                ref _predictedRotRight, ref _firstRotRight,
                RightHand);

            LeftHand = RunInference(
                leftCtrl, nearestL, categoryL,
                _workerLeft,
                _featMeanLeft, _featStdLeft, _tgtMeanLeft, _tgtStdLeft,
                _rotMeanLeft, _rotStdLeft,
                ref _smoothLeft, ref _firstLeft,
                ref _prevWristLeft, ref _prevTimeLeft,
                ref _predictedRotLeft, ref _firstRotLeft,
                LeftHand);
        }

        void OnDestroy()
        {
            _workerRight?.Dispose();
            _workerLeft?.Dispose();
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
            HandPose current)
        {
            if (ctrl == null || nearestObj == null || worker == null)
            {
                if (Time.frameCount % 60 == 0)
                    DLog($"[SKIP] ctrl={(ctrl?.name ?? "NULL")}  nearestObj={(nearestObj?.name ?? "NULL")}  worker={worker != null}");
                return current;
            }

            bool doLog = (Time.frameCount % 60 == 0);

            // 1. Positions and world-frame direction in HOT3D coordinate space
            Vector3 wristPosH = ToHOT3D(ctrl.position);
            Vector3 objPosH   = ToHOT3D(nearestObj.position);
            Vector3 relWorld  = objPosH - wristPosH;
            float   dist      = relWorld.magnitude;

            if (doLog)
                DLog($"[POS] ctrl={ctrl.position:F3}  obj={nearestObj.position:F3}  dist={dist:F3}m  cat={categoryId}");

            if (dist < 1e-6f) return current;

            Vector3 dirWorld = relWorld / dist;   // HOT3D world-frame unit vector

            // 2. Object-local direction — which face of the object is being approached
            Quaternion objRotH   = ToHOT3DQuat(nearestObj.rotation);
            Vector3    dirObjLoc = Quaternion.Inverse(objRotH) * dirWorld;

            // 3. Approach speed (wrist velocity dot approach direction)
            float now = Time.realtimeSinceStartup;
            float approachSpeed = 0f;
            if (prevTime >= 0f)
            {
                float dt = now - prevTime;
                if (dt > 1e-4f)
                {
                    Vector3 velWorld = (wristPosH - prevWristH) / dt;
                    approachSpeed = Vector3.Dot(velWorld, dirWorld);
                }
            }
            prevWristH = wristPosH;
            prevTime   = now;

            // 4. Grip category and bbox
            int     grip = BopToGrip.TryGetValue(categoryId, out int g) ? g : 0;
            float[] bbox = BopToBbox.TryGetValue(categoryId, out float[] b) ? b : DefaultBbox;
            string[] gripNames = { "Power", "Precision", "Palmar", "Pinch" };

            // 5. Assemble raw 15-dim feature
            //    [dir_world(3), dir_obj_local(3), dist(1), approach_speed(1), grip_oh(4), bbox(3)]
            float[] feat = new float[15];
            feat[0] = dirWorld.x;  feat[1] = dirWorld.y;  feat[2] = dirWorld.z;
            feat[3] = dirObjLoc.x; feat[4] = dirObjLoc.y; feat[5] = dirObjLoc.z;
            feat[6] = dist;
            feat[7] = approachSpeed;
            feat[8]  = grip == 0 ? 1f : 0f;
            feat[9]  = grip == 1 ? 1f : 0f;
            feat[10] = grip == 2 ? 1f : 0f;
            feat[11] = grip == 3 ? 1f : 0f;
            feat[12] = bbox[0]; feat[13] = bbox[1]; feat[14] = bbox[2];

            if (doLog)
                DLog($"[FEAT_RAW] dir=({feat[0]:F3},{feat[1]:F3},{feat[2]:F3})  " +
                     $"objLoc=({feat[3]:F3},{feat[4]:F3},{feat[5]:F3})  " +
                     $"dist={feat[6]:F3}  spd={feat[7]:F3}  " +
                     $"grip={gripNames[grip]}  bbox=({feat[12]:F3},{feat[13]:F3},{feat[14]:F3})");

            // 6. Normalize
            for (int i = 0; i < 15; i++)
                feat[i] = (feat[i] - featMean[i]) / featStd[i];

            // 7. Split → spatial (8) and object (7) tensors
            float[] spatialInput = new float[] {
                feat[0], feat[1], feat[2], feat[3],
                feat[4], feat[5], feat[6], feat[7]
            };
            float[] objectInput = new float[] {
                feat[8], feat[9], feat[10], feat[11], feat[12], feat[13], feat[14]
            };

            // 8. Run ONNX model
            using var spatialTensor = new Tensor<float>(new TensorShape(1, 8), spatialInput);
            using var objectTensor  = new Tensor<float>(new TensorShape(1, 7), objectInput);

            worker.SetInput("spatial_input", spatialTensor);
            worker.SetInput("object_input",  objectTensor);
            worker.Schedule();

            var anglesTensor = worker.PeekOutput("joint_angles") as Tensor<float>;
            var rotTensor    = worker.PeekOutput("wrist_rot_6d")  as Tensor<float>;

            if (anglesTensor == null)
            {
                DLog("[ERROR] PeekOutput('joint_angles') returned null.");
                return current;
            }

            using var angleCpu = anglesTensor.ReadbackAndClone();

            // 9. Denormalize 22 UME angles
            float[] angles = new float[22];
            for (int i = 0; i < 22; i++)
                angles[i] = angleCpu[0, i] * tgtStd[i] + tgtMean[i];

            if (doLog)
            {
                string umeDeg = string.Join(" ", System.Array.ConvertAll(angles, a => (a * Mathf.Rad2Deg).ToString("F1")));
                DLog($"[UME_DENORM(deg)] [{umeDeg}]");
            }

            // 10. EMA smoothing over 22 UME angles
            if (firstFrame)
            {
                Array.Copy(angles, smooth, 22);
                firstFrame = false;
            }
            else
            {
                for (int i = 0; i < 22; i++)
                    smooth[i] = emaAlpha * angles[i] + (1f - emaAlpha) * smooth[i];
            }

            // 11. Map 22 UME → 15 MANO flexion angles
            var pose = new HandPose();
            for (int m = 0; m < 15; m++)
                pose.ManoJointAngles[m] = smooth[UmeToMano[m]];

            for (int f = 0; f < 5; f++)
                pose.UmeAbductionAngles[f] = smooth[UmeAbductionIdx[f]];

            // Per-finger confidence (EMA instability proxy)
            for (int f = 0; f < 5; f++)
            {
                float sumDelta = 0f;
                int start = UmeFingerStart[f];
                for (int j = start; j < start + 4; j++)
                    sumDelta += Mathf.Abs(angles[j] - smooth[j]);
                pose.FingerConfidence[f] = 1f / (1f + sumDelta * 4f);
            }

            // 12. Decode wrist rotation from 6D output → Unity Quaternion
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

                if (doLog)
                    DLog($"[ROT] dir_unity={dirWorldUnity:F3}  qRel={qRel.eulerAngles:F1}  " +
                         $"wristEuler={predictedRot.eulerAngles:F1}");
            }

            // Fill context fields
            pose.WristPosition     = wristPosH;
            pose.WristRotation     = predictedRot;
            pose.ApproachDirection = dirWorld;
            pose.ApproachDistance  = dist;
            pose.GripCategory      = grip;

            if (doLog)
            {
                string[] manoNames = {
                    "Th.MCP","Th.PIP","Th.DIP",
                    "Idx.MCP","Idx.PIP","Idx.DIP",
                    "Mid.MCP","Mid.PIP","Mid.DIP",
                    "Rng.MCP","Rng.PIP","Rng.DIP",
                    "Pnk.MCP","Pnk.PIP","Pnk.DIP"
                };
                var sb = new System.Text.StringBuilder("[MANO(deg)]");
                for (int m = 0; m < 15; m++)
                    sb.Append($"  {manoNames[m]}={pose.ManoJointAngles[m] * Mathf.Rad2Deg:F1}");
                DLog(sb.ToString());
                DLog($"[CONTEXT] wristPos={wristPosH:F3}  wristEuler={predictedRot.eulerAngles:F1}  " +
                     $"approachDir=({dirWorld.x:F3},{dirWorld.y:F3},{dirWorld.z:F3})  dist={dist:F3}m  grip={gripNames[grip]}");
            }

            return pose;
        }

        // -----------------------------------------------------------------------
        // Coordinate frame conversion: Unity (left-handed) ↔ HOT3D (right-handed)
        // -----------------------------------------------------------------------
        private static Vector3    ToHOT3D(Vector3 v)        => new Vector3(v.x, v.y, -v.z);
        private static Quaternion ToHOT3DQuat(Quaternion q) => new Quaternion(q.x, q.y, -q.z, q.w);

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
