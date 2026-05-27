using System;
using System.Collections.Generic;
using UnityEngine;
using Unity.InferenceEngine;

namespace AuraXR
{
    /// <summary>
    /// Runs AuraXRModel (11-dim MLP) inference each frame via Unity Sentis.
    ///
    /// Model: two branches → 22 UME joint angles per hand.
    ///   Input  "spatial_input": [dir_x, dir_y, dir_z, distance]  — direction in wrist frame
    ///   Input  "object_input":  [grip_oh(4), bbox(3)]
    ///   Output "joint_angles":  [22 joint angles, normalized]    — denorm with model_meta.json
    ///
    /// Feature layout (11 dims):
    ///   [0..2]  direction  — unit vector from wrist to object, in wrist local frame (HOT3D frame)
    ///   [3]     distance   — metres, wrist to object centroid
    ///   [4..7]  grip_onehot — Power / Precision / Palmar / Pinch  (HOT3D BOP categories)
    ///   [8..10] bbox       — object half-extents x,y,z in metres
    ///
    /// UME→MANO mapping (22 UME flexion+abduction → 15 MANO flexion only):
    ///   UME per finger: [abduction, MCP, PIP, DIP] × 5 fingers + 2 placeholders
    ///   MANO keeps only flexion: Thumb[0,2,3], Index[5,6,7], Mid[9,10,11], Ring[13,14,15], Pinky[17,18,19]
    ///
    /// Coordinate frame: HOT3D is right-handed Y-up. Unity is left-handed Y-up.
    ///   Conversion: negate Z of positions and quaternion Z component.
    ///
    /// Public output read by HandRigController / AuraXRHandRenderer each frame:
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
        [Tooltip("Provides controller transforms and nearest object data (shared with other components)")]
        public AuraXRFeatureAssembler featureAssembler;

        [Header("Virtual Hand Anchors")]
        [Tooltip("Root Transform of the left virtual hand — moved to match left controller each frame")]
        public Transform virtualHandLeft;
        [Tooltip("Root Transform of the right virtual hand — moved to match right controller each frame")]
        public Transform virtualHandRight;

        [Header("Inference Rate")]
        [Tooltip("Run inference every N frames to match ~30 FPS training rate at 72 Hz")]
        public int inferenceEveryNFrames = 2;

        [Header("Temporal Smoothing")]
        [Range(0f, 1f)]
        public float emaAlpha = 0.35f;

        [Header("Debug")]
        public bool debugBypassModel = false;

        // -----------------------------------------------------------------------
        // Public output — read by HandRigController / AuraXRHandRenderer each frame
        // -----------------------------------------------------------------------
        public HandPose LeftHand  { get; private set; } = new HandPose();
        public HandPose RightHand { get; private set; } = new HandPose();

        // -----------------------------------------------------------------------
        // Runtime state
        // -----------------------------------------------------------------------
        private Worker _workerRight, _workerLeft;

        // Normalization stats (11 feature dims, 22 target dims)
        private float[] _featMeanRight, _featStdRight, _tgtMeanRight, _tgtStdRight;
        private float[] _featMeanLeft,  _featStdLeft,  _tgtMeanLeft,  _tgtStdLeft;

        // EMA smoothed UME joint angles (22 per hand)
        private float[] _smoothRight = new float[22];
        private float[] _smoothLeft  = new float[22];
        private bool    _firstRight  = true, _firstLeft = true;

        private int _frameCounter;

        // File-based debug log (bypasses console character limit)
        private string _debugLogPath;
        private System.IO.StreamWriter _debugWriter;

        // UME joint indices that map to MANO flexion angles (skip abduction at 1,4,8,12,16)
        // UME per finger: [abduction, MCP, PIP, DIP] — Thumb[0-3], Index[4-7], Mid[8-11], Ring[12-15], Pinky[16-19]
        // Note: Thumb index 0 = CMC/MCP flex (not abduction) — see mean ~16° in training data
        private static readonly int[] UmeToMano = { 0, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19 };

        // -----------------------------------------------------------------------
        // BOP object → grip category + bbox (from grip_categories.py)
        // grip: 0=Power, 1=Precision, 2=Palmar, 3=Pinch
        // -----------------------------------------------------------------------
        private static readonly Dictionary<int, int> BopToGrip = new Dictionary<int, int>
        {
            { 1,2},{ 2,2},{ 3,0},{ 4,0},{ 5,0},{ 6,0},{ 7,0},{ 8,1},{ 9,0},
            {10,3},{11,0},{12,3},{13,0},{14,0},{15,3},{16,1},{17,3},{18,3},
            {19,0},{20,0},{21,0},{22,0},{23,0},{24,0},{25,1},{26,0},{27,0},
            {28,2},{29,2},{30,2},{31,2},{32,2},{33,2},
        };

        // Half-extents (x,y,z) in metres
        private static readonly Dictionary<int, float[]> BopToBbox = new Dictionary<int, float[]>
        {
            { 1, new float[]{0.043f,0.043f,0.065f}},
            { 2, new float[]{0.040f,0.040f,0.055f}},
            { 3, new float[]{0.030f,0.030f,0.100f}},
            { 4, new float[]{0.030f,0.030f,0.100f}},
            { 5, new float[]{0.035f,0.035f,0.120f}},
            { 6, new float[]{0.035f,0.035f,0.100f}},
            { 7, new float[]{0.030f,0.030f,0.110f}},
            { 8, new float[]{0.020f,0.020f,0.080f}},
            { 9, new float[]{0.045f,0.040f,0.060f}},
            {10, new float[]{0.030f,0.020f,0.015f}},
            {11, new float[]{0.030f,0.030f,0.100f}},
            {12, new float[]{0.025f,0.020f,0.010f}},
            {13, new float[]{0.035f,0.035f,0.095f}},
            {14, new float[]{0.025f,0.025f,0.080f}},
            {15, new float[]{0.020f,0.015f,0.008f}},
            {16, new float[]{0.015f,0.015f,0.070f}},
            {17, new float[]{0.020f,0.010f,0.005f}},
            {18, new float[]{0.015f,0.010f,0.005f}},
            {19, new float[]{0.030f,0.030f,0.080f}},
            {20, new float[]{0.030f,0.030f,0.090f}},
            {21, new float[]{0.030f,0.030f,0.090f}},
            {22, new float[]{0.028f,0.028f,0.085f}},
            {23, new float[]{0.028f,0.028f,0.085f}},
            {24, new float[]{0.028f,0.028f,0.085f}},
            {25, new float[]{0.015f,0.012f,0.060f}},
            {26, new float[]{0.030f,0.030f,0.095f}},
            {27, new float[]{0.030f,0.030f,0.090f}},
            {28, new float[]{0.025f,0.025f,0.040f}},
            {29, new float[]{0.030f,0.030f,0.050f}},
            {30, new float[]{0.025f,0.025f,0.035f}},
            {31, new float[]{0.025f,0.030f,0.045f}},
            {32, new float[]{0.020f,0.020f,0.035f}},
            {33, new float[]{0.030f,0.025f,0.040f}},
        };

        private static readonly float[] DefaultBbox = new float[] { 0.03f, 0.03f, 0.08f };

        // -----------------------------------------------------------------------
        // Unity lifecycle
        // -----------------------------------------------------------------------
        void Start()
        {
            // Open per-session debug log file (survives console truncation)
            _debugLogPath = System.IO.Path.Combine(
                Application.persistentDataPath,
                $"auraxr_debug_{System.DateTime.Now:yyyy_MM_dd_HH_mm_ss}.txt");
            _debugWriter = new System.IO.StreamWriter(_debugLogPath, append: false) { AutoFlush = true };
            DLog($"=== AuraXRInferenceManager debug log started ===");
            DLog($"persistentDataPath: {Application.persistentDataPath}");
            Debug.Log($"[AuraXR] Debug log → {_debugLogPath}");

            // Auto-load from Resources if not assigned in Inspector
            if (rightModelAsset == null)
                rightModelAsset = Resources.Load<ModelAsset>("auraxr_right");
            if (leftModelAsset == null)
                leftModelAsset  = Resources.Load<ModelAsset>("auraxr_left");
            if (rightMetaJson == null)
                rightMetaJson   = Resources.Load<TextAsset>("model_meta_right");
            if (leftMetaJson == null)
                leftMetaJson    = Resources.Load<TextAsset>("model_meta_left");

            if (featureAssembler == null)
                featureAssembler = FindAnyObjectByType<AuraXRFeatureAssembler>();

            LoadModel(rightModelAsset, rightMetaJson,
                out _workerRight, out _featMeanRight, out _featStdRight,
                out _tgtMeanRight, out _tgtStdRight, "right");

            LoadModel(leftModelAsset, leftMetaJson,
                out _workerLeft, out _featMeanLeft, out _featStdLeft,
                out _tgtMeanLeft, out _tgtStdLeft, "left");
        }

        void Update()
        {
            Transform rightCtrl   = featureAssembler?.rightControllerTransform;
            Transform leftCtrl    = featureAssembler?.leftControllerTransform;

            // Anchor virtual hands to controller positions every frame
            if (virtualHandLeft  != null && leftCtrl  != null)
                virtualHandLeft.SetPositionAndRotation(leftCtrl.position, leftCtrl.rotation);
            if (virtualHandRight != null && rightCtrl != null)
                virtualHandRight.SetPositionAndRotation(rightCtrl.position, rightCtrl.rotation);

            _frameCounter++;
            if (_frameCounter % inferenceEveryNFrames != 0) return;

            if (debugBypassModel)
            {
                // Half-curl test pose: all joints at 0.5 rad (~28°).
                // Fingers should visibly curl. If they do → rig OK, turn this OFF and check logs.
                // If they stay flat → HandRigController.fingerJoints not wired or wrong bone axis.
                var testR = new HandPose();
                var testL = new HandPose();
                for (int i = 0; i < 15; i++) { testR.ManoJointAngles[i] = 0.5f; testL.ManoJointAngles[i] = 0.5f; }
                RightHand = testR;
                LeftHand  = testL;
                if (_frameCounter % 120 == 0)
                    DLog("[BYPASS] debugBypassModel=true — all joints 0.5 rad. Disable to use inference.");
                return;
            }

            Transform nearestR    = featureAssembler?.nearestObjectRight;
            Transform nearestL    = featureAssembler?.nearestObjectLeft;
            int       categoryR   = featureAssembler?.nearestObjectCategoryRight ?? 0;
            int       categoryL   = featureAssembler?.nearestObjectCategoryLeft  ?? 0;

            if (_frameCounter % 60 == 0)
                DLog($"[WIRE] nearestR={(nearestR?.name ?? "NULL")} cat={categoryR}  " +
                     $"nearestL={(nearestL?.name ?? "NULL")} cat={categoryL}  " +
                     $"rightCtrl={(rightCtrl?.name ?? "NULL")}  leftCtrl={(leftCtrl?.name ?? "NULL")}  " +
                     $"fasmbl={featureAssembler != null}  wkrR={_workerRight != null}  wkrL={_workerLeft != null}");

            RightHand = RunInference(
                rightCtrl, nearestR, categoryR,
                _workerRight, _featMeanRight, _featStdRight, _tgtMeanRight, _tgtStdRight,
                ref _smoothRight, ref _firstRight,
                RightHand);

            LeftHand = RunInference(
                leftCtrl, nearestL, categoryL,
                _workerLeft, _featMeanLeft, _featStdLeft, _tgtMeanLeft, _tgtStdLeft,
                ref _smoothLeft, ref _firstLeft,
                LeftHand);
        }

        void OnDestroy()
        {
            _workerRight?.Dispose();
            _workerLeft?.Dispose();
            _debugWriter?.Close();
        }

        private void DLog(string msg)
        {
            string line = $"[{Time.frameCount:D6}] {msg}";
            _debugWriter?.WriteLine(line);
            Debug.Log($"[AuraXR] {msg}");
        }

        // -----------------------------------------------------------------------
        // Inference for one hand — returns updated HandPose (or unchanged on skip)
        // -----------------------------------------------------------------------
        private HandPose RunInference(
            Transform ctrl, Transform nearestObj, int categoryId,
            Worker worker,
            float[] featMean, float[] featStd, float[] tgtMean, float[] tgtStd,
            ref float[] smooth, ref bool firstFrame,
            HandPose current)
        {
            if (ctrl == null || nearestObj == null || worker == null)
            {
                if (Time.frameCount % 60 == 0)
                    DLog($"[SKIP] ctrl={(ctrl?.name ?? "NULL")}  nearestObj={(nearestObj?.name ?? "NULL")}  worker={worker != null}");
                return current;
            }

            bool doLog = (Time.frameCount % 60 == 0);

            // 1. Compute direction to object in wrist-local HOT3D frame
            Vector3 wristPosUnity = ctrl.position;
            Vector3 objPosUnity   = nearestObj.position;
            Vector3 wristPosH     = ToHOT3D(wristPosUnity);
            Quaternion wristRotH  = ToHOT3DQuat(ctrl.rotation);
            Vector3 objPosH       = ToHOT3D(objPosUnity);

            Vector3 relWorld = objPosH - wristPosH;
            float dist = relWorld.magnitude;

            if (doLog)
                DLog($"[POS] ctrl(Unity)={wristPosUnity:F3}  obj(Unity)={objPosUnity:F3}  " +
                     $"ctrl(HOT3D)={wristPosH:F3}  obj(HOT3D)={objPosH:F3}  " +
                     $"dist={dist:F3}m  cat={categoryId}  obj={nearestObj.name}");

            if (dist < 1e-6f) return current;

            Vector3 relLocal = Quaternion.Inverse(wristRotH) * relWorld;
            Vector3 dir = relLocal / dist;

            // 2. Look up grip category and bbox
            int grip = BopToGrip.TryGetValue(categoryId, out int g) ? g : 0;
            float[] bbox = BopToBbox.TryGetValue(categoryId, out float[] b) ? b : DefaultBbox;
            string[] gripNames = { "Power", "Precision", "Palmar", "Pinch" };

            // 3. Assemble raw 11-dim feature: [dir(3), dist(1), grip_oh(4), bbox(3)]
            float[] feat = new float[11];
            feat[0] = dir.x; feat[1] = dir.y; feat[2] = dir.z;
            feat[3] = dist;
            feat[4] = grip == 0 ? 1f : 0f; // Power
            feat[5] = grip == 1 ? 1f : 0f; // Precision
            feat[6] = grip == 2 ? 1f : 0f; // Palmar
            feat[7] = grip == 3 ? 1f : 0f; // Pinch
            feat[8] = bbox[0]; feat[9] = bbox[1]; feat[10] = bbox[2];

            if (doLog)
                DLog($"[FEAT_RAW] dir=({feat[0]:F3},{feat[1]:F3},{feat[2]:F3})  dist={feat[3]:F3}  " +
                     $"grip={gripNames[grip]}({grip}) oh=[{feat[4]},{feat[5]},{feat[6]},{feat[7]}]  " +
                     $"bbox=({feat[8]:F3},{feat[9]:F3},{feat[10]:F3})");

            // 4. Normalize
            float[] rawFeat = (float[])feat.Clone();
            for (int i = 0; i < 11; i++)
                feat[i] = (feat[i] - featMean[i]) / featStd[i];

            if (doLog)
            {
                string normStr = string.Join(" ", System.Array.ConvertAll(feat, a => a.ToString("F3")));
                DLog($"[FEAT_NORM] [{normStr}]");
            }

            // 5. Split into spatial (4) and object (7) inputs
            float[] spatialInput = new float[] { feat[0], feat[1], feat[2], feat[3] };
            float[] objectInput  = new float[] { feat[4], feat[5], feat[6], feat[7], feat[8], feat[9], feat[10] };

            // 6. Run model
            using var spatialTensor = new Tensor<float>(new TensorShape(1, 4), spatialInput);
            using var objectTensor  = new Tensor<float>(new TensorShape(1, 7), objectInput);

            worker.SetInput("spatial_input", spatialTensor);
            worker.SetInput("object_input",  objectTensor);
            worker.Schedule();

            var outTensor = worker.PeekOutput("joint_angles") as Tensor<float>;
            if (outTensor == null)
            {
                DLog("[ERROR] PeekOutput('joint_angles') returned null — check ONNX model output name.");
                return current;
            }
            using var cpu = outTensor.ReadbackAndClone();

            if (doLog)
            {
                var rawOut = new float[22];
                for (int i = 0; i < 22; i++) rawOut[i] = cpu[0, i];
                string rawOutStr = string.Join(" ", System.Array.ConvertAll(rawOut, a => a.ToString("F4")));
                DLog($"[MODEL_RAW_OUT] [{rawOutStr}]");
            }

            // 7. Denormalize 22 UME angles
            float[] angles = new float[22];
            for (int i = 0; i < 22; i++)
                angles[i] = cpu[0, i] * tgtStd[i] + tgtMean[i];

            if (doLog)
            {
                string umeStr = string.Join(" ", System.Array.ConvertAll(angles, a => a.ToString("F3")));
                DLog($"[UME_DENORM(rad)] [{umeStr}]");
                string umeDeg = string.Join(" ", System.Array.ConvertAll(angles, a => (a * Mathf.Rad2Deg).ToString("F1")));
                DLog($"[UME_DENORM(deg)] [{umeDeg}]");
            }

            // 8. EMA smoothing over 22 UME angles
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

            // 9. Map 22 UME → 15 MANO flexion angles
            var pose = new HandPose();
            for (int m = 0; m < 15; m++)
                pose.ManoJointAngles[m] = smooth[UmeToMano[m]];

            if (doLog)
            {
                string[] manoNames = { "Th.MCP","Th.PIP","Th.DIP","Idx.MCP","Idx.PIP","Idx.DIP",
                                       "Mid.MCP","Mid.PIP","Mid.DIP","Rng.MCP","Rng.PIP","Rng.DIP",
                                       "Pnk.MCP","Pnk.PIP","Pnk.DIP" };
                var sb = new System.Text.StringBuilder();
                sb.Append("[MANO(deg)]");
                for (int m = 0; m < 15; m++)
                    sb.Append($"  {manoNames[m]}={pose.ManoJointAngles[m] * Mathf.Rad2Deg:F1}");
                DLog(sb.ToString());
            }

            return pose;
        }

        // -----------------------------------------------------------------------
        // Coordinate frame conversion: Unity (left-handed) → HOT3D (right-handed)
        // -----------------------------------------------------------------------
        private static Vector3 ToHOT3D(Vector3 v)
            => new Vector3(v.x, v.y, -v.z);

        private static Quaternion ToHOT3DQuat(Quaternion q)
            => new Quaternion(-q.x, -q.y, q.z, q.w);

        // -----------------------------------------------------------------------
        // Load model + parse meta JSON
        // -----------------------------------------------------------------------
        private static void LoadModel(
            ModelAsset asset, TextAsset metaJson,
            out Worker worker,
            out float[] featMean, out float[] featStd,
            out float[] tgtMean,  out float[] tgtStd,
            string tag)
        {
            worker = null;
            featMean = featStd = tgtMean = tgtStd = null;

            if (asset == null)
            {
                Debug.LogError($"[AuraXR] {tag} model asset not assigned.");
                return;
            }
            if (metaJson == null)
            {
                Debug.LogError($"[AuraXR] {tag} meta JSON not assigned.");
                return;
            }

            var model = ModelLoader.Load(asset);
            worker = new Worker(model, BackendType.GPUCompute);

            var meta = JsonUtility.FromJson<ModelMeta>(metaJson.text);
            featMean = meta.feature_mean;
            featStd  = meta.feature_std;
            tgtMean  = meta.target_mean;
            tgtStd   = meta.target_std;

            for (int i = 0; i < featStd.Length; i++)
                if (featStd[i] < 1e-6f) featStd[i] = 1f;
            for (int i = 0; i < tgtStd.Length; i++)
                if (tgtStd[i] < 1e-6f) tgtStd[i] = 1f;

            Debug.Log($"[AuraXR] {tag} model loaded. feat_dims={featMean.Length} tgt_dims={tgtMean.Length}");
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
        }
    }

    // -----------------------------------------------------------------------
    // Shared output type — read by HandRigController / AuraXRHandRenderer
    // -----------------------------------------------------------------------
    [Serializable]
    public class HandPose
    {
        public float[]     ManoJointAngles;   // 15 floats, MANO order (Thumb MCP/PIP/DIP, Index MCP/PIP/DIP, ...)
        public float[]     ManoShapeBetas;    // 10 zeros (shape not predicted)
        public Vector3     WristPosition;
        public Quaternion  WristRotation;
        public Vector3     DeltaPosition;
        public Quaternion  DeltaRotation;

        public HandPose()
        {
            ManoJointAngles = new float[15];
            ManoShapeBetas  = new float[10];
            WristPosition   = Vector3.zero;
            WristRotation   = Quaternion.identity;
            DeltaPosition   = Vector3.zero;
            DeltaRotation   = Quaternion.identity;
        }
    }
}
