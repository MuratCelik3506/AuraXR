using System;
using UnityEngine;
using Unity.InferenceEngine;

namespace AuraXR
{
    /// <summary>
    /// Runs IntentFormer inference every N frames via Unity Sentis.
    /// Decodes the 78-dim output into per-hand pose data.
    ///
    /// Output layout (matches intentformer_meta.json):
    ///   [0..14]   mano_pose_h0   (15 joint angles)
    ///   [15..24]  mano_betas_h0  (10 shape params)
    ///   [25..27]  wrist_t_h0     (world position, metres)
    ///   [28..31]  wrist_q_h0     (quaternion w,x,y,z)
    ///   [32..34]  delta_t_h0     (controller→wrist offset, metres)
    ///   [35..38]  delta_q_h0     (controller→wrist quaternion w,x,y,z)
    ///   [39..77]  same for hand 1
    /// </summary>
    public class AuraXRInferenceManager : MonoBehaviour
    {
        [Header("Model assets")]
        [Tooltip("Drag intentformer.onnx here")]
        public ModelAsset modelAsset;

        [Header("Dependencies")]
        public AuraXRMetaLoader    metaLoader;
        public AuraXRFeatureAssembler featureAssembler;

        [Header("Virtual hand anchors (assigned by scene setup)")]
        public Transform virtualHandLeft;
        public Transform virtualHandRight;

        [Header("Frame Rate Handling")]
        [Tooltip("Run inference every N frames (default 2-3 to match ~30 FPS training rate at 72 Hz)")]
        public int inferenceEveryNFrames = 2;

        [Header("Debug")]
        [Tooltip("Skip ONNX output — hand tracks controller directly. Use to test grab/visibility with untrained model.")]
        public bool debugBypassModel = false;

        // -----------------------------------------------------------------------
        // Public output — read by hand animators each frame
        // -----------------------------------------------------------------------
        public HandPose LeftHand  { get; private set; } = new HandPose();
        public HandPose RightHand { get; private set; } = new HandPose();

        // -----------------------------------------------------------------------
        // Sentis runtime
        // -----------------------------------------------------------------------
        private Model           _runtimeModel;
        private Worker          _worker;
        private Tensor<float>   _inputTensor;

        // Flat buffer [T * F] reused each frame
        private float[] _flatWindow;

        // -----------------------------------------------------------------------
        // Frame skipping & interpolation state
        // -----------------------------------------------------------------------
        private int _frameCounter = 0;
        private HandPose _rawLeftHand   = new HandPose();
        private HandPose _rawRightHand  = new HandPose();
        private HandPose _lastLeftHand  = new HandPose();
        private HandPose _lastRightHand = new HandPose();
        private HandPose _prevLeftHand  = new HandPose();
        private HandPose _prevRightHand = new HandPose();
        private float _blendAlpha = 0f;

        void Awake()
        {
            if (modelAsset == null)
            {
                Debug.LogError("[AuraXR] modelAsset not assigned.");
                return;
            }

            _runtimeModel = ModelLoader.Load(modelAsset);
            _worker       = new Worker(_runtimeModel, BackendType.GPUCompute);
            _flatWindow   = new float[AuraXRFeatureAssembler.WindowFrames * AuraXRFeatureAssembler.FeatureDim];

            Debug.Log("[AuraXR] Model loaded. Input: [1,16,96]  Output: [1,78]");
        }

        void OnDestroy()
        {
            _worker?.Dispose();
            _inputTensor?.Dispose();
        }

        void LateUpdate()
        {
            if (!metaLoader.IsReady || !featureAssembler.IsReady) return;

            _frameCounter++;

            if (_frameCounter % inferenceEveryNFrames == 0)
            {
                RunInference();

                _prevLeftHand  = _lastLeftHand;
                _prevRightHand = _lastRightHand;
                _lastLeftHand  = _rawLeftHand;
                _lastRightHand = _rawRightHand;
                // Reset alpha so we lerp smoothly from old pose to new pose over the next N frames
                _blendAlpha = inferenceEveryNFrames <= 1 ? 1f : 0f;
            }
            else
            {
                // Step alpha forward: reaches 1.0 just as the next inference fires
                float step = inferenceEveryNFrames <= 1 ? 1f : 1f / (inferenceEveryNFrames - 1);
                _blendAlpha = Mathf.Min(1f, _blendAlpha + step);
            }

            LeftHand  = InterpolateHandPose(_prevLeftHand,  _lastLeftHand,  _blendAlpha);
            RightHand = InterpolateHandPose(_prevRightHand, _lastRightHand, _blendAlpha);

            ApplyToAnchor(virtualHandLeft,  LeftHand,  featureAssembler.leftControllerTransform);
            ApplyToAnchor(virtualHandRight, RightHand, featureAssembler.rightControllerTransform);
        }

        // -----------------------------------------------------------------------
        // Run the ONNX model and store decoded poses in _rawLeftHand / _rawRightHand
        // -----------------------------------------------------------------------
        private void RunInference()
        {
            if (debugBypassModel)
            {
                _rawLeftHand  = new HandPose();
                _rawRightHand = new HandPose();
                return;
            }

            // 1. Collect feature window (raw, pre-normalisation)
            featureAssembler.CopyWindowFlat(_flatWindow);

            int T = AuraXRFeatureAssembler.WindowFrames;
            int F = AuraXRFeatureAssembler.FeatureDim;

#if UNITY_EDITOR
            // ── INPUT LOG ────────────────────────────────────────────────────────
            // Snapshot the latest frame (index T-1) before we overwrite with normalised values.
            // Layout: [0..2]=posL [3..6]=rotL [7]=gripL [8]=trigL
            //         [9..11]=posR [12..15]=rotR [16]=gripR [17]=trigR
            int latestBase = (T - 1) * F;
            Vector3    dbgPosL = new Vector3(_flatWindow[latestBase+0], _flatWindow[latestBase+1], _flatWindow[latestBase+2]);
            Quaternion dbgRotL = new Quaternion(_flatWindow[latestBase+4], _flatWindow[latestBase+5], _flatWindow[latestBase+6], _flatWindow[latestBase+3]);
            float      dbgGripL = _flatWindow[latestBase+7], dbgTrigL = _flatWindow[latestBase+8];
            Vector3    dbgPosR = new Vector3(_flatWindow[latestBase+9], _flatWindow[latestBase+10], _flatWindow[latestBase+11]);
            Quaternion dbgRotR = new Quaternion(_flatWindow[latestBase+13], _flatWindow[latestBase+14], _flatWindow[latestBase+15], _flatWindow[latestBase+12]);
            float      dbgGripR = _flatWindow[latestBase+16], dbgTrigR = _flatWindow[latestBase+17];

            Debug.Log($"[AuraXR|INPUT-L] frame={Time.frameCount}  pos={dbgPosL:F3}  rot={dbgRotL.eulerAngles:F1}  grip={dbgGripL:F2}  trigger={dbgTrigL:F2}");
            Debug.Log($"[AuraXR|INPUT-R] frame={Time.frameCount}  pos={dbgPosR:F3}  rot={dbgRotR.eulerAngles:F1}  grip={dbgGripR:F2}  trigger={dbgTrigR:F2}");
#endif

            // 2. Normalise window in-place
            for (int t = 0; t < T; t++)
                for (int f = 0; f < F; f++)
                {
                    int idx = t * F + f;
                    _flatWindow[idx] = (_flatWindow[idx] - metaLoader.FeatureMean[f])
                                       / metaLoader.FeatureStd[f];
                }

            // 3. Build Sentis input tensor [1, 16, 96]
            _inputTensor?.Dispose();
            _inputTensor = new Tensor<float>(new TensorShape(1, T, F), _flatWindow);

            // 4. Run inference
            _worker.Schedule(_inputTensor);
            var outputTensor = _worker.PeekOutput("pose") as Tensor<float>;
            using var cpuTensor = outputTensor.ReadbackAndClone();

            // 5. Copy raw output (pre-denorm)
            float[] raw = new float[78];
            for (int i = 0; i < 78; i++)
                raw[i] = cpuTensor[0, i];

#if UNITY_EDITOR
            // ── RAW OUTPUT LOG (pre-denorm) ───────────────────────────────────────
            // If these are identical every frame the model is stuck / collapsed to mean.
            string preL  = $"{raw[0]:F4} {raw[1]:F4} {raw[2]:F4} {raw[3]:F4} {raw[4]:F4}";
            string preR  = $"{raw[39]:F4} {raw[40]:F4} {raw[41]:F4} {raw[42]:F4} {raw[43]:F4}";
            string preDL = $"{raw[32]:F4} {raw[33]:F4} {raw[34]:F4}";
            string preDR = $"{raw[71]:F4} {raw[72]:F4} {raw[73]:F4}";
            Debug.Log($"[AuraXR|RAW-PRE-DENORM]\n" +
                      $"  L pose[0..4]: {preL}  delta_t: {preDL}\n" +
                      $"  R pose[0..4]: {preR}  delta_t: {preDR}");
#endif

            // 6. De-normalise
            metaLoader.DenormaliseTarget(raw);

            // 7. Decode into raw HandPose structs
            _rawLeftHand  = DecodeHand(raw, offset: 0);
            _rawRightHand = DecodeHand(raw, offset: 39);

#if UNITY_EDITOR
            // ── OUTPUT LOG (post-denorm, decoded) ────────────────────────────────
            string lDeg = string.Join(" ", System.Array.ConvertAll(_rawLeftHand.ManoJointAngles,
                              a => (a * Mathf.Rad2Deg).ToString("F1")));
            string rDeg = string.Join(" ", System.Array.ConvertAll(_rawRightHand.ManoJointAngles,
                              a => (a * Mathf.Rad2Deg).ToString("F1")));

            Debug.Log($"[AuraXR|OUT-L] frame={Time.frameCount}  wristPos={_rawLeftHand.WristPosition:F3}  wristRot={_rawLeftHand.WristRotation.eulerAngles:F1}  deltaPos={_rawLeftHand.DeltaPosition:F4}");
            Debug.Log($"[AuraXR|OUT-L-FINGERS] {lDeg}");
            Debug.Log($"[AuraXR|OUT-R] frame={Time.frameCount}  wristPos={_rawRightHand.WristPosition:F3}  wristRot={_rawRightHand.WristRotation.eulerAngles:F1}  deltaPos={_rawRightHand.DeltaPosition:F4}");
            Debug.Log($"[AuraXR|OUT-R-FINGERS] {rDeg}");

            // ── ANCHOR PLACEMENT LOG ─────────────────────────────────────────────
            string lCtrl  = featureAssembler.leftControllerTransform  != null ? featureAssembler.leftControllerTransform.position.ToString("F3")  : "NULL";
            string rCtrl  = featureAssembler.rightControllerTransform != null ? featureAssembler.rightControllerTransform.position.ToString("F3") : "NULL";
            string lAnchor = virtualHandLeft  != null ? virtualHandLeft.position.ToString("F3")  : "NULL";
            string rAnchor = virtualHandRight != null ? virtualHandRight.position.ToString("F3") : "NULL";
            Debug.Log($"[AuraXR|ANCHOR] L ctrl={lCtrl} -> anchor={lAnchor}   R ctrl={rCtrl} -> anchor={rAnchor}");
#endif
        }

        // -----------------------------------------------------------------------
        // Lerp/slerp between two poses for smooth inter-frame display
        // -----------------------------------------------------------------------
        private static HandPose InterpolateHandPose(HandPose from, HandPose to, float t)
        {
            if (from.ManoJointAngles == null) return to;
            if (to.ManoJointAngles   == null) return from;

            var result = new HandPose();
            result.WristPosition = Vector3.Lerp(from.WristPosition, to.WristPosition, t);
            result.WristRotation = Quaternion.Slerp(from.WristRotation, to.WristRotation, t);
            result.DeltaPosition = Vector3.Lerp(from.DeltaPosition, to.DeltaPosition, t);
            result.DeltaRotation = Quaternion.Slerp(from.DeltaRotation, to.DeltaRotation, t);

            result.ManoJointAngles = new float[to.ManoJointAngles.Length];
            for (int i = 0; i < to.ManoJointAngles.Length; i++)
                result.ManoJointAngles[i] = Mathf.Lerp(from.ManoJointAngles[i], to.ManoJointAngles[i], t);

            return result;
        }

        // -----------------------------------------------------------------------
        // Decode one hand's 39-dim block starting at 'offset'
        // -----------------------------------------------------------------------
        private static HandPose DecodeHand(float[] raw, int offset)
        {
            var p = new HandPose();

            // MANO joint angles (15)
            p.ManoJointAngles = new float[15];
            Array.Copy(raw, offset, p.ManoJointAngles, 0, 15);

            // Shape betas (10)
            p.ManoShapeBetas = new float[10];
            Array.Copy(raw, offset + 15, p.ManoShapeBetas, 0, 10);

            // Wrist world position
            p.WristPosition = new Vector3(raw[offset + 25], raw[offset + 26], raw[offset + 27]);

            // Wrist quaternion (w,x,y,z)
            var wq = new Quaternion(raw[offset + 29], raw[offset + 30], raw[offset + 31], raw[offset + 28]);
            p.WristRotation = wq.normalized;

            // Controller→wrist offset translation
            p.DeltaPosition = new Vector3(raw[offset + 32], raw[offset + 33], raw[offset + 34]);

            // Controller→wrist offset quaternion (w,x,y,z)
            var dq = new Quaternion(raw[offset + 36], raw[offset + 37], raw[offset + 38], raw[offset + 35]);
            p.DeltaRotation = dq.normalized;

            return p;
        }

        // -----------------------------------------------------------------------
        // Place virtual hand anchor using delta from controller
        // VirtualWrist = ControllerPos + delta_t
        // VirtualWristRot = ControllerRot * delta_q
        // -----------------------------------------------------------------------
        private static void ApplyToAnchor(Transform anchor, HandPose pose, Transform controller)
        {
            if (anchor == null || controller == null) return;

            anchor.position = controller.position + pose.DeltaPosition;
            anchor.rotation = controller.rotation * pose.DeltaRotation;
        }
    }

    // -----------------------------------------------------------------------
    // Data holder for one hand's decoded output
    // -----------------------------------------------------------------------
    [Serializable]
    public class HandPose
    {
        public float[]     ManoJointAngles;   // 15 floats
        public float[]     ManoShapeBetas;    // 10 floats
        public Vector3     WristPosition;     // world space
        public Quaternion  WristRotation;
        public Vector3     DeltaPosition;     // controller→wrist
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
