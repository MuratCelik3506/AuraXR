/*
 * IntentPredictor.cs
 * ==================
 * Unity Sentis bridge — runs the IntentFormer CoreML model in real-time.
 *
 * Responsibilities:
 *   1. Maintain a sliding circular buffer of hand-pose + object-pose frames.
 *   2. Invoke the Sentis engine asynchronously every `inferenceIntervalMs` ms.
 *   3. Publish per-class probabilities and the Top-1 action intent.
 *   4. Fire the OnIntentConfident event when confidence ≥ ghostingThreshold.
 *
 * Requirements:
 *   • Unity 2023.2+ with com.unity.sentis package installed.
 *   • IntentFormer.mlpackage embedded in Assets/StreamingAssets/ OR
 *     converted to sentis format via the Sentis Model Asset Importer.
 *
 * Usage:
 *   Attach to any persistent GameObject.
 *   Assign the model asset and hand/object tracking sources in the Inspector.
 */

using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Sentis;

namespace XRIntent
{
    /// <summary>
    /// Predicted intent payload published each inference cycle.
    /// </summary>
    [Serializable]
    public class IntentResult
    {
        public int   TopClass;          // 0-indexed action class (0..35)
        public float Confidence;        // max softmax prob
        public float[] Probabilities;   // full distribution (36 elements)
        public float ObsRatio;          // observation ratio used
        public float TTC;               // estimated Time-to-Contact (seconds)
        public float[] PredPose;        // predicted next-frame joints (126 elements)
        public long  InferenceLatencyUs;// inference latency in microseconds
    }

    /// <summary>
    /// Unity Sentis inference bridge for the IntentFormer model.
    /// </summary>
    public class IntentPredictor : MonoBehaviour
    {
        // ── Inspector ────────────────────────────────────────────
        [Header("Model")]
        [Tooltip("Drag the imported Sentis model asset here.")]
        public ModelAsset modelAsset;

        [Header("Inference Settings")]
        [Range(16.67f, 200f)]
        [Tooltip("How often (ms) to run inference. Default = 33ms ≈ 30 fps.")]
        public float inferenceIntervalMs = 33f;

        [Range(0.51f, 0.99f)]
        [Tooltip("Confidence threshold that triggers the Ghosting UX.")]
        public float ghostingThreshold = 0.65f;

        [Tooltip("Number of frames in the sliding window (must match model training).")]
        public int windowSize = 30;

        [Tooltip("Observation ratio sent to the model as context.")]
        [Range(0.1f, 0.5f)]
        public float currentObsRatio = 0.25f;

        [Header("Data Sources")]
        [Tooltip("Script providing 21-joint left hand data each frame.")]
        public HandPoseProvider leftHandProvider;
        [Tooltip("Script providing 21-joint right hand data each frame.")]
        public HandPoseProvider rightHandProvider;
        [Tooltip("Script providing the target object RT matrix each frame.")]
        public ObjectPoseProvider objectPoseProvider;

        // ── Events ───────────────────────────────────────────────
        /// <summary>Fired after every inference cycle.</summary>
        public event Action<IntentResult> OnIntentUpdated;

        /// <summary>
        /// Fired when max confidence ≥ ghostingThreshold.
        /// Parameter = the IntentResult that crossed the threshold.
        /// </summary>
        public event Action<IntentResult> OnIntentConfident;

        // ── Internal state ───────────────────────────────────────
        private const int NumJoints   = 21;
        private const int NumHands    = 2;
        private const int PosDim      = NumHands * NumJoints * 3; // 126
        private const int HandFlatDim = PosDim * 3;               // 378 (pos + vel + acc)
        private const int ObjRtDim    = 16;
        private const int NumClasses  = 36;

        private IWorker    _worker;
        private Model      _runtimeModel;

        // Circular buffer — each slot: float[126] + float[16]
        private Queue<float[]> _handBuffer = new();
        private Queue<float[]> _objBuffer  = new();

        private float _inferenceTimer;
        private bool  _isModelReady;

        // ── Lifecycle ────────────────────────────────────────────

        private void Awake()
        {
            if (modelAsset == null)
            {
                Debug.LogError("[IntentPredictor] No model asset assigned!");
                return;
            }

            _runtimeModel = ModelLoader.Load(modelAsset);
            _worker      = WorkerFactory.CreateWorker(BackendType.GPUCompute,
                                                      _runtimeModel);
            _isModelReady = true;
            Debug.Log("[IntentPredictor] Model loaded. Worker ready.");
        }

        private void OnDestroy()
        {
            _worker?.Dispose();
        }

        // ── Per-frame update ─────────────────────────────────────

        private void Update()
        {
            if (!_isModelReady) return;

            // 1. Collect a new frame into the circular buffer
            CollectFrame();

            // 2. Throttled inference
            _inferenceTimer += Time.deltaTime * 1000f;  // ms
            if (_inferenceTimer >= inferenceIntervalMs && _handBuffer.Count >= windowSize)
            {
                _inferenceTimer = 0f;
                RunInference();
            }
        }

        // ── Feature collection ───────────────────────────────────

        private void CollectFrame()
        {
            // Build hand_flat: wrist-relative, both hands flattened (126 floats)
            float[] handFlat = BuildHandFlat();
            float[] objRt    = objectPoseProvider != null
                ? objectPoseProvider.GetCurrentRT()
                : new float[ObjRtDim];

            _handBuffer.Enqueue(handFlat);
            _objBuffer.Enqueue(objRt);

            // Keep only the last `windowSize` frames
            while (_handBuffer.Count > windowSize) _handBuffer.Dequeue();
            while (_objBuffer.Count  > windowSize) _objBuffer.Dequeue();
        }

        private float[] BuildHandFlat()
        {
            float[] flat = new float[PosDim];

            // Left hand: joints 0..62 (21 × 3)
            if (leftHandProvider != null)
            {
                Vector3[] lJoints = leftHandProvider.GetJointsWorldSpace();
                Vector3   lWrist  = lJoints[0];
                for (int j = 0; j < NumJoints; j++)
                {
                    Vector3 rel = lJoints[j] - lWrist;   // wrist-relative
                    int off = j * 3;
                    flat[off]     = rel.x;
                    flat[off + 1] = rel.y;
                    flat[off + 2] = rel.z;
                }
            }

            // Right hand: joints 63..125 (21 × 3)
            if (rightHandProvider != null)
            {
                Vector3[] rJoints = rightHandProvider.GetJointsWorldSpace();
                Vector3   rWrist  = rJoints[0];
                for (int j = 0; j < NumJoints; j++)
                {
                    Vector3 rel = rJoints[j] - rWrist;
                    int off = NumJoints * 3 + j * 3;
                    flat[off]     = rel.x;
                    flat[off + 1] = rel.y;
                    flat[off + 2] = rel.z;
                }
            }

            return flat;
        }

            return flat;
        }

        // ── Sentis inference ──────────────────────────────────────

        private void RunInference()
        {
            long t0 = System.Diagnostics.Stopwatch.GetTimestamp();

            // 1. Prepare Kinematic Features (T, 378)
            float[] kinematicBuffer = new float[windowSize * HandFlatDim];
            float[][] bufferArray = _handBuffer.ToArray();

            for (int t = 0; t < windowSize; t++)
            {
                // Position (0..125)
                Array.Copy(bufferArray[t], 0, kinematicBuffer, t * HandFlatDim, PosDim);
                
                // Velocity (126..251): x[t] - x[t-1]
                if (t > 0)
                {
                    for (int d = 0; d < PosDim; d++)
                        kinematicBuffer[t * HandFlatDim + PosDim + d] = 
                            bufferArray[t][d] - bufferArray[t-1][d];
                }
                
                // Acceleration (252..377): v[t] - v[t-1]
                if (t > 1)
                {
                    for (int d = 0; d < PosDim; d++)
                    {
                        float v_now = bufferArray[t][d] - bufferArray[t-1][d];
                        float v_prev = bufferArray[t-1][d] - bufferArray[t-2][d];
                        kinematicBuffer[t * HandFlatDim + 2 * PosDim + d] = v_now - v_prev;
                    }
                }
            }

            float[] objRts = new float[windowSize * ObjRtDim];
            int i = 0;
            foreach (float[] o in _objBuffer)
            {
                Array.Copy(o, 0, objRts, i * ObjRtDim, ObjRtDim);
                i++;
            }

            // Create Sentis tensors
            using var tHand = new TensorFloat(
                new TensorShape(1, windowSize, HandFlatDim), kinematicBuffer);
            using var tObj  = new TensorFloat(
                new TensorShape(1, windowSize, ObjRtDim),   objRts);
            using var tObs  = new TensorFloat(
                new TensorShape(1), new[] { currentObsRatio });

            // Execute model
            _worker.SetInput("hand_flat",  tHand);
            _worker.SetInput("obj_rt",     tObj);
            _worker.SetInput("obs_ratio",  tObs);
            _worker.Schedule();

            // Read outputs
            using var logitsTensor = _worker.PeekOutput("logits") as TensorFloat;
            logitsTensor?.MakeReadable();

            using var poseTensor = _worker.PeekOutput("pred_pose") as TensorFloat;
            poseTensor?.MakeReadable();

            float[] logits = new float[NumClasses];
            for (int c = 0; c < NumClasses; c++)
                logits[c] = logitsTensor?[0, c] ?? 0f;

            float[] predPose = new float[PosDim];
            for (int d = 0; d < PosDim; d++)
                predPose[d] = poseTensor?[0, d] ?? 0f;

            // Softmax
            float[] probs = Softmax(logits);

            // Top-1
            int   topClass = 0;
            float topProb  = 0f;
            for (int c = 0; c < NumClasses; c++)
                if (probs[c] > topProb) { topProb = probs[c]; topClass = c; }

            long latencyTicks = System.Diagnostics.Stopwatch.GetTimestamp() - t0;
            long latencyUs    = latencyTicks * 1_000_000L
                / System.Diagnostics.Stopwatch.Frequency;

            var result = new IntentResult
            {
                TopClass           = topClass,
                Confidence         = topProb,
                Probabilities      = probs,
                ObsRatio           = currentObsRatio,
                TTC                = EstimateTTC(),
                PredPose           = predPose,
                InferenceLatencyUs = latencyUs,
            };

            OnIntentUpdated?.Invoke(result);

            if (topProb >= ghostingThreshold)
                OnIntentConfident?.Invoke(result);
        }

        // ── Helpers ──────────────────────────────────────────────

        private static float[] Softmax(float[] logits)
        {
            float max = float.NegativeInfinity;
            foreach (float v in logits) if (v > max) max = v;

            float[] exps = new float[logits.Length];
            float sum = 0f;
            for (int i = 0; i < logits.Length; i++)
            {
                exps[i] = MathF.Exp(logits[i] - max);
                sum += exps[i];
            }
            for (int i = 0; i < exps.Length; i++)
                exps[i] /= sum;
            return exps;
        }

        /// <summary>
        /// Rough TTC estimate based on current obs_ratio.
        /// If obs_ratio = 0.25, then 75% of the motion remains.
        /// Assuming fixed action length of `windowSize / obs_ratio` frames at 30 fps.
        /// </summary>
        private float EstimateTTC()
        {
            if (currentObsRatio <= 0f) return 0f;
            float totalFrames     = windowSize / currentObsRatio;
            float remainingFrames = totalFrames * (1f - currentObsRatio);
            return remainingFrames / 30f;   // seconds at 30 fps
        }
    }
}
