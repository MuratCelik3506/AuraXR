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
        private const int HandFlatDim = NumHands * NumJoints * 3; // 126
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
            float[] flat = new float[HandFlatDim];

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

        // ── Sentis inference ──────────────────────────────────────

        private void RunInference()
        {
            long t0 = System.Diagnostics.Stopwatch.GetTimestamp();

            // Flatten the circular buffer into (1, T, D) tensors
            float[] handFlats = new float[windowSize * HandFlatDim];
            float[] objRts    = new float[windowSize * ObjRtDim];

            int i = 0;
            foreach (float[] h in _handBuffer)
            {
                Array.Copy(h, 0, handFlats, i * HandFlatDim, HandFlatDim);
                i++;
            }
            i = 0;
            foreach (float[] o in _objBuffer)
            {
                Array.Copy(o, 0, objRts, i * ObjRtDim, ObjRtDim);
                i++;
            }

            // Create Sentis tensors
            using var tHand = new TensorFloat(
                new TensorShape(1, windowSize, HandFlatDim), handFlats);
            using var tObj  = new TensorFloat(
                new TensorShape(1, windowSize, ObjRtDim),   objRts);
            using var tObs  = new TensorFloat(
                new TensorShape(1), new[] { currentObsRatio });

            // Execute model
            _worker.SetInput("hand_flat",  tHand);
            _worker.SetInput("obj_rt",     tObj);
            _worker.SetInput("obs_ratio",  tObs);
            _worker.Schedule();

            // Read logits
            using var logitsTensor = _worker.PeekOutput("logits") as TensorFloat;
            logitsTensor?.MakeReadable();

            float[] logits = new float[NumClasses];
            for (int c = 0; c < NumClasses; c++)
                logits[c] = logitsTensor?[0, c] ?? 0f;

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
