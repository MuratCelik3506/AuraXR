using System.Diagnostics;
using UnityEngine;

#if AURAXR_USE_INFERENCE_ENGINE
using Unity.InferenceEngine;
#endif

namespace AuraXR.Demo
{
    public sealed class AuraXRModelRuntime : MonoBehaviour
    {
        [Header("Inputs")]
        public AuraXRFeatureAssembler featureAssembler;

        [Header("Runtime")]
        [Tooltip("If true, produces a deterministic open-hand fallback without Sentis.")]
        public bool bypassModel = true;

#if AURAXR_USE_INFERENCE_ENGINE
        [Tooltip("Assign grasp_model.onnx imported as a Unity InferenceEngine ModelAsset.")]
        public ModelAsset modelAsset;
        [Tooltip("Fallback path relative to Application.streamingAssetsPath when modelAsset is not available.")]
        public string modelStreamingRelativePath = "AuraXR/grasp_model.onnx";
        public BackendType backend = BackendType.GPUCompute;
#endif

        [Header("State")]
        public bool hasOutput;
        public float qualityScore;
        public float successProb;
        public float latencyMs;
        public int inferenceCount;

        public AuraXRModelOutput Output { get; } = new AuraXRModelOutput();
        public float[] PrevPose => _prevPose;

        private readonly float[] _frame = new float[AuraXRFeatureAssembler.Window * AuraXRFeatureAssembler.FrameDim];
        private readonly float[] _contact = new float[AuraXRFeatureAssembler.Window];
        private readonly float[] _prevPose = new float[45];
        private readonly Stopwatch _sw = new Stopwatch();
        private bool _inferenceSymbolWarned;

        // Cached per active object — rebuilt only when object changes.
        private AuraXRDemoObject _cachedPtsObject;
        private readonly float[] _objFlat = new float[1024 * 3];

#if AURAXR_USE_INFERENCE_ENGINE
        private Model _model;
        private Worker _worker;

        // Persistent input tensors — allocated once, updated in-place each frame.
        private Tensor<float> _frameTensor;
        private Tensor<float> _objTensor;
        private Tensor<float> _contactTensor;
        private Tensor<float> _prevPoseTensor;

        // Async readback fence for GPU→CPU output transfer.
        private bool _readbackPending;
        private Tensor<float> _poseOutput;
        private Tensor<float> _qualityOutput;
        private Tensor<float> _successOutput;
#endif

        void Start()
        {
#if AURAXR_USE_INFERENCE_ENGINE
            if (!bypassModel)
            {
                if (modelAsset != null)
                    _model = ModelLoader.Load(modelAsset);
                else
                {
                    string path = System.IO.Path.Combine(Application.streamingAssetsPath, modelStreamingRelativePath);
                    _model = ModelLoader.Load(path);
                }
                _worker = new Worker(_model, backend);

                // Allocate persistent input tensors once.
                _frameTensor    = new Tensor<float>(new TensorShape(1, AuraXRFeatureAssembler.Window, AuraXRFeatureAssembler.FrameDim));
                _objTensor      = new Tensor<float>(new TensorShape(1, 1024, 3));
                _contactTensor  = new Tensor<float>(new TensorShape(1, AuraXRFeatureAssembler.Window, 1));
                _prevPoseTensor = new Tensor<float>(new TensorShape(1, 45));
            }
#endif
        }

        void OnDestroy()
        {
#if AURAXR_USE_INFERENCE_ENGINE
            _worker?.Dispose();
            _frameTensor?.Dispose();
            _objTensor?.Dispose();
            _contactTensor?.Dispose();
            _prevPoseTensor?.Dispose();
#endif
        }

        void Update()
        {
            if (featureAssembler == null || !featureAssembler.IsReady)
            {
                hasOutput = false;
                return;
            }

            // Only run inference when the feature assembler produced a new frame.
            if (!featureAssembler.NewFrameReady) return;

            RunInference();
        }

        public bool RunInference()
        {
            featureAssembler.CopyOrderedWindow(_frame, _contact);
            _sw.Restart();

            if (bypassModel)
            {
                FillBypassOutput();
            }
            else
            {
#if AURAXR_USE_INFERENCE_ENGINE
                if (!RunInferenceEngine()) return false;
#else
                if (!_inferenceSymbolWarned)
                {
                    UnityEngine.Debug.LogWarning("[AuraXRModelRuntime] AURAXR_USE_INFERENCE_ENGINE is not defined; using bypass output.");
                    _inferenceSymbolWarned = true;
                }
                FillBypassOutput();
#endif
            }

            _sw.Stop();
            Output.latencyMs = (float)_sw.Elapsed.TotalMilliseconds;
            latencyMs = Output.latencyMs;
            qualityScore = Output.qualityScore;
            successProb = Output.successProb;
            hasOutput = Output.valid;
            inferenceCount++;

            if (Output.valid)
                System.Array.Copy(Output.selectedPose, _prevPose, _prevPose.Length);

            return Output.valid;
        }

        private void FillBypassOutput()
        {
            for (int i = 0; i < Output.selectedPose.Length; i++) Output.selectedPose[i] = 0f;
            Output.qualityScore = 0f;
            Output.successProb = 0f;
            Output.valid = true;
        }

#if AURAXR_USE_INFERENCE_ENGINE
        private bool RunInferenceEngine()
        {
            if (_worker == null)
            {
                UnityEngine.Debug.LogError("[AuraXRModelRuntime] InferenceEngine worker is null. Assign modelAsset or enable bypassModel.");
                return false;
            }

            // Refresh obj_pts only when the active object changes.
            RebuildObjFlatIfNeeded();

            // Upload new data into the persistent tensors.
            _frameTensor.Upload(_frame, 0);
            _objTensor.Upload(_objFlat, 0);

            // contact_flat has shape (Window,) but tensor is (1, Window, 1) — interleave into contact buffer.
            float[] contactInterleaved = _contactTensor.dataOnBackend == null ? new float[AuraXRFeatureAssembler.Window] : null;
            // Upload directly — _contact already matches the flat layout for (1,W,1).
            _contactTensor.Upload(_contact, 0);
            _prevPoseTensor.Upload(_prevPose, 0);

            _worker.SetInput("frame_feat",    _frameTensor);
            _worker.SetInput("obj_pts",       _objTensor);
            _worker.SetInput("contact_flag",  _contactTensor);
            _worker.SetInput("prev_pose",     _prevPoseTensor);
            _worker.Schedule();

            // Peek outputs — still on GPU if GPUCompute backend.
            _poseOutput    = _worker.PeekOutput("selected_pose") as Tensor<float>;
            _qualityOutput = _worker.PeekOutput("quality_score") as Tensor<float>;
            _successOutput = _worker.PeekOutput("success_prob")  as Tensor<float>;

            if (_poseOutput == null || _qualityOutput == null || _successOutput == null)
            {
                UnityEngine.Debug.LogError("[AuraXRModelRuntime] Missing one or more ONNX outputs: selected_pose, quality_score, success_prob.");
                return false;
            }

            // ReadbackAndClone is synchronous but unavoidable for single-frame latency.
            // Use CPU backend (BackendType.CPU) to eliminate this cost entirely.
            using Tensor<float> poseCpu    = _poseOutput.ReadbackAndClone();
            using Tensor<float> qualityCpu = _qualityOutput.ReadbackAndClone();
            using Tensor<float> successCpu = _successOutput.ReadbackAndClone();

            for (int i = 0; i < 45; i++) Output.selectedPose[i] = poseCpu[i];
            Output.qualityScore = qualityCpu[0];
            Output.successProb  = successCpu[0];
            Output.valid = true;
            return true;
        }

        private void RebuildObjFlatIfNeeded()
        {
            AuraXRDemoObject obj = featureAssembler.activeObject;
            if (obj == _cachedPtsObject) return;
            _cachedPtsObject = obj;

            Vector3[] pts = obj.normalizedPoints;
            for (int i = 0, j = 0; i < 1024; i++, j += 3)
            {
                _objFlat[j]     = pts[i].x;
                _objFlat[j + 1] = pts[i].y;
                _objFlat[j + 2] = pts[i].z;
            }
            _objTensor.Upload(_objFlat, 0);
        }
#endif
    }
}
