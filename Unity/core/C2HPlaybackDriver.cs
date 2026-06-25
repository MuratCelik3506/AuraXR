using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text;
using Unity.InferenceEngine;
using UnityEngine;

namespace AuraXR
{
    /// <summary>
    /// Minimal non-VR playback driver for the HOT3D c2h_step.onnx integration test.
    /// It replays Assets/StreamingAssets/c2h_playback.json, feeds LSTM h/c state back
    /// frame by frame, and optionally drives AuraXRInferenceManager.RightHand so the
    /// existing HandRigController/MANODecoder path can animate the hand rig.
    /// </summary>
    [DefaultExecutionOrder(300)]
    public class C2HPlaybackDriver : MonoBehaviour
    {
        [Header("Inputs")]
        public ModelAsset modelAsset;
        public TextAsset playbackJson;
        public string streamingAssetsFile = "c2h_playback.json";

        [Header("Playback")]
        public bool runParityOnStart = true;
        public bool playOnStart = true;
        public bool loop = true;
        [Min(1f)] public float playbackFps = 30f;
        public int startFrame = 0;

        [Header("Scene Targets")]
        public AuraXRInferenceManager inferenceManager;
        public Transform wristRoot;
        public Transform objectRoot;
        public bool driveRightHand = true;
        public bool applyWristPose = true;
        public bool applyObjectPose = true;

        [Header("Debug")]
        public float parityThreshold = 1e-3f;
        public bool verboseFrameLog = false;

        public PlaybackParityResult LastParity { get; private set; }
        public int CurrentFrame => _frameIndex;
        public int FrameCount => _playback?.frames?.Count ?? 0;

        private Worker _worker;
        private PlaybackData _playback;
        private float[] _h;
        private float[] _c;
        private int _hidden = 256;
        private int _frameIndex;
        private float _accum;
        private bool _ready;

        private const string InputFeat = "feat";
        private const string InputCategory = "category";
        private const string InputH = "h0";
        private const string InputC = "c0";
        private const string OutputPca = "pca15";
        private const string OutputH = "hn";
        private const string OutputC = "cn";

        private void Awake()
        {
            if (inferenceManager == null)
                inferenceManager = FindAnyObjectByType<AuraXRInferenceManager>();
        }

        private void Start()
        {
            if (!Initialize())
                return;

            if (runParityOnStart)
                RunParityCheck();

            if (playOnStart)
                PlayFrom(startFrame);
            else
                StepToFrame(Mathf.Clamp(startFrame, 0, FrameCount - 1));
        }

        private void Update()
        {
            if (!_ready || !playOnStart || FrameCount == 0)
                return;

            _accum += Time.deltaTime;
            float frameDt = 1f / Mathf.Max(1f, playbackFps);
            while (_accum >= frameDt)
            {
                _accum -= frameDt;
                int next = _frameIndex + 1;
                if (next >= FrameCount)
                {
                    if (!loop)
                    {
                        playOnStart = false;
                        return;
                    }
                    PlayFrom(0);
                    return;
                }
                StepToFrame(next);
            }
        }

        private void OnDestroy()
        {
            _worker?.Dispose();
            _worker = null;
        }

        [ContextMenu("Initialize")]
        public bool Initialize()
        {
            if (_ready && _worker != null && _playback != null && _playback.frames.Count > 0)
                return true;

            _worker?.Dispose();
            _worker = null;
            _ready = false;

            if (modelAsset == null)
                modelAsset = FindModelAsset();
            if (modelAsset == null)
            {
                Debug.LogError("[C2HPlayback] c2h_step ModelAsset not assigned/found.");
                return false;
            }

            string json = LoadPlaybackJson();
            if (string.IsNullOrEmpty(json))
                return false;

            _playback = C2HPlaybackJson.Parse(json);
            if (_playback.frames.Count == 0)
            {
                Debug.LogError("[C2HPlayback] Playback contains no frames.");
                return false;
            }

            _hidden = _playback.hidden > 0 ? _playback.hidden : 256;
            _h = new float[_hidden];
            _c = new float[_hidden];

            var model = ModelLoader.Load(modelAsset);
            _worker = new Worker(model, BackendType.CPU);
            Debug.Log($"[C2HPlayback] Loaded model IO: {DescribeModel(model)}");
            Debug.Log($"[C2HPlayback] Loaded {_playback.frames.Count} frames, object={_playback.objectName}, hidden={_hidden}.");

            _ready = true;
            return true;
        }

        [ContextMenu("Run Parity Check")]
        public PlaybackParityResult RunParityCheck()
        {
            if (!Initialize())
                return default;

            ResetState();
            float maxDiff = 0f;
            int maxFrame = -1;
            int maxDim = -1;
            float unityAtMax = 0f;
            float expectedAtMax = 0f;

            for (int t = 0; t < _playback.frames.Count; t++)
            {
                float[] pca = RunInference(_playback.frames[t], updateState: true);
                for (int i = 0; i < 15; i++)
                {
                    float diff = Mathf.Abs(pca[i] - _playback.frames[t].pcaExpected[i]);
                    if (diff > maxDiff)
                    {
                        maxDiff = diff;
                        maxFrame = t;
                        maxDim = i;
                        unityAtMax = pca[i];
                        expectedAtMax = _playback.frames[t].pcaExpected[i];
                    }
                }
            }

            LastParity = new PlaybackParityResult
            {
                pass = maxDiff < parityThreshold,
                maxAbsDiff = maxDiff,
                frame = maxFrame,
                dim = maxDim,
                unityValue = unityAtMax,
                expectedValue = expectedAtMax,
                frames = _playback.frames.Count,
                objectName = _playback.objectName
            };

            string status = LastParity.pass ? "PASS" : "FAIL";
            Debug.Log($"[C2HPlayback] Parity {status}: frames={LastParity.frames} object={LastParity.objectName} " +
                      $"maxAbsDiff={LastParity.maxAbsDiff:E6} frame={LastParity.frame} dim={LastParity.dim} " +
                      $"unity={LastParity.unityValue:E6} expected={LastParity.expectedValue:E6}");

            ResetState();
            return LastParity;
        }

        [ContextMenu("Play From Start")]
        public void PlayFromStart()
        {
            PlayFrom(0);
        }

        public void PlayFrom(int frame)
        {
            if (!Initialize())
                return;

            ResetState();
            int target = Mathf.Clamp(frame, 0, FrameCount - 1);
            for (int i = 0; i <= target; i++)
                StepToFrame(i);
            playOnStart = true;
            _accum = 0f;
        }

        public void StepToFrame(int frame)
        {
            if (!Initialize())
                return;

            _frameIndex = Mathf.Clamp(frame, 0, FrameCount - 1);
            PlaybackFrame f = _playback.frames[_frameIndex];
            float[] pca = RunInference(f, updateState: true);
            ApplyFrame(f, pca);

            if (verboseFrameLog)
            {
                Debug.Log($"[C2HPlayback] frame={_frameIndex}/{FrameCount - 1} " +
                          $"pca0={pca[0]:F4} wrist={Hot3DPositionToUnity(f.wristT):F4} obj={Hot3DPositionToUnity(f.objT):F4}");
            }
        }

        private float[] RunInference(PlaybackFrame frame, bool updateState)
        {
            using var featTensor = new Tensor<float>(new TensorShape(1, 1, 13), frame.feat);
            using var catTensor = new Tensor<int>(new TensorShape(1, 1), new[] { frame.category });
            using var hTensor = new Tensor<float>(new TensorShape(1, 1, _hidden), _h);
            using var cTensor = new Tensor<float>(new TensorShape(1, 1, _hidden), _c);

            _worker.SetInput(InputFeat, featTensor);
            _worker.SetInput(InputCategory, catTensor);
            _worker.SetInput(InputH, hTensor);
            _worker.SetInput(InputC, cTensor);
            _worker.Schedule();

            var pcaTensor = _worker.PeekOutput(OutputPca) as Tensor<float>;
            var hnTensor = _worker.PeekOutput(OutputH) as Tensor<float>;
            var cnTensor = _worker.PeekOutput(OutputC) as Tensor<float>;
            if (pcaTensor == null || hnTensor == null || cnTensor == null)
                throw new InvalidOperationException("[C2HPlayback] Missing ONNX output. Expected pca15, hn, cn.");

            using var pcaCpu = pcaTensor.ReadbackAndClone();
            using var hnCpu = hnTensor.ReadbackAndClone();
            using var cnCpu = cnTensor.ReadbackAndClone();

            var pca = new float[15];
            for (int i = 0; i < 15; i++)
                pca[i] = pcaCpu[0, 0, i];

            if (updateState)
            {
                for (int i = 0; i < _hidden; i++)
                {
                    _h[i] = hnCpu[0, 0, i];
                    _c[i] = cnCpu[0, 0, i];
                }
            }

            return pca;
        }

        private void ApplyFrame(PlaybackFrame frame, float[] pca)
        {
            Vector3 wristPos = Hot3DPositionToUnity(frame.wristT);
            Quaternion wristRot = Hot3DRotationToUnity(frame.wristQ);
            Vector3 objPos = Hot3DPositionToUnity(frame.objT);
            Quaternion objRot = Hot3DRotationToUnity(frame.objQ);

            if (applyWristPose && wristRoot != null)
            {
                wristRoot.SetPositionAndRotation(wristPos, wristRot);
            }

            if (applyObjectPose && objectRoot != null)
            {
                objectRoot.SetPositionAndRotation(objPos, objRot);
            }

            if (inferenceManager == null)
                return;

            HandPose pose = driveRightHand ? inferenceManager.RightHand : inferenceManager.LeftHand;
            if (pose == null)
                pose = new HandPose();

            Array.Copy(pca, pose.ManoJointAngles, 15);
            pose.WristPosition = wristPos;
            pose.WristRotation = wristRot;
            pose.ApproachDirection = (objPos - wristPos).sqrMagnitude > 1e-8f
                ? (objPos - wristPos).normalized
                : Vector3.forward;
            pose.ApproachDistance = Vector3.Distance(wristPos, objPos);
            pose.GripCategory = frame.category;

            if (driveRightHand)
                inferenceManager.RightHand = pose;
            else
                inferenceManager.LeftHand = pose;
        }

        private void ResetState()
        {
            if (_h != null) Array.Clear(_h, 0, _h.Length);
            if (_c != null) Array.Clear(_c, 0, _c.Length);
            _frameIndex = 0;
            _accum = 0f;
        }

        private string LoadPlaybackJson()
        {
            if (playbackJson != null)
                return playbackJson.text;

            string path = Path.Combine(Application.streamingAssetsPath, streamingAssetsFile);
            if (!File.Exists(path))
            {
                Debug.LogError($"[C2HPlayback] Missing playback JSON: {path}");
                return null;
            }

            return File.ReadAllText(path);
        }

        private static ModelAsset FindModelAsset()
        {
#if UNITY_EDITOR
            string[] guids = UnityEditor.AssetDatabase.FindAssets("c2h_step t:ModelAsset");
            if (guids.Length == 0)
                return null;
            string path = UnityEditor.AssetDatabase.GUIDToAssetPath(guids[0]);
            return UnityEditor.AssetDatabase.LoadAssetAtPath<ModelAsset>(path);
#else
            return null;
#endif
        }

        private static string DescribeModel(Model model)
        {
            var sb = new StringBuilder();
            sb.Append("inputs:");
            foreach (var input in model.inputs)
                sb.Append(' ').Append(input.name).Append(" shape=").Append(input.shape);
            sb.Append(" | outputs:");
            foreach (var output in model.outputs)
                sb.Append(' ').Append(output.name);
            return sb.ToString();
        }

        public static Vector3 Hot3DPositionToUnity(float[] v)
        {
            return new Vector3(v[0], v[1], -v[2]);
        }

        public static Quaternion Hot3DRotationToUnity(float[] q)
        {
            return new Quaternion(q[0], q[1], -q[2], q[3]);
        }
    }

    [Serializable]
    public struct PlaybackParityResult
    {
        public bool pass;
        public float maxAbsDiff;
        public int frame;
        public int dim;
        public float unityValue;
        public float expectedValue;
        public int frames;
        public string objectName;
    }

    internal class PlaybackData
    {
        public string objectName;
        public int hidden;
        public readonly List<PlaybackFrame> frames = new List<PlaybackFrame>();
    }

    internal class PlaybackFrame
    {
        public float[] feat;
        public int category;
        public float[] pcaExpected;
        public float[] pcaGt;
        public float[] wristT;
        public float[] wristQ;
        public float[] objT;
        public float[] objQ;
    }

    internal static class C2HPlaybackJson
    {
        private static readonly CultureInfo Inv = CultureInfo.InvariantCulture;

        public static PlaybackData Parse(string json)
        {
            var data = new PlaybackData
            {
                objectName = ExtractString(json, "object", "unknown"),
                hidden = ExtractInt(json, "hidden", 256)
            };

            int key = json.IndexOf("\"frames\"", StringComparison.Ordinal);
            if (key < 0)
                return data;

            int arrayStart = json.IndexOf('[', key);
            int arrayEnd = FindMatching(json, arrayStart, '[', ']');
            int i = arrayStart + 1;
            while (i < arrayEnd)
            {
                int objStart = json.IndexOf('{', i);
                if (objStart < 0 || objStart >= arrayEnd)
                    break;

                int objEnd = FindMatching(json, objStart, '{', '}');
                string frameJson = json.Substring(objStart, objEnd - objStart + 1);
                data.frames.Add(new PlaybackFrame
                {
                    feat = ExtractFloatArray(frameJson, "feat", 13),
                    category = ExtractInt(frameJson, "category", 0),
                    pcaExpected = ExtractFloatArray(frameJson, "pca_expected", 15),
                    pcaGt = ExtractFloatArray(frameJson, "pca_gt", 15),
                    wristT = ExtractFloatArray(frameJson, "wrist_t", 3),
                    wristQ = ExtractFloatArray(frameJson, "wrist_q", 4),
                    objT = ExtractFloatArray(frameJson, "obj_t", 3),
                    objQ = ExtractFloatArray(frameJson, "obj_q", 4)
                });
                i = objEnd + 1;
            }

            return data;
        }

        private static int ExtractInt(string json, string key, int fallback)
        {
            int keyPos = json.IndexOf("\"" + key + "\"", StringComparison.Ordinal);
            if (keyPos < 0)
                return fallback;

            int colon = json.IndexOf(':', keyPos);
            int start = colon + 1;
            while (start < json.Length && char.IsWhiteSpace(json[start]))
                start++;

            int end = start;
            while (end < json.Length && (char.IsDigit(json[end]) || json[end] == '-'))
                end++;

            return int.TryParse(json.Substring(start, end - start), NumberStyles.Integer, Inv, out int value)
                ? value
                : fallback;
        }

        private static string ExtractString(string json, string key, string fallback)
        {
            int keyPos = json.IndexOf("\"" + key + "\"", StringComparison.Ordinal);
            if (keyPos < 0)
                return fallback;

            int colon = json.IndexOf(':', keyPos);
            int start = json.IndexOf('"', colon + 1) + 1;
            int end = json.IndexOf('"', start);
            return start > 0 && end > start ? json.Substring(start, end - start) : fallback;
        }

        private static float[] ExtractFloatArray(string json, string key, int expected)
        {
            int keyPos = json.IndexOf("\"" + key + "\"", StringComparison.Ordinal);
            if (keyPos < 0)
                throw new FormatException("Missing key " + key);

            int start = json.IndexOf('[', keyPos);
            int end = FindMatching(json, start, '[', ']');
            string[] parts = json.Substring(start + 1, end - start - 1).Split(',');
            if (parts.Length != expected)
                throw new FormatException($"{key} length {parts.Length} != {expected}");

            var values = new float[expected];
            for (int i = 0; i < expected; i++)
                values[i] = float.Parse(parts[i], NumberStyles.Float, Inv);
            return values;
        }

        private static int FindMatching(string s, int start, char open, char close)
        {
            int depth = 0;
            bool inString = false;
            for (int i = start; i < s.Length; i++)
            {
                char ch = s[i];
                if (ch == '"' && (i == 0 || s[i - 1] != '\\'))
                    inString = !inString;
                if (inString)
                    continue;
                if (ch == open)
                    depth++;
                else if (ch == close && --depth == 0)
                    return i;
            }

            return s.Length - 1;
        }
    }
}
