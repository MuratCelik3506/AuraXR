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

        [Header("Scene Alignment")]
        [Tooltip("Align HOT3D world coordinates to the current Unity scene by keeping the first playback object pose at the object's scene position.")]
        public bool alignToSceneObjectAnchor = true;
        [Tooltip("If true, capture objectRoot's current scene pose as the anchor during Initialize(). Disable this when sceneObjectAnchorPosition is configured explicitly.")]
        public bool captureAnchorOnInitialize = false;
        public Vector3 sceneObjectAnchorPosition;
        public Quaternion sceneObjectAnchorRotation = Quaternion.identity;

        [Header("Skeleton Visualization")]
        [Tooltip("Draw the predicted (free-running) MANO hand as a procedural sphere+bone skeleton from the exported pred_joints. " +
                 "Lets multiple scenario drivers render their own hand side by side without an OVR rig.")]
        public bool renderSkeleton = false;
        [Tooltip("Also draw the ground-truth hand (gt_joints) as a second, ghost-colored skeleton for comparison.")]
        public bool renderGroundTruth = false;
        public float jointSphereRadius = 0.008f;
        public float boneThickness = 0.004f;
        public Color predColor = new Color(0.20f, 0.80f, 1.00f, 1f);
        public Color gtColor = new Color(1.00f, 0.55f, 0.15f, 1f);

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
        private Vector3 _playbackObjectAnchorPosition;
        private Quaternion _playbackObjectAnchorRotation = Quaternion.identity;
        private Vector3 _sceneTranslationOffset;
        private Quaternion _sceneRotationOffset = Quaternion.identity;
        private bool _alignmentReady;

        private const int JointCount = 16;   // MANO wrist + 15 finger joints (from src/mano_torch.py)
        private Transform _skeletonRoot;
        private Transform[] _predJointT;
        private Transform[] _predBoneT;
        private Transform[] _gtJointT;
        private Transform[] _gtBoneT;

        // Model now outputs 45-dim MANO finger axis-angle (aa45). The legacy "pca15"
        // naming is kept only in field names for back-compat; the real dim is 45.
        private const int PoseDim = 45;
        private const string InputFeat = "feat";
        private const string InputCategory = "category";
        private const string InputH = "h0";
        private const string InputC = "c0";
        private const string OutputPca = "aa45";
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
            ConfigureSceneAlignment();

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
                for (int i = 0; i < PoseDim; i++)
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
                throw new InvalidOperationException("[C2HPlayback] Missing ONNX output. Expected aa45, hn, cn.");

            using var pcaCpu = pcaTensor.ReadbackAndClone();
            using var hnCpu = hnTensor.ReadbackAndClone();
            using var cnCpu = cnTensor.ReadbackAndClone();

            var pca = new float[PoseDim];
            for (int i = 0; i < PoseDim; i++)
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
            Vector3 wristPos = TransformPlaybackPosition(Hot3DPositionToUnity(frame.wristT));
            Quaternion wristRot = Hot3DRotationToUnity(frame.wristQ);
            wristRot = TransformPlaybackRotation(wristRot);
            Vector3 objPos = TransformPlaybackPosition(Hot3DPositionToUnity(frame.objT));
            Quaternion objRot = Hot3DRotationToUnity(frame.objQ);
            objRot = TransformPlaybackRotation(objRot);

            if (applyWristPose && wristRoot != null)
            {
                wristRoot.SetPositionAndRotation(wristPos, wristRot);
            }

            if (applyObjectPose && objectRoot != null)
            {
                objectRoot.SetPositionAndRotation(objPos, objRot);
            }

            if (renderSkeleton)
                UpdateSkeleton(frame);

            if (inferenceManager == null)
                return;

            HandPose pose = driveRightHand ? inferenceManager.RightHand : inferenceManager.LeftHand;
            if (pose == null)
                pose = new HandPose();

            // pca is now 45-dim aa; ManoJointAngles may be shorter (legacy 15). Copy what fits
            // so the shared-rig path (default driver) does not throw. Scenario drivers skip this
            // block entirely (inferenceManager == null) and render via pred_joints instead.
            Array.Copy(pca, pose.ManoJointAngles, Math.Min(pca.Length, pose.ManoJointAngles.Length));
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

        // ── Procedural skeleton visualization ──────────────────────────────────
        // Renders the exported MANO joint world-positions (pred_joints / gt_joints)
        // as spheres + bone cylinders so each scenario driver shows its own hand
        // without an OVR rig. Joints are HOT3D world coords → converted + scene-aligned
        // exactly like wrist/object poses.

        private void UpdateSkeleton(PlaybackFrame frame)
        {
            int[] parents = _playback?.jointParents;
            if (frame.predJoints == null || parents == null || parents.Length < JointCount)
                return;

            EnsureSkeleton();

            UpdateChain(frame.predJoints, parents, _predJointT, _predBoneT);

            bool showGt = renderGroundTruth && frame.gtJoints != null;
            SetChainActive(_gtJointT, _gtBoneT, showGt);
            if (showGt)
                UpdateChain(frame.gtJoints, parents, _gtJointT, _gtBoneT);
        }

        private void EnsureSkeleton()
        {
            if (_skeletonRoot != null)
                return;

            var rootGo = new GameObject($"{name} Skeleton");
            rootGo.transform.SetParent(transform, worldPositionStays: false);
            _skeletonRoot = rootGo.transform;

            BuildChain("pred", predColor, out _predJointT, out _predBoneT);
            BuildChain("gt", gtColor, out _gtJointT, out _gtBoneT);
            SetChainActive(_gtJointT, _gtBoneT, renderGroundTruth);
        }

        private void BuildChain(string label, Color color, out Transform[] joints, out Transform[] bones)
        {
            Material mat = MakeMaterial(color);
            joints = new Transform[JointCount];
            bones = new Transform[JointCount];   // index j holds the bone from parents[j] to j (j>=1)

            for (int j = 0; j < JointCount; j++)
            {
                var sphere = CreatePrimitiveNoCollider(PrimitiveType.Sphere, $"{label}_joint_{j}", mat);
                sphere.SetParent(_skeletonRoot, worldPositionStays: false);
                float d = jointSphereRadius * 2f;
                sphere.localScale = new Vector3(d, d, d);
                joints[j] = sphere;

                if (j >= 1)
                {
                    var bone = CreatePrimitiveNoCollider(PrimitiveType.Cylinder, $"{label}_bone_{j}", mat);
                    bone.SetParent(_skeletonRoot, worldPositionStays: false);
                    bones[j] = bone;
                }
            }
        }

        private void UpdateChain(float[] joints48, int[] parents, Transform[] jointT, Transform[] boneT)
        {
            for (int j = 0; j < JointCount; j++)
            {
                if (jointT[j] != null)
                    jointT[j].position = WorldJoint(joints48, j);
            }

            for (int j = 1; j < JointCount; j++)
            {
                int p = parents[j];
                if (boneT[j] == null || p < 0 || p >= JointCount)
                    continue;

                Vector3 a = jointT[p].position;
                Vector3 b = jointT[j].position;
                Vector3 dir = b - a;
                float len = dir.magnitude;
                // Unity cylinder is 2 units tall along local Y → half-length scale on Y.
                boneT[j].position = (a + b) * 0.5f;
                boneT[j].rotation = len > 1e-6f
                    ? Quaternion.FromToRotation(Vector3.up, dir)
                    : Quaternion.identity;
                boneT[j].localScale = new Vector3(boneThickness, Mathf.Max(len * 0.5f, 1e-5f), boneThickness);
            }
        }

        private Vector3 WorldJoint(float[] joints48, int j)
        {
            int b = j * 3;
            Vector3 hot3d = new Vector3(joints48[b], joints48[b + 1], -joints48[b + 2]);
            return TransformPlaybackPosition(hot3d);
        }

        private static void SetChainActive(Transform[] jointT, Transform[] boneT, bool active)
        {
            if (jointT == null)
                return;
            foreach (var t in jointT)
                if (t != null && t.gameObject.activeSelf != active) t.gameObject.SetActive(active);
            if (boneT != null)
                foreach (var t in boneT)
                    if (t != null && t.gameObject.activeSelf != active) t.gameObject.SetActive(active);
        }

        private static Transform CreatePrimitiveNoCollider(PrimitiveType type, string name, Material mat)
        {
            var go = GameObject.CreatePrimitive(type);
            go.name = name;
            var col = go.GetComponent<Collider>();
            if (col != null) Destroy(col);
            var rend = go.GetComponent<Renderer>();
            if (rend != null) rend.sharedMaterial = mat;
            return go.transform;
        }

        private static Material MakeMaterial(Color color)
        {
            Shader shader = Shader.Find("Universal Render Pipeline/Lit")
                            ?? Shader.Find("Standard")
                            ?? Shader.Find("Unlit/Color")
                            ?? Shader.Find("Sprites/Default");
            var mat = new Material(shader) { color = color };
            return mat;
        }

        private void ConfigureSceneAlignment()
        {
            _alignmentReady = false;
            _sceneRotationOffset = Quaternion.identity;
            _sceneTranslationOffset = Vector3.zero;

            if (!alignToSceneObjectAnchor || _playback == null || _playback.frames.Count == 0)
                return;

            PlaybackFrame first = _playback.frames[0];
            _playbackObjectAnchorPosition = Hot3DPositionToUnity(first.objT);
            _playbackObjectAnchorRotation = Hot3DRotationToUnity(first.objQ);

            if (captureAnchorOnInitialize && objectRoot != null)
            {
                sceneObjectAnchorPosition = objectRoot.position;
                sceneObjectAnchorRotation = objectRoot.rotation;
            }

            _sceneRotationOffset = sceneObjectAnchorRotation * Quaternion.Inverse(_playbackObjectAnchorRotation);
            _sceneTranslationOffset = sceneObjectAnchorPosition - (_sceneRotationOffset * _playbackObjectAnchorPosition);
            _alignmentReady = true;

            Debug.Log($"[C2HPlayback] Scene alignment enabled. playbackObj0={_playbackObjectAnchorPosition:F4} " +
                      $"sceneObj0={sceneObjectAnchorPosition:F4} offset={_sceneTranslationOffset:F4}");
        }

        private Vector3 TransformPlaybackPosition(Vector3 playbackPosition)
        {
            if (!alignToSceneObjectAnchor || !_alignmentReady)
                return playbackPosition;
            return _sceneRotationOffset * playbackPosition + _sceneTranslationOffset;
        }

        private Quaternion TransformPlaybackRotation(Quaternion playbackRotation)
        {
            if (!alignToSceneObjectAnchor || !_alignmentReady)
                return playbackRotation;
            return _sceneRotationOffset * playbackRotation;
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
        public int[] jointParents;   // MANO kintree, length 16 ([-1,0,1,2,...]); null if absent
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
        public float[] predJoints;   // 16*3 HOT3D world joints (free-running); null if absent
        public float[] gtJoints;     // 16*3 HOT3D world joints (ground truth); null if absent
    }

    internal static class C2HPlaybackJson
    {
        private static readonly CultureInfo Inv = CultureInfo.InvariantCulture;

        public static PlaybackData Parse(string json)
        {
            var data = new PlaybackData
            {
                objectName = ExtractString(json, "object", "unknown"),
                hidden = ExtractInt(json, "hidden", 256),
                jointParents = TryExtractIntArray(json, "joint_parents")
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
                    pcaExpected = ExtractFloatArray(frameJson, "pca_expected", 45),
                    pcaGt = ExtractFloatArray(frameJson, "pca_gt", 45),
                    wristT = ExtractFloatArray(frameJson, "wrist_t", 3),
                    wristQ = ExtractFloatArray(frameJson, "wrist_q", 4),
                    objT = ExtractFloatArray(frameJson, "obj_t", 3),
                    objQ = ExtractFloatArray(frameJson, "obj_q", 4),
                    predJoints = TryExtractFloatArray(frameJson, "pred_joints", 16 * 3),
                    gtJoints = TryExtractFloatArray(frameJson, "gt_joints", 16 * 3)
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

        /// <summary>Like ExtractFloatArray but returns null if the key is absent (optional fields).</summary>
        private static float[] TryExtractFloatArray(string json, string key, int expected)
        {
            int keyPos = json.IndexOf("\"" + key + "\"", StringComparison.Ordinal);
            if (keyPos < 0)
                return null;
            return ExtractFloatArray(json, key, expected);
        }

        /// <summary>Parse an integer array (e.g. joint_parents). Returns null if the key is absent.</summary>
        private static int[] TryExtractIntArray(string json, string key)
        {
            int keyPos = json.IndexOf("\"" + key + "\"", StringComparison.Ordinal);
            if (keyPos < 0)
                return null;

            int start = json.IndexOf('[', keyPos);
            int end = FindMatching(json, start, '[', ']');
            string[] parts = json.Substring(start + 1, end - start - 1).Split(',');
            var values = new int[parts.Length];
            for (int i = 0; i < parts.Length; i++)
                values[i] = int.Parse(parts[i].Trim(), NumberStyles.Integer, Inv);
            return values;
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
