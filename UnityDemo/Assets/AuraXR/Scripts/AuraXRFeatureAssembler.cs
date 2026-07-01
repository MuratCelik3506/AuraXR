using System;
using UnityEngine;

namespace AuraXR.Demo
{
    public sealed class AuraXRFeatureAssembler : MonoBehaviour
    {
        public const int Window = 16;
        public const int FrameDim = 13;

        [Header("References")]
        public Transform wrist;
        [Tooltip("Optional palm/fingertip probes used for HUD distance and contact. The wrist transform still drives the model pose input.")]
        public Transform[] contactProbes;
        [Tooltip("Objects that can become the model target. When enabled, the nearest one is assigned to activeObject automatically.")]
        public AuraXRDemoObject[] candidateObjects;
        public bool useNearestObject = true;
        public AuraXRDemoObject activeObject;
        public TextAsset modelStatsJson;
        [Tooltip("Relative to Application.streamingAssetsPath, used when modelStatsJson is not imported as TextAsset.")]
        public string modelStatsRelativePath = "AuraXR/model_stats.json";

        [Header("Sampling")]
        [Tooltip("HOT3D temporal stats are effectively 30 Hz. Keep feature sampling at 30 Hz for first demo.")]
        public float featureHz = 30f;
        public float contactThresholdM = 0.02f;
        public bool usePhysicsContact = true;

        [Header("Debug")]
        public int activeWindowFill;
        public string modelState = "warming_up";
        public float distanceM;
        public float lastContactFlag;

        public AuraXRModelStats Stats { get; private set; }
        public bool IsReady => activeWindowFill >= Window && activeObject != null && wrist != null;
        // True for exactly one Update() after a new frame was sampled; ModelRuntime reads this.
        public bool NewFrameReady { get; private set; }
        public float[] FrameFeatFlat => _frameFeatFlat;
        public float[] ContactFlagFlat => _contactFlagFlat;
        public float[] LastFrameFeat => _lastFrameFeat;

        private readonly float[] _frameFeatFlat = new float[Window * FrameDim];
        private readonly float[] _contactFlagFlat = new float[Window];
        private readonly float[] _lastFrameFeat = new float[FrameDim];
        private readonly float[] _scratchRaw = new float[FrameDim];
        private int _writeIndex;
        private float _accum;
        private float _lastSampleTime = -1f;
        private Vector3 _prevRelPos;
        private bool _hasPrevRelPos;

        // Cached per active object to avoid GetComponentInChildren every sample.
        private AuraXRDemoObject _cachedColObject;
        private Collider _cachedCollider;

        // Reusable OverlapSphere buffer — size 8 is enough for typical scenes.
        private readonly Collider[] _overlapBuffer = new Collider[8];

        void Awake()
        {
            if (modelStatsJson != null)
            {
                Stats = JsonUtility.FromJson<AuraXRModelStats>(modelStatsJson.text);
                Stats.Validate();
            }
            else if (!string.IsNullOrEmpty(modelStatsRelativePath))
            {
                string path = System.IO.Path.Combine(Application.streamingAssetsPath, modelStatsRelativePath);
                Stats = JsonUtility.FromJson<AuraXRModelStats>(System.IO.File.ReadAllText(path));
                Stats.Validate();
            }
        }

        void Update()
        {
            NewFrameReady = false;

            if (Stats == null || wrist == null)
            {
                modelState = "missing_refs";
                return;
            }

            UpdateNearestObject();

            if (activeObject == null)
            {
                modelState = "missing_refs";
                return;
            }

            if (activeObject.normalizedPoints == null || activeObject.normalizedPoints.Length != 1024)
            {
                activeObject.LoadPoints(Stats);
            }

            // Invalidate collider cache when the active object changes.
            if (_cachedColObject != activeObject)
            {
                _cachedColObject = activeObject;
                _cachedCollider = activeObject.GetComponentInChildren<Collider>();
                ResetTemporalWindow();
            }

            float sampleDt = 1f / Mathf.Max(featureHz, 1f);
            _accum += Time.deltaTime;
            if (_accum + 1e-6f < sampleDt) return;
            _accum = 0f;
            SampleFrame();
            NewFrameReady = true;
        }

        private void UpdateNearestObject()
        {
            if (!useNearestObject || candidateObjects == null || candidateObjects.Length == 0) return;

            AuraXRDemoObject nearest = null;
            float best = float.PositiveInfinity;
            for (int i = 0; i < candidateObjects.Length; i++)
            {
                AuraXRDemoObject candidate = candidateObjects[i];
                if (candidate == null) continue;

                float distance = ComputeDistanceToCandidateM(candidate);
                if (distance < best)
                {
                    best = distance;
                    nearest = candidate;
                }
            }

            if (nearest != null && nearest != activeObject)
                activeObject = nearest;
        }

        private float ComputeDistanceToCandidateM(AuraXRDemoObject candidate)
        {
            float best = float.PositiveInfinity;
            Collider col = candidate.GetComponentInChildren<Collider>();
            if (col != null)
            {
                AccumulateClosestDistance(col, wrist, ref best);
                if (contactProbes != null)
                {
                    for (int i = 0; i < contactProbes.Length; i++)
                        AccumulateClosestDistance(col, contactProbes[i], ref best);
                }
                return best;
            }

            AccumulatePointDistance(candidate.transform.position, wrist, ref best);
            if (contactProbes != null)
            {
                for (int i = 0; i < contactProbes.Length; i++)
                    AccumulatePointDistance(candidate.transform.position, contactProbes[i], ref best);
            }
            return best;
        }

        private void ResetTemporalWindow()
        {
            Array.Clear(_frameFeatFlat, 0, _frameFeatFlat.Length);
            Array.Clear(_contactFlagFlat, 0, _contactFlagFlat.Length);
            Array.Clear(_lastFrameFeat, 0, _lastFrameFeat.Length);
            _writeIndex = 0;
            activeWindowFill = 0;
            _lastSampleTime = -1f;
            _prevRelPos = Vector3.zero;
            _hasPrevRelPos = false;
            modelState = "warming_up";
            NewFrameReady = false;
        }

        private void SampleFrame()
        {
            Transform obj = activeObject.transform;
            Vector3 currentRelPos = obj.InverseTransformPoint(wrist.position);
            float now = Time.time;
            float dt = _lastSampleTime > 0f ? Mathf.Max(now - _lastSampleTime, 1e-6f) : 1f / Mathf.Max(featureHz, 1f);
            Vector3 prevRelPos = _hasPrevRelPos ? _prevRelPos : currentRelPos;

            distanceM = ComputeDistanceM(obj);
            AuraXRMath.RelativePoseToFrame13(wrist, obj, prevRelPos, dt, distanceM, _scratchRaw);
            Stats.NormalizeFrameInPlace(_scratchRaw);

            float contact = distanceM <= contactThresholdM ? 1f : 0f;
            if (usePhysicsContact && PhysicsContactLikely()) contact = 1f;
            lastContactFlag = contact;

            int row = _writeIndex * FrameDim;
            Array.Copy(_scratchRaw, 0, _frameFeatFlat, row, FrameDim);
            _contactFlagFlat[_writeIndex] = contact;
            Array.Copy(_scratchRaw, _lastFrameFeat, FrameDim);

            _writeIndex = (_writeIndex + 1) % Window;
            activeWindowFill = Mathf.Min(activeWindowFill + 1, Window);
            modelState = activeWindowFill >= Window ? "running" : "warming_up";

            _prevRelPos = currentRelPos;
            _hasPrevRelPos = true;
            _lastSampleTime = now;
        }

        public void CopyOrderedWindow(float[] frameOut, float[] contactOut)
        {
            if (frameOut == null || frameOut.Length != Window * FrameDim) throw new ArgumentException("frameOut wrong length.");
            if (contactOut == null || contactOut.Length != Window) throw new ArgumentException("contactOut wrong length.");

            int start = activeWindowFill < Window ? 0 : _writeIndex;
            for (int t = 0; t < Window; t++)
            {
                int src = (start + t) % Window;
                Array.Copy(_frameFeatFlat, src * FrameDim, frameOut, t * FrameDim, FrameDim);
                contactOut[t] = _contactFlagFlat[src];
            }
        }

        private float ComputeDistanceM(Transform obj)
        {
            float best = float.PositiveInfinity;
            if (_cachedCollider != null)
            {
                AccumulateClosestDistance(_cachedCollider, wrist, ref best);
                if (contactProbes != null)
                {
                    for (int i = 0; i < contactProbes.Length; i++)
                        AccumulateClosestDistance(_cachedCollider, contactProbes[i], ref best);
                }
                return best;
            }

            AccumulatePointDistance(obj.position, wrist, ref best);
            if (contactProbes != null)
            {
                for (int i = 0; i < contactProbes.Length; i++)
                    AccumulatePointDistance(obj.position, contactProbes[i], ref best);
            }
            return best;
        }

        private static void AccumulateClosestDistance(Collider col, Transform probe, ref float best)
        {
            if (probe == null) return;
            Vector3 closest = col.ClosestPoint(probe.position);
            best = Mathf.Min(best, Vector3.Distance(probe.position, closest));
        }

        private static void AccumulatePointDistance(Vector3 target, Transform probe, ref float best)
        {
            if (probe == null) return;
            best = Mathf.Min(best, Vector3.Distance(probe.position, target));
        }

        private bool PhysicsContactLikely()
        {
            if (_cachedCollider == null) return false;
            if (ProbeOverlapsObject(wrist)) return true;
            if (contactProbes != null)
            {
                for (int i = 0; i < contactProbes.Length; i++)
                {
                    if (ProbeOverlapsObject(contactProbes[i])) return true;
                }
            }
            return false;
        }

        private bool ProbeOverlapsObject(Transform probe)
        {
            if (probe == null) return false;
            // Broad-phase: skip expensive OverlapSphere if probe is far from object bounds.
            Bounds bounds = _cachedCollider.bounds;
            bounds.Expand(contactThresholdM * 2f);
            if (!bounds.Contains(probe.position)) return false;

            int count = Physics.OverlapSphereNonAlloc(probe.position, contactThresholdM, _overlapBuffer);
            for (int i = 0; i < count; i++)
            {
                if (_overlapBuffer[i] == _cachedCollider || _overlapBuffer[i].transform.IsChildOf(activeObject.transform))
                    return true;
            }
            return false;
        }
    }
}
