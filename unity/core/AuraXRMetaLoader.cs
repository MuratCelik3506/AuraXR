using System;
using System.IO;
using UnityEngine;

namespace AuraXR
{
    /// <summary>
    /// Loads intentformer_meta.json and exposes normalisation stats for runtime use.
    /// Attach to any persistent GameObject (e.g. GameManager).
    /// </summary>
    public class AuraXRMetaLoader : MonoBehaviour
    {
        [Tooltip("Drag intentformer_meta.json here (TextAsset)")]
        public TextAsset metaJson;

        // Parsed at Awake — ready before other components need them
        public float[] FeatureMean  { get; private set; }
        public float[] FeatureStd   { get; private set; }
        public float[] TargetMean   { get; private set; }
        public float[] TargetStd    { get; private set; }

        public int FeatureDim   { get; private set; }
        public int TargetDim    { get; private set; }
        public int WindowFrames { get; private set; }

        public bool IsReady { get; private set; }

        void Awake()
        {
            if (metaJson == null)
            {
                Debug.LogError("[AuraXR] metaJson is not assigned. Drag intentformer_meta.json to the inspector.");
                return;
            }

            var root = JsonUtility.FromJson<MetaRoot>(metaJson.text);
            FeatureDim   = root.feature_dim;
            TargetDim    = root.target_dim;
            WindowFrames = root.T;

            FeatureMean = root.feature_mean;
            FeatureStd  = root.feature_std;
            TargetMean  = root.target_mean;
            TargetStd   = root.target_std;

            // Guard against all-zeros std (untrained model export)
            for (int i = 0; i < FeatureStd.Length; i++)
                if (FeatureStd[i] < 1e-6f) FeatureStd[i] = 1f;
            for (int i = 0; i < TargetStd.Length; i++)
                if (TargetStd[i] < 1e-6f) TargetStd[i] = 1f;

            IsReady = true;
            Debug.Log($"[AuraXR] Meta loaded. Feature={FeatureDim}  Target={TargetDim}  T={WindowFrames}");
        }

        // -----------------------------------------------------------------------
        // Normalise a raw feature vector in-place: (x - mean) / std
        // -----------------------------------------------------------------------
        public void NormaliseFeature(float[] feature)
        {
            for (int i = 0; i < feature.Length && i < FeatureDim; i++)
                feature[i] = (feature[i] - FeatureMean[i]) / FeatureStd[i];
        }

        // -----------------------------------------------------------------------
        // De-normalise a raw model output in-place: x * std + mean
        // -----------------------------------------------------------------------
        public void DenormaliseTarget(float[] target)
        {
            for (int i = 0; i < target.Length && i < TargetDim; i++)
                target[i] = target[i] * TargetStd[i] + TargetMean[i];
        }

        // -----------------------------------------------------------------------
        // JSON schema
        // -----------------------------------------------------------------------
        [Serializable]
        private class MetaRoot
        {
            public int     T;
            public int     feature_dim;
            public int     target_dim;
            public float[] feature_mean;
            public float[] feature_std;
            public float[] target_mean;
            public float[] target_std;
        }
    }
}
