using UnityEngine;

namespace AuraXR.Demo
{
    public sealed class AuraXRBlendController : MonoBehaviour
    {
        public AuraXRFeatureAssembler featureAssembler;
        public AuraXRHandRetargeter handRetargeter;

        [Header("Thresholds")]
        public float farThresholdM = 0.10f;
        public float nearThresholdM = 0.02f;
        public float releaseThresholdM = 0.05f;
        public int freeToTransitionFrames = 5;
        public int transitionToGraspFrames = 3;

        [Header("State")]
        [Range(0f, 1f)] public float blendWeight;
        public string phase = "free";

        private int _nearFarCount;
        private int _nearGraspCount;

        void LateUpdate()
        {
            if (featureAssembler == null || handRetargeter == null) return;

            float d = featureAssembler.distanceM;
            bool contact = featureAssembler.lastContactFlag > 0.5f;

            if (phase == "free")
            {
                _nearFarCount = d < farThresholdM ? _nearFarCount + 1 : 0;
                if (_nearFarCount >= freeToTransitionFrames) phase = "transition";
            }
            else if (phase == "transition")
            {
                _nearGraspCount = d <= nearThresholdM || contact ? _nearGraspCount + 1 : 0;
                if (_nearGraspCount >= transitionToGraspFrames) phase = "grasp";
                if (d > farThresholdM)
                {
                    phase = "free";
                    _nearFarCount = 0;
                    _nearGraspCount = 0;
                }
            }
            else if (phase == "grasp")
            {
                if (d > releaseThresholdM && !contact)
                {
                    phase = "transition";
                    _nearGraspCount = 0;
                }
            }

            float raw = (farThresholdM - d) / Mathf.Max(farThresholdM - nearThresholdM, 1e-6f);
            blendWeight = phase == "free" ? 0f : AuraXRMath.SmoothStep01(raw);
            handRetargeter.Apply(blendWeight);
        }
    }
}
