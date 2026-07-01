using UnityEngine;
using UnityEngine.UI;

namespace AuraXR.Demo
{
    public sealed class AuraXRDemoHUD : MonoBehaviour
    {
        public AuraXRFeatureAssembler featureAssembler;
        public AuraXRModelRuntime modelRuntime;
        public AuraXRBlendController blendController;
        public Text targetText;
        public TextMesh worldText;

        private int _lastInferenceCount = -1;
        private string _cachedText = string.Empty;

        void OnGUI()
        {
            if (targetText != null) return;
            GUI.color = Color.white;
            GUI.Label(new Rect(16, 16, 520, 180), _cachedText);
        }

        void Update()
        {
            // Rebuild text only when new inference data is available (30 Hz), not every render frame.
            int count = modelRuntime != null ? modelRuntime.inferenceCount : 0;
            if (count == _lastInferenceCount) return;
            _lastInferenceCount = count;

            _cachedText = BuildText();
            if (targetText != null) targetText.text = _cachedText;
            if (worldText != null) worldText.text = _cachedText;
        }

        private string BuildText()
        {
            string objectId = featureAssembler != null && featureAssembler.activeObject != null
                ? featureAssembler.activeObject.objectId
                : "none";
            string state = featureAssembler != null ? featureAssembler.modelState : "missing";
            int fill = featureAssembler != null ? featureAssembler.activeWindowFill : 0;
            float distCm = featureAssembler != null ? featureAssembler.distanceM * 100f : 0f;
            float contact = featureAssembler != null ? featureAssembler.lastContactFlag : 0f;
            float blend = blendController != null ? blendController.blendWeight : 0f;
            float quality = modelRuntime != null ? modelRuntime.qualityScore : 0f;
            float success = modelRuntime != null ? modelRuntime.successProb : 0f;
            float latency = modelRuntime != null ? modelRuntime.latencyMs : 0f;

            return
                $"AuraXR Model Demo\n" +
                $"object_id: {objectId}\n" +
                $"model_state: {state} ({fill}/16)\n" +
                $"distance_cm: {distCm:F2}\n" +
                $"blend_weight: {blend:F3}\n" +
                $"contact_flag: {contact:F0}\n" +
                $"quality_score: {quality:F3}\n" +
                $"success_prob: {success:F3}\n" +
                $"latency_ms: {latency:F2}";
        }
    }
}
