using UnityEngine;

namespace AuraXR
{
    /// <summary>
    /// Highlights interactable objects when a virtual hand (or controller fallback) is nearby.
    /// Uses emission highlight — does not change base material color, so objects are never
    /// permanently yellow regardless of editor material settings.
    /// </summary>
    public class GraspIndicator : MonoBehaviour
    {
        [Tooltip("Distance at which highlight activates (metres)")]
        public float highlightDistance = 0.15f;

        [Tooltip("Minimum grip value (0–1) required to show highlight. " +
                 "Prevents false highlight when hand is nearby but not gripping.")]
        [Range(0f, 1f)]
        public float gripThreshold = 0.35f;

        [Tooltip("Highlight emission colour")]
        public Color highlightColor = new Color(1f, 0.85f, 0f, 1f);

        private Material _mat;
        private bool _isHighlighted;

        [Header("Hand References (assign in Inspector)")]
        public Transform leftHandRig;
        public Transform rightHandRig;

        private AuraXRFeatureAssembler _featureAssembler;

        static readonly int EmissionColorID = Shader.PropertyToID("_EmissionColor");

        void Start()
        {
            var r = GetComponent<Renderer>();
            if (r != null)
            {
                _mat = r.material;
                // Make sure emission is off at start — removes any editor-set yellow
                _mat.SetColor(EmissionColorID, Color.black);
                _mat.DisableKeyword("_EMISSION");
            }

            _featureAssembler = FindAnyObjectByType<AuraXRFeatureAssembler>();

            if (leftHandRig  == null) Debug.LogWarning($"[GraspIndicator] '{gameObject.name}': leftHandRig not assigned.");
            if (rightHandRig == null) Debug.LogWarning($"[GraspIndicator] '{gameObject.name}': rightHandRig not assigned.");
        }

        void LateUpdate()
        {
            if (_mat == null) return;

            // Require BOTH proximity AND sufficient grip — prevents false highlight
            // when the wrist is nearby but the hand is open.
            float leftGrip  = _featureAssembler?.LastLeftGrip  ?? 0f;
            float rightGrip = _featureAssembler?.LastRightGrip ?? 0f;

            bool shouldHighlight =
                (CheckHand(leftHandRig,  _featureAssembler?.leftControllerTransform)  && leftGrip  >= gripThreshold)
             || (CheckHand(rightHandRig, _featureAssembler?.rightControllerTransform) && rightGrip >= gripThreshold);

            if (shouldHighlight == _isHighlighted) return;

            _isHighlighted = shouldHighlight;
            if (_isHighlighted)
            {
                _mat.SetColor(EmissionColorID, highlightColor);
                _mat.EnableKeyword("_EMISSION");
            }
            else
            {
                _mat.SetColor(EmissionColorID, Color.black);
                _mat.DisableKeyword("_EMISSION");
            }
        }

        bool CheckHand(Transform handRig, Transform controllerFallback)
        {
            // Prefer virtual hand wrist (ONNX-driven), fall back to controller
            Transform probe = handRig ?? controllerFallback;
            if (probe == null) return false;
            return Vector3.Distance(probe.position, transform.position) < highlightDistance;
        }

        void OnDestroy()
        {
            if (_mat != null) Destroy(_mat);
        }
    }
}
