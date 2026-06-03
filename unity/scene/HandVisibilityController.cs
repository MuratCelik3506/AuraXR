using UnityEngine;

namespace AuraXR
{
    /// <summary>
    /// Controls hand visibility based on proximity to objects.
    /// Fades in when close, fades out when far.
    /// </summary>
    public class HandVisibilityController : MonoBehaviour
    {
        public Renderer handRenderer;
        public ProximityDetector proximityDetector;

        [Tooltip("Distance at which hand becomes fully visible")]
        public float fadeInDistance = 0.3f;

        [Tooltip("Distance at which hand becomes invisible")]
        public float fadeOutDistance = 1.0f;

        private Material _handMaterial;

        void Start()
        {
            if (handRenderer != null)
                _handMaterial = handRenderer.material;
        }

        // Wrist placement and alpha are now handled by AuraXRInferenceManager (position)
        // and HandProximityVisibility (fade). This script is kept as a hook for future use.
    }
}
