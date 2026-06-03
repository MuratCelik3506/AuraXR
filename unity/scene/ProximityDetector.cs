using UnityEngine;

namespace AuraXR
{
    /// <summary>
    /// Finds nearest InteractableObject to each controller each frame.
    /// Updates AuraXRFeatureAssembler.nearestObjectLeft/Right accordingly.
    /// </summary>
    public class ProximityDetector : MonoBehaviour
    {
        public AuraXRFeatureAssembler featureAssembler;

        [Tooltip("Search radius in metres")]
        public float searchRadius = 2.0f;

        private InteractableObject[] _interactableObjects;

        void Start()
        {
            _interactableObjects = FindObjectsByType<InteractableObject>(FindObjectsInactive.Exclude);
            if (_interactableObjects.Length == 0)
                Debug.LogWarning("[AuraXR] No InteractableObjects found in scene!");
        }

        void LateUpdate()
        {
            if (featureAssembler == null) return;

            if (featureAssembler.leftControllerTransform != null)
            {
                var (nearestObj, _) = FindNearest(featureAssembler.leftControllerTransform.position);
                featureAssembler.nearestObjectLeft           = nearestObj != null ? nearestObj.transform : null;
                featureAssembler.nearestObjectCategoryLeft   = nearestObj?.categoryId ?? 0;
            }

            if (featureAssembler.rightControllerTransform != null)
            {
                var (nearestObj, _) = FindNearest(featureAssembler.rightControllerTransform.position);
                featureAssembler.nearestObjectRight          = nearestObj != null ? nearestObj.transform : null;
                featureAssembler.nearestObjectCategoryRight  = nearestObj?.categoryId ?? 0;
            }
        }

        private (InteractableObject, float) FindNearest(Vector3 from)
        {
            InteractableObject nearest = null;
            float minDistance = searchRadius;

            foreach (var obj in _interactableObjects)
            {
                if (obj == null) continue;
                float distance = Vector3.Distance(from, obj.transform.position);
                if (distance < minDistance)
                {
                    minDistance = distance;
                    nearest = obj;
                }
            }

            return (nearest, minDistance);
        }
    }
}
