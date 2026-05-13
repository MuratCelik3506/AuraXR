using UnityEngine;

namespace AuraXR
{
    /// <summary>
    /// Marks this GameObject as an interactable object.
    /// The feature assembler will detect nearby objects and use their properties as input.
    /// </summary>
    public class InteractableObject : MonoBehaviour
    {
        [Tooltip("HOT3D object category ID (1–33). 0 = unknown.")]
        [Range(0, 33)]
        public int categoryId = 0;

        [Tooltip("Human-readable category name (e.g., 'bottle', 'mug')")]
        public string categoryName = "unknown";

        public Collider objectCollider;

        void Start()
        {
            objectCollider = GetComponent<Collider>();
            if (objectCollider == null)
                Debug.LogWarning($"[AuraXR] InteractableObject '{gameObject.name}' has no Collider!");
        }

        public Bounds GetBounds()
        {
            var renderer = GetComponent<Renderer>();
            if (renderer != null) return renderer.bounds;
            if (objectCollider != null) return objectCollider.bounds;
            return new Bounds(transform.position, Vector3.one * 0.1f);
        }
    }
}
