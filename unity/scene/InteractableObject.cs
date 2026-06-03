using UnityEngine;

namespace AuraXR
{
    /// <summary>
    /// Marks this GameObject as an interactable object.
    /// The feature assembler will detect nearby objects and use their properties as input.
    /// </summary>
    public class InteractableObject : MonoBehaviour
    {
        // HOT3D BOP object IDs (use these exact values to match training data):
        //   8  = mug_patterned   (handle_grasp)   POC ★★★
        //   9  = mug_white       (handle_grasp)   POC ★★★
        //  10  = can_soup        (cylindrical)    POC ★★★
        //  13  = bottle_mustard  (bottle_grasp)   POC ★★★
        //  33  = dvd_remote      (precision)      POC ★★★
        [Tooltip("HOT3D BOP category ID (1–33). Must match training data. 0 = unknown.")]
        [Range(0, 33)]
        public int categoryId = 0;

        [Tooltip("HOT3D object name (e.g., 'bottle_mustard', 'mug_patterned')")]
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
