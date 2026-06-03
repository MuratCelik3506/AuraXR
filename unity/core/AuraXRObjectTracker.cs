using System.Collections.Generic;
using UnityEngine;

namespace AuraXR
{
    /// <summary>
    /// Each frame finds the nearest AuraXRTrackable to the left and right controllers,
    /// then updates AuraXRFeatureAssembler so the model receives correct object context.
    ///
    /// Setup:
    ///   1. Attach this component to any persistent GameObject.
    ///   2. Assign the Assembler reference.
    ///   3. Add AuraXRTrackable to every interactable object in the scene and set its CategoryId.
    ///      Category IDs must match HOT3D BOP IDs (1–33). Use 0 for unknown/generic props.
    /// </summary>
    public class AuraXRObjectTracker : MonoBehaviour
    {
        [Header("Dependencies")]
        public AuraXRFeatureAssembler assembler;

        [Header("Search radius (metres) — objects beyond this are ignored")]
        public float searchRadius = 2.0f;

        // All trackables register/unregister themselves
        private static readonly List<AuraXRTrackable> _all = new();

        public static void Register(AuraXRTrackable t)   => _all.Add(t);
        public static void Unregister(AuraXRTrackable t) => _all.Remove(t);

        void LateUpdate()
        {
            if (assembler == null) return;

            Transform leftCtrl  = assembler.leftControllerTransform;
            Transform rightCtrl = assembler.rightControllerTransform;

            AuraXRTrackable nearestLeft  = FindNearest(leftCtrl,  searchRadius);
            AuraXRTrackable nearestRight = FindNearest(rightCtrl, searchRadius);

            assembler.nearestObjectLeft          = nearestLeft?.transform;
            assembler.nearestObjectCategoryLeft  = nearestLeft?.CategoryId ?? 0;

            assembler.nearestObjectRight         = nearestRight?.transform;
            assembler.nearestObjectCategoryRight = nearestRight?.CategoryId ?? 0;
        }

        private static AuraXRTrackable FindNearest(Transform origin, float maxDist)
        {
            if (origin == null) return null;

            AuraXRTrackable best    = null;
            float           bestDist = maxDist;

            foreach (var t in _all)
            {
                if (t == null || !t.gameObject.activeInHierarchy) continue;
                float d = Vector3.Distance(origin.position, t.transform.position);
                if (d < bestDist)
                {
                    bestDist = d;
                    best     = t;
                }
            }
            return best;
        }
    }


    /// <summary>
    /// Attach to any interactable GameObject. Registers it with AuraXRObjectTracker.
    /// CategoryId must match the HOT3D BOP object ID (1–33).
    /// </summary>
    public class AuraXRTrackable : MonoBehaviour
    {
        [Tooltip("HOT3D BOP category ID (1–33). 0 = unknown/generic.")]
        public int CategoryId = 0;

        void OnEnable()  => AuraXRObjectTracker.Register(this);
        void OnDisable() => AuraXRObjectTracker.Unregister(this);
    }
}
