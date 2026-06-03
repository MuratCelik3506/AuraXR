using System;
using UnityEngine;

namespace AuraXR
{
    /// <summary>
    /// Auto-configures Rigidbody physics on every InteractableObject at scene start.
    /// Objects that fall below fallResetY teleport back to spawn position.
    ///
    /// Logical additions:
    ///   • MassOverride[]     — set different mass per object (bottle heavier than cup)
    ///   • onObjectReset      — event so other systems react when an object is reset
    ///   • Snap zone clearing — when an object is reset, its SnapZone is cleared too
    ///   • ResetAll()         — public method for external reset triggers (e.g. restart button)
    /// </summary>
    public class PhysicsObjectSetup : MonoBehaviour
    {
        [Header("Rigidbody Defaults")]
        [Tooltip("Default mass in kg (used when no MassOverride is set).")]
        public float defaultMass         = 0.4f;
        public float defaultDrag         = 0.5f;
        public float defaultAngularDrag  = 1.5f;
        public float maxAngularVelocity  = 8f;

        [Header("Per-Object Mass Overrides")]
        [Tooltip("Set a specific mass for individual objects (e.g. bottle = 0.5 kg, cup = 0.2 kg).")]
        public MassOverride[] massOverrides;

        [Header("Fall Reset")]
        [Tooltip("Objects that fall below this world-Y teleport back to spawn.")]
        public float fallResetY = -0.5f;

        // Fired each time an object is reset — subscribe in code, not Inspector
        public event Action<InteractableObject> onObjectReset;

        private InteractableObject[] _objects;
        private Vector3[]            _spawnPositions;
        private Quaternion[]         _spawnRotations;
        private SnapZone[]           _snapZones;

        // ── Unity ─────────────────────────────────────────────────────────────

        void Start()
        {
            _objects        = FindObjectsByType<InteractableObject>(FindObjectsInactive.Exclude);
            _snapZones      = FindObjectsByType<SnapZone>(FindObjectsInactive.Exclude);
            _spawnPositions = new Vector3[_objects.Length];
            _spawnRotations = new Quaternion[_objects.Length];

            for (int i = 0; i < _objects.Length; i++)
            {
                _spawnPositions[i] = _objects[i].transform.position;
                _spawnRotations[i] = _objects[i].transform.rotation;
                EnsurePhysics(_objects[i]);
            }
        }

        void FixedUpdate()
        {
            for (int i = 0; i < _objects.Length; i++)
            {
                if (_objects[i] == null) continue;
                if (_objects[i].transform.position.y < fallResetY)
                    ResetObject(i);
            }
        }

        // ── Public ────────────────────────────────────────────────────────────

        /// <summary>Reset every interactable object to its spawn position and clear all snap zones.</summary>
        public void ResetAll()
        {
            for (int i = 0; i < _objects.Length; i++)
                ResetObject(i);
        }

        // ── Private ───────────────────────────────────────────────────────────

        private void EnsurePhysics(InteractableObject obj)
        {
            var rb = obj.GetComponent<Rigidbody>();
            if (rb == null) rb = obj.gameObject.AddComponent<Rigidbody>();

            rb.mass                   = GetMass(obj);
            rb.linearDamping          = defaultDrag;
            rb.angularDamping         = defaultAngularDrag;
            rb.maxAngularVelocity     = maxAngularVelocity;
            rb.interpolation          = RigidbodyInterpolation.Interpolate;
            rb.collisionDetectionMode = CollisionDetectionMode.Continuous;

            if (obj.GetComponent<Collider>() == null)
                obj.gameObject.AddComponent<BoxCollider>();
        }

        private float GetMass(InteractableObject obj)
        {
            if (massOverrides != null)
                foreach (var mo in massOverrides)
                    if (mo.obj == obj) return mo.mass;
            return defaultMass;
        }

        private void ResetObject(int i)
        {
            var obj = _objects[i];

            // Clear any snap zone that currently holds this object
            foreach (var zone in _snapZones)
                if (zone != null && zone.SnappedObject == obj)
                    zone.ClearSnap();

            var rb = obj.GetComponent<Rigidbody>();
            if (rb != null)
            {
                rb.isKinematic    = true;
                rb.linearVelocity  = Vector3.zero;
                rb.angularVelocity = Vector3.zero;
            }

            obj.transform.position = _spawnPositions[i];
            obj.transform.rotation = _spawnRotations[i];

            if (rb != null) rb.isKinematic = false;

            onObjectReset?.Invoke(obj);
            Debug.Log($"[AuraXR] Reset '{obj.name}' to spawn position.");
        }

        // ── Nested Types ──────────────────────────────────────────────────────

        [Serializable]
        public struct MassOverride
        {
            [Tooltip("The specific interactable object.")]
            public InteractableObject obj;
            [Tooltip("Mass in kg.")]
            public float mass;
        }
    }
}
