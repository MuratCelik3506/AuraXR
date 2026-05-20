using System.Collections;
using UnityEngine;
using UnityEngine.Events;

namespace AuraXR
{
    /// <summary>
    /// A glowing placement zone on the table surface.
    /// When an InteractableObject is released slowly enough inside this trigger
    /// it snaps smoothly to the zone centre.
    ///
    /// New logical additions:
    ///   • acceptedObject  — only snaps a specific object (null = accept any)
    ///   • IsSnapped       — readable by ScenarioKitchenTask to advance task state
    ///   • SnappedObject   — lets PhysicsObjectSetup clear the zone on fall-reset
    ///   • onSnapped       — UnityEvent so other systems react without polling
    ///
    /// Setup:
    ///   1. Create an empty GameObject on the table surface, e.g. "SnapZone_Bottle".
    ///   2. Add this component — it auto-makes the collider a trigger.
    ///   3. Assign acceptedObject if this zone is for a specific item.
    /// </summary>
    [RequireComponent(typeof(Collider))]
    public class SnapZone : MonoBehaviour
    {
        [Header("Snap Settings")]
        [Tooltip("Only accept this specific object. Leave null to accept any InteractableObject.")]
        public InteractableObject acceptedObject;

        [Tooltip("Objects moving slower than this (m/s) snap to centre when released inside.")]
        public float snapVelocityThreshold = 1.2f;

        [Tooltip("Height offset above zone pivot (keeps object sitting on the surface).")]
        public Vector3 snapOffset = new Vector3(0f, 0.05f, 0f);

        [Header("Ring Visual")]
        public float ringRadius  = 0.12f;
        public Color idleColor   = new Color(0.3f, 0.7f, 1.0f, 0.55f);
        public Color activeColor = new Color(0.0f, 1.0f, 0.5f, 1.00f);
        public float pulseSpeed  = 3.5f;

        [Header("Events")]
        [Tooltip("Fired when an object successfully snaps. Passes the snapped InteractableObject.")]
        public UnityEvent<InteractableObject> onSnapped;

        // ── State ─────────────────────────────────────────────────────────────

        /// <summary>True once an object has been snapped into this zone.</summary>
        public bool IsSnapped => _isSnapped;

        /// <summary>The currently snapped object, or null.</summary>
        public InteractableObject SnappedObject => _snappedObject;

        private bool               _isSnapped;
        private InteractableObject _snappedObject;
        private InteractableObject _candidate;
        private Coroutine          _snapCoroutine;
        private LineRenderer       _ring;

        const int k_Segments = 40;

        // ── Unity ─────────────────────────────────────────────────────────────

        void Start()
        {
            GetComponent<Collider>().isTrigger = true;
            BuildRing();
        }

        void Update()
        {
            RefreshRingColor();
            TrySnap();
        }

        void OnTriggerEnter(Collider other)
        {
            if (_isSnapped) return;
            var io = other.GetComponent<InteractableObject>();
            if (io == null) return;
            if (acceptedObject != null && io != acceptedObject) return;
            _candidate = io;
        }

        void OnTriggerExit(Collider other)
        {
            var io = other.GetComponent<InteractableObject>();
            if (io != null && io == _candidate) _candidate = null;
        }

        // ── Public ────────────────────────────────────────────────────────────

        /// <summary>
        /// Unsnap and re-open the zone (call this on task reset or fall-reset).
        /// Also restores the snapped object's Rigidbody to non-kinematic.
        /// </summary>
        public void ClearSnap()
        {
            if (_snappedObject != null)
            {
                var rb = _snappedObject.GetComponent<Rigidbody>();
                if (rb != null) rb.isKinematic = false;
            }

            _isSnapped     = false;
            _snappedObject = null;
            _candidate     = null;

            if (_snapCoroutine != null)
            {
                StopCoroutine(_snapCoroutine);
                _snapCoroutine = null;
            }
        }

        // ── Private ───────────────────────────────────────────────────────────

        private void TrySnap()
        {
            if (_candidate == null || _isSnapped || _snapCoroutine != null) return;

            var rb = _candidate.GetComponent<Rigidbody>();
            if (rb == null || rb.isKinematic) return;

            if (rb.linearVelocity.magnitude < snapVelocityThreshold)
                _snapCoroutine = StartCoroutine(AnimateSnap(_candidate));
        }

        private IEnumerator AnimateSnap(InteractableObject obj)
        {
            _isSnapped     = true;
            _snappedObject = obj;
            _candidate     = null;

            var rb = obj.GetComponent<Rigidbody>();
            if (rb != null)
            {
                rb.linearVelocity  = Vector3.zero;
                rb.angularVelocity = Vector3.zero;
                rb.isKinematic     = true;
            }

            Vector3 start  = obj.transform.position;
            Vector3 target = transform.position + snapOffset;
            float   t      = 0f;

            while (t < 1f)
            {
                t += Time.deltaTime * 8f;
                obj.transform.position = Vector3.Lerp(start, target, Mathf.SmoothStep(0f, 1f, t));
                yield return null;
            }
            obj.transform.position = target;

            if (rb != null) rb.isKinematic = false;

            onSnapped?.Invoke(obj);
            _snapCoroutine = null;

            Debug.Log($"[SnapZone] '{obj.name}' snapped to '{gameObject.name}'.");
        }

        // ── Ring ──────────────────────────────────────────────────────────────

        private void BuildRing()
        {
            var go = new GameObject("SnapRing");
            go.transform.SetParent(transform, false);

            _ring = go.AddComponent<LineRenderer>();
            _ring.loop              = true;
            _ring.positionCount     = k_Segments;
            _ring.widthMultiplier   = 0.006f;
            _ring.useWorldSpace     = false;
            _ring.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
            _ring.receiveShadows    = false;

            var mat = new Material(Shader.Find("Sprites/Default"));
            mat.color      = idleColor;
            _ring.material = mat;

            for (int i = 0; i < k_Segments; i++)
            {
                float a = (float)i / k_Segments * Mathf.PI * 2f;
                _ring.SetPosition(i, new Vector3(
                    Mathf.Cos(a) * ringRadius, 0.003f,
                    Mathf.Sin(a) * ringRadius));
            }
        }

        private void RefreshRingColor()
        {
            if (_ring == null) return;

            Color target;
            float width;

            if (_isSnapped)
            {
                target = activeColor;
                width  = 0.006f;
            }
            else if (_candidate != null)
            {
                float pulse = Mathf.Sin(Time.time * pulseSpeed) * 0.5f + 0.5f;
                target = Color.Lerp(idleColor, activeColor, pulse);
                width  = 0.005f + pulse * 0.004f;
            }
            else
            {
                target = idleColor;
                width  = 0.005f;
            }

            _ring.startColor      = target;
            _ring.endColor        = target;
            _ring.widthMultiplier = width;
        }

        void OnDrawGizmosSelected()
        {
            Gizmos.color = idleColor;
            Gizmos.DrawWireSphere(transform.position + snapOffset, ringRadius);
        }
    }
}
