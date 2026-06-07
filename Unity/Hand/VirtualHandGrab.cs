using UnityEngine;

namespace AuraXR
{
    /// <summary>
    /// Grab system for ONNX-driven virtual hands.
    /// - Intent: OVR controller grip trigger (> gripThreshold)
    /// - Proximity: virtual hand wrist Transform (LeftHandRig / RightHandRig)
    /// - On grab: object becomes kinematic and follows wrist each frame
    /// - On release: object gets throw velocity from wrist movement
    ///
    /// Attach to GameManager. Assign leftHandWrist = LeftHandRig, rightHandWrist = RightHandRig.
    /// </summary>
    public class VirtualHandGrab : MonoBehaviour
    {
        [Header("Virtual Hand Wrists (drag LeftHandRig / RightHandRig)")]
        public Transform leftHandWrist;
        public Transform rightHandWrist;

        [Header("Physical Controllers (drag OVRCameraRig LeftHandAnchor / RightHandAnchor)")]
        [Tooltip("Used for grab proximity detection. Defaults to leftHandWrist if left null.")]
        public Transform leftController;
        [Tooltip("Used for grab proximity detection. Defaults to rightHandWrist if left null.")]
        public Transform rightController;

        [Header("Grab Settings")]
        [Tooltip("Radius around wrist to search for objects (metres)")]
        public float grabRadius = 0.15f;

        [Tooltip("Grip trigger threshold (0–1). Lower = easier to grab.")]
        public float gripThreshold = 0.5f;

        [Tooltip("Also accept index-finger trigger for grab (more natural for some users)")]
        public bool acceptIndexTrigger = true;

        [Tooltip("Multiply wrist velocity when throwing")]
        [Range(0f, 5f)]
        public float throwMultiplier = 1.5f;

        [Header("Proximity Auto-Close")]
        [Tooltip("When enabled: hand automatically blends toward grab pose whenever a " +
                 "graspable object is within grabRadius — no button press required. " +
                 "The pose is driven by the model; this only drives the grab-blend flag " +
                 "in HandRigController so the hand visually closes as it approaches.")]
        public bool proximityAutoClose = true;

        [Tooltip("Distance inside grabRadius at which grab-blend reaches 1 (fully closed). " +
                 "Objects farther than grabRadius → blend 0. At this distance → blend 1.")]
        public float proximityFullCloseRadius = 0.04f;

        // Grab state — read by HandRigController to drive grab-pose blend
        public bool  IsGrabbingLeft   { get; private set; }
        public bool  IsGrabbingRight  { get; private set; }

        // Smooth proximity blend (0 = open, 1 = fully closed).
        // HandRigController uses this when proximityAutoClose is on.
        public float ProximityBlendLeft  { get; private set; }
        public float ProximityBlendRight { get; private set; }

        // Currently held objects
        private InteractableObject _heldLeft;
        private InteractableObject _heldRight;
        private InteractableObject[] _allInteractables;

        // Grab offsets so the object keeps its relative position to the wrist
        private Vector3    _leftGrabPosOffset;
        private Quaternion _leftGrabRotOffset;
        private Vector3    _rightGrabPosOffset;
        private Quaternion _rightGrabRotOffset;

        // Previous grip states for edge detection
        private bool _leftGripWas;
        private bool _rightGripWas;

        // Previous wrist positions for throw velocity
        private Vector3 _leftWristPrev;
        private Vector3 _rightWristPrev;

        void Start()
        {
            _allInteractables = FindObjectsByType<InteractableObject>(FindObjectsInactive.Exclude);
            if (_allInteractables.Length == 0)
                Debug.LogError("[VirtualHandGrab] No InteractableObjects found in scene! Add an InteractableObject component to every grabable object.");
            if (leftHandWrist  == null)
                Debug.LogError("[VirtualHandGrab] leftHandWrist is not assigned! Drag LeftHandRig into the Inspector field.");
            if (rightHandWrist == null)
                Debug.LogError("[VirtualHandGrab] rightHandWrist is not assigned! Drag RightHandRig into the Inspector field.");

            if (leftHandWrist  != null) _leftWristPrev  = leftHandWrist.position;
            if (rightHandWrist != null) _rightWristPrev = rightHandWrist.position;
        }

        void Update()
        {
            // Grip trigger (middle-finger squeeze) OR index trigger — whichever the user prefers.
            bool leftGrip  = OVRInput.Get(OVRInput.Axis1D.PrimaryHandTrigger,  OVRInput.Controller.LTouch) > gripThreshold
                          || (acceptIndexTrigger && OVRInput.Get(OVRInput.Axis1D.PrimaryIndexTrigger, OVRInput.Controller.LTouch) > gripThreshold);
            bool rightGrip = OVRInput.Get(OVRInput.Axis1D.PrimaryHandTrigger,  OVRInput.Controller.RTouch) > gripThreshold
                          || (acceptIndexTrigger && OVRInput.Get(OVRInput.Axis1D.PrimaryIndexTrigger, OVRInput.Controller.RTouch) > gripThreshold);

            Transform leftProximity  = leftController  != null ? leftController  : leftHandWrist;
            Transform rightProximity = rightController != null ? rightController : rightHandWrist;

            if (leftHandWrist != null)
            {
                HandleHand(leftHandWrist, leftProximity,  ref _heldLeft,  leftGrip,  _leftGripWas,  _leftWristPrev);
                _leftWristPrev  = leftHandWrist.position;
            }

            if (rightHandWrist != null)
            {
                HandleHand(rightHandWrist, rightProximity, ref _heldRight, rightGrip, _rightGripWas, _rightWristPrev);
                _rightWristPrev = rightHandWrist.position;
            }

            _leftGripWas  = leftGrip;
            _rightGripWas = rightGrip;

            // ── Proximity auto-close blend ────────────────────────────────────
            if (proximityAutoClose)
            {
                ProximityBlendLeft  = CalcProximityBlend(leftProximity);
                ProximityBlendRight = CalcProximityBlend(rightProximity);
            }
            else
            {
                ProximityBlendLeft  = 0f;
                ProximityBlendRight = 0f;
            }
        }

        /// <summary>
        /// Returns 0 when no object is within grabRadius, and smoothly rises to 1
        /// as the nearest object approaches proximityFullCloseRadius.
        /// </summary>
        float CalcProximityBlend(Transform probe)
        {
            if (probe == null) return 0f;
            var nearest = FindNearest(probe.position);
            if (nearest == null) return 0f;

            float dist  = Vector3.Distance(probe.position, nearest.transform.position);
            // Remap: grabRadius → 0,  proximityFullCloseRadius → 1
            float range = grabRadius - proximityFullCloseRadius;
            if (range <= 0f) return 1f;
            return Mathf.Clamp01(1f - (dist - proximityFullCloseRadius) / range);
        }

        void LateUpdate()
        {
            // Keep held objects attached to wrist after inference has updated hand position
            if (_heldLeft  != null && leftHandWrist  != null)
                FollowWrist(_heldLeft,  leftHandWrist,  _leftGrabPosOffset,  _leftGrabRotOffset);
            if (_heldRight != null && rightHandWrist != null)
                FollowWrist(_heldRight, rightHandWrist, _rightGrabPosOffset, _rightGrabRotOffset);
        }

        // -----------------------------------------------------------------------

        void HandleHand(Transform wrist, Transform proximity, ref InteractableObject held,
                        bool grip, bool gripWas, Vector3 wristPrev)
        {
            bool isLeft       = wrist == leftHandWrist;
            bool justPressed  =  grip && !gripWas;
            bool justReleased = !grip &&  gripWas;

            if (justPressed && held == null)
            {
                var nearest = FindNearest(proximity.position);
                if (nearest != null)
                {
                    if (isLeft)
                        Grab(nearest, wrist, ref held, ref _leftGrabPosOffset,  ref _leftGrabRotOffset);
                    else
                        Grab(nearest, wrist, ref held, ref _rightGrabPosOffset, ref _rightGrabRotOffset);
                    Debug.Log($"[VirtualHandGrab] Grabbed {nearest.gameObject.name}");
                }
                else
                {
                    // Trigger squeeze even with no object — hand still closes visually
                    if (isLeft) IsGrabbingLeft  = true;
                    else        IsGrabbingRight = true;
                }
            }
            else if (justReleased)
            {
                if (held != null)
                {
                    var throwVel = (wrist.position - wristPrev) / Time.deltaTime;
                    Release(ref held, throwVel, isLeft);
                }
                else
                {
                    if (isLeft) IsGrabbingLeft  = false;
                    else        IsGrabbingRight = false;
                }
            }
        }

        InteractableObject FindNearest(Vector3 origin)
        {
            float bestDist = grabRadius;
            InteractableObject best = null;
            foreach (var io in _allInteractables)
            {
                if (io == null) continue;
                float d = Vector3.Distance(origin, io.transform.position);
                if (d < bestDist)
                {
                    bestDist = d;
                    best = io;
                }
            }
            return best;
        }

        void Grab(InteractableObject obj, Transform wrist, ref InteractableObject held,
                  ref Vector3 posOffset, ref Quaternion rotOffset)
        {
            var rb = obj.GetComponent<Rigidbody>();
            if (rb != null)
            {
                rb.isKinematic     = true;
                rb.linearVelocity  = Vector3.zero;
                rb.angularVelocity = Vector3.zero;
            }
            posOffset = wrist.InverseTransformPoint(obj.transform.position);
            rotOffset = Quaternion.Inverse(wrist.rotation) * obj.transform.rotation;
            held = obj;

            if (wrist == leftHandWrist)  IsGrabbingLeft  = true;
            else                         IsGrabbingRight = true;
        }

        void Release(ref InteractableObject held, Vector3 wristVelocity, bool isLeft)
        {
            if (isLeft) IsGrabbingLeft  = false;
            else        IsGrabbingRight = false;

            var rb = held.GetComponent<Rigidbody>();
            if (rb != null)
            {
                rb.isKinematic    = false;
                rb.linearVelocity = wristVelocity * throwMultiplier;
            }
            held = null;
        }

        void FollowWrist(InteractableObject obj, Transform wrist, Vector3 posOffset, Quaternion rotOffset)
        {
            obj.transform.position = wrist.TransformPoint(posOffset);
            obj.transform.rotation = wrist.rotation * rotOffset;
        }
    }
}
