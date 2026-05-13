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

        [Header("Grab Settings")]
        [Tooltip("Radius around wrist to search for objects (metres)")]
        public float grabRadius = 0.15f;

        [Tooltip("Grip trigger threshold (0–1)")]
        public float gripThreshold = 0.7f;

        [Tooltip("Multiply wrist velocity when throwing")]
        [Range(0f, 5f)]
        public float throwMultiplier = 1.5f;

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
            bool leftGrip  = OVRInput.Get(OVRInput.Axis1D.PrimaryHandTrigger,
                                           OVRInput.Controller.LTouch) > gripThreshold;
            bool rightGrip = OVRInput.Get(OVRInput.Axis1D.PrimaryHandTrigger,
                                           OVRInput.Controller.RTouch) > gripThreshold;

            if (leftHandWrist != null)
            {
                HandleHand(leftHandWrist,  ref _heldLeft,  leftGrip,  _leftGripWas,  _leftWristPrev);
                _leftWristPrev  = leftHandWrist.position;
            }

            if (rightHandWrist != null)
            {
                HandleHand(rightHandWrist, ref _heldRight, rightGrip, _rightGripWas, _rightWristPrev);
                _rightWristPrev = rightHandWrist.position;
            }

            _leftGripWas  = leftGrip;
            _rightGripWas = rightGrip;
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

        void HandleHand(Transform wrist, ref InteractableObject held,
                        bool grip, bool gripWas, Vector3 wristPrev)
        {
            bool justPressed  =  grip && !gripWas;
            bool justReleased = !grip &&  gripWas;

            if (justPressed && held == null)
            {
                var nearest = FindNearest(wrist.position);
                if (nearest != null)
                {
                    if (wrist == leftHandWrist)
                        Grab(nearest, wrist, ref held, ref _leftGrabPosOffset,  ref _leftGrabRotOffset);
                    else
                        Grab(nearest, wrist, ref held, ref _rightGrabPosOffset, ref _rightGrabRotOffset);
                    Debug.Log($"[VirtualHandGrab] Grabbed {nearest.gameObject.name}");
                }
            }
            else if (justReleased && held != null)
            {
                var throwVel = (wrist.position - wristPrev) / Time.deltaTime;
                Release(ref held, throwVel);
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
            // Store grab-time offset so the object stays in the same relative position
            posOffset = wrist.InverseTransformPoint(obj.transform.position);
            rotOffset = Quaternion.Inverse(wrist.rotation) * obj.transform.rotation;
            held = obj;
        }

        void Release(ref InteractableObject held, Vector3 wristVelocity)
        {
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
