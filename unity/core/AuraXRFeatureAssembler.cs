using UnityEngine;

namespace AuraXR
{
    /// <summary>
    /// Runtime data hub: exposes controller transforms, nearest interactable objects,
    /// and current grip/trigger values to all AuraXR components.
    ///
    /// The 96-dim ring buffer and visual embedding placeholder that were here previously
    /// have been removed — the inference pipeline assembles its own 15-dim feature directly.
    /// </summary>
    public class AuraXRFeatureAssembler : MonoBehaviour
    {
        [Header("XR Node references (auto-detected if left null)")]
        public Transform leftControllerTransform;
        public Transform rightControllerTransform;

        [Header("Nearest interactable objects (set by ProximityDetector each frame)")]
        public Transform nearestObjectLeft;
        public Transform nearestObjectRight;
        [Tooltip("HOT3D BOP category IDs (1–33, 0 = unknown)")]
        public int nearestObjectCategoryLeft  = 0;
        public int nearestObjectCategoryRight = 0;

        // Current grip / trigger values — read by SessionDataLogger and VirtualHandGrab each frame
        public float LastLeftGrip     { get; private set; }
        public float LastRightGrip    { get; private set; }
        public float LastLeftTrigger  { get; private set; }
        public float LastRightTrigger { get; private set; }

        void Update()
        {
            LastLeftGrip     = OVRInput.Get(OVRInput.Axis1D.PrimaryHandTrigger,  OVRInput.Controller.LTouch);
            LastLeftTrigger  = OVRInput.Get(OVRInput.Axis1D.PrimaryIndexTrigger, OVRInput.Controller.LTouch);
            LastRightGrip    = OVRInput.Get(OVRInput.Axis1D.PrimaryHandTrigger,  OVRInput.Controller.RTouch);
            LastRightTrigger = OVRInput.Get(OVRInput.Axis1D.PrimaryIndexTrigger, OVRInput.Controller.RTouch);
        }
    }
}
