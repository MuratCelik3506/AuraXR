using System.Linq;
using UnityEngine;
#if ENABLE_INPUT_SYSTEM
using UnityEngine.InputSystem;
#endif

namespace AuraXR.Demo
{
    /// <summary>
    /// Editor/MetaXR-Simulator fallback controls.
    ///
    /// Keeps controller/hand visuals visible when no tracked controller pose is supplied,
    /// and provides simple keyboard movement so the demo is testable before real XR input
    /// plumbing is verified.
    /// </summary>
    public sealed class AuraXRDesktopSimulatorControls : MonoBehaviour
    {
        [Header("Rig")]
        public Transform xrRigRoot;
        public Transform cameraOffset;
        public Transform leftController;
        public Transform rightController;
        public Transform rightWristForModel;

        [Header("Movement")]
        public float rigMoveSpeed = 1.2f;
        public float wristMoveSpeed = 0.45f;
        public bool forceVisibleDefaultPose = true;

        [Header("Default Local Poses")]
        public Vector3 leftControllerLocal = new Vector3(-0.28f, -0.25f, 0.55f);
        public Vector3 rightControllerLocal = new Vector3(0.28f, -0.25f, 0.55f);

        void Start()
        {
            AutoWireIfNeeded();
            ApplyVisibleDefaultPose();
        }

        void Update()
        {
            AutoWireIfNeeded();
            if (forceVisibleDefaultPose) EnsureVisualsActive();

            float dt = Time.deltaTime;
            if (xrRigRoot != null)
            {
                Vector3 move = Vector3.zero;
                if (IsPressed(KeyCode.W)) move += Vector3.forward;
                if (IsPressed(KeyCode.S)) move += Vector3.back;
                if (IsPressed(KeyCode.A)) move += Vector3.left;
                if (IsPressed(KeyCode.D)) move += Vector3.right;
                if (IsPressed(KeyCode.Q)) move += Vector3.down;
                if (IsPressed(KeyCode.E)) move += Vector3.up;
                xrRigRoot.position += move * rigMoveSpeed * dt;
            }

            Transform wrist = rightWristForModel != null ? rightWristForModel : rightController;
            if (wrist != null)
            {
                Vector3 move = Vector3.zero;
                if (IsPressed(KeyCode.I)) move += Vector3.forward;
                if (IsPressed(KeyCode.K)) move += Vector3.back;
                if (IsPressed(KeyCode.J)) move += Vector3.left;
                if (IsPressed(KeyCode.L)) move += Vector3.right;
                if (IsPressed(KeyCode.U)) move += Vector3.up;
                if (IsPressed(KeyCode.O)) move += Vector3.down;
                wrist.position += move * wristMoveSpeed * dt;
            }
        }

        private static bool IsPressed(KeyCode key)
        {
#if ENABLE_INPUT_SYSTEM
            Keyboard kb = Keyboard.current;
            if (kb == null) return false;
            switch (key)
            {
                case KeyCode.W: return kb.wKey.isPressed;
                case KeyCode.S: return kb.sKey.isPressed;
                case KeyCode.A: return kb.aKey.isPressed;
                case KeyCode.D: return kb.dKey.isPressed;
                case KeyCode.Q: return kb.qKey.isPressed;
                case KeyCode.E: return kb.eKey.isPressed;
                case KeyCode.I: return kb.iKey.isPressed;
                case KeyCode.K: return kb.kKey.isPressed;
                case KeyCode.J: return kb.jKey.isPressed;
                case KeyCode.L: return kb.lKey.isPressed;
                case KeyCode.U: return kb.uKey.isPressed;
                case KeyCode.O: return kb.oKey.isPressed;
                default: return false;
            }
#else
            return Input.GetKey(key);
#endif
        }

        public void ApplyVisibleDefaultPose()
        {
            if (leftController != null) leftController.localPosition = leftControllerLocal;
            if (rightController != null) rightController.localPosition = rightControllerLocal;
            EnsureVisualsActive();
        }

        private void EnsureVisualsActive()
        {
            ActivateChildrenContaining(leftController, "Visual");
            ActivateChildrenContaining(rightController, "Visual");
            ActivateChildrenContaining(leftController, "UniversalController");
            ActivateChildrenContaining(rightController, "UniversalController");
        }

        private static void ActivateChildrenContaining(Transform root, string token)
        {
            if (root == null) return;
            foreach (Transform t in root.GetComponentsInChildren<Transform>(true))
            {
                if (t.name.Contains(token)) t.gameObject.SetActive(true);
            }
        }

        private void AutoWireIfNeeded()
        {
            if (xrRigRoot == null)
            {
                GameObject rig = GameObject.Find("AuraXR_XR_Origin_Hands");
                if (rig != null) xrRigRoot = rig.transform;
            }
            if (xrRigRoot == null) return;

            Transform[] all = xrRigRoot.GetComponentsInChildren<Transform>(true);
            if (cameraOffset == null) cameraOffset = all.FirstOrDefault(t => t.name == "Camera Offset");
            if (leftController == null) leftController = all.FirstOrDefault(t => t.name == "Left Controller");
            if (rightController == null) rightController = all.FirstOrDefault(t => t.name == "Right Controller");
            if (rightWristForModel == null) rightWristForModel = rightController;
        }
    }
}
