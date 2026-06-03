using System.Collections;
using UnityEngine;

// Attach to OVRCameraRig (add CharacterController to the same object).
// Inspector assignments:
//   cameraTransform       -> CenterEyeAnchor (inside OVRCameraRig)
//   headCollisionLayers   -> "Environment" layer (walls, table, floor)
public class ThumbstickLocomotion : MonoBehaviour
{
    [Header("Movement")]
    public Transform cameraTransform;
    public float moveSpeed  = 1.5f;
    public float turnSpeed  = 60f;
    public float gravity    = -9.81f;

    [Header("Jump  (A button / Space in simulator)")]
    public float jumpHeight = 1.0f;

    [Header("Crouch  (B button / B in simulator)")]
    public float standHeight  = 1.8f;
    public float crouchHeight = 0.9f;

    [Header("Head Collision (prevents bending/leaning through walls)")]
    public LayerMask headCollisionLayers;   // assign the Environment layer
    public float headRadius = 0.12f;

    private CharacterController _cc;
    private float _verticalVelocity = 0f;
    private bool  _ready        = false;
    private bool  _crouching    = false;
    private bool  _crouchPressed = false;
    private float _crouchPressTime = -1f;

    void Awake()
    {
        _cc = GetComponent<CharacterController>();
        if (_cc == null)
        {
            Debug.LogError("[ThumbstickLocomotion] No CharacterController on this GameObject. " +
                           "Add CharacterController to OVRCameraRig, then attach this script to it.", this);
            enabled = false;
            return;
        }
        _cc.height = standHeight;
        _cc.center = new Vector3(0, standHeight / 2f, 0);
    }

    IEnumerator Start()
    {
        // Skip 3 frames so OVR finishes its own initialization.
        yield return null;
        yield return null;
        yield return null;

        // Settle CC onto the floor before enabling movement.
        int limit = 30;
        while (!_cc.isGrounded && limit-- > 0)
        {
            _cc.Move(Vector3.down * 0.05f);
            yield return null;
        }

        _verticalVelocity = -1f;
        _ready = true;
    }

    void LateUpdate()
    {
        if (!_ready) return;

        HandleTurn();
        HandleCrouch();
        HandleMovement();
        ClampHeadPosition();   // must run last, after OVR updates camera
    }

    void HandleTurn()
    {
        Vector2 turn = OVRInput.Get(OVRInput.Axis2D.PrimaryThumbstick, OVRInput.Controller.RTouch);
        transform.Rotate(Vector3.up, turn.x * turnSpeed * Time.deltaTime);
    }

    void HandleCrouch()
    {
        // B button (right controller) — toggle crouch
        // Meta XR Simulator: B key on keyboard
        _crouchPressed = OVRInput.GetDown(OVRInput.Button.Two, OVRInput.Controller.RTouch);
        if (_crouchPressed)
        {
            _crouchPressTime = Time.time;
            _crouching = !_crouching;
            float h = _crouching ? crouchHeight : standHeight;
            _cc.height = h;
            _cc.center = new Vector3(0, h / 2f, 0);
        }
    }

    void HandleMovement()
    {
        Vector2 move = OVRInput.Get(OVRInput.Axis2D.PrimaryThumbstick, OVRInput.Controller.LTouch);

        Vector3 forward = cameraTransform.forward; forward.y = 0f; forward.Normalize();
        Vector3 right   = cameraTransform.right;   right.y   = 0f; right.Normalize();
        Vector3 moveDir = (forward * move.y + right * move.x) * moveSpeed;

        if (_cc.isGrounded)
        {
            _verticalVelocity = -1f;

            // A button (right controller) — jump
            // Meta XR Simulator: A key on keyboard
            bool recentCrouch = (Time.time - _crouchPressTime) < 0.15f;
            if (!recentCrouch && OVRInput.GetDown(OVRInput.Button.One, OVRInput.Controller.RTouch))
                _verticalVelocity = Mathf.Sqrt(jumpHeight * -2f * gravity);
        }
        else
        {
            _verticalVelocity += gravity * Time.deltaTime;
        }

        moveDir.y = _verticalVelocity;
        _cc.Move(moveDir * Time.deltaTime);
    }

    // Detects when the physical head (headset) has moved into geometry (e.g. leaning into
    // a wall or bending over the table) and pushes the player root horizontally to compensate.
    // headCollisionLayers should be set to the "Environment" layer in the Inspector.
    // If not set (value = 0), falls back to Physics.DefaultRaycastLayers so walls still block.
    void ClampHeadPosition()
    {
        if (cameraTransform == null) return;

        int mask = headCollisionLayers != 0 ? headCollisionLayers.value : Physics.DefaultRaycastLayers;

        Vector3 headPos = cameraTransform.position;
        Collider[] hits = Physics.OverlapSphere(headPos, headRadius, mask,
                                                QueryTriggerInteraction.Ignore);

        foreach (Collider hit in hits)
        {
            if (hit is MeshCollider mc && !mc.convex) continue;
            Vector3 closest  = hit.ClosestPoint(headPos);
            Vector3 pushDir  = headPos - closest;
            float   overlap  = headRadius - pushDir.magnitude;

            if (overlap <= 0f || pushDir.sqrMagnitude < 1e-6f) continue;

            // Push only horizontally so gravity / floor detection is unaffected.
            Vector3 push = pushDir.normalized * overlap;
            push.y = 0f;
            transform.position += push;
        }
    }
}
