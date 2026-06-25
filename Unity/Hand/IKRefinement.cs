using UnityEngine;

namespace AuraXR
{
    /// <summary>
    /// Contact-aware pose refinement: takes the model's predicted finger angles as a
    /// prior, then runs a per-finger correction pass that pushes fingertips out of
    /// interpenetration with the nearest object's collider.
    ///
    /// Detection strategy: "raycast from behind"
    ///   For each DIP bone, shoot a ray from 3 cm BEHIND the fingertip toward the tip.
    ///   If the ray hits a collider surface BEFORE reaching the tip position, the tip
    ///   has passed through the surface — penetration depth = (tipDist - hitDist).
    ///   This works even when the finger is already INSIDE the collider (SphereCast from
    ///   inside is unreliable for most collider types).
    ///
    /// Correction:
    ///   Compute the approximate angle reduction needed to pull the tip back to the surface.
    ///   Apply to DIP first. If DIP alone isn't sufficient (e.g. deep penetration) also
    ///   correct PIP by half the remaining delta.
    ///
    /// Execution order 1500 — runs after HandRigController (500) and HandSkeletonAnchor (1000).
    /// Performance: 5 fingers × 2 raycasts (DIP probe + PIP probe) = 10 raycasts/frame ≈ 0.1 ms.
    /// </summary>
    [DefaultExecutionOrder(1500)]
    public class IKRefinement : MonoBehaviour
    {
        [Header("References")]
        [Tooltip("The HandRigController on this same GameObject")]
        public HandRigController rigController;
        [Tooltip("Feature assembler — used to find nearest object. Auto-found if null.")]
        public AuraXRFeatureAssembler featureAssembler;

        [Header("Enable")]
        public bool enableIK = true;
        [Tooltip("Draw debug rays in Scene view (green = clear, red = penetrating)")]
        public bool drawDebugRays = false;

        [Header("Geometry")]
        [Range(0.005f, 0.04f)]
        [Tooltip("Estimated length of the distal phalanx (m). Controls how far ahead to cast.")]
        public float distalPhalanxLen = 0.022f;   // 22 mm

        [Range(0.01f, 0.05f)]
        [Tooltip("How far behind the fingertip the ray originates (m). Must be > phalanx length to always cross the surface.")]
        public float rearOffset = 0.03f;           // 30 mm

        [Header("Correction")]
        [Range(0f, 60f)]
        [Tooltip("Max DIP correction per frame (degrees). Prevents sudden snapping.")]
        public float maxDipCorrDeg = 20f;

        [Range(0f, 30f)]
        [Tooltip("Max PIP correction per frame (degrees). Applied only if DIP correction alone is insufficient.")]
        public float maxPipCorrDeg = 10f;

        [Range(0f, 1f)]
        [Tooltip("EMA factor for smoothing the correction across frames. 0 = no smoothing, 0.8 = heavy. Prevents flicker.")]
        public float correctionSmoothing = 0.5f;

        [Header("Filtering")]
        [Tooltip("LayerMask for objects considered for contact. Assign to your InteractableObjects layer.")]
        public LayerMask contactLayerMask = ~0;

        // MANO order DIP indices: Thumb=2, Index=5, Middle=8, Ring=11, Pinky=14
        private static readonly int[] DIP_IDX = { 2, 5, 8, 11, 14 };
        // PIP indices (one before DIP): Thumb=1, Index=4, Middle=7, Ring=10, Pinky=13
        private static readonly int[] PIP_IDX = { 1, 4, 7, 10, 13 };

        // EMA-smoothed correction per finger (radians)
        private readonly float[] _smoothCorrRad = new float[5];

        // Public read for HandPoseDebugger overlay
        public float[] LastCorrectionRad => _smoothCorrRad;

        void Start()
        {
            if (rigController    == null) rigController    = GetComponent<HandRigController>();
            if (featureAssembler == null) featureAssembler = FindAnyObjectByType<AuraXRFeatureAssembler>();
        }

        void LateUpdate()
        {
            if (!enableIK || rigController == null || rigController.fingerJoints == null) return;

            Transform nearest = NearestObject();
            if (nearest == null)
            {
                // Decay corrections when hand is away from any object
                for (int f = 0; f < 5; f++)
                    _smoothCorrRad[f] *= 0.85f;
                return;
            }

            Collider col = nearest.GetComponentInChildren<Collider>();
            if (col == null) col = nearest.GetComponent<Collider>();
            if (col == null) return;

            for (int f = 0; f < 5; f++)
            {
                int dipIdx = DIP_IDX[f];
                int pipIdx = PIP_IDX[f];

                if (dipIdx >= rigController.fingerJoints.Length) continue;
                Transform dip = rigController.fingerJoints[dipIdx];
                if (dip == null) continue;

                // Bone forward direction (long axis = local +X in OVR skeleton convention)
                Vector3 boneForward = (dip.rotation * Vector3.right).normalized;

                // Estimated fingertip world position
                Vector3 tipPos     = dip.position + boneForward * distalPhalanxLen;

                // Ray origin: behind the tip by rearOffset
                Vector3 rayOrigin  = tipPos - boneForward * rearOffset;
                float   rayDist    = rearOffset + distalPhalanxLen;

                bool penetrating = false;
                float penetrationM = 0f;
                Vector3 surfaceNormal = -boneForward;

                if (Physics.Raycast(rayOrigin, boneForward, out RaycastHit hit,
                                    rayDist, contactLayerMask, QueryTriggerInteraction.Ignore))
                {
                    // How far along the ray is the surface vs. where the tip is?
                    float tipDistAlongRay = rearOffset;  // tip is at rearOffset from origin
                    if (hit.distance < tipDistAlongRay)
                    {
                        penetrating   = true;
                        penetrationM  = tipDistAlongRay - hit.distance;
                        surfaceNormal = hit.normal;
                    }
                }

                if (drawDebugRays)
                    Debug.DrawRay(rayOrigin, boneForward * rayDist,
                                  penetrating ? Color.red : Color.green, 0.02f);

                float targetCorrRad = 0f;
                if (penetrating && penetrationM > 0.001f)
                {
                    // Angle to retract DIP so tip retreats by penetrationM.
                    // Approximate: Δtip ≈ L * sin(Δθ)  →  Δθ ≈ asin(Δtip / L)
                    float sinArg = Mathf.Clamp01(penetrationM / Mathf.Max(distalPhalanxLen, 0.001f));
                    targetCorrRad = Mathf.Asin(sinArg);
                }

                // EMA smooth
                _smoothCorrRad[f] = correctionSmoothing * _smoothCorrRad[f]
                                  + (1f - correctionSmoothing) * targetCorrRad;

                float corrRad = _smoothCorrRad[f];
                if (corrRad < 0.001f) continue;

                // ── Apply DIP correction ───────────────────────────────────────
                float dipCorrDeg = Mathf.Min(corrRad * Mathf.Rad2Deg, maxDipCorrDeg);
                ApplyAngleCorrection(dip, -dipCorrDeg, rigController.bendAxis);

                // ── Apply PIP correction if deep penetration ───────────────────
                // Give PIP half the remaining correction (up to maxPipCorrDeg)
                float remainingDeg = (corrRad * Mathf.Rad2Deg - dipCorrDeg);
                if (remainingDeg > 0.5f && pipIdx < rigController.fingerJoints.Length)
                {
                    Transform pip = rigController.fingerJoints[pipIdx];
                    if (pip != null)
                    {
                        float pipCorrDeg = Mathf.Min(remainingDeg * 0.5f, maxPipCorrDeg);
                        ApplyAngleCorrection(pip, -pipCorrDeg, rigController.bendAxis);
                    }
                }
            }
        }

        // Rotate joint around the same axis as HandRigController uses for flexion.
        // deltaDeg < 0 = extend (open), > 0 = flex (close).
        private static void ApplyAngleCorrection(Transform bone, float deltaDeg, Vector3 localBendAxis)
        {
            // bendAxis is in bone-local space; transform to world for AngleAxis
            Vector3 axisWorld = bone.rotation * localBendAxis;
            bone.rotation = Quaternion.AngleAxis(deltaDeg, axisWorld) * bone.rotation;
        }

        private Transform NearestObject()
        {
            if (featureAssembler == null) return null;
            return rigController.isLeftHand
                ? featureAssembler.nearestObjectLeft
                : featureAssembler.nearestObjectRight;
        }
    }
}
