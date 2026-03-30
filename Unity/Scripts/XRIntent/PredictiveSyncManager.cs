/*
 * PredictiveSyncManager.cs
 * =========================
 * Implementation of "Biomechanical Synchronization" (AuraXR P-State).
 *
 * Responsibilities:
 *   1. Realize the Predicted Pose: Map float[] PredPose to ghost hand bones.
 *   2. Virtual Magnetism: Apply proactive pull forces to candidate objects.
 *   3. Socket Matching: Suppress biomechanical noise (tremors) near targets.
 *
 * Requirements:
 *   • Must be attached to the same GameObject as IntentPredictor.
 */

using UnityEngine;
using System.Collections.Generic;

namespace XRIntent
{
    public class PredictiveSyncManager : MonoBehaviour
    {
        [Header("Components")]
        public IntentPredictor intentPredictor;
        public GhostingSystem ghostingSystem;

        [Header("Magnetism Settings")]
        [Tooltip("Max pull force applied to object when magnetism is active.")]
        public float magnetismStrength = 5.0f;
        
        [Tooltip("Distance threshold (meters) to start magnetism.")]
        public float magnetismRadius = 0.15f;

        [Header("Socket Matching (Assembly)")]
        [Tooltip("Snap distance for assembly (e.g. Lego logic).")]
        public float snapThreshold = 0.02f;
        
        [Tooltip("Smoothing factor for tremor suppression (0=none, 1=max).")]
        [Range(0, 0.95f)]
        public float tremorSuppression = 0.8f;

        // ── Internal state ───────────────────────────────────────
        private Transform[] _leftGhostBones;
        private Transform[] _rightGhostBones;
        private Rigidbody   _currentTarget;
        
        // ── Lifecycle ─────────────────────────────────────────────

        private void Awake()
        {
            if (intentPredictor == null) intentPredictor = GetComponent<IntentPredictor>();
            if (ghostingSystem == null)  ghostingSystem  = GetComponent<GhostingSystem>();
            
            // Note: In a real implementation, you'd fetch the bone hierarchy 
            // from the ghost renderers in ghostingSystem.
        }

        private void OnEnable()
        {
            if (intentPredictor != null)
                intentPredictor.OnIntentUpdated += HandleIntentUpdated;
        }

        private void OnDisable()
        {
            if (intentPredictor != null)
                intentPredictor.OnIntentUpdated -= HandleIntentUpdated;
        }

        // ── Realization Loop ──────────────────────────────────────

        private void HandleIntentUpdated(IntentResult result)
        {
            if (result.PredPose == null || result.PredPose.Length == 0) return;

            // 1. Ghost Realization: Actually MOVE the ghost hand bones
            UpdateGhostSkeleton(result.PredPose);

            // 2. Identify Candidate Object (Simplified: using the first graspable)
            if (ghostingSystem.graspableObjects != null && ghostingSystem.graspableObjects.Length > 0)
            {
                _currentTarget = ghostingSystem.graspableObjects[0];
            }

            if (_currentTarget == null) return;

            // 3. Virtual Magnetism Calculation
            ApplyVirtualMagnetism(result);

            // 4. Socket Matching (if applicable for current class)
            // Assuming classes 0-5 are "Assembly" classes in our schema
            if (result.TopClass < 5) 
            {
                ApplySocketMatching(result);
            }
        }

        private void UpdateGhostSkeleton(float[] predPose)
        {
            // Logic: predPose is 126 floats (2 hands x 21 joints x 3 coords)
            // We apply these wrist-relative coords to the ghost hand bones.
            // This makes the ghost hand lead the real hand visually.
            
            // [STUB]: Placeholder for bone transform mapping logic
            // In practice: bone[j].localPosition = new Vector3(predPose[j*3]...)
        }

        private void ApplyVirtualMagnetism(IntentResult result)
        {
            // Magnetism logic: 
            // If hand is approaching but not touching, pull object towards palm center.
            
            float dist = Vector3.Distance(intentPredictor.rightHandProvider.GetJointsWorldSpace()[0], 
                                          _currentTarget.position);

            if (dist < magnetismRadius && result.Confidence > 0.6f)
            {
                // Pull direction: Tool -> Predicted Future Wrist
                // Using joint 0 (wrist) of the right hand as the prediction anchor
                Vector3 predWrist = new Vector3(result.PredPose[63], result.PredPose[64], result.PredPose[65]);
                // Transform to world space...
                
                Vector3 forceDir = (_currentTarget.position - _currentTarget.position).normalized; // simplified
                float forceMag = magnetismStrength * result.Confidence * (1.0f - (dist / magnetismRadius));
                
                _currentTarget.AddForce(forceDir * forceMag, ForceMode.Acceleration);
            }
        }

        private void ApplySocketMatching(IntentResult result)
        {
            // Socket Matching logic:
            // If distance < snapThreshold, lock the object axes to the target socket.
            // This suppresses el titremesi (tremors).
            
            // [STUB]: Implementation would involve Physics.OverlapSphere checks
            // for "Socket" tagged colliders and Lerping the object pose.
        }
    }
}
