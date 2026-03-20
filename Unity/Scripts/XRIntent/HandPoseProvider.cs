/*
 * HandPoseProvider.cs
 * ====================
 * Abstract base class for supplying 21-joint hand pose data to IntentPredictor.
 *
 * Override this with your XR tracking implementation (e.g. OpenXR, Meta SDK,
 * Apple ARKit, or Ultraleap).
 *
 * Usage:
 *   public class OpenXRHandProvider : HandPoseProvider { ... }
 */

using UnityEngine;

namespace XRIntent
{
    /// <summary>
    /// Provides 21 skeletal joint positions in world space.
    /// Joint ordering follows the MANO / H2O convention:
    ///   0 = Wrist
    ///   1-4   = Thumb (CMC, MCP, IP, Tip)
    ///   5-8   = Index
    ///   9-12  = Middle
    ///   13-16 = Ring
    ///   17-20 = Pinky
    /// </summary>
    public abstract class HandPoseProvider : MonoBehaviour
    {
        public const int JointCount = 21;

        /// <summary>
        /// Returns 21 joint positions in world space.
        /// Joint[0] is the wrist (used as origin for wrist-relative normalization).
        /// </summary>
        public abstract Vector3[] GetJointsWorldSpace();

        /// <summary>Whether this hand is currently tracked.</summary>
        public abstract bool IsTracked { get; }
    }
}
