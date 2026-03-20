/*
 * ObjectPoseProvider.cs
 * ======================
 * Abstract base class for supplying the flattened 4×4 RT matrix of the
 * target object to IntentPredictor.
 *
 * In a real XR setup this could come from:
 *   • ARKit object tracking
 *   • Known rigid body pose in scene graph
 *   • Marker-based tracking (e.g. ArUco)
 */

using UnityEngine;

namespace XRIntent
{
    /// <summary>
    /// Provides the flattened (column-major) 4×4 RT matrix of the closest
    /// graspable object, matching the H2O obj_pose_rt format.
    /// Returns a float[16] array.
    /// </summary>
    public abstract class ObjectPoseProvider : MonoBehaviour
    {
        /// <summary>
        /// Returns the 4×4 RT matrix as a flat float[16] array (column-major).
        /// Returns float[16] of zeros if no object is currently tracked.
        /// </summary>
        public abstract float[] GetCurrentRT();

        // ── Convenience: build RT from Unity Transform ────────────
        /// <summary>
        /// Helper: convert a Unity Transform into a column-major float[16]
        /// matching the H2O obj_pose_rt format.
        /// </summary>
        protected static float[] TransformToRT(Transform t)
        {
            Matrix4x4 m = Matrix4x4.TRS(t.position, t.rotation, Vector3.one);
            float[] rt  = new float[16];
            for (int col = 0; col < 4; col++)
                for (int row = 0; row < 4; row++)
                    rt[col * 4 + row] = m[row, col];
            return rt;
        }
    }
}
