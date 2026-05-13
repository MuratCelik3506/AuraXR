using System;
using System.IO;
using UnityEngine;

namespace AuraXR
{
    /// <summary>
    /// Logs hand poses, controller data, and interaction events to CSV for analysis.
    /// </summary>
    public class SessionDataLogger : MonoBehaviour
    {
        [Header("References")]
        public AuraXRInferenceManager inferenceManager;
        public AuraXRFeatureAssembler featureAssembler;

        [Header("Logging")]
        public bool enableLogging = true;

        private StreamWriter _csvWriter;
        private float _sessionStartTime;

        void Start()
        {
            if (!enableLogging) return;

            string logPath = $"{Application.persistentDataPath}/auraxr_session_{DateTime.Now:yyyy_MM_dd_HH_mm_ss}.csv";
            _csvWriter = new StreamWriter(logPath, false);
            _csvWriter.WriteLine(
                "timestamp,left_grip,left_trigger,right_grip,right_trigger," +
                "left_hand_x,left_hand_y,left_hand_z,right_hand_x,right_hand_y,right_hand_z," +
                "left_obj_category,right_obj_category");
            _csvWriter.Flush();

            _sessionStartTime = Time.time;
            Debug.Log($"[AuraXR] Session logging started: {logPath}");
        }

        void LateUpdate()
        {
            if (!enableLogging || _csvWriter == null) return;
            if (inferenceManager == null || featureAssembler == null) return;

            float timestamp = Time.time - _sessionStartTime;
            var leftPose  = inferenceManager.LeftHand;
            var rightPose = inferenceManager.RightHand;

            string row = $"{timestamp:F3}," +
                $"{featureAssembler.LastLeftGrip:F3},{featureAssembler.LastLeftTrigger:F3}," +
                $"{featureAssembler.LastRightGrip:F3},{featureAssembler.LastRightTrigger:F3}," +
                $"{leftPose.WristPosition.x:F3},{leftPose.WristPosition.y:F3},{leftPose.WristPosition.z:F3}," +
                $"{rightPose.WristPosition.x:F3},{rightPose.WristPosition.y:F3},{rightPose.WristPosition.z:F3}," +
                $"{featureAssembler.nearestObjectCategoryLeft}," +
                $"{featureAssembler.nearestObjectCategoryRight}";

            _csvWriter.WriteLine(row);

            if (Time.frameCount % 100 == 0)
                _csvWriter.Flush();
        }

        void OnDestroy()
        {
            if (_csvWriter != null)
            {
                _csvWriter.Flush();
                _csvWriter.Close();
                Debug.Log("[AuraXR] Session log saved.");
            }
        }
    }
}
