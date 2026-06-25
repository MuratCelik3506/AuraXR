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
        [Tooltip("Use labels like E0_static, E1_learned, E2_state_machine for replay comparison.")]
        public string sessionLabel = "E1_learned";

        [Tooltip("Write every N frames. 1 = every frame.")]
        [Range(1, 30)]
        public int sampleEveryNFrames = 1;

        private StreamWriter _csvWriter;
        private float _sessionStartTime;
        private float[] _prevLeftAngles;
        private float[] _prevRightAngles;
        private float[] _prevLeftDelta;
        private float[] _prevRightDelta;
        private string _prevLeftObject = "";
        private string _prevRightObject = "";

        void Start()
        {
            if (!enableLogging) return;

            string logFolder = System.IO.Path.Combine(Application.persistentDataPath, "Logs");
            System.IO.Directory.CreateDirectory(logFolder);
            string logPath = System.IO.Path.Combine(logFolder, $"auraxr_session_{DateTime.Now:yyyy_MM_dd_HH_mm_ss}.csv");
            _csvWriter = new StreamWriter(logPath, false);
            _csvWriter.WriteLine(
                "timestamp,frame,session_label,left_grip,left_trigger,right_grip,right_trigger," +
                "left_obj_name,right_obj_name,left_obj_category,right_obj_category," +
                "left_obj_distance_m,right_obj_distance_m,left_object_switched,right_object_switched," +
                "left_approach_distance_m,right_approach_distance_m," +
                "left_angle_delta_deg,right_angle_delta_deg,left_angle_jerk_deg,right_angle_jerk_deg," +
                "left_hand_x,left_hand_y,left_hand_z,right_hand_x,right_hand_y,right_hand_z," +
                "left_controller_x,left_controller_y,left_controller_z,right_controller_x,right_controller_y,right_controller_z");
            _csvWriter.Flush();

            _sessionStartTime = Time.time;
            Debug.Log($"[AuraXR] Session logging started: {logPath}");
        }

        void LateUpdate()
        {
            if (!enableLogging || _csvWriter == null) return;
            if (sampleEveryNFrames > 1 && Time.frameCount % sampleEveryNFrames != 0) return;
            if (inferenceManager == null || featureAssembler == null) return;

            float timestamp = Time.time - _sessionStartTime;
            var leftPose  = inferenceManager.LeftHand;
            var rightPose = inferenceManager.RightHand;

            string leftObjName = featureAssembler.nearestObjectLeft != null
                ? featureAssembler.nearestObjectLeft.name : "NONE";
            string rightObjName = featureAssembler.nearestObjectRight != null
                ? featureAssembler.nearestObjectRight.name : "NONE";
            bool leftSwitched = leftObjName != _prevLeftObject;
            bool rightSwitched = rightObjName != _prevRightObject;
            _prevLeftObject = leftObjName;
            _prevRightObject = rightObjName;

            float leftObjDist = DistanceToObject(featureAssembler.leftControllerTransform, featureAssembler.nearestObjectLeft);
            float rightObjDist = DistanceToObject(featureAssembler.rightControllerTransform, featureAssembler.nearestObjectRight);

            float leftDelta = MeanAngleDeltaDeg(leftPose.ManoJointAngles, ref _prevLeftAngles);
            float rightDelta = MeanAngleDeltaDeg(rightPose.ManoJointAngles, ref _prevRightAngles);
            float leftJerk = MeanDeltaJerkDeg(leftDelta, ref _prevLeftDelta);
            float rightJerk = MeanDeltaJerkDeg(rightDelta, ref _prevRightDelta);

            Vector3 leftCtrl = featureAssembler.leftControllerTransform != null
                ? featureAssembler.leftControllerTransform.position : Vector3.zero;
            Vector3 rightCtrl = featureAssembler.rightControllerTransform != null
                ? featureAssembler.rightControllerTransform.position : Vector3.zero;

            string row = $"{timestamp:F3},{Time.frameCount},{sessionLabel}," +
                $"{featureAssembler.LastLeftGrip:F3},{featureAssembler.LastLeftTrigger:F3}," +
                $"{featureAssembler.LastRightGrip:F3},{featureAssembler.LastRightTrigger:F3}," +
                $"{leftObjName},{rightObjName}," +
                $"{featureAssembler.nearestObjectCategoryLeft}," +
                $"{featureAssembler.nearestObjectCategoryRight}," +
                $"{leftObjDist:F4},{rightObjDist:F4}," +
                $"{(leftSwitched ? 1 : 0)},{(rightSwitched ? 1 : 0)}," +
                $"{leftPose.ApproachDistance:F4},{rightPose.ApproachDistance:F4}," +
                $"{leftDelta:F3},{rightDelta:F3},{leftJerk:F3},{rightJerk:F3}," +
                $"{leftPose.WristPosition.x:F3},{leftPose.WristPosition.y:F3},{leftPose.WristPosition.z:F3}," +
                $"{rightPose.WristPosition.x:F3},{rightPose.WristPosition.y:F3},{rightPose.WristPosition.z:F3}," +
                $"{leftCtrl.x:F3},{leftCtrl.y:F3},{leftCtrl.z:F3}," +
                $"{rightCtrl.x:F3},{rightCtrl.y:F3},{rightCtrl.z:F3}";

            _csvWriter.WriteLine(row);

            if (Time.frameCount % 100 == 0)
                _csvWriter.Flush();
        }

        private static float DistanceToObject(Transform controller, Transform obj)
        {
            if (controller == null || obj == null) return -1f;
            return Vector3.Distance(controller.position, obj.position);
        }

        private static float MeanAngleDeltaDeg(float[] angles, ref float[] previous)
        {
            if (angles == null || angles.Length == 0) return 0f;
            if (previous == null || previous.Length != angles.Length)
            {
                previous = (float[])angles.Clone();
                return 0f;
            }

            float sum = 0f;
            for (int i = 0; i < angles.Length; i++)
            {
                sum += Mathf.Abs(angles[i] - previous[i]) * Mathf.Rad2Deg;
                previous[i] = angles[i];
            }
            return sum / angles.Length;
        }

        private static float MeanDeltaJerkDeg(float delta, ref float[] previousDelta)
        {
            if (previousDelta == null || previousDelta.Length == 0)
            {
                previousDelta = new float[] { delta };
                return 0f;
            }
            float jerk = Mathf.Abs(delta - previousDelta[0]);
            previousDelta[0] = delta;
            return jerk;
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
