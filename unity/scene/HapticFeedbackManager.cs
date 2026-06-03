using UnityEngine;
using UnityEngine.XR;
using System.Collections.Generic;

namespace AuraXR
{
    /// <summary>
    /// Triggers haptic pulses on Quest 3 controllers when hands are near interactable objects.
    /// </summary>
    public class HapticFeedbackManager : MonoBehaviour
    {
        public AuraXRFeatureAssembler featureAssembler;

        [Tooltip("Distance at which haptic pulse starts (metres)")]
        public float hapticTriggerDistance = 0.12f;

        [Tooltip("Haptic pulse amplitude (0–1)")]
        [Range(0f, 1f)]
        public float hapticAmplitude = 0.3f;

        [Tooltip("Haptic pulse duration in seconds")]
        public float hapticDuration = 0.05f;

        [Tooltip("Minimum interval between pulses (seconds) to avoid constant vibration")]
        public float hapticCooldown = 0.2f;

        private float _lastHapticLeft = 0f;
        private float _lastHapticRight = 0f;
        private InputDevice _leftDevice;
        private InputDevice _rightDevice;

        void Start()
        {
            var leftDevices = new List<InputDevice>();
            InputDevices.GetDevicesWithCharacteristics(
                InputDeviceCharacteristics.Left | InputDeviceCharacteristics.Controller, leftDevices);
            if (leftDevices.Count > 0) _leftDevice = leftDevices[0];

            var rightDevices = new List<InputDevice>();
            InputDevices.GetDevicesWithCharacteristics(
                InputDeviceCharacteristics.Right | InputDeviceCharacteristics.Controller, rightDevices);
            if (rightDevices.Count > 0) _rightDevice = rightDevices[0];
        }

        void LateUpdate()
        {
            if (featureAssembler == null) return;

            if (featureAssembler.nearestObjectLeft != null
                && featureAssembler.leftControllerTransform != null)
            {
                float dist = Vector3.Distance(
                    featureAssembler.leftControllerTransform.position,
                    featureAssembler.nearestObjectLeft.position);

                if (dist < hapticTriggerDistance && Time.time - _lastHapticLeft > hapticCooldown)
                {
                    TriggerHaptic(_leftDevice, hapticAmplitude, hapticDuration);
                    _lastHapticLeft = Time.time;
                }
            }

            if (featureAssembler.nearestObjectRight != null
                && featureAssembler.rightControllerTransform != null)
            {
                float dist = Vector3.Distance(
                    featureAssembler.rightControllerTransform.position,
                    featureAssembler.nearestObjectRight.position);

                if (dist < hapticTriggerDistance && Time.time - _lastHapticRight > hapticCooldown)
                {
                    TriggerHaptic(_rightDevice, hapticAmplitude, hapticDuration);
                    _lastHapticRight = Time.time;
                }
            }
        }

        private void TriggerHaptic(InputDevice device, float amplitude, float duration)
        {
            if (device.isValid)
                device.SendHapticImpulse(0, amplitude, duration);
        }
    }
}
