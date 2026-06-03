using UnityEngine;
using TMPro;

namespace AuraXR
{
    /// <summary>
    /// Displays current task instruction and elapsed timer on a World Space Canvas.
    /// </summary>
    public class UITaskDisplay : MonoBehaviour
    {
        [Header("UI Elements")]
        public TextMeshProUGUI instructionText;
        public TextMeshProUGUI timerText;

        [Header("Timer")]
        public bool timerRunning = false;
        private float _startTime;

        void LateUpdate()
        {
            if (timerRunning && timerText != null)
            {
                float elapsed = Time.time - _startTime;
                timerText.text = $"{(int)(elapsed / 60):00}:{(elapsed % 60):00.0}";
            }
        }

        public void UpdateInstruction(string message)
        {
            if (instructionText != null)
                instructionText.text = message;
        }

        public void StartTimer()
        {
            _startTime = Time.time;
            timerRunning = true;
        }

        public void StopTimer() => timerRunning = false;
    }
}
