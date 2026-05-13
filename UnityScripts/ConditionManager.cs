using UnityEngine;

namespace AuraXR
{
    /// <summary>
    /// Switches between experimental conditions at session start.
    /// Condition persists via PlayerPrefs so it survives reloads on-device.
    /// </summary>
    public class ConditionManager : MonoBehaviour
    {
        public enum Condition { VirtualHands, Controller, StaticPose }

        [Tooltip("Override condition here for testing — ignored on-device if PlayerPrefs is set")]
        public Condition debugCondition = Condition.VirtualHands;

        [Header("References")]
        public GameObject leftHandRig;
        public GameObject rightHandRig;
        public GameObject leftControllerModel;
        public GameObject rightControllerModel;
        public AuraXRInferenceManager inferenceManager;

        public static Condition ActiveCondition { get; private set; }

        void Awake()
        {
            int stored = PlayerPrefs.GetInt("AuraXR_Condition", (int)debugCondition);
            ActiveCondition = (Condition)stored;
            ApplyCondition(ActiveCondition);
            Debug.Log($"[AuraXR] Condition: {ActiveCondition}");
        }

        public static void SetCondition(Condition c)
        {
            PlayerPrefs.SetInt("AuraXR_Condition", (int)c);
            PlayerPrefs.Save();
        }

        private void ApplyCondition(Condition c)
        {
            bool virtualHands = (c == Condition.VirtualHands || c == Condition.StaticPose);
            bool controllers  = (c == Condition.Controller);
            bool freeze       = (c == Condition.StaticPose);

            if (leftHandRig  != null) leftHandRig.SetActive(virtualHands);
            if (rightHandRig != null) rightHandRig.SetActive(virtualHands);
            if (leftControllerModel  != null) leftControllerModel.SetActive(controllers);
            if (rightControllerModel != null) rightControllerModel.SetActive(controllers);

            if (inferenceManager != null)
                inferenceManager.enabled = !freeze;
        }
    }
}
