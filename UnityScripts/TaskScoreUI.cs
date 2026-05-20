using UnityEngine;
using TMPro;

namespace AuraXR
{
    /// <summary>
    /// Gameful step-by-step task display: live checklist, timer, and star rating at completion.
    ///
    /// Logical additions:
    ///   • task reference  — auto-wires onStateChange + onTaskComplete at Start(),
    ///                       so no manual inspector event wiring is needed.
    ///   • Direct methods  — ScenarioKitchenTask also calls these directly (double-safe).
    ///
    /// Canvas Setup:
    ///   World Space Canvas, ~1 m in front of player at eye height.
    ///   Add four TMP labels for steps, one timer, one stars, one result line.
    /// </summary>
    public class TaskScoreUI : MonoBehaviour
    {
        [Header("Step Labels (TMP)")]
        public TextMeshProUGUI step1Text;
        public TextMeshProUGUI step2Text;
        public TextMeshProUGUI step3Text;
        public TextMeshProUGUI step4Text;

        [Header("Status (TMP)")]
        public TextMeshProUGUI timerText;
        public TextMeshProUGUI starsText;
        public TextMeshProUGUI resultText;

        [Header("Auto-Wire (optional)")]
        [Tooltip("Assign the ScenarioKitchenTask here to auto-hook its events at Start().")]
        public ScenarioKitchenTask task;

        [Header("Step Descriptions")]
        public string[] stepLabels =
        {
            "Pick up the mustard bottle",
            "Pour into the mug",
            "Place bottle back",
            "Pick up the mug"
        };

        [Header("Star Thresholds (seconds)")]
        public float threeStarTime = 20f;
        public float twoStarTime   = 45f;

        private float _startTime;
        private bool  _running;
        private int   _activeStep;

        const string k_Done    = "<color=#00FF88>✓  </color>";
        const string k_Active  = "<color=#FFD700>▶  </color>";
        const string k_Pending = "<color=#888888>○  </color>";

        // ── Unity ─────────────────────────────────────────────────────────────

        void Start()
        {
            // Auto-wire events so inspector wiring is not mandatory
            if (task != null)
            {
                task.onStateChange.AddListener(OnStateChanged);
                task.onTaskComplete.AddListener(OnTaskComplete);
            }

            if (starsText  != null) starsText.text  = "";
            if (resultText != null) resultText.text = "";
            RenderSteps(0);
        }

        void OnDestroy()
        {
            if (task != null)
            {
                task.onStateChange.RemoveListener(OnStateChanged);
                task.onTaskComplete.RemoveListener(OnTaskComplete);
            }
        }

        void LateUpdate()
        {
            if (!_running || timerText == null) return;
            float e = Time.time - _startTime;
            timerText.text = $"<mspace=0.55em>{(int)(e / 60):00}:{(e % 60):00.0}</mspace>";
        }

        // ── Public API ────────────────────────────────────────────────────────

        public void StartTask()
        {
            _startTime  = Time.time;
            _running    = true;
            _activeStep = 0;
            if (starsText  != null) starsText.text  = "";
            if (resultText != null) resultText.text = "";
            RenderSteps(0);
        }

        /// <summary>Called by ScenarioKitchenTask.onStateChange (auto-wired or via Inspector).</summary>
        public void OnStateChanged(ScenarioKitchenTask.TaskState state)
        {
            int step = state switch
            {
                ScenarioKitchenTask.TaskState.PickBottle  => 0,
                ScenarioKitchenTask.TaskState.PourBottle  => 1,
                ScenarioKitchenTask.TaskState.PlaceBottle => 2,
                ScenarioKitchenTask.TaskState.PickCup     => 3,
                ScenarioKitchenTask.TaskState.Done        => 4,
                _                                         => _activeStep
            };
            _activeStep = step;
            RenderSteps(step);
        }

        /// <summary>Called by ScenarioKitchenTask.onTaskComplete (auto-wired or via Inspector).</summary>
        public void OnTaskComplete()
        {
            _running = false;
            float elapsed = Time.time - _startTime;

            int stars = elapsed <= threeStarTime ? 3
                      : elapsed <= twoStarTime   ? 2
                      : 1;

            if (starsText != null)
                starsText.text =
                    "<color=#FFD700>" + new string('★', stars)     + "</color>" +
                    "<color=#444444>" + new string('★', 3 - stars) + "</color>";

            if (resultText != null)
                resultText.text = $"Completed in <b>{elapsed:F1}s</b>";

            if (timerText != null)
            {
                float e = elapsed;
                timerText.text =
                    $"<mspace=0.55em>{(int)(e / 60):00}:{(e % 60):00.0}</mspace>";
            }
        }

        // ── Private ───────────────────────────────────────────────────────────

        private void RenderSteps(int activeIdx)
        {
            SetField(step1Text, 0, activeIdx);
            SetField(step2Text, 1, activeIdx);
            SetField(step3Text, 2, activeIdx);
            SetField(step4Text, 3, activeIdx);
        }

        private void SetField(TextMeshProUGUI field, int stepIndex, int activeIdx)
        {
            if (field == null) return;
            string label = stepIndex < stepLabels.Length
                ? stepLabels[stepIndex]
                : $"Step {stepIndex + 1}";

            field.text = stepIndex < activeIdx  ? k_Done    + $"<s>{label}</s>"
                       : stepIndex == activeIdx ? k_Active  + label
                       :                         k_Pending + label;
        }
    }
}
