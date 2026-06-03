using UnityEngine;
using UnityEngine.Events;

namespace AuraXR
{
    /// <summary>
    /// State machine for the kitchen interaction task used in the user study.
    /// States: Idle → PickBottle → PourBottle → PlaceBottle → PickCup → Done
    ///
    /// Logical additions:
    ///   • bottleSnapZone  — PlaceBottle state advances when snap zone reports IsSnapped
    ///                       (falls back to grip-release check if no zone assigned)
    ///   • scoreUI         — directly calls TaskScoreUI.StartTask() so the UI is
    ///                       always in sync regardless of event wiring order
    ///   • ResetTask()     — public reset method; clears snap zones, restores Idle state
    /// </summary>
    public class ScenarioKitchenTask : MonoBehaviour
    {
        public enum TaskState { Idle, PickBottle, PourBottle, PlaceBottle, PickCup, Done }

        [Header("Scene Objects")]
        public InteractableObject bottle;
        public InteractableObject cup;

        [Header("Snap Zones")]
        [Tooltip("Zone where the bottle must be placed. If assigned, PlaceBottle state waits for snap.")]
        public SnapZone bottleSnapZone;

        [Header("State")]
        public TaskState currentState = TaskState.Idle;
        public float taskStartTime;
        public float taskCompletionTime;

        [Header("Events")]
        public UnityEvent onTaskComplete;
        public UnityEvent<TaskState> onStateChange;

        [Header("References")]
        public AuraXRFeatureAssembler featureAssembler;
        public UITaskDisplay uiDisplay;
        public TaskScoreUI scoreUI;
        public SoundManager soundManager;

        [Tooltip("Distance threshold (m) for hand-to-object proximity to trigger state transitions")]
        public float gripThreshold = 0.15f;
        public float gripInputThreshold = 0.7f;

        [Tooltip("If true, task starts automatically on scene load (useful for user study)")]
        public bool autoStart = false;

        private bool _taskStarted = false;

        // ── Unity ─────────────────────────────────────────────────────────────

        void Start()
        {
            if (autoStart) StartTask();
        }

        void Update()
        {
            if (!_taskStarted) return;

            switch (currentState)
            {
                case TaskState.Idle:
                    break;

                case TaskState.PickBottle:
                    if (IsHandNear(bottle, isRight: true) && GetRightGrip() > gripInputThreshold)
                        ChangeState(TaskState.PourBottle);
                    break;

                case TaskState.PourBottle:
                    if (IsTilted(bottle.transform, 60f))
                        ChangeState(TaskState.PlaceBottle);
                    break;

                case TaskState.PlaceBottle:
                    // Use snap zone if assigned, otherwise fall back to grip-release check
                    bool placed = bottleSnapZone != null
                        ? bottleSnapZone.IsSnapped
                        : GetRightGrip() < 0.3f;
                    if (placed)
                        ChangeState(TaskState.PickCup);
                    break;

                case TaskState.PickCup:
                    if ((IsHandNear(cup, isRight: true) || IsHandNear(cup, isRight: false))
                        && (GetRightGrip() > gripInputThreshold || GetLeftGrip() > gripInputThreshold))
                    {
                        ChangeState(TaskState.Done);
                        taskCompletionTime = Time.time - taskStartTime;
                        Debug.Log($"[AuraXR] Task complete in {taskCompletionTime:F1}s");
                        onTaskComplete?.Invoke();
                    }
                    break;
            }
        }

        // ── Public ────────────────────────────────────────────────────────────

        public void StartTask()
        {
            _taskStarted  = true;
            taskStartTime = Time.time;

            uiDisplay?.StartTimer();
            scoreUI?.StartTask();           // sync gameful UI

            ChangeState(TaskState.PickBottle);
            Debug.Log("[AuraXR] Kitchen task started.");
        }

        /// <summary>Resets state to Idle and clears all snap zones. Call before a new trial.</summary>
        public void ResetTask()
        {
            _taskStarted  = false;
            currentState  = TaskState.Idle;
            taskStartTime = 0f;

            bottleSnapZone?.ClearSnap();

            uiDisplay?.StopTimer();
            uiDisplay?.UpdateInstruction("");

            Debug.Log("[AuraXR] Kitchen task reset.");
        }

        // ── Private ───────────────────────────────────────────────────────────

        private void ChangeState(TaskState newState)
        {
            currentState = newState;
            onStateChange?.Invoke(newState);
            uiDisplay?.UpdateInstruction(GetInstruction(newState));
            scoreUI?.OnStateChanged(newState);  // direct call — no event wiring required

            if (soundManager != null)
            {
                switch (newState)
                {
                    case TaskState.PourBottle:  soundManager.PlayPickup();   break;
                    case TaskState.PlaceBottle: soundManager.PlayPour();     break;
                    case TaskState.PickCup:     soundManager.PlayPlace();    break;
                    case TaskState.Done:        soundManager.PlayComplete(); break;
                }
            }
        }

        private bool IsHandNear(InteractableObject obj, bool isRight)
        {
            if (obj == null || featureAssembler == null) return false;
            var ctrl = isRight ? featureAssembler.rightControllerTransform
                               : featureAssembler.leftControllerTransform;
            if (ctrl == null) return false;
            return Vector3.Distance(ctrl.position, obj.transform.position) < gripThreshold;
        }

        private bool IsTilted(Transform t, float degreesThreshold) =>
            Vector3.Angle(t.up, Vector3.up) > degreesThreshold;

        private float GetRightGrip() =>
            featureAssembler != null ? featureAssembler.LastRightGrip : 0f;

        private float GetLeftGrip() =>
            featureAssembler != null ? featureAssembler.LastLeftGrip : 0f;

        private string GetInstruction(TaskState state) => state switch
        {
            TaskState.PickBottle  => "Pick up the mustard bottle with your right hand.",
            TaskState.PourBottle  => "Tilt the bottle to pour into the mug.",
            TaskState.PlaceBottle => "Place the bottle back on the table.",
            TaskState.PickCup     => "Pick up the mug.",
            TaskState.Done        => "Task complete! Thank you.",
            _                     => ""
        };
    }
}
