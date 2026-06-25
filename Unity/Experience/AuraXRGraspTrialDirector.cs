using TMPro;
using UnityEngine;

namespace AuraXR
{
    /// <summary>
    /// Small gamified trial loop for the controller-to-hand vertical slice.
    /// The user is asked to approach one target object at a time and hold the
    /// controller/virtual wrist near it while AuraXR proximity and model features run.
    /// </summary>
    [DefaultExecutionOrder(300)]
    public class AuraXRGraspTrialDirector : MonoBehaviour
    {
        [Header("References")]
        public AuraXRFeatureAssembler featureAssembler;
        public AuraXRInferenceManager inferenceManager;
        public TMP_Text instructionText;
        public TMP_Text statusText;
        public TMP_Text feedbackText;
        public TMP_Text activeTargetText;
        public Renderer progressBarFill;
        public Transform[] targetObjects;
        public Renderer[] targetMarkers;
        public Renderer[] targetBeacons;
        public TMP_Text[] targetLabels;

        [Header("Trial Rules")]
        public float successDistance = 0.16f;
        public float holdSeconds = 0.6f;
        public float retargetDelay = 0.8f;
        public float roundSeconds = 75f;
        public int pointsPerTarget = 100;
        public int streakBonus = 25;
        public bool requireNearestObjectMatch = true;
        public bool preferRightHand = true;

        private int _targetIndex;
        private float _holdTimer;
        private float _nextTargetAt;
        private float _roundStartedAt;
        private float _lastSuccessAt = -10f;
        private int _score;
        private int _streak;

        void Awake()
        {
            if (featureAssembler == null) featureAssembler = FindAnyObjectByType<AuraXRFeatureAssembler>();
            if (inferenceManager == null) inferenceManager = FindAnyObjectByType<AuraXRInferenceManager>();
        }

        void Start()
        {
            _roundStartedAt = Time.time;
            SelectTarget(0);
            UpdateHud(0f, Mathf.Infinity, "-", null, 0, false);
        }

        void Update()
        {
            if (featureAssembler == null || targetObjects == null || targetObjects.Length == 0) return;

            UpdateTargetVisuals();

            if (Time.time < _nextTargetAt)
            {
                UpdateHud(0f, Mathf.Infinity, "-", null, 0, false);
                return;
            }

            Transform target = targetObjects[Mathf.Clamp(_targetIndex, 0, targetObjects.Length - 1)];
            if (target == null) return;

            Transform probe = SelectProbe(target, out string side, out float distance);
            Transform nearest = side == "L" ? featureAssembler.nearestObjectLeft : featureAssembler.nearestObjectRight;
            int categoryId = side == "L" ? featureAssembler.nearestObjectCategoryLeft : featureAssembler.nearestObjectCategoryRight;
            bool match = !requireNearestObjectMatch || nearest == target;
            bool onTarget = probe != null && distance <= successDistance && match;

            _holdTimer = onTarget
                ? _holdTimer + Time.deltaTime
                : Mathf.MoveTowards(_holdTimer, 0f, Time.deltaTime * 2f);

            if (_holdTimer >= holdSeconds)
            {
                _streak++;
                _score += pointsPerTarget + Mathf.Max(0, _streak - 1) * streakBonus;
                _holdTimer = 0f;
                _lastSuccessAt = Time.time;
                _nextTargetAt = Time.time + retargetDelay;
                SelectTarget((_targetIndex + 1) % targetObjects.Length);
            }

            UpdateHud(Mathf.Clamp01(_holdTimer / holdSeconds), distance, side, nearest, categoryId, match);
        }

        private Transform SelectProbe(Transform target, out string side, out float distance)
        {
            Transform left = featureAssembler.leftControllerTransform;
            Transform right = featureAssembler.rightControllerTransform;

            float leftDistance = left != null ? DistanceToTarget(left.position, target) : Mathf.Infinity;
            float rightDistance = right != null ? DistanceToTarget(right.position, target) : Mathf.Infinity;

            if (preferRightHand && right != null && (rightDistance < successDistance * 1.5f || left == null))
            {
                side = "R";
                distance = rightDistance;
                return right;
            }

            if (leftDistance < rightDistance)
            {
                side = "L";
                distance = leftDistance;
                return left;
            }

            side = "R";
            distance = rightDistance;
            return right;
        }

        private static float DistanceToTarget(Vector3 point, Transform target)
        {
            Collider col = target.GetComponentInChildren<Collider>();
            return col != null
                ? Vector3.Distance(point, col.ClosestPoint(point))
                : Vector3.Distance(point, target.position);
        }

        private void SelectTarget(int index)
        {
            _targetIndex = index;
            for (int i = 0; targetMarkers != null && i < targetMarkers.Length; i++)
            {
                Renderer marker = targetMarkers[i];
                if (marker == null) continue;

                bool active = i == _targetIndex;
                Color color = active
                    ? new Color(0.05f, 0.85f, 1f, 0.72f)
                    : new Color(0.18f, 0.22f, 0.24f, 0.22f);
                SetRendererColor(marker, color);
            }

            for (int i = 0; targetLabels != null && i < targetLabels.Length; i++)
            {
                if (targetLabels[i] == null) continue;
                targetLabels[i].color = i == _targetIndex
                    ? new Color(1f, 0.93f, 0.35f, 1f)
                    : new Color(0.72f, 0.74f, 0.76f, 0.72f);
            }
        }

        private void UpdateHud(float progress, float distance, string side, Transform nearest, int categoryId, bool match)
        {
            Transform target = targetObjects != null && targetObjects.Length > 0
                ? targetObjects[Mathf.Clamp(_targetIndex, 0, targetObjects.Length - 1)]
                : null;

            if (instructionText != null)
            {
                int round = targetObjects != null && targetObjects.Length > 0 ? _targetIndex + 1 : 0;
                int total = targetObjects != null ? targetObjects.Length : 0;
                instructionText.text = "GOREV " + round + "/" + total
                                     + "\n1  ISIKLI HEDEFE YAKLAS"
                                     + "\n2  EL GORUNUNCE HALKANIN ICINDE TUT"
                                     + "\n3  DOLANA KADAR BEKLE";
            }

            if (statusText != null)
            {
                string d = float.IsInfinity(distance) ? "--" : distance.ToString("0.00") + "m";
                string nearestName = nearest != null ? nearest.name : "NONE";
                float remaining = Mathf.Max(0f, roundSeconds - (Time.time - _roundStartedAt));
                string modelMode = inferenceManager != null && inferenceManager.debugStaticNeutralPose ? "Static"
                    : inferenceManager != null && inferenceManager.debugBypassModel ? "Bypass"
                    : "Model/Runtime";

                statusText.text = "SKOR " + _score
                                + "\nSURE " + Mathf.CeilToInt(remaining) + "s"
                                + "\nTUTUS " + Mathf.RoundToInt(progress * 100f) + "%"
                                + "\nHEDEF " + GripName(categoryId)
                                + "\n" + (match ? "DOGRU OBJE" : "HEDEFI ARA");

                if (Time.frameCount % 120 == 0 && !string.IsNullOrEmpty(modelMode) && nearestName.Length > 0)
                {
                    // Keep the compact runtime diagnostics available without making the HUD unreadable.
                    Debug.Log("[AuraXRTrial] mode=" + modelMode + " hand=" + side + " dist=" + d + " nearest=" + nearestName);
                }
            }

            if (feedbackText != null)
            {
                if (Time.time - _lastSuccessAt < 0.7f)
                    feedbackText.text = "+" + (pointsPerTarget + Mathf.Max(0, _streak - 1) * streakBonus) + "  SIRADAKI HEDEF";
                else if (progress > 0.05f)
                    feedbackText.text = "TUT  " + Mathf.RoundToInt(progress * 100f) + "%";
                else
                    feedbackText.text = "ISIKLI HEDEFE GIT";
            }

            if (activeTargetText != null)
            {
                activeTargetText.text = target != null
                    ? "BURAYA TUT\n" + CleanTargetName(target.name)
                    : "BURAYA TUT";
            }

            if (progressBarFill != null)
            {
                Vector3 scale = progressBarFill.transform.localScale;
                progressBarFill.transform.localScale = new Vector3(Mathf.Lerp(0.02f, 0.70f, progress), scale.y, scale.z);
                Color color = Color.Lerp(new Color(0.95f, 0.20f, 0.12f, 1f), new Color(0.20f, 0.95f, 0.32f, 1f), progress);
                SetRendererColor(progressBarFill, color);
            }
        }

        private void UpdateTargetVisuals()
        {
            for (int i = 0; targetMarkers != null && i < targetMarkers.Length; i++)
            {
                Renderer marker = targetMarkers[i];
                if (marker == null) continue;

                bool active = i == _targetIndex;
                float pulse = active ? 1f + Mathf.Sin(Time.time * 7f) * 0.10f : 1f;
                marker.transform.localScale = new Vector3(0.27f * pulse, 0.006f, 0.27f * pulse);

                Color color = active
                    ? Color.Lerp(new Color(0.05f, 0.85f, 1f, 0.55f), new Color(1f, 0.93f, 0.20f, 0.78f), Mathf.PingPong(Time.time * 2.4f, 1f))
                    : new Color(0.18f, 0.22f, 0.24f, 0.22f);
                SetRendererColor(marker, color);
            }

            for (int i = 0; targetBeacons != null && i < targetBeacons.Length; i++)
            {
                Renderer beacon = targetBeacons[i];
                if (beacon == null) continue;

                bool active = i == _targetIndex;
                beacon.enabled = active;
                if (!active) continue;

                float alpha = 0.22f + Mathf.PingPong(Time.time * 1.8f, 0.18f);
                SetRendererColor(beacon, new Color(0.10f, 0.85f, 1f, alpha));
            }

            if (activeTargetText != null && targetObjects != null && targetObjects.Length > 0)
            {
                Transform target = targetObjects[Mathf.Clamp(_targetIndex, 0, targetObjects.Length - 1)];
                if (target != null)
                {
                    activeTargetText.transform.position = target.position + new Vector3(0f, 0.46f, 0f);
                    activeTargetText.transform.rotation = Quaternion.Euler(24f, 180f, 0f);
                }
            }
        }

        private static string CleanTargetName(string raw)
        {
            if (raw.Contains("mug")) return "KULPLU KUPA";
            if (raw.Contains("bottle")) return "SISE";
            if (raw.Contains("carton")) return "KUTU";
            return raw;
        }

        private static string GripName(int categoryId)
        {
            switch (categoryId)
            {
                case 1:
                case 23:
                case 25:
                case 27:
                case 30:
                case 31:
                    return "Pinch";
                case 3:
                case 20:
                case 24:
                case 28:
                case 29:
                case 33:
                    return "Wide";
                case 4:
                case 5:
                case 6:
                case 32:
                    return "Precision";
                case 0:
                    return "Unknown";
                default:
                    return "Power";
            }
        }

        private static void SetRendererColor(Renderer renderer, Color color)
        {
            Material mat = renderer.material;
            if (mat.HasProperty("_BaseColor")) mat.SetColor("_BaseColor", color);
            if (mat.HasProperty("_Color")) mat.SetColor("_Color", color);
            mat.color = color;
        }
    }
}
