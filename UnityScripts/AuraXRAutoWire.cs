using UnityEngine;
using TMPro;

namespace AuraXR
{
    /// <summary>
    /// Runs before every other AuraXR script (ExecutionOrder -200) and fills in
    /// every Inspector reference that would otherwise need manual drag-and-drop.
    ///
    /// What it does:
    ///   1. Adds PhysicsObjectSetup to GameManager if missing.
    ///   2. Creates a SnapZone on the table for the Bottle and wires it to ScenarioKitchenTask.
    ///   3. Creates a TaskScoreUI panel on the TaskCanvas and wires it back to the task.
    ///   4. Logs exactly what it wired (or skips if already assigned).
    ///
    /// Add this component to any GameObject in the scene (e.g. GameManager).
    /// It destroys itself after Awake() to leave no runtime overhead.
    /// </summary>
    [DefaultExecutionOrder(-200)]
    public class AuraXRAutoWire : MonoBehaviour
    {
        [Header("Optional Overrides (leave 0/null to use scene defaults)")]
        [Tooltip("Height of the snap zone above the TableTop surface (metres).")]
        public float snapZoneHeightAboveTable = 0.06f;

        [Tooltip("X offset of the bottle snap zone from the table centre.")]
        public float bottleSnapZoneX = -0.25f;

        void Awake()
        {
            var task        = FindAny<ScenarioKitchenTask>();
            var canvas      = FindGoByName("TaskCanvas");
            var tableTop    = FindGoByName("TableTop");
            var gameManager = FindGoByName("GameManager");

            if (task == null)
            {
                Debug.LogError("[AutoWire] ScenarioKitchenTask not found — aborting.");
                return;
            }

            WirePhysicsSetup(gameManager ?? gameObject, task);
            WireSnapZone(task, tableTop);
            WireScoreUI(task, canvas);

            Debug.Log("[AutoWire] Wiring complete. Self-destroying.");
            Destroy(this);   // zero runtime cost after startup
        }

        // ── PhysicsObjectSetup ────────────────────────────────────────────────

        private void WirePhysicsSetup(GameObject host, ScenarioKitchenTask task)
        {
            var existing = FindAny<PhysicsObjectSetup>();
            if (existing != null) { Log("PhysicsObjectSetup already present — skipped."); return; }

            var setup = host.AddComponent<PhysicsObjectSetup>();

            // HOT3D-calibrated masses:
            //   bottle_mustard (ID 13) ~0.5 kg  — sauce bottle, full
            //   mug_patterned  (ID 8)  ~0.35 kg — ceramic mug
            if (task.bottle != null && task.cup != null)
            {
                setup.massOverrides = new[]
                {
                    new PhysicsObjectSetup.MassOverride { obj = task.bottle, mass = 0.5f  },
                    new PhysicsObjectSetup.MassOverride { obj = task.cup,    mass = 0.35f }
                };
            }

            Log("Added PhysicsObjectSetup to " + host.name);
        }

        // ── SnapZone ──────────────────────────────────────────────────────────

        private void WireSnapZone(ScenarioKitchenTask task, GameObject tableTop)
        {
            if (task.bottleSnapZone != null) { Log("bottleSnapZone already assigned — skipped."); return; }
            if (task.bottle == null)         { Debug.LogWarning("[AutoWire] task.bottle is null — cannot create snap zone."); return; }

            // Determine table surface Y
            float surfaceY = DetermineSurfaceY(tableTop, task.bottle.transform.position.y);

            // Create zone GameObject
            var zoneGO = new GameObject("SnapZone_Bottle");

            // Parent under TableTop if available, otherwise scene root
            if (tableTop != null)
                zoneGO.transform.SetParent(tableTop.transform, worldPositionStays: true);

            zoneGO.transform.position = new Vector3(
                task.bottle.transform.position.x + bottleSnapZoneX,
                surfaceY,
                task.bottle.transform.position.z);

            // Box trigger collider
            var col  = zoneGO.AddComponent<BoxCollider>();
            col.size      = new Vector3(0.18f, 0.14f, 0.18f);
            col.isTrigger = true;

            // SnapZone component
            var zone = zoneGO.AddComponent<SnapZone>();
            zone.acceptedObject = task.bottle;
            zone.snapOffset     = new Vector3(0f, snapZoneHeightAboveTable, 0f);

            // Wire back to task
            task.bottleSnapZone = zone;

            Log($"Created SnapZone_Bottle at {zoneGO.transform.position} and wired to ScenarioKitchenTask.");
        }

        private static float DetermineSurfaceY(GameObject tableTop, float fallback)
        {
            if (tableTop == null) return fallback;

            var r = tableTop.GetComponent<Renderer>();
            if (r != null) return r.bounds.max.y;

            var c = tableTop.GetComponent<Collider>();
            if (c != null) return c.bounds.max.y;

            return tableTop.transform.position.y + 0.02f;
        }

        // ── TaskScoreUI ───────────────────────────────────────────────────────

        private void WireScoreUI(ScenarioKitchenTask task, GameObject canvasGO)
        {
            if (task.scoreUI != null) { Log("scoreUI already assigned — skipped."); return; }

            // Try to find an existing TaskScoreUI first
            var existing = FindAny<TaskScoreUI>();
            if (existing != null)
            {
                task.scoreUI     = existing;
                existing.task    = task;
                Log("Found existing TaskScoreUI and wired to task.");
                return;
            }

            if (canvasGO == null)
            {
                Debug.LogWarning("[AutoWire] TaskCanvas not found — TaskScoreUI will not be created.");
                return;
            }

            // Build a small score panel inside the existing TaskCanvas
            var panel = BuildScorePanel(canvasGO);

            var scoreUI  = panel.AddComponent<TaskScoreUI>();
            scoreUI.task = task;
            task.scoreUI = scoreUI;

            Log("Created TaskScoreUI panel inside TaskCanvas and wired to ScenarioKitchenTask.");
        }

        /// <summary>Creates a vertical TMP panel with 4 step labels + timer + stars + result.</summary>
        private static GameObject BuildScorePanel(GameObject canvasGO)
        {
            var panel = new GameObject("ScorePanel");
            panel.transform.SetParent(canvasGO.transform, worldPositionStays: false);

            var rect = panel.AddComponent<RectTransform>();
            rect.anchoredPosition = new Vector2(160f, 0f);   // right side of canvas
            rect.sizeDelta        = new Vector2(300f, 360f);

            // Grab the canvas font size so labels look consistent
            var scoreUI = panel.AddComponent<TaskScoreUI>();
            Destroy(scoreUI);   // we'll re-add after attaching labels below

            string[] names = { "Step1", "Step2", "Step3", "Step4", "Timer", "Stars", "Result" };
            float[]  yPos  = {  130f,    90f,    50f,   10f,  -50f, -100f,  -140f };

            TextMeshProUGUI[] texts = new TextMeshProUGUI[names.Length];
            for (int i = 0; i < names.Length; i++)
                texts[i] = MakeTMP(names[i], panel.transform, yPos[i]);

            var ui = panel.AddComponent<TaskScoreUI>();
            ui.step1Text  = texts[0];
            ui.step2Text  = texts[1];
            ui.step3Text  = texts[2];
            ui.step4Text  = texts[3];
            ui.timerText  = texts[4];
            ui.starsText  = texts[5];
            ui.resultText = texts[6];

            return panel;
        }

        private static TextMeshProUGUI MakeTMP(string name, Transform parent, float yPos)
        {
            var go   = new GameObject(name);
            go.transform.SetParent(parent, worldPositionStays: false);

            var rect = go.AddComponent<RectTransform>();
            rect.anchoredPosition = new Vector2(0f, yPos);
            rect.sizeDelta        = new Vector2(280f, 36f);

            var tmp  = go.AddComponent<TextMeshProUGUI>();
            tmp.fontSize  = 18f;
            tmp.color     = Color.white;
            tmp.alignment = TextAlignmentOptions.Left;
            tmp.text      = name;   // placeholder — will be overwritten at runtime
            return tmp;
        }

        // ── Helpers ───────────────────────────────────────────────────────────

        private static T FindAny<T>() where T : Component =>
            Object.FindAnyObjectByType<T>(FindObjectsInactive.Exclude);

        private static GameObject FindGoByName(string n)
        {
            var go = GameObject.Find(n);
            if (go == null) Debug.LogWarning($"[AutoWire] '{n}' not found in scene.");
            return go;
        }

        private static void Log(string msg) =>
            Debug.Log($"[AutoWire] {msg}");
    }
}
