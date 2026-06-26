using System.IO;
using AuraXR;
using UnityEditor;
using Unity.InferenceEngine;
using UnityEngine;

namespace AuraXR.Editor
{
    public static class C2HPlaybackEditor
    {
        [MenuItem("AuraXR/Faz 3A/Create C2H Playback Driver")]
        public static void CreatePlaybackDriver()
        {
            var existing = Object.FindAnyObjectByType<C2HPlaybackDriver>();
            GameObject go = existing != null
                ? existing.gameObject
                : new GameObject("C2H Playback Driver");

            var driver = go.GetComponent<C2HPlaybackDriver>();
            if (driver == null)
                driver = go.AddComponent<C2HPlaybackDriver>();

            driver.modelAsset = FindC2HModel();
            driver.inferenceManager = Object.FindAnyObjectByType<AuraXRInferenceManager>();
            driver.wristRoot = FindRightHandRig();
            driver.objectRoot = FindOrCreatePlaybackObject().transform;
            driver.runParityOnStart = true;
            driver.playOnStart = true;
            driver.loop = true;
            driver.playbackFps = 30f;
            driver.driveRightHand = true;
            driver.alignToSceneObjectAnchor = true;
            driver.captureAnchorOnInitialize = false;
            if (driver.objectRoot != null)
            {
                driver.sceneObjectAnchorPosition = driver.objectRoot.position;
                driver.sceneObjectAnchorRotation = driver.objectRoot.rotation;
            }

            Selection.activeGameObject = go;
            EditorUtility.SetDirty(go);
            Debug.Log("[C2HPlaybackEditor] C2H Playback Driver ready in scene.");
        }

        // Must match SCENARIOS in src/export_unity_playback.py.
        private static readonly string[] Scenarios =
        {
            "mug_white", "mouse", "bowl", "can_soup", "mug_patterned",
        };

        private const float ScenarioSpacing = 0.45f;   // metres between adjacent scenario anchors
        private const string ScenarioPrefix = "C2H Scenario";

        [MenuItem("AuraXR/Faz 3A/Create All Playback Scenarios")]
        public static void CreateAllScenarios()
        {
            ModelAsset model = FindC2HModel();
            string sa = Application.streamingAssetsPath;
            int placed = 0;
            int total = 0;

            for (int i = 0; i < Scenarios.Length; i++)
            {
                string obj = Scenarios[i];
                string file = $"c2h_playback_{obj}.json";
                if (!File.Exists(Path.Combine(sa, file)))
                {
                    Debug.LogWarning($"[C2HPlaybackEditor] Skipping '{obj}' — {file} not found in StreamingAssets. " +
                                     "Run: python src/export_unity_playback.py");
                    continue;
                }

                total++;
                // Spread scenarios along X, centred on the table front edge.
                float x = (i - (Scenarios.Length - 1) * 0.5f) * ScenarioSpacing;
                var anchor = new Vector3(x, 1.0f, 0.4f);
                CreateScenarioDriver(obj, file, model, anchor);
                placed++;
            }

            Debug.Log($"[C2HPlaybackEditor] Created {placed}/{total} playback scenarios " +
                      $"(of {Scenarios.Length} listed). Each renders its own predicted+GT hand skeleton.");
        }

        private static void CreateScenarioDriver(string obj, string file, ModelAsset model, Vector3 anchor)
        {
            string goName = $"{ScenarioPrefix} — {obj}";
            GameObject go = GameObject.Find(goName) ?? new GameObject(goName);
            go.transform.SetPositionAndRotation(Vector3.zero, Quaternion.identity);

            var driver = go.GetComponent<C2HPlaybackDriver>() ?? go.AddComponent<C2HPlaybackDriver>();
            driver.modelAsset = model;
            driver.playbackJson = null;
            driver.streamingAssetsFile = file;

            // Standalone skeleton mode — do not drive the shared OVR hand rig.
            driver.inferenceManager = null;
            driver.driveRightHand = false;
            driver.applyWristPose = false;
            driver.wristRoot = null;
            driver.renderSkeleton = true;
            driver.renderGroundTruth = true;

            // Each scenario gets its own object marker at its anchor.
            driver.objectRoot = CreateObjectMarker(go.transform, obj, anchor).transform;
            driver.applyObjectPose = true;

            driver.runParityOnStart = true;
            driver.playOnStart = true;
            driver.loop = true;
            driver.playbackFps = 30f;

            // Anchor the playback world so this scenario sits at its slot in the scene.
            driver.alignToSceneObjectAnchor = true;
            driver.captureAnchorOnInitialize = false;
            driver.sceneObjectAnchorPosition = anchor;
            driver.sceneObjectAnchorRotation = Quaternion.identity;

            AddLabel(go.transform, obj, anchor + Vector3.up * 0.18f);

            EditorUtility.SetDirty(go);
        }

        private static GameObject CreateObjectMarker(Transform parent, string obj, Vector3 anchor)
        {
            var marker = GameObject.CreatePrimitive(PrimitiveType.Cube);
            marker.name = $"Object ({obj})";
            marker.transform.SetParent(parent, worldPositionStays: false);
            marker.transform.position = anchor;
            marker.transform.localScale = new Vector3(0.06f, 0.06f, 0.06f);
            Object.DestroyImmediate(marker.GetComponent<Collider>());

            Shader shader = Shader.Find("Universal Render Pipeline/Lit")
                            ?? Shader.Find("Standard")
                            ?? Shader.Find("Unlit/Color");
            var mat = new Material(shader) { color = new Color(0.7f, 0.7f, 0.72f, 1f) };
            marker.GetComponent<Renderer>().sharedMaterial = mat;
            return marker;
        }

        private static void AddLabel(Transform parent, string text, Vector3 pos)
        {
            var go = new GameObject("Label");
            go.transform.SetParent(parent, worldPositionStays: false);
            go.transform.position = pos;
            var tm = go.AddComponent<TextMesh>();
            tm.text = text;
            tm.characterSize = 0.02f;
            tm.fontSize = 64;
            tm.anchor = TextAnchor.LowerCenter;
            tm.alignment = TextAlignment.Center;
            tm.color = Color.white;
        }

        [MenuItem("AuraXR/Faz 3A/Run C2H Parity Check")]
        public static void RunParityCheck()
        {
            var driver = Object.FindAnyObjectByType<C2HPlaybackDriver>();
            if (driver == null)
            {
                CreatePlaybackDriver();
                driver = Object.FindAnyObjectByType<C2HPlaybackDriver>();
            }

            if (driver == null)
            {
                Debug.LogError("[C2HPlaybackEditor] Could not create/find C2HPlaybackDriver.");
                return;
            }

            driver.modelAsset = driver.modelAsset != null ? driver.modelAsset : FindC2HModel();
            PlaybackParityResult parity = driver.RunParityCheck();
            if (!parity.pass)
                Debug.LogError($"[C2HPlaybackEditor] Parity failed: maxAbsDiff={parity.maxAbsDiff:E6}");
        }

        private static ModelAsset FindC2HModel()
        {
            string[] guids = AssetDatabase.FindAssets("c2h_step t:ModelAsset");
            if (guids.Length == 0)
                return null;

            string path = AssetDatabase.GUIDToAssetPath(guids[0]);
            return AssetDatabase.LoadAssetAtPath<ModelAsset>(path);
        }

        private static Transform FindRightHandRig()
        {
            GameObject rightRig = GameObject.Find("RightHandRig");
            return rightRig != null ? rightRig.transform : null;
        }

        private static GameObject FindOrCreatePlaybackObject()
        {
            GameObject mug = GameObject.Find("Target_01_mug_white_hook");
            if (mug != null)
                return mug;

            GameObject obj = GameObject.Find("C2H Playback Object");
            if (obj != null)
                return obj;

            obj = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
            obj.name = "C2H Playback Object";
            obj.transform.localScale = new Vector3(0.08f, 0.12f, 0.08f);

            Renderer renderer = obj.GetComponent<Renderer>();
            Shader shader = Shader.Find("Universal Render Pipeline/Lit")
                            ?? Shader.Find("Unlit/Color")
                            ?? Shader.Find("Sprites/Default");
            var mat = new Material(shader);
            mat.color = new Color(0.85f, 0.86f, 0.82f, 1f);
            renderer.sharedMaterial = mat;

            Undo.RegisterCreatedObjectUndo(obj, "Create C2H Playback Object");
            return obj;
        }
    }
}
