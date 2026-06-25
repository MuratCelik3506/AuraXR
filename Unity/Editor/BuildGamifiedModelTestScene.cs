using System.IO;
using TMPro;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.SceneManagement;

namespace AuraXR.EditorTools
{
    /// <summary>
    /// Builds a separate gamified scene for ROADMAP phase 0.5/4 testing.
    /// Menu: AuraXR/Build Gamified Model Test Scene.
    /// </summary>
    public static class BuildGamifiedModelTestScene
    {
        public const string ScenePath = "Assets/Scenes/AuraXR_GamifiedModelTest.unity";

        [MenuItem("AuraXR/Build Gamified Model Test Scene", priority = 12)]
        public static void Build()
        {
            Scene scene = EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);

            BuildLighting();
            BuildFloor();
            BuildTable();
            Transform[] targets = BuildTargets();
            Renderer[] markers = BuildTargetVisuals(targets, out Renderer[] beacons, out TMP_Text[] labels);
            BuildRigFallbackIfNeeded();
            InstantiateHandRig("LeftHandRig", "OVRCustomHandPrefab_L");
            InstantiateHandRig("RightHandRig", "OVRCustomHandPrefab_R");
            BuildGameManager(targets, markers, beacons, labels);

            Directory.CreateDirectory(Path.GetDirectoryName(ScenePath));
            EditorSceneManager.SaveScene(scene, ScenePath);
            AssetDatabase.SaveAssets();
            AssetDatabase.Refresh();
            EditorSceneManager.OpenScene(ScenePath, OpenSceneMode.Single);

            Debug.Log("[AuraXR] Built gamified model test scene: " + ScenePath);
        }

        private static void BuildLighting()
        {
            GameObject sunGo = new GameObject("SunLight");
            Light sun = sunGo.AddComponent<Light>();
            sun.type = LightType.Directional;
            sun.intensity = 1.15f;
            sun.color = new Color(1f, 0.96f, 0.9f);
            sunGo.transform.rotation = Quaternion.Euler(55f, -35f, 0f);
        }

        private static void BuildFloor()
        {
            GameObject floor = GameObject.CreatePrimitive(PrimitiveType.Plane);
            floor.name = "Floor";
            floor.transform.localScale = new Vector3(2f, 1f, 2f);
            Paint(floor, new Color(0.26f, 0.27f, 0.29f));
        }

        private static void BuildTable()
        {
            GameObject top = GameObject.CreatePrimitive(PrimitiveType.Cube);
            top.name = "ModelTest_Table";
            top.transform.position = new Vector3(0f, 0.73f, 0.65f);
            top.transform.localScale = new Vector3(1.35f, 0.06f, 0.72f);
            Paint(top, new Color(0.42f, 0.31f, 0.22f));

            Vector3[] legs =
            {
                new Vector3(-0.58f, 0.36f, 0.32f),
                new Vector3( 0.58f, 0.36f, 0.32f),
                new Vector3(-0.58f, 0.36f, 0.98f),
                new Vector3( 0.58f, 0.36f, 0.98f)
            };

            foreach (Vector3 pos in legs)
            {
                GameObject leg = GameObject.CreatePrimitive(PrimitiveType.Cube);
                leg.name = "TableLeg";
                leg.transform.position = pos;
                leg.transform.localScale = new Vector3(0.055f, 0.72f, 0.055f);
                Paint(leg, new Color(0.34f, 0.25f, 0.18f));
            }
        }

        private static Transform[] BuildTargets()
        {
            return new[]
            {
                BuildTarget("Target_01_mug_white_hook", 9, PrimitiveType.Cylinder, new Vector3(-0.36f, 0.82f, 0.63f), new Vector3(0.085f, 0.055f, 0.085f), new Color(0.90f, 0.92f, 0.88f)),
                BuildTarget("Target_02_bottle_mustard_power", 13, PrimitiveType.Cylinder, new Vector3(0.00f, 0.89f, 0.65f), new Vector3(0.070f, 0.130f, 0.070f), new Color(0.95f, 0.78f, 0.12f)),
                BuildTarget("Target_03_carton_milk_wide", 17, PrimitiveType.Cube, new Vector3(0.36f, 0.86f, 0.65f), new Vector3(0.110f, 0.170f, 0.090f), new Color(0.35f, 0.65f, 0.95f))
            };
        }

        private static Transform BuildTarget(string objectName, int categoryId, PrimitiveType primitive, Vector3 position, Vector3 scale, Color color)
        {
            GameObject go = TryInstantiateObjectPrefab(objectName, categoryId);
            if (go == null)
            {
                go = GameObject.CreatePrimitive(primitive);
                go.name = objectName;
                go.transform.localScale = scale;
                Paint(go, color);
            }

            go.transform.position = position;
            go.transform.rotation = Quaternion.identity;

            Collider existing = go.GetComponentInChildren<Collider>();
            if (existing == null)
            {
                MeshFilter mf = go.GetComponentInChildren<MeshFilter>();
                if (mf != null)
                {
                    MeshCollider mc = mf.gameObject.AddComponent<MeshCollider>();
                    mc.convex = true;
                }
                else
                {
                    go.AddComponent<BoxCollider>();
                }
            }

            Rigidbody rb = go.GetComponent<Rigidbody>();
            if (rb == null) rb = go.AddComponent<Rigidbody>();
            rb.mass = 0.25f;
            rb.linearDamping = 0.6f;
            rb.angularDamping = 1.0f;

            InteractableObject io = go.GetComponent<InteractableObject>();
            if (io == null) io = go.AddComponent<InteractableObject>();
            io.categoryId = categoryId;
            io.categoryName = objectName;
            return go.transform;
        }

        private static Renderer[] BuildTargetVisuals(Transform[] targets, out Renderer[] beacons, out TMP_Text[] labels)
        {
            Renderer[] markers = new Renderer[targets.Length];
            beacons = new Renderer[targets.Length];
            labels = new TMP_Text[targets.Length];
            for (int i = 0; i < targets.Length; i++)
            {
                GameObject marker = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
                marker.name = targets[i].name + "_Marker";
                marker.transform.position = new Vector3(targets[i].position.x, 0.766f, targets[i].position.z);
                marker.transform.localScale = new Vector3(0.27f, 0.006f, 0.27f);
                Object.DestroyImmediate(marker.GetComponent<Collider>());

                Renderer renderer = marker.GetComponent<Renderer>();
                renderer.sharedMaterial = TransparentMaterial(new Color(0.05f, 0.85f, 1f, 0.22f));
                markers[i] = renderer;

                GameObject beacon = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
                beacon.name = targets[i].name + "_Beacon";
                beacon.transform.position = new Vector3(targets[i].position.x, 1.18f, targets[i].position.z);
                beacon.transform.localScale = new Vector3(0.070f, 0.50f, 0.070f);
                Object.DestroyImmediate(beacon.GetComponent<Collider>());
                Renderer beaconRenderer = beacon.GetComponent<Renderer>();
                beaconRenderer.sharedMaterial = TransparentMaterial(new Color(0.10f, 0.85f, 1f, 0.20f));
                beacons[i] = beaconRenderer;

                labels[i] = CreateFloatingLabel(
                    "Label_" + targets[i].name,
                    CleanTargetName(targets[i].name),
                    targets[i].position + new Vector3(0f, 0.28f, 0f));
            }
            return markers;
        }

        private static void BuildRigFallbackIfNeeded()
        {
            if (TryInstantiatePrefab(new[] { "OVRCameraRig", "OVRCameraRigInteraction", "[BuildingBlock] Camera Rig" }) != null)
                return;

            GameObject rig = new GameObject("OVRCameraRig_Fallback");
            Transform tracking = new GameObject("TrackingSpace").transform;
            tracking.SetParent(rig.transform, false);
            tracking.localPosition = new Vector3(0f, 1.55f, 0f);

            GameObject cameraGo = new GameObject("CenterEyeAnchor");
            cameraGo.transform.SetParent(tracking, false);
            Camera camera = cameraGo.AddComponent<Camera>();
            camera.tag = "MainCamera";
            camera.nearClipPlane = 0.03f;
            cameraGo.AddComponent<AudioListener>();

            Transform left = new GameObject("LeftHandAnchor").transform;
            left.SetParent(tracking, false);
            left.localPosition = new Vector3(-0.22f, -0.22f, 0.35f);

            Transform right = new GameObject("RightHandAnchor").transform;
            right.SetParent(tracking, false);
            right.localPosition = new Vector3(0.22f, -0.22f, 0.35f);
        }

        private static void BuildGameManager(Transform[] targets, Renderer[] markers, Renderer[] beacons, TMP_Text[] labels)
        {
            GameObject gm = new GameObject("GameManager");
            AuraXRFeatureAssembler assembler = gm.AddComponent<AuraXRFeatureAssembler>();
            ProximityDetector detector = gm.AddComponent<ProximityDetector>();
            detector.searchRadius = 1.0f;

            VirtualHandGrab grab = gm.AddComponent<VirtualHandGrab>();
            grab.grabRadius = 0.22f;
            grab.proximityAutoClose = true;

            HandProximityVisibility visibility = gm.AddComponent<HandProximityVisibility>();
            visibility.showDistance = 0.42f;
            visibility.hideDistance = 0.62f;

            AuraXRInferenceManager inference = gm.AddComponent<AuraXRInferenceManager>();
            inference.lockWristToController = true;
            inference.lockWristRotationToController = true;
            inference.inferenceEveryNFrames = 2;
            inference.useLSTMModel = true;

            gm.AddComponent<ObjectAwareHandGuide>();
            gm.AddComponent<ControllerVisualFader>();
            gm.AddComponent<AuraXRLogger>();
            gm.AddComponent<SessionDataLogger>();
            gm.AddComponent<ObjectSDFDatabase>();
            gm.AddComponent<SDFGridDatabase>();
            gm.AddComponent<AuraXRAutoWire>();

            GameObject hud = BuildHud(out TMP_Text instruction, out TMP_Text status, out TMP_Text feedback, out Renderer progressFill);
            TMP_Text activeTargetText = CreateActiveTargetText();
            AuraXRGraspTrialDirector director = gm.AddComponent<AuraXRGraspTrialDirector>();
            director.featureAssembler = assembler;
            director.inferenceManager = inference;
            director.instructionText = instruction;
            director.statusText = status;
            director.feedbackText = feedback;
            director.activeTargetText = activeTargetText;
            director.progressBarFill = progressFill;
            director.targetObjects = targets;
            director.targetMarkers = markers;
            director.targetBeacons = beacons;
            director.targetLabels = labels;
            director.preferRightHand = true;

            hud.transform.SetParent(gm.transform, true);
        }

        private static GameObject BuildHud(out TMP_Text instruction, out TMP_Text status, out TMP_Text feedback, out Renderer progressFill)
        {
            GameObject hud = new GameObject("WorldHUD");
            hud.transform.position = new Vector3(0f, 1.62f, 1.08f);
            hud.transform.rotation = Quaternion.Euler(14f, 180f, 0f);

            GameObject panel = GameObject.CreatePrimitive(PrimitiveType.Cube);
            panel.name = "HUD_Backplate";
            panel.transform.SetParent(hud.transform, false);
            panel.transform.localPosition = new Vector3(0f, 0.02f, 0.03f);
            panel.transform.localScale = new Vector3(1.38f, 0.72f, 0.018f);
            Object.DestroyImmediate(panel.GetComponent<Collider>());
            Paint(panel, new Color(0.04f, 0.05f, 0.06f, 0.92f));

            instruction = CreateText("InstructionText", hud.transform, new Vector3(-0.50f, 0.18f, -0.01f), 0.115f, TextAlignmentOptions.Left);
            instruction.rectTransform.sizeDelta = new Vector2(0.86f, 0.46f);

            status = CreateText("StatusText", hud.transform, new Vector3(0.41f, 0.16f, -0.01f), 0.080f, TextAlignmentOptions.Left);
            status.rectTransform.sizeDelta = new Vector2(0.42f, 0.42f);

            feedback = CreateText("FeedbackText", hud.transform, new Vector3(0f, -0.22f, -0.01f), 0.120f, TextAlignmentOptions.Center);
            feedback.rectTransform.sizeDelta = new Vector2(1.20f, 0.18f);
            feedback.color = new Color(1f, 0.93f, 0.35f);

            GameObject progressBack = GameObject.CreatePrimitive(PrimitiveType.Cube);
            progressBack.name = "HoldProgress_Back";
            progressBack.transform.SetParent(hud.transform, false);
            progressBack.transform.localPosition = new Vector3(0f, -0.34f, -0.01f);
            progressBack.transform.localScale = new Vector3(0.76f, 0.045f, 0.018f);
            Object.DestroyImmediate(progressBack.GetComponent<Collider>());
            Paint(progressBack, new Color(0.18f, 0.20f, 0.22f));

            GameObject progress = GameObject.CreatePrimitive(PrimitiveType.Cube);
            progress.name = "HoldProgress_Fill";
            progress.transform.SetParent(hud.transform, false);
            progress.transform.localPosition = new Vector3(0f, -0.34f, -0.028f);
            progress.transform.localScale = new Vector3(0.02f, 0.050f, 0.020f);
            Object.DestroyImmediate(progress.GetComponent<Collider>());
            Paint(progress, new Color(0.95f, 0.20f, 0.12f));
            progressFill = progress.GetComponent<Renderer>();
            return hud;
        }

        private static TMP_Text CreateText(string name, Transform parent, Vector3 localPosition, float fontSize, TextAlignmentOptions alignment)
        {
            GameObject go = new GameObject(name);
            go.transform.SetParent(parent, false);
            go.transform.localPosition = localPosition;
            TMP_Text text = go.AddComponent<TextMeshPro>();
            text.fontSize = fontSize;
            text.alignment = alignment;
            text.rectTransform.sizeDelta = new Vector2(0.56f, 0.30f);
            text.color = Color.white;
            return text;
        }

        private static TMP_Text CreateFloatingLabel(string name, string label, Vector3 position)
        {
            GameObject go = new GameObject(name);
            go.transform.position = position;
            go.transform.rotation = Quaternion.Euler(30f, 180f, 0f);
            TMP_Text text = go.AddComponent<TextMeshPro>();
            text.text = label;
            text.fontSize = 0.105f;
            text.alignment = TextAlignmentOptions.Center;
            text.rectTransform.sizeDelta = new Vector2(0.70f, 0.18f);
            text.color = new Color(0.72f, 0.74f, 0.76f, 0.72f);
            return text;
        }

        private static TMP_Text CreateActiveTargetText()
        {
            GameObject go = new GameObject("ActiveTarget_Callout");
            go.transform.position = new Vector3(0f, 1.25f, 0.65f);
            go.transform.rotation = Quaternion.Euler(24f, 180f, 0f);
            TMP_Text text = go.AddComponent<TextMeshPro>();
            text.text = "BURAYA TUT";
            text.fontSize = 0.135f;
            text.alignment = TextAlignmentOptions.Center;
            text.rectTransform.sizeDelta = new Vector2(0.90f, 0.28f);
            text.color = new Color(1f, 0.93f, 0.20f, 1f);
            return text;
        }

        private static string CleanTargetName(string raw)
        {
            if (raw.Contains("mug")) return "HOOK / MUG";
            if (raw.Contains("bottle")) return "POWER / BOTTLE";
            if (raw.Contains("carton")) return "WIDE / CARTON";
            return raw;
        }

        private static GameObject TryInstantiateObjectPrefab(string objectName, int categoryId)
        {
            string[] queries =
            {
                objectName.Replace("Target_01_", "").Replace("Target_02_", "").Replace("Target_03_", "").Split('_')[0],
                "bop" + categoryId.ToString("D2"),
                objectName
            };

            foreach (string query in queries)
            {
                string[] guids = AssetDatabase.FindAssets(query + " t:Prefab");
                foreach (string guid in guids)
                {
                    GameObject prefab = AssetDatabase.LoadAssetAtPath<GameObject>(AssetDatabase.GUIDToAssetPath(guid));
                    if (prefab == null) continue;
                    GameObject instance = (GameObject)PrefabUtility.InstantiatePrefab(prefab);
                    instance.name = objectName;
                    return instance;
                }
            }
            return null;
        }

        private static GameObject InstantiateHandRig(string exactName, string fallbackName)
        {
            GameObject go = TryInstantiatePrefab(new[] { exactName, fallbackName });
            if (go != null) go.name = exactName;
            return go;
        }

        private static GameObject TryInstantiatePrefab(string[] nameCandidates)
        {
            foreach (string name in nameCandidates)
            {
                string[] guids = AssetDatabase.FindAssets(name + " t:Prefab");
                foreach (string guid in guids)
                {
                    string path = AssetDatabase.GUIDToAssetPath(guid);
                    if (Path.GetFileNameWithoutExtension(path) != name) continue;
                    GameObject prefab = AssetDatabase.LoadAssetAtPath<GameObject>(path);
                    if (prefab == null) continue;
                    return (GameObject)PrefabUtility.InstantiatePrefab(prefab);
                }
            }
            return null;
        }

        private static void Paint(GameObject go, Color color)
        {
            Renderer renderer = go.GetComponent<Renderer>();
            if (renderer == null) return;
            Material mat = new Material(Shader.Find("Universal Render Pipeline/Lit") ?? Shader.Find("Standard"));
            mat.color = color;
            if (mat.HasProperty("_BaseColor")) mat.SetColor("_BaseColor", color);
            renderer.sharedMaterial = mat;
        }

        private static Material TransparentMaterial(Color color)
        {
            Material mat = new Material(Shader.Find("Universal Render Pipeline/Lit") ?? Shader.Find("Standard"));
            mat.color = color;
            if (mat.HasProperty("_BaseColor")) mat.SetColor("_BaseColor", color);
            if (mat.HasProperty("_Surface")) mat.SetFloat("_Surface", 1f);
            mat.renderQueue = 3000;
            return mat;
        }
    }
}
