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

            Selection.activeGameObject = go;
            EditorUtility.SetDirty(go);
            Debug.Log("[C2HPlaybackEditor] C2H Playback Driver ready in scene.");
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
