using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using AuraXR.Demo;

namespace AuraXR.Demo.Editor
{
    public static class AuraXRDemoSceneBuilder
    {
        [MenuItem("AuraXR/Build Model Demo Scene")]
        public static void BuildScene()
        {
            var scene = EditorSceneManager.NewScene(NewSceneSetup.DefaultGameObjects, NewSceneMode.Single);
            scene.name = "AuraXR_ModelDemo";

            GameObject root = new GameObject("AuraXR_ModelDemo");
            GameObject table = GameObject.CreatePrimitive(PrimitiveType.Cube);
            table.name = "DemoTable";
            table.transform.SetParent(root.transform);
            table.transform.position = new Vector3(0f, -0.05f, 0.55f);
            table.transform.localScale = new Vector3(0.8f, 0.05f, 0.6f);

            GameObject objectRoot = new GameObject("DemoObjectRoot");
            objectRoot.transform.SetParent(root.transform);

            AuraXRDemoObject mug = CreateDemoObject("ObjectSlot_Mug", "mug_white", PrimitiveType.Cylinder, new Vector3(-0.22f, 0.06f, 0.55f), objectRoot.transform);
            CreateDemoObject("ObjectSlot_Box", "can_parmesan", PrimitiveType.Cube, new Vector3(0f, 0.06f, 0.55f), objectRoot.transform);
            CreateDemoObject("ObjectSlot_Tool", "spatula_red", PrimitiveType.Capsule, new Vector3(0.22f, 0.06f, 0.55f), objectRoot.transform);

            GameObject handRoot = new GameObject("HandRoot");
            handRoot.transform.SetParent(root.transform);
            GameObject wrist = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            wrist.name = "TrackedWristAnchor";
            wrist.transform.SetParent(handRoot.transform);
            wrist.transform.position = new Vector3(0f, 0.18f, 0.25f);
            wrist.transform.localScale = Vector3.one * 0.035f;

            GameObject predictedRig = new GameObject("PredictedHandRig");
            predictedRig.transform.SetParent(handRoot.transform);

            GameObject runtime = new GameObject("AuraXRModelRuntime");
            runtime.transform.SetParent(root.transform);
            var assembler = runtime.AddComponent<AuraXRFeatureAssembler>();
            var model = runtime.AddComponent<AuraXRModelRuntime>();
            var retargeter = runtime.AddComponent<AuraXRHandRetargeter>();
            var blend = runtime.AddComponent<AuraXRBlendController>();
            var hud = runtime.AddComponent<AuraXRDemoHUD>();
            var logger = runtime.AddComponent<AuraXRDemoLogger>();

            assembler.wrist = wrist.transform;
            assembler.activeObject = mug;
            model.featureAssembler = assembler;
            retargeter.modelRuntime = model;
            blend.featureAssembler = assembler;
            blend.handRetargeter = retargeter;
            hud.featureAssembler = assembler;
            hud.modelRuntime = model;
            hud.blendController = blend;
            logger.featureAssembler = assembler;
            logger.modelRuntime = model;
            logger.blendController = blend;

            Selection.activeGameObject = root;
            EditorSceneManager.MarkSceneDirty(scene);
            Debug.Log("[AuraXR] Built AuraXR_ModelDemo hierarchy. Assign model_stats.json, object point cloud TextAssets, ONNX ModelAsset, and MANO bones in Inspector.");
        }

        private static AuraXRDemoObject CreateDemoObject(string name, string objectId, PrimitiveType primitive, Vector3 position, Transform parent)
        {
            GameObject go = GameObject.CreatePrimitive(primitive);
            go.name = name;
            go.transform.SetParent(parent);
            go.transform.position = position;
            go.transform.localScale = new Vector3(0.08f, 0.08f, 0.08f);
            var demo = go.AddComponent<AuraXRDemoObject>();
            demo.objectId = objectId;
            demo.bboxDiagonalM = 0.1f;
            demo.pointCloudRelativePath = $"AuraXR/objects/{objectId}_pts.bytes";
            return demo;
        }
    }
}
