using UnityEngine;

namespace AuraXR
{
    /// <summary>
    /// Procedurally builds a clean AuraXR demo environment when the scene starts:
    ///   • A floor + a kitchen-style table the user stands in front of.
    ///   • 5 interactable props placed on the table (HOT3D BOP category IDs that the
    ///     model was trained on — mug_patterned, can_soup, bottle_mustard, dvd_remote,
    ///     holder_black). Each is given a Rigidbody, MeshCollider, and InteractableObject.
    ///   • A "GameManager" GameObject hosting AuraXRFeatureAssembler, ProximityDetector,
    ///     VirtualHandGrab, HandProximityVisibility, AuraXRInferenceManager, and
    ///     AuraXRAutoWire. AutoWire takes over Inspector-reference wiring on Awake().
    ///
    /// This runs at [DefaultExecutionOrder(-300)] — BEFORE AuraXRAutoWire (-200) — so by
    /// the time AutoWire looks for InteractableObject and the manager components, they
    /// already exist.
    ///
    /// Drop this component on ANY empty GameObject in the scene (or let
    /// SpawnIfMissing=true create one for you). It self-destructs after running.
    /// </summary>
    [DefaultExecutionOrder(-300)]
    public class AuraXRSceneSetup : MonoBehaviour
    {
        // Auto-bootstrap: runs on any scene load even without a scene GameObject.
        // Creates a temporary host, adds AuraXRSceneSetup, then it self-destructs after Awake.
        [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterSceneLoad)]
        static void AutoBootstrap()
        {
            if (Object.FindAnyObjectByType<AuraXRSceneSetup>() != null) return;
            if (Object.FindAnyObjectByType<AuraXRInferenceManager>() != null) return;
            var host = new GameObject("AuraXRBootstrap");
            host.AddComponent<AuraXRSceneSetup>();
        }

        [Header("What to build")]
        public bool buildFloor       = true;
        public bool buildTable       = true;
        public bool buildProps       = true;
        public bool buildGameManager = true;

        [Header("Table placement")]
        [Tooltip("Table center in world space — props will sit on top.")]
        public Vector3 tablePosition = new Vector3(0f, 0.75f, 0.6f);
        [Tooltip("(width, thickness, depth) in metres.")]
        public Vector3 tableSize     = new Vector3(1.2f, 0.05f, 0.6f);

        [Header("Materials")]
        public Color floorColor = new Color(0.20f, 0.20f, 0.22f);
        public Color tableColor = new Color(0.55f, 0.40f, 0.25f);

        void Awake()
        {
            var root = new GameObject("AuraXR_Scene");

            if (buildFloor)       BuildFloor(root.transform);
            if (buildTable)       BuildTable(root.transform);
            if (buildProps)       BuildProps(root.transform);
            if (buildGameManager) BuildGameManager();

            Debug.Log("[SceneSetup] Environment built. Self-destroying.");
            Destroy(this);
        }

        // ── Static geometry ─────────────────────────────────────────────────

        void BuildFloor(Transform parent)
        {
            var floor = GameObject.CreatePrimitive(PrimitiveType.Plane);
            floor.name = "Floor";
            floor.transform.SetParent(parent, false);
            floor.transform.localScale = new Vector3(2f, 1f, 2f); // 20m × 20m
            Paint(floor, floorColor);
        }

        void BuildTable(Transform parent)
        {
            var table = new GameObject("Table");
            table.transform.SetParent(parent, false);
            table.transform.position = tablePosition;

            var top = GameObject.CreatePrimitive(PrimitiveType.Cube);
            top.name = "TableTop";
            top.transform.SetParent(table.transform, false);
            top.transform.localScale = tableSize;
            Paint(top, tableColor);

            // Four legs
            float legH = tablePosition.y - tableSize.y * 0.5f;
            float legR = 0.04f;
            Vector3[] legXZ = {
                new Vector3( tableSize.x * 0.45f, -legH * 0.5f - tableSize.y * 0.5f,  tableSize.z * 0.45f),
                new Vector3(-tableSize.x * 0.45f, -legH * 0.5f - tableSize.y * 0.5f,  tableSize.z * 0.45f),
                new Vector3( tableSize.x * 0.45f, -legH * 0.5f - tableSize.y * 0.5f, -tableSize.z * 0.45f),
                new Vector3(-tableSize.x * 0.45f, -legH * 0.5f - tableSize.y * 0.5f, -tableSize.z * 0.45f),
            };
            for (int i = 0; i < 4; i++)
            {
                var leg = GameObject.CreatePrimitive(PrimitiveType.Cube);
                leg.name = $"Leg_{i}";
                leg.transform.SetParent(table.transform, false);
                leg.transform.localPosition = legXZ[i];
                leg.transform.localScale    = new Vector3(legR, legH, legR);
                Paint(leg, tableColor * 0.85f);
            }
        }

        // ── Interactable props ──────────────────────────────────────────────

        // Props to spawn: (bop_id, name, fallback_primitive, fallback_scale, fallback_color)
        // GLB loaded from Resources/Objects/bop{id:D2}_{name}.glb when available.
        private static readonly (int id, string name, PrimitiveType prim, Vector3 scale, Color color)[] PropDefs =
        {
            ( 8, "mug_patterned",  PrimitiveType.Cylinder, new Vector3(0.09f, 0.05f, 0.09f), new Color(0.90f, 0.45f, 0.25f)),
            (10, "can_soup",       PrimitiveType.Cylinder, new Vector3(0.07f, 0.05f, 0.07f), new Color(0.85f, 0.20f, 0.20f)),
            (13, "bottle_mustard", PrimitiveType.Cylinder, new Vector3(0.07f, 0.10f, 0.07f), new Color(0.95f, 0.80f, 0.10f)),
            (33, "dvd_remote",     PrimitiveType.Cube,     new Vector3(0.05f, 0.02f, 0.18f), new Color(0.10f, 0.10f, 0.10f)),
            ( 1, "holder_black",   PrimitiveType.Cube,     new Vector3(0.06f, 0.15f, 0.06f), new Color(0.20f, 0.20f, 0.20f)),
        };

        void BuildProps(Transform parent)
        {
            float yTop     = tablePosition.y + tableSize.y * 0.5f;
            float xSpacing = tableSize.x / 6f;
            float z        = tablePosition.z;

            for (int i = 0; i < PropDefs.Length; i++)
            {
                var (id, pname, prim, scale, color) = PropDefs[i];
                float x = (i - (PropDefs.Length - 1) * 0.5f) * xSpacing;

                GameObject go = SpawnProp(id, pname, prim, scale, color);
                go.transform.SetParent(parent, false);

                // Place on table surface: GLB meshes are centred at origin; estimate halfH from bbox size.
                var bounds = GetMeshBounds(go);
                float halfH = bounds.size.y * 0.5f;
                go.transform.position = new Vector3(tablePosition.x + x, yTop + halfH + 0.001f, z);

                // Convex physics collider + rigidbody
                foreach (var c in go.GetComponentsInChildren<Collider>()) Object.Destroy(c);
                var mc = go.AddComponent<MeshCollider>();
                mc.convex   = true;
                mc.isTrigger = false;

                var rb = go.AddComponent<Rigidbody>();
                rb.useGravity     = true;
                rb.mass           = 0.3f;
                rb.linearDamping  = 0.5f;
                rb.angularDamping = 1.0f;

                var io = go.AddComponent<InteractableObject>();
                io.categoryId   = id;
                io.categoryName = pname;
            }
        }

        /// Try to instantiate a HOT3D GLB from Resources; fall back to Unity primitive.
        private static GameObject SpawnProp(int id, string pname,
            PrimitiveType fallbackPrim, Vector3 fallbackScale, Color fallbackColor)
        {
            string resPath = $"Objects/bop{id:D2}_{pname}";
            var prefab = Resources.Load<GameObject>(resPath);
            if (prefab != null)
            {
                var go = Object.Instantiate(prefab);
                go.name = pname;
                return go;
            }

            // Fallback: coloured primitive
            var pgo = GameObject.CreatePrimitive(fallbackPrim);
            pgo.name = pname;
            pgo.transform.localScale = fallbackScale;
            Paint(pgo, fallbackColor);
            return pgo;
        }

        private static Bounds GetMeshBounds(GameObject go)
        {
            var renderers = go.GetComponentsInChildren<Renderer>();
            if (renderers.Length == 0) return new Bounds(Vector3.zero, Vector3.one * 0.1f);
            var b = renderers[0].bounds;
            for (int i = 1; i < renderers.Length; i++) b.Encapsulate(renderers[i].bounds);
            return b;
        }

        // ── GameManager ─────────────────────────────────────────────────────

        void BuildGameManager()
        {
            // If a GameManager already exists, just ensure components are present.
            var existing = GameObject.Find("GameManager");
            var gm = existing != null ? existing : new GameObject("GameManager");

            Ensure<AuraXRFeatureAssembler>(gm);
            Ensure<ProximityDetector>(gm);
            Ensure<VirtualHandGrab>(gm);
            Ensure<HandProximityVisibility>(gm);
            Ensure<AuraXRInferenceManager>(gm);
            Ensure<AuraXRLogger>(gm);
            Ensure<SessionDataLogger>(gm);
            Ensure<ObjectSDFDatabase>(gm);
            Ensure<SDFGridDatabase>(gm);
            // AutoWire runs at -200, just after this script (-300). It will fill in refs.
            Ensure<AuraXRAutoWire>(gm);
        }

        // ── Helpers ─────────────────────────────────────────────────────────

        static void Paint(GameObject go, Color c)
        {
            var r = go.GetComponent<Renderer>();
            if (r == null) return;
            // URP Lit if available, else Standard.
            var shader = Shader.Find("Universal Render Pipeline/Lit");
            if (shader == null) shader = Shader.Find("Standard");
            var mat = new Material(shader);
            if (mat.HasProperty("_BaseColor")) mat.SetColor("_BaseColor", c);
            mat.color = c;
            r.sharedMaterial = mat;
        }

        static T Ensure<T>(GameObject go) where T : Component
        {
            var c = go.GetComponent<T>();
            if (c == null) c = go.AddComponent<T>();
            return c;
        }
    }
}
