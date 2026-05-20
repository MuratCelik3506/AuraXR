using UnityEngine;

namespace AuraXR
{
    /// <summary>
    /// Procedurally builds a kitchen-table environment at Awake.
    /// Creates: floor plane, table top + 4 legs, repositions objects onto the surface.
    ///
    /// Logical additions:
    ///   • autoCreateSnapZones — spawns a SnapZone at each object position, and wires
    ///                           acceptedObject so each zone only accepts its own item
    ///   • SnapZones property  — lets ScenarioKitchenTask (or inspector) retrieve the
    ///                           created zones without a separate FindObjectsByType call
    ///   • decorativeObjects   — extra non-interactable props for scene dressing
    ///
    /// Setup:
    ///   1. Add this to an empty "Environment" GameObject.
    ///   2. Drag Bottle and Cup into objectsOnTable[].
    ///   3. If autoCreateSnapZones = true, assign SnapZones[0] to ScenarioKitchenTask.bottleSnapZone.
    /// </summary>
    public class TableEnvironmentBuilder : MonoBehaviour
    {
        [Header("Table Geometry")]
        [Tooltip("Table top: width (X), thickness (Y), depth (Z)")]
        public Vector3 tableTopSize  = new Vector3(1.2f, 0.04f, 0.65f);
        public float   tableHeight   = 0.75f;
        public float   legWidth      = 0.05f;
        [Tooltip("World position of the table floor-centre (player faces +Z).")]
        public Vector3 tablePosition = new Vector3(0f, 0f, 0.65f);

        [Header("Objects to Place on Table")]
        [Tooltip("Assigned GameObjects are repositioned to sit on the table surface at Awake.")]
        public GameObject[] objectsOnTable;
        [Tooltip("X-offset from table centre for each object (same order as objectsOnTable).")]
        public float[]      objectXOffsets;

        [Header("Snap Zones")]
        [Tooltip("Auto-create a SnapZone trigger at each object's table position.")]
        public bool    autoCreateSnapZones = true;
        [Tooltip("Size of each generated snap zone trigger collider.")]
        public Vector3 snapZoneSize        = new Vector3(0.18f, 0.12f, 0.18f);

        [Header("Decorative Props (non-interactable scene dressing)")]
        [Tooltip("Extra GameObjects to position along the back edge of the table.")]
        public GameObject[] decorativeObjects;
        [Tooltip("X-offsets for decorative objects (back edge of table, z offset = tableTopSize.z * 0.35).")]
        public float[]      decorXOffsets;

        [Header("Materials (optional — built-in colours used as fallback)")]
        public Material tableTopMaterial;
        public Material tableLegMaterial;
        public Material floorMaterial;

        [Header("Floor")]
        public float floorSize = 6f;

        // ── Accessors ─────────────────────────────────────────────────────────

        /// <summary>
        /// Snap zones created for each entry in objectsOnTable (same order).
        /// Assign e.g. SnapZones[0] to ScenarioKitchenTask.bottleSnapZone.
        /// Null until Awake() completes.
        /// </summary>
        public SnapZone[] SnapZones { get; private set; }

        // ── Unity ─────────────────────────────────────────────────────────────

        void Awake()
        {
            BuildFloor();
            BuildTable();
            PlaceObjectsOnTable();
            if (decorativeObjects != null && decorativeObjects.Length > 0)
                PlaceDecorativeObjects();
        }

        // ── Floor ─────────────────────────────────────────────────────────────

        private void BuildFloor()
        {
            var floor = GameObject.CreatePrimitive(PrimitiveType.Plane);
            floor.name = "Floor";
            floor.transform.SetParent(transform);
            floor.transform.position   = Vector3.zero;
            floor.transform.localScale = Vector3.one * (floorSize / 10f);
            ApplyMaterialOrColor(floor, floorMaterial, new Color(0.86f, 0.83f, 0.79f));
        }

        // ── Table ─────────────────────────────────────────────────────────────

        private void BuildTable()
        {
            var root = new GameObject("KitchenTable");
            root.transform.SetParent(transform);
            root.transform.position = tablePosition;

            float surfaceY = tableHeight + tableTopSize.y * 0.5f;

            var top = MakeBox("TableTop", root.transform,
                new Vector3(0f, surfaceY, 0f), tableTopSize);
            ApplyMaterialOrColor(top, tableTopMaterial, new Color(0.56f, 0.36f, 0.18f));

            float hx      = tableTopSize.x * 0.5f - legWidth * 0.5f - 0.04f;
            float hz      = tableTopSize.z * 0.5f - legWidth * 0.5f - 0.04f;
            float legH    = tableHeight - tableTopSize.y;
            var   legSize = new Vector3(legWidth, legH, legWidth);

            SpawnLeg(root.transform, new Vector3( hx, legH * 0.5f,  hz), legSize);
            SpawnLeg(root.transform, new Vector3(-hx, legH * 0.5f,  hz), legSize);
            SpawnLeg(root.transform, new Vector3( hx, legH * 0.5f, -hz), legSize);
            SpawnLeg(root.transform, new Vector3(-hx, legH * 0.5f, -hz), legSize);
        }

        private void SpawnLeg(Transform parent, Vector3 localPos, Vector3 size)
        {
            var leg = MakeBox("Leg", parent, localPos, size);
            ApplyMaterialOrColor(leg, tableLegMaterial, new Color(0.40f, 0.25f, 0.11f));
        }

        // ── Object Placement ──────────────────────────────────────────────────

        private void PlaceObjectsOnTable()
        {
            if (objectsOnTable == null || objectsOnTable.Length == 0) return;

            int count = objectsOnTable.Length;
            if (autoCreateSnapZones) SnapZones = new SnapZone[count];

            float surfaceTopY = tablePosition.y + tableHeight + tableTopSize.y;

            for (int i = 0; i < count; i++)
            {
                float xOff = GetOffset(objectXOffsets, i, count);

                if (objectsOnTable[i] != null)
                {
                    float halfH = GetHalfHeight(objectsOnTable[i]);
                    objectsOnTable[i].transform.position =
                        tablePosition + new Vector3(xOff, tableHeight + tableTopSize.y + halfH, 0f);

                    var rb = objectsOnTable[i].GetComponent<Rigidbody>();
                    if (rb != null) { rb.linearVelocity = Vector3.zero; rb.angularVelocity = Vector3.zero; }
                }

                if (autoCreateSnapZones)
                {
                    var zone = CreateSnapZone(i, xOff, surfaceTopY);
                    // Wire accepted object only when the slot has an InteractableObject
                    if (objectsOnTable[i] != null)
                        zone.acceptedObject = objectsOnTable[i].GetComponent<InteractableObject>();
                    SnapZones[i] = zone;
                }
            }
        }

        private SnapZone CreateSnapZone(int index, float xOff, float surfaceTopY)
        {
            var go = new GameObject($"SnapZone_{index}");
            go.transform.SetParent(transform);
            go.transform.position = tablePosition + new Vector3(xOff, surfaceTopY, 0f);

            // Box trigger collider
            var col = go.AddComponent<BoxCollider>();
            col.size      = snapZoneSize;
            col.isTrigger = true;

            var zone = go.AddComponent<SnapZone>();
            zone.snapOffset = new Vector3(0f, snapZoneSize.y * 0.5f + 0.01f, 0f);
            return zone;
        }

        // ── Decorative Props ──────────────────────────────────────────────────

        private void PlaceDecorativeObjects()
        {
            float surfaceTopY = tablePosition.y + tableHeight + tableTopSize.y;
            float backZ       = tableTopSize.z * 0.35f;

            for (int i = 0; i < decorativeObjects.Length; i++)
            {
                if (decorativeObjects[i] == null) continue;
                float xOff  = GetOffset(decorXOffsets, i, decorativeObjects.Length);
                float halfH = GetHalfHeight(decorativeObjects[i]);
                decorativeObjects[i].transform.position =
                    tablePosition + new Vector3(xOff, tableHeight + tableTopSize.y + halfH, backZ);
            }
        }

        // ── Helpers ───────────────────────────────────────────────────────────

        private static float GetOffset(float[] offsets, int i, int total)
        {
            if (offsets != null && i < offsets.Length) return offsets[i];
            return (i - (total - 1) * 0.5f) * 0.28f;
        }

        private static float GetHalfHeight(GameObject go)
        {
            var r = go.GetComponentInChildren<Renderer>();
            if (r != null) return r.bounds.extents.y;
            var c = go.GetComponent<Collider>();
            if (c != null) return c.bounds.extents.y;
            return 0.06f;
        }

        private static GameObject MakeBox(string name, Transform parent,
                                          Vector3 localPos, Vector3 size)
        {
            var go = GameObject.CreatePrimitive(PrimitiveType.Cube);
            go.name = name;
            go.transform.SetParent(parent);
            go.transform.localPosition = localPos;
            go.transform.localScale    = size;
            return go;
        }

        private static void ApplyMaterialOrColor(GameObject go, Material mat, Color fallback)
        {
            if (mat != null) { go.GetComponent<Renderer>().material = mat; return; }

            var shader = Shader.Find("Universal Render Pipeline/Lit")
                      ?? Shader.Find("Standard");

            var m = shader != null ? new Material(shader) : go.GetComponent<Renderer>().material;
            m.color = fallback;
            go.GetComponent<Renderer>().material = m;
        }
    }
}
