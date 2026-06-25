using System;
using System.Collections.Generic;
using UnityEngine;

namespace AuraXR
{
    /// <summary>
    /// Loads pre-baked SDF grids from Unity Resources/SDFGrids/ and answers
    /// per-frame trilinear SDF queries for each HOT3D BOP object.
    ///
    /// Replaces the bbox approximation in AuraXRInferenceManager with exact
    /// signed-distance-function values matching training (sdf_utils.py).
    ///
    /// Files in Resources/SDFGrids/:
    ///   sdf_manifest.json   — bounds + metadata (33 objects)
    ///   sdf_bop{N:D2}.bytes — raw float32 LE, 32³ = 32768 floats per object
    ///
    /// Usage (called from AuraXRInferenceManager every frame):
    ///   float[] feat4 = sdfGrid.Query(bopId, wristInObjLocal);
    ///   // returns [sdf_value, grad_x, grad_y, grad_z] in metres, object-local frame
    /// </summary>
    public class SDFGridDatabase : MonoBehaviour
    {
        // Auto-loaded from Resources if not assigned
        public TextAsset manifestJson;

        private const int N = 32;
        private const int GRID_FLOATS = N * N * N;  // 32768

        private struct GridEntry
        {
            public float[] data;      // GRID_FLOATS floats, row-major [x,y,z]
            public Vector3 boundsMin; // object-local metres
            public Vector3 boundsMax;
        }

        private readonly Dictionary<int, GridEntry> _grids = new();
        private static readonly float[] _zeros4 = new float[4];

        void Awake() => Load();

        private void Load()
        {
            if (manifestJson == null)
                manifestJson = Resources.Load<TextAsset>("SDFGrids/sdf_manifest");
            if (manifestJson == null)
            {
                Debug.LogError("[SDFGridDB] sdf_manifest.json not found in Resources/SDFGrids/.");
                return;
            }

            var manifest = JsonUtility.FromJson<ManifestRoot>(manifestJson.text);
            if (manifest?.objects == null) { Debug.LogError("[SDFGridDB] Failed to parse manifest."); return; }

            int loaded = 0;
            foreach (var obj in manifest.objects)
            {
                string binPath = $"SDFGrids/sdf_bop{obj.bop_id:D2}";
                var asset = Resources.Load<TextAsset>(binPath);
                if (asset == null)
                {
                    Debug.LogWarning($"[SDFGridDB] {binPath}.bytes not found — skipping bop{obj.bop_id}.");
                    continue;
                }

                byte[] raw = asset.bytes;
                if (raw.Length != GRID_FLOATS * 4)
                {
                    Debug.LogWarning($"[SDFGridDB] bop{obj.bop_id}: unexpected byte count {raw.Length} (expected {GRID_FLOATS * 4}).");
                    continue;
                }

                float[] floats = new float[GRID_FLOATS];
                Buffer.BlockCopy(raw, 0, floats, 0, raw.Length);

                _grids[obj.bop_id] = new GridEntry
                {
                    data      = floats,
                    boundsMin = new Vector3(obj.bounds_min[0], obj.bounds_min[1], obj.bounds_min[2]),
                    boundsMax = new Vector3(obj.bounds_max[0], obj.bounds_max[1], obj.bounds_max[2]),
                };
                loaded++;
            }
            Debug.Log($"[SDFGridDB] Loaded {loaded}/{manifest.objects.Length} SDF grids.");
        }

        /// <summary>
        /// Query SDF at <paramref name="pointObjLocal"/> (object-local HOT3D frame, metres).
        /// Returns float[4] = [sdf_value, grad_x, grad_y, grad_z].
        /// Matches Python sdf_utils.query_sdf_and_gradient exactly.
        /// </summary>
        public float[] Query(int bopId, Vector3 pointObjLocal)
        {
            if (!_grids.TryGetValue(bopId, out GridEntry g)) return _zeros4;

            Vector3 lo = g.boundsMin, hi = g.boundsMax;
            Vector3 p  = pointObjLocal;

            // Clamp to grid bounds (extrapolate linearly outside)
            Vector3 pClamped = new Vector3(
                Mathf.Clamp(p.x, lo.x, hi.x),
                Mathf.Clamp(p.y, lo.y, hi.y),
                Mathf.Clamp(p.z, lo.z, hi.z));
            float extraDist = (p - pClamped).magnitude;

            // Convert clamped point to fractional grid coordinates [0, N-1]
            Vector3 range = hi - lo;
            float fx = (pClamped.x - lo.x) / range.x * (N - 1);
            float fy = (pClamped.y - lo.y) / range.y * (N - 1);
            float fz = (pClamped.z - lo.z) / range.z * (N - 1);

            float sdfBoundary = Trilinear(g.data, fx, fy, fz);
            float sdfVal = sdfBoundary + extraDist;

            float gx, gy, gz;
            if (extraDist > 1e-6f)
            {
                // Gradient: unit vector from boundary toward query point
                Vector3 outDir = (p - pClamped).normalized;
                gx = outDir.x; gy = outDir.y; gz = outDir.z;
            }
            else
            {
                // Central differences, eps = 1 voxel in each axis
                // Cell sizes in metres
                float cx = range.x / (N - 1);
                float cy = range.y / (N - 1);
                float cz = range.z / (N - 1);
                const float eps = 1f;
                gx = (Trilinear(g.data, fx + eps, fy, fz) - Trilinear(g.data, fx - eps, fy, fz)) / (2f * eps * cx);
                gy = (Trilinear(g.data, fx, fy + eps, fz) - Trilinear(g.data, fx, fy - eps, fz)) / (2f * eps * cy);
                gz = (Trilinear(g.data, fx, fy, fz + eps) - Trilinear(g.data, fx, fy, fz - eps)) / (2f * eps * cz);
            }

            return new float[] { sdfVal, gx, gy, gz };
        }

        // Trilinear interpolation — grid indexed as [x, y, z] row-major (x outer)
        private static float Trilinear(float[] data, float fx, float fy, float fz)
        {
            fx = Mathf.Clamp(fx, 0f, N - 1f);
            fy = Mathf.Clamp(fy, 0f, N - 1f);
            fz = Mathf.Clamp(fz, 0f, N - 1f);

            int x0 = Mathf.Min((int)fx, N - 2);
            int y0 = Mathf.Min((int)fy, N - 2);
            int z0 = Mathf.Min((int)fz, N - 2);
            int x1 = x0 + 1, y1 = y0 + 1, z1 = z0 + 1;

            float dx = fx - x0, dy = fy - y0, dz = fz - z0;
            float mx = 1f - dx, my = 1f - dy, mz = 1f - dz;

            return data[Idx(x0,y0,z0)] * mx*my*mz
                 + data[Idx(x1,y0,z0)] * dx*my*mz
                 + data[Idx(x0,y1,z0)] * mx*dy*mz
                 + data[Idx(x0,y0,z1)] * mx*my*dz
                 + data[Idx(x1,y1,z0)] * dx*dy*mz
                 + data[Idx(x1,y0,z1)] * dx*my*dz
                 + data[Idx(x0,y1,z1)] * mx*dy*dz
                 + data[Idx(x1,y1,z1)] * dx*dy*dz;
        }

        // Row-major index: grid[x, y, z] = data[x*N*N + y*N + z]
        private static int Idx(int x, int y, int z) => x * N * N + y * N + z;

        // JSON deserialization
        [Serializable] private class ManifestRoot    { public ManifestEntry[] objects; }
        [Serializable] private class ManifestEntry
        {
            public int      bop_id;
            public string   name;
            public int      grid_size;
            public float[]  bounds_min;
            public float[]  bounds_max;
        }
    }
}
