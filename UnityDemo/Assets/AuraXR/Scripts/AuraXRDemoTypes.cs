using System;
using System.Collections.Generic;
using System.IO;
using UnityEngine;

namespace AuraXR.Demo
{
    [Serializable]
    public sealed class AuraXRModelStats
    {
        public string source;
        public string[] input_feature_order;
        public float[] input_mean;
        public float[] input_std;
        public float[] pts_mean;
        public float[] pts_std;
        public string units;
        public string rotation_format;

        public void Validate()
        {
            if (input_mean == null || input_mean.Length != 13) throw new InvalidDataException("input_mean must have 13 values.");
            if (input_std == null || input_std.Length != 13) throw new InvalidDataException("input_std must have 13 values.");
            if (pts_mean == null || pts_mean.Length != 3) throw new InvalidDataException("pts_mean must have 3 values.");
            if (pts_std == null || pts_std.Length != 3) throw new InvalidDataException("pts_std must have 3 values.");
        }

        public void NormalizeFrameInPlace(float[] frame13)
        {
            Validate();
            if (frame13 == null || frame13.Length != 13) throw new ArgumentException("frame13 must have 13 values.");
            for (int i = 0; i < 13; i++)
            {
                float std = Mathf.Max(Mathf.Abs(input_std[i]), 1e-6f);
                frame13[i] = (frame13[i] - input_mean[i]) / std;
            }
        }

        public Vector3 NormalizePoint(Vector3 p)
        {
            Validate();
            return new Vector3(
                (p.x - pts_mean[0]) / Mathf.Max(Mathf.Abs(pts_std[0]), 1e-6f),
                (p.y - pts_mean[1]) / Mathf.Max(Mathf.Abs(pts_std[1]), 1e-6f),
                (p.z - pts_mean[2]) / Mathf.Max(Mathf.Abs(pts_std[2]), 1e-6f)
            );
        }
    }

    [Serializable]
    public sealed class AuraXRObjectEntry
    {
        public string object_id;
        public string source;
        public int source_count;
        public int point_count;
        public string points_asset;
        public float[] bbox_min;
        public float[] bbox_max;
        public float[] bbox_size;
        public float bbox_diagonal_m;
    }

    [Serializable]
    public sealed class AuraXRObjectsManifest
    {
        public int n_points;
        public string point_format;
        public AuraXRObjectEntry[] objects;
    }

    public sealed class AuraXRModelOutput
    {
        public readonly float[] selectedPose = new float[45];
        public float qualityScore;
        public float successProb;
        public float latencyMs;
        public bool valid;
    }

    public static class AuraXRMath
    {
        public static Quaternion AxisAngleToQuaternion(float x, float y, float z, bool flipZForUnity = true)
        {
            if (flipZForUnity) z = -z;
            float angle = Mathf.Sqrt(x * x + y * y + z * z);
            if (angle < 1e-8f) return Quaternion.identity;
            return Quaternion.AngleAxis(angle * Mathf.Rad2Deg, new Vector3(x / angle, y / angle, z / angle));
        }

        public static void RelativePoseToFrame13(
            Transform wrist,
            Transform obj,
            Vector3 previousRelPos,
            float deltaTime,
            float distanceM,
            float[] frame13)
        {
            if (frame13 == null || frame13.Length != 13) throw new ArgumentException("frame13 must have 13 values.");

            Quaternion relRot = Quaternion.Inverse(obj.rotation) * wrist.rotation;
            Matrix4x4 r = Matrix4x4.Rotate(relRot);
            Vector3 relPos = obj.InverseTransformPoint(wrist.position);
            Vector3 relVel = deltaTime > 1e-6f ? (relPos - previousRelPos) / deltaTime : Vector3.zero;

            frame13[0] = relPos.x;
            frame13[1] = relPos.y;
            frame13[2] = relPos.z;

            Vector4 c0 = r.GetColumn(0);
            Vector4 c1 = r.GetColumn(1);
            frame13[3] = c0.x;
            frame13[4] = c0.y;
            frame13[5] = c0.z;
            frame13[6] = c1.x;
            frame13[7] = c1.y;
            frame13[8] = c1.z;

            frame13[9] = relVel.x;
            frame13[10] = relVel.y;
            frame13[11] = relVel.z;
            frame13[12] = distanceM;
        }

        public static float SmoothStep01(float t)
        {
            t = Mathf.Clamp01(t);
            return t * t * (3f - 2f * t);
        }

        public static float[] LoadFloat32Bytes(TextAsset asset, int expectedCount)
        {
            if (asset == null) throw new ArgumentNullException(nameof(asset));
            byte[] bytes = asset.bytes;
            if (bytes.Length != expectedCount * sizeof(float))
            {
                throw new InvalidDataException($"{asset.name} has {bytes.Length} bytes, expected {expectedCount * sizeof(float)}.");
            }

            float[] values = new float[expectedCount];
            Buffer.BlockCopy(bytes, 0, values, 0, bytes.Length);
            return values;
        }

        public static Vector3[] ToVector3Points(float[] xyz)
        {
            if (xyz == null || xyz.Length % 3 != 0) throw new ArgumentException("xyz length must be divisible by 3.");
            Vector3[] pts = new Vector3[xyz.Length / 3];
            for (int i = 0, j = 0; i < pts.Length; i++, j += 3)
            {
                pts[i] = new Vector3(xyz[j], xyz[j + 1], xyz[j + 2]);
            }
            return pts;
        }
    }

    public sealed class AuraXRDemoObject : MonoBehaviour
    {
        public string objectId;
        public TextAsset pointCloudBytes;
        [Tooltip("Relative to Application.streamingAssetsPath, used when pointCloudBytes is not imported as TextAsset.")]
        public string pointCloudRelativePath;
        public float bboxDiagonalM;
        [NonSerialized] public Vector3[] rawPoints;
        [NonSerialized] public Vector3[] normalizedPoints;

        public void LoadPoints(AuraXRModelStats stats)
        {
            float[] xyz;
            if (pointCloudBytes != null)
            {
                xyz = AuraXRMath.LoadFloat32Bytes(pointCloudBytes, 1024 * 3);
            }
            else
            {
                string path = Path.Combine(Application.streamingAssetsPath, pointCloudRelativePath ?? string.Empty);
                byte[] bytes = File.ReadAllBytes(path);
                if (bytes.Length != 1024 * 3 * sizeof(float))
                {
                    throw new InvalidDataException($"{path} has {bytes.Length} bytes, expected {1024 * 3 * sizeof(float)}.");
                }
                xyz = new float[1024 * 3];
                Buffer.BlockCopy(bytes, 0, xyz, 0, bytes.Length);
            }
            rawPoints = AuraXRMath.ToVector3Points(xyz);
            normalizedPoints = new Vector3[rawPoints.Length];
            for (int i = 0; i < rawPoints.Length; i++)
            {
                normalizedPoints[i] = stats.NormalizePoint(rawPoints[i]);
            }
        }
    }
}
