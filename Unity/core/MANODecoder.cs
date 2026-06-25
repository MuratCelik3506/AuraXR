using System;
using System.Collections.Generic;
using UnityEngine;

namespace AuraXR
{
    /// <summary>
    /// Decodes MANO PCA hand pose (15-dim) to per-bone quaternions for hand rig animation.
    ///
    /// Pipeline:
    ///   pose_pca (15,) → axis_angle (45,) → 15 × Quaternion
    ///
    /// Step 1 — PCA decode (matches Python smplx):
    ///   axis_angle_45 = mean_pose_45 + pose_pca_15 @ pca_matrix_15x45
    ///
    /// Step 2 — Axis-angle → Quaternion for each joint (3 floats per joint):
    ///   angle  = ||aa_3||
    ///   axis   = aa_3 / angle  (if angle > ε, else use (0,0,1))
    ///   q      = AngleAxis(angle_rad, axis)  — in MANO rest frame (right-handed Y-up)
    ///
    /// Step 3 — Coordinate frame:
    ///   MANO is right-handed. Unity is left-handed.
    ///   Conversion: negate z-component of axis (mirrors the handedness flip).
    ///   This matches the HOT3D frame conversion used everywhere else in AuraXR.
    ///
    /// Usage:
    ///   1. Export PCA JSON from Python:
    ///      python src/mano_fk.py --export_pca data/mano_models/pca_right.json right
    ///      → Copy pca_right.json into Unity Resources/ folder
    ///
    ///   2. Call LoadFromJson() once (e.g., in Awake / Start):
    ///      var decoder = new MANODecoder();
    ///      decoder.LoadFromJson(Resources.Load<TextAsset>("pca_right"));
    ///
    ///   3. Each frame:
    ///      Quaternion[] boneRots = decoder.Decode(handPose.ManoJointAngles);
    ///      for (int j = 0; j < 15; j++) boneTransforms[j].localRotation = boneRots[j];
    ///
    /// MANO joint order (15 joints):
    ///   Index [0..2]:  Thumb  MCP, PIP, DIP
    ///   Index [3..5]:  Index  MCP, PIP, DIP
    ///   Index [6..8]:  Middle MCP, PIP, DIP
    ///   Index [9..11]: Ring   MCP, PIP, DIP
    ///   Index [12..14]:Pinky  MCP, PIP, DIP
    /// </summary>
    public class MANODecoder
    {
        private const int PCA_DIM  = 15;
        private const int AA_DIM   = 45;
        private const int N_JOINTS = 15;

        // PCA basis: [PCA_DIM × AA_DIM] stored row-major
        private float[] _pcaMatrix;   // [PCA_DIM * AA_DIM]
        private float[] _meanPose;    // [AA_DIM]
        private bool    _loaded = false;

        // Scratch buffer to avoid per-frame allocs
        private readonly float[]       _axisAngle = new float[AA_DIM];
        private readonly Quaternion[]  _result     = new Quaternion[N_JOINTS];

        // ── Loading ────────────────────────────────────────────────────────────

        /// <summary>Load PCA matrix and mean pose from a JSON TextAsset.</summary>
        /// <remarks>JSON schema (produced by src/mano_fk.py --export_pca):
        /// { "pca_matrix": [[...15 rows of 45 floats...]], "mean_pose": [...45 floats...] }</remarks>
        public bool LoadFromJson(TextAsset jsonAsset)
        {
            if (jsonAsset == null)
            {
                Debug.LogWarning("[MANODecoder] TextAsset is null — decoder not loaded.");
                return false;
            }
            return LoadFromJson(jsonAsset.text);
        }

        public bool LoadFromJson(string json)
        {
            try
            {
                var data = JsonUtility.FromJson<PCAJson>(json);
                if (data == null || data.pca_matrix == null || data.mean_pose == null)
                {
                    Debug.LogError("[MANODecoder] Failed to parse PCA JSON.");
                    return false;
                }

                if (data.pca_matrix.Length != PCA_DIM || data.mean_pose.Length != AA_DIM)
                {
                    Debug.LogError($"[MANODecoder] Unexpected dimensions: " +
                                   $"pca_matrix rows={data.pca_matrix.Length} (expected {PCA_DIM}), " +
                                   $"mean_pose={data.mean_pose.Length} (expected {AA_DIM})");
                    return false;
                }

                // Flatten [PCA_DIM][AA_DIM] → [PCA_DIM * AA_DIM]
                _pcaMatrix = new float[PCA_DIM * AA_DIM];
                for (int r = 0; r < PCA_DIM; r++)
                {
                    if (data.pca_matrix[r] == null || data.pca_matrix[r].values.Length != AA_DIM)
                    {
                        Debug.LogError($"[MANODecoder] pca_matrix row {r} has wrong length.");
                        return false;
                    }
                    Array.Copy(data.pca_matrix[r].values, 0, _pcaMatrix, r * AA_DIM, AA_DIM);
                }
                _meanPose = data.mean_pose;
                _loaded   = true;
                Debug.Log("[MANODecoder] Loaded PCA basis (15×45) + mean pose (45).");
                return true;
            }
            catch (Exception e)
            {
                Debug.LogError($"[MANODecoder] Exception while loading JSON: {e.Message}");
                return false;
            }
        }

        /// <returns>True if the PCA basis has been successfully loaded.</returns>
        public bool IsLoaded => _loaded;

        // ── Decoding ───────────────────────────────────────────────────────────

        /// <summary>
        /// Decode MANO PCA pose (15-dim) to per-joint quaternions (15 joints).
        /// Returns the internal result buffer — copy if you need to store across frames.
        /// </summary>
        public Quaternion[] Decode(float[] posePca)
        {
            if (!_loaded)
            {
                for (int j = 0; j < N_JOINTS; j++) _result[j] = Quaternion.identity;
                return _result;
            }

            // Step 1: axis_angle = mean_pose + posePca @ pcaMatrix
            // _axisAngle[k] = _meanPose[k] + sum_i(posePca[i] * _pcaMatrix[i * AA_DIM + k])
            Array.Copy(_meanPose, _axisAngle, AA_DIM);
            for (int i = 0; i < PCA_DIM; i++)
            {
                float pc = posePca[i];
                int   rowOffset = i * AA_DIM;
                for (int k = 0; k < AA_DIM; k++)
                    _axisAngle[k] += pc * _pcaMatrix[rowOffset + k];
            }

            // Step 2: convert each 3-element axis-angle to Quaternion
            for (int j = 0; j < N_JOINTS; j++)
            {
                int   base_  = j * 3;
                float ax = _axisAngle[base_];
                float ay = _axisAngle[base_ + 1];
                float az = _axisAngle[base_ + 2];

                float angle = Mathf.Sqrt(ax*ax + ay*ay + az*az);
                if (angle < 1e-6f)
                {
                    _result[j] = Quaternion.identity;
                }
                else
                {
                    float invAngle = 1f / angle;
                    // Step 3: MANO right-handed → Unity left-handed: negate z of axis
                    Vector3 axis = new Vector3(ax * invAngle, ay * invAngle, -az * invAngle);
                    _result[j]   = Quaternion.AngleAxis(angle * Mathf.Rad2Deg, axis);
                }
            }

            return _result;
        }

        // ── Identity fallback ──────────────────────────────────────────────────

        /// <summary>Returns identity rotations for all 15 joints (open-hand neutral).</summary>
        public Quaternion[] IdentityPose()
        {
            for (int j = 0; j < N_JOINTS; j++) _result[j] = Quaternion.identity;
            return _result;
        }

        // ── JSON schema ────────────────────────────────────────────────────────
        // JsonUtility cannot deserialize jagged arrays, so we use a row wrapper.

        [Serializable]
        private class FloatRow
        {
            public float[] values;
        }

        [Serializable]
        private class PCAJson
        {
            public FloatRow[] pca_matrix;   // [PCA_DIM] rows, each with [AA_DIM] values
            public float[]    mean_pose;    // [AA_DIM]
        }
    }

    // ── Optional MonoBehaviour wrapper ─────────────────────────────────────────
    /// <summary>
    /// MonoBehaviour that wraps MANODecoder for Inspector-wired bone list.
    /// Attach to the virtual hand root; assign the 15 MANO bone transforms.
    ///
    /// MANO bone order (15, corresponds to ManoJointAngles):
    ///   [0-2]  Thumb  MCP/PIP/DIP
    ///   [3-5]  Index  MCP/PIP/DIP
    ///   [6-8]  Middle MCP/PIP/DIP
    ///   [9-11] Ring   MCP/PIP/DIP
    ///   [12-14]Pinky  MCP/PIP/DIP
    /// </summary>
    public class MANOHandController : MonoBehaviour
    {
        [Header("PCA JSON (from src/mano_fk.py --export_pca)")]
        [Tooltip("Drag pca_right.json or pca_left.json here (Resources/...)")]
        public TextAsset pcaJson;

        [Header("MANO Bone Transforms (15 joints)")]
        [Tooltip("Assign 15 transforms in MANO order: Thumb[MCP,PIP,DIP], Index[...], Mid[...], Ring[...], Pinky[...]")]
        public Transform[] manoBones = new Transform[15];

        [Header("Inference Source")]
        [Tooltip("AuraXRInferenceManager in scene — auto-found if null")]
        public AuraXRInferenceManager inferenceManager;

        [Header("Hand Side")]
        public bool isRightHand = true;

        private MANODecoder _decoder;

        void Awake()
        {
            _decoder = new MANODecoder();
            if (pcaJson != null)
                _decoder.LoadFromJson(pcaJson);
            else
                Debug.LogWarning("[MANOHandController] pcaJson not assigned — bones will stay in rest pose.");

            if (inferenceManager == null)
                inferenceManager = FindAnyObjectByType<AuraXRInferenceManager>();
        }

        void LateUpdate()
        {
            if (inferenceManager == null) return;

            var handPose = isRightHand ? inferenceManager.RightHand : inferenceManager.LeftHand;
            if (handPose == null || handPose.ManoJointAngles == null) return;

            Quaternion[] boneRots = _decoder.Decode(handPose.ManoJointAngles);

            for (int j = 0; j < 15 && j < manoBones.Length; j++)
            {
                if (manoBones[j] != null)
                    manoBones[j].localRotation = boneRots[j];
            }
        }
    }
}
