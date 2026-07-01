using UnityEngine;

namespace AuraXR.Demo
{
    public sealed class AuraXRHandRetargeter : MonoBehaviour
    {
        [Header("Model")]
        public AuraXRModelRuntime modelRuntime;

        [Header("Bones in MANO order")]
        [Tooltip("15 transforms: index MCP/PIP/DIP, middle, ring, pinky, thumb CMC/MCP/IP.")]
        public Transform[] manoBones = new Transform[15];

        [Header("Retarget")]
        public bool flipZForUnity = true;
        public Vector3[] localEulerOffsets = new Vector3[15];
        [Tooltip("Use anatomy-aware curl limits instead of applying MANO axis-angle rotations directly.")]
        public bool useCalibratedRetarget = true;
        [Range(0f, 1f)] public float modelInfluence = 0.35f;
        [Range(0f, 1f)] public float proceduralFallback = 0.75f;
        public float smoothingHz = 18f;

        public Quaternion[] PredictedRotations => _predicted;
        public float[] JointCurlDegrees => _jointCurlDegrees;

        private readonly Quaternion[] _predicted = new Quaternion[15];
        private readonly Quaternion[] _smoothed = new Quaternion[15];
        private readonly float[] _jointCurlDegrees = new float[15];

        private static readonly float[] OpenCurl =
        {
            4f, 3f, 2f,
            3f, 2f, 2f,
            4f, 3f, 2f,
            6f, 4f, 3f,
            10f, 8f, 6f
        };

        private static readonly float[] GraspCurl =
        {
            52f, 70f, 48f,
            58f, 76f, 52f,
            56f, 74f, 50f,
            48f, 64f, 44f,
            42f, 48f, 34f
        };

        private static readonly float[] MinCurl =
        {
            0f, 0f, 0f,
            0f, 0f, 0f,
            0f, 0f, 0f,
            0f, 0f, 0f,
            0f, 0f, 0f
        };

        private static readonly float[] MaxCurl =
        {
            72f, 94f, 72f,
            78f, 98f, 76f,
            78f, 96f, 74f,
            70f, 88f, 66f,
            62f, 70f, 56f
        };

        private static readonly float[] Spread =
        {
            -8f, 0f, 0f,
            -2f, 0f, 0f,
            5f, 0f, 0f,
            12f, 0f, 0f,
            -48f, -8f, 0f
        };

        void Awake()
        {
            for (int i = 0; i < _predicted.Length; i++)
            {
                _predicted[i] = Quaternion.identity;
                _smoothed[i] = Quaternion.identity;
            }
        }

        void LateUpdate()
        {
            if (modelRuntime == null || !modelRuntime.hasOutput) return;
            Decode(modelRuntime.Output.selectedPose);
        }

        public void Decode(float[] aa45)
        {
            if (aa45 == null || aa45.Length != 45) return;
            for (int joint = 0; joint < 15; joint++)
            {
                int k = joint * 3;
                Quaternion q = AuraXRMath.AxisAngleToQuaternion(aa45[k], aa45[k + 1], aa45[k + 2], flipZForUnity);
                if (localEulerOffsets != null && joint < localEulerOffsets.Length)
                {
                    q *= Quaternion.Euler(localEulerOffsets[joint]);
                }
                _predicted[joint] = q;
                _jointCurlDegrees[joint] = EstimateCurlDegrees(aa45[k], aa45[k + 1], aa45[k + 2], joint);
            }
        }

        public void Apply(float blendWeight)
        {
            blendWeight = Mathf.Clamp01(blendWeight);
            float alpha = 1f - Mathf.Exp(-Mathf.Max(1f, smoothingHz) * Time.deltaTime);
            for (int i = 0; i < 15; i++)
            {
                Transform bone = manoBones != null && i < manoBones.Length ? manoBones[i] : null;
                if (bone == null) continue;

                Quaternion target = useCalibratedRetarget
                    ? BuildCalibratedTarget(i, blendWeight)
                    : _predicted[i];

                _smoothed[i] = Quaternion.Slerp(_smoothed[i], target, alpha);
                bone.localRotation = Quaternion.Slerp(Quaternion.identity, _smoothed[i], blendWeight);
            }
        }

        private Quaternion BuildCalibratedTarget(int joint, float blendWeight)
        {
            float modelCurl = Mathf.Clamp(_jointCurlDegrees[joint], MinCurl[joint], MaxCurl[joint]);
            float fallbackCurl = Mathf.Lerp(OpenCurl[joint], GraspCurl[joint], blendWeight * proceduralFallback);
            float curl = Mathf.Lerp(fallbackCurl, modelCurl, modelInfluence);
            curl = Mathf.Clamp(curl, MinCurl[joint], MaxCurl[joint]);

            float spread = joint % 3 == 0 ? Spread[joint] * Mathf.Lerp(0.35f, 1f, blendWeight) : Spread[joint];

            if (joint >= 12)
            {
                return Quaternion.Euler(-curl * 0.65f, spread, -curl * 0.25f);
            }
            return Quaternion.Euler(-curl, spread, 0f);
        }

        private static float EstimateCurlDegrees(float x, float y, float z, int joint)
        {
            float sign = joint >= 12 ? Mathf.Sign(x + z * 0.5f) : Mathf.Sign(x);
            if (Mathf.Approximately(sign, 0f)) sign = 1f;
            float radians = Mathf.Sqrt(x * x + y * y + z * z);
            return Mathf.Abs(radians * Mathf.Rad2Deg * sign);
        }
    }
}
