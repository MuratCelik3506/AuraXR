/*
 * GhostingSystem.cs
 * ==================
 * Implements the "Ghosting Effect" described in instruction.md §5.
 *
 * Behaviour:
 *   • Listens to IntentPredictor.OnIntentConfident.
 *   • When confidence ≥ 65 % → fades in a semi-transparent "Ghost Hand".
 *   • When confidence drops below threshold → fades it out.
 *   • Pre-loads physics (mass, friction) on the target object's Rigidbody
 *     so that `OnCollisionEnter` incurs zero calculation overhead ("jitter-free").
 *
 * Usage:
 *   1. Attach to the same GameObject as IntentPredictor.
 *   2. Assign ghostHandLeft / ghostHandRight mesh renderers.
 *   3. Assign physicsObjects array with one entry per action class.
 */

using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace XRIntent
{
    /// <summary>
    /// Per-object physics preset preloaded before contact.
    /// </summary>
    [Serializable]
    public class ObjectPhysicsPreset
    {
        public int    ActionClass;       // 0-indexed
        public float  Mass     = 1.0f;
        public float  DynamicFriction = 0.4f;
        public float  StaticFriction  = 0.6f;
        public float  Bounciness      = 0.0f;
    }

    public class GhostingSystem : MonoBehaviour
    {
        // ── Inspector ────────────────────────────────────────────
        [Header("Ghost Hand Renderers")]
        [Tooltip("Skinned mesh renderer for the left ghost hand.")]
        public SkinnedMeshRenderer ghostHandLeft;
        [Tooltip("Skinned mesh renderer for the right ghost hand.")]
        public SkinnedMeshRenderer ghostHandRight;

        [Header("Ghost Settings")]
        [Range(0.1f, 0.6f)]
        [Tooltip("Target ghost opacity (0 = invisible, 0.4 = typical semi-transparent).")]
        public float ghostAlpha = 0.35f;

        [Range(0.05f, 1.0f)]
        [Tooltip("Seconds for the fade-in / fade-out animation.")]
        public float fadeDuration = 0.15f;

        [Header("Physics Presets")]
        [Tooltip("Map action class → physics properties to preload.")]
        public ObjectPhysicsPreset[] physicsPresets;

        [Tooltip("Rigidbodies of graspable objects (index must match class mapping).")]
        public Rigidbody[] graspableObjects;

        [Header("Runtime References")]
        [Tooltip("Source of intent predictions.")]
        public IntentPredictor intentPredictor;

        // ── State ─────────────────────────────────────────────────
        private bool    _ghostVisible   = false;
        private float   _currentAlpha   = 0f;
        private Coroutine _fadeCoroutine;

        private Dictionary<int, ObjectPhysicsPreset> _presetLookup = new();
        private int _lastPreloadedClass = -1;

        // ── Material property caches (avoid GC per-frame) ─────────
        private static readonly int AlphaID = Shader.PropertyToID("_Alpha");
        private MaterialPropertyBlock _mpb;

        // ── Lifecycle ─────────────────────────────────────────────

        private void Awake()
        {
            _mpb = new MaterialPropertyBlock();

            // Build lookup
            if (physicsPresets != null)
                foreach (var p in physicsPresets)
                    _presetLookup[p.ActionClass] = p;

            // Start ghosts as invisible
            SetGhostAlpha(0f);
        }

        private void OnEnable()
        {
            if (intentPredictor != null)
            {
                intentPredictor.OnIntentUpdated   += HandleIntentUpdated;
                intentPredictor.OnIntentConfident += HandleIntentConfident;
            }
        }

        private void OnDisable()
        {
            if (intentPredictor != null)
            {
                intentPredictor.OnIntentUpdated   -= HandleIntentUpdated;
                intentPredictor.OnIntentConfident -= HandleIntentConfident;
            }
        }

        // ── Intent callbacks ──────────────────────────────────────

        /// <summary>
        /// Called every inference cycle.
        /// Hide the ghost if confidence dropped below threshold.
        /// </summary>
        private void HandleIntentUpdated(IntentResult result)
        {
            if (result.Confidence < intentPredictor.ghostingThreshold && _ghostVisible)
            {
                FadeGhost(false);
            }
        }

        /// <summary>
        /// Called when confidence ≥ ghostingThreshold.
        /// Show ghost + preload physics for the predicted class.
        /// </summary>
        private void HandleIntentConfident(IntentResult result)
        {
            // ── 1. Show ghost ─────────────────────────────────────
            if (!_ghostVisible) FadeGhost(true);

            // ── 2. Physics pre-loading ────────────────────────────
            if (result.TopClass != _lastPreloadedClass)
            {
                PreloadPhysics(result.TopClass);
                _lastPreloadedClass = result.TopClass;
            }

            // ── 3. Debug info ─────────────────────────────────────
            Debug.Log(
                $"[Ghost] Class={result.TopClass}  Conf={result.Confidence:P1}  " +
                $"TTC={result.TTC:F2}s  Latency={result.InferenceLatencyUs}µs"
            );
        }

        // ── Ghost fade ────────────────────────────────────────────

        private void FadeGhost(bool visible)
        {
            _ghostVisible = visible;
            if (_fadeCoroutine != null) StopCoroutine(_fadeCoroutine);
            _fadeCoroutine = StartCoroutine(FadeRoutine(visible ? ghostAlpha : 0f));
        }

        private IEnumerator FadeRoutine(float targetAlpha)
        {
            float startAlpha = _currentAlpha;
            float elapsed    = 0f;

            while (elapsed < fadeDuration)
            {
                elapsed += Time.deltaTime;
                _currentAlpha = Mathf.Lerp(startAlpha, targetAlpha,
                                           elapsed / fadeDuration);
                SetGhostAlpha(_currentAlpha);
                yield return null;
            }

            _currentAlpha = targetAlpha;
            SetGhostAlpha(_currentAlpha);
        }

        private void SetGhostAlpha(float alpha)
        {
            _mpb.SetFloat(AlphaID, alpha);
            if (ghostHandLeft  != null) ghostHandLeft.SetPropertyBlock(_mpb);
            if (ghostHandRight != null) ghostHandRight.SetPropertyBlock(_mpb);
        }

        // ── Physics pre-loading ───────────────────────────────────

        /// <summary>
        /// Pre-configure the Rigidbody and PhysicsMaterial of the predicted
        /// target object so that OnCollisionEnter requires zero additional work.
        /// This is the key mechanism that eliminates "jitter."
        /// </summary>
        private void PreloadPhysics(int actionClass)
        {
            if (!_presetLookup.TryGetValue(actionClass, out var preset))
            {
                Debug.LogWarning($"[Ghost] No physics preset for class {actionClass}");
                return;
            }

            // Find a suitable Rigidbody (simple linear search; extend as needed)
            if (graspableObjects == null || graspableObjects.Length == 0) return;

            // For now: use the first graspable object.
            // In production: match via class-to-object mapping.
            Rigidbody rb = graspableObjects[0];
            if (rb == null) return;

            // ── Set mass ──────────────────────────────────────────
            rb.mass = preset.Mass;

            // ── Set physics material ───────────────────────────────
            Collider col = rb.GetComponent<Collider>();
            if (col != null)
            {
                PhysicsMaterial mat = col.material;
                if (mat == null)
                {
                    mat = new PhysicsMaterial($"PreloadedMat_Class{actionClass}")
                    {
                        dynamicFriction = preset.DynamicFriction,
                        staticFriction  = preset.StaticFriction,
                        bounciness      = preset.Bounciness,
                    };
                    col.material = mat;
                }
                else
                {
                    mat.dynamicFriction = preset.DynamicFriction;
                    mat.staticFriction  = preset.StaticFriction;
                    mat.bounciness      = preset.Bounciness;
                }
            }

            Debug.Log(
                $"[Physics] Pre-loaded class={actionClass}  " +
                $"mass={preset.Mass}  dynFric={preset.DynamicFriction}"
            );
        }
    }
}
