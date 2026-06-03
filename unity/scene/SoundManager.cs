using UnityEngine;

namespace AuraXR
{
    /// <summary>
    /// Plays audio clips for task events: pickup, pour, place, task complete.
    /// </summary>
    public class SoundManager : MonoBehaviour
    {
        [Header("Audio Clips")]
        public AudioClip pickupSound;
        public AudioClip pourSound;
        public AudioClip placeSound;
        public AudioClip taskCompleteSound;

        private AudioSource _audioSource;

        void Start()
        {
            _audioSource = gameObject.AddComponent<AudioSource>();
            _audioSource.spatialBlend = 1.0f;
        }

        public void PlayPickup()   => Play(pickupSound);
        public void PlayPour()     => Play(pourSound);
        public void PlayPlace()    => Play(placeSound);
        public void PlayComplete() => Play(taskCompleteSound);

        private void Play(AudioClip clip)
        {
            if (clip != null && _audioSource != null)
                _audioSource.PlayOneShot(clip);
        }
    }
}
