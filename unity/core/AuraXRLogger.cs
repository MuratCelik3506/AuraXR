using System;
using System.IO;
using UnityEngine;

namespace AuraXR
{
    /// <summary>
    /// Captures Unity log messages and writes them to Application.persistentDataPath/Logs/.
    /// Add this component once to a persistent GameObject in the scene.
    /// </summary>
    public class AuraXRLogger : MonoBehaviour
    {
        [Header("Filter")]
        [Tooltip("Only capture logs that start with [AuraXR]. Disable to capture everything.")]
        public bool auraXROnly = true;

        private StreamWriter _writer;

        public static string LogFolder { get; private set; }

        void Awake()
        {
            LogFolder = Path.Combine(Application.persistentDataPath, "Logs");
            Directory.CreateDirectory(LogFolder);

            string path = Path.Combine(LogFolder, $"auraxr_{DateTime.Now:yyyy_MM_dd_HH_mm_ss}.log");
            _writer = new StreamWriter(path, append: false) { AutoFlush = true };

            Application.logMessageReceived += HandleLog;
            Debug.Log($"[AuraXR] Logger active → {path}");
        }

        void OnDestroy()
        {
            Application.logMessageReceived -= HandleLog;
            _writer?.Close();
        }

        private void HandleLog(string condition, string stackTrace, LogType type)
        {
            if (auraXROnly && !condition.StartsWith("[AuraXR]")) return;

            _writer.WriteLine($"[{DateTime.Now:HH:mm:ss.fff}][{type}] {condition}");

            if (type == LogType.Error || type == LogType.Exception)
                _writer.WriteLine(stackTrace);
        }
    }
}
