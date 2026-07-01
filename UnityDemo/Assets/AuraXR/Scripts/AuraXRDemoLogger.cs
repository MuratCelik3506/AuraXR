using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text;
using System.Threading;
using UnityEngine;

namespace AuraXR.Demo
{
    public sealed class AuraXRDemoLogger : MonoBehaviour
    {
        public AuraXRFeatureAssembler featureAssembler;
        public AuraXRModelRuntime modelRuntime;
        public AuraXRBlendController blendController;
        public bool logEveryInference = true;
        public string fileName = "auraxr_demo_log.jsonl";
        [Tooltip("Number of log lines to buffer before flushing to disk on background thread.")]
        public int flushBatchSize = 30;

        private string _path;
        private int _lastInferenceCount = -1;

        // Lines queued for background write — accessed from main thread (Enqueue) and worker thread (Dequeue).
        private readonly Queue<string> _queue = new Queue<string>();
        private readonly object _lock = new object();
        private Thread _writerThread;
        private volatile bool _running;
        private readonly AutoResetEvent _signal = new AutoResetEvent(false);

        void Start()
        {
            _path = Path.Combine(Application.persistentDataPath, fileName);
            Debug.Log($"[AuraXRDemoLogger] Writing JSONL to {_path}");

            _running = true;
            _writerThread = new Thread(WriterLoop) { IsBackground = true, Name = "AuraXRLogger" };
            _writerThread.Start();
        }

        void OnDestroy()
        {
            _running = false;
            _signal.Set();
            _writerThread?.Join(2000);
        }

        void LateUpdate()
        {
            if (!logEveryInference || featureAssembler == null || modelRuntime == null || !modelRuntime.hasOutput) return;
            if (_lastInferenceCount == modelRuntime.inferenceCount) return;
            _lastInferenceCount = modelRuntime.inferenceCount;
            EnqueueRecord();
        }

        public void AppendRecord() => EnqueueRecord();

        private void EnqueueRecord()
        {
            string line = BuildRecord();
            lock (_lock)
            {
                _queue.Enqueue(line);
                if (_queue.Count >= flushBatchSize)
                    _signal.Set();
            }
        }

        // Background thread: drains the queue and appends to disk in batches.
        private void WriterLoop()
        {
            var sb = new StringBuilder(65536);
            while (_running)
            {
                _signal.WaitOne(500);

                sb.Clear();
                lock (_lock)
                {
                    while (_queue.Count > 0)
                        sb.AppendLine(_queue.Dequeue());
                }
                if (sb.Length > 0)
                {
                    try { File.AppendAllText(_path, sb.ToString()); }
                    catch (Exception ex) { UnityEngine.Debug.LogError($"[AuraXRDemoLogger] Write failed: {ex.Message}"); }
                }
            }

            // Flush remaining on shutdown.
            sb.Clear();
            lock (_lock)
            {
                while (_queue.Count > 0)
                    sb.AppendLine(_queue.Dequeue());
            }
            if (sb.Length > 0)
            {
                try { File.AppendAllText(_path, sb.ToString()); }
                catch { }
            }
        }

        private string BuildRecord()
        {
            AuraXRDemoObject obj = featureAssembler.activeObject;
            AuraXRModelOutput output = modelRuntime.Output;
            var sb = new StringBuilder(4096);
            sb.Append('{');
            WriteString(sb, "timestamp", DateTime.UtcNow.ToString("o", CultureInfo.InvariantCulture)); sb.Append(',');
            WriteString(sb, "object_id", obj != null ? obj.objectId : "none"); sb.Append(',');
            WriteNumber(sb, "window_size", AuraXRFeatureAssembler.Window); sb.Append(',');
            WriteNumber(sb, "distance_cm", featureAssembler.distanceM * 100f); sb.Append(',');
            WriteNumber(sb, "blend_weight", blendController != null ? blendController.blendWeight : 0f); sb.Append(',');
            WriteNumber(sb, "quality_score", output.qualityScore); sb.Append(',');
            WriteNumber(sb, "success_prob", output.successProb); sb.Append(',');
            WriteNumber(sb, "latency_ms", output.latencyMs); sb.Append(',');
            WriteArray(sb, "last_frame_feat", featureAssembler.LastFrameFeat); sb.Append(',');
            WriteArray(sb, "selected_pose", output.selectedPose); sb.Append(',');
            WriteNumber(sb, "contact_flag", featureAssembler.lastContactFlag);
            sb.Append('}');
            return sb.ToString();
        }

        private static void WriteString(StringBuilder sb, string key, string value)
        {
            sb.Append('\"').Append(key).Append("\":\"").Append(value.Replace("\"", "\\\"")).Append('\"');
        }

        private static void WriteNumber(StringBuilder sb, string key, float value)
        {
            sb.Append('\"').Append(key).Append("\":").Append(value.ToString("G9", CultureInfo.InvariantCulture));
        }

        private static void WriteNumber(StringBuilder sb, string key, int value)
        {
            sb.Append('\"').Append(key).Append("\":").Append(value);
        }

        private static void WriteArray(StringBuilder sb, string key, float[] values)
        {
            sb.Append('\"').Append(key).Append("\":[");
            if (values != null)
            {
                for (int i = 0; i < values.Length; i++)
                {
                    if (i > 0) sb.Append(',');
                    sb.Append(values[i].ToString("G9", CultureInfo.InvariantCulture));
                }
            }
            sb.Append(']');
        }
    }
}
