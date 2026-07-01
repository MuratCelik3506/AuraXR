# Deney Takip Tablosu

Bu belge hangi komutu ne sırayla çalıştıracağınızı, hangi dosyaları ürettiğini ve nereye kaydettiğini gösterir.

> **Dosya isimlendirmesi ve deney bütçeleri koda karşı doğrulandı** (2026-07-01). Tüm checkpoint, log ve eval yolları `orchestrate.py → train_grasp.py → evaluate.py` zinciriyle eşleşiyor. `experiments.json` güncel hedef: Phase 1 = 50 epoch, Phase 2 = 50 epoch.

> **Veri/model notu:** HOT3D `stats.json` artık `pts_mean/pts_std` içeriyor; HOT3D ve OakInk `obj_pts` PointNet'e normalize edilmiş unit-std ölçekte gider. `obj_pts_contact` ise eğitim/eval temas kayıpları için normalize edilmeden wrist-frame metre olarak kalır. Eski checkpointler bu stats öncesi eğitildiyse kullanılmamalıdır.

---

## Dosya İsimlendirme Kuralı

```
checkpoints/<name>/phase<N>_best.pt      ← en iyi val_rec checkpoint
checkpoints/<name>/phase<N>_latest.pt    ← son epoch checkpoint (resume için)
results/<name>/train_phase<N>_log.json   ← epoch-by-epoch eğitim metrikleri
results/<name>/eval.json                 ← eval sonucu {"oakink":…,"hot3d":…}
results/<name>/summary.json              ← orchestrate özeti
results/master_summary.json              ← tüm deneylerin özeti
```

Multi-seed: `name` → `<name>_seed<N>` (örn. `checkpoints/full_seed42/phase2_best.pt`)

---

## Faz Ayrımı

| Faz | Veri | Epoch | Komut |
|-----|------|-------|-------|
| Phase 1 | OakInk statik (~8921 train sample) | 50 | `--max-phase 1` ile durdur |
| Phase 2 | HOT3D temporal + %30 OakInk replay | 50 | `--max-phase` olmadan devam et |

```bash
# Önce sadece phase 1:
python src/training/orchestrate.py --max-phase 1

# Sonra phase 2 (phase 1 "done" görünür, atlanır):
python src/training/orchestrate.py
```

---

## AŞAMA 0 — Veri Hazırlığı

| # | Komut | Çıktı | Durum |
|---|-------|-------|-------|
| 0-A | `python src/preprocessing/rebuild_split.py` | `split.json` → `seen_test`(917) + `unseen_test`(872) | ✅ Tamamlandı |
| 0-B | `python src/preprocessing/recompute_normalization_stats.py` | OakInk `input_mean/std`, HOT3D `pts_mean/std` güncellendi | ✅ Tamamlandı |

---

## AŞAMA 1 — Temel & Mimari Ablation

### Phase 1 Durumu

| Deney | Epoch | Checkpoint | Durum |
|-------|-------|------------|-------|
| `full` | 50 | `checkpoints/full/phase1_best.pt` | ✅ |
| `ablation_static_only` | 50 | `checkpoints/ablation_static_only/phase1_best.pt` | ✅ |
| `ablation_singleframe_mlp` | 50 | `checkpoints/ablation_singleframe_mlp/phase1_best.pt` | ✅ |
| `ablation_gru_t1` | 50 | `checkpoints/ablation_gru_t1/phase1_best.pt` | ✅ |
| `ablation_no_obj` | 50 | `checkpoints/ablation_no_obj/phase1_best.pt` | ✅ |
| `ablation_bbox_obj` | 50 | `checkpoints/ablation_bbox_obj/phase1_best.pt` | ⬜ |
| `ablation_no_attention` | 50 | `checkpoints/ablation_no_attention/phase1_best.pt` | ⬜ |
| `ablation_no_film` | 50 | `checkpoints/ablation_no_film/phase1_best.pt` | ⬜ |
| `ablation_no_vel` | 50 | `checkpoints/ablation_no_vel/phase1_best.pt` | ✅ |
| `ablation_no_aug` | 50 | `checkpoints/ablation_no_aug/phase1_best.pt` | ⬜ |

```bash
python src/training/orchestrate.py --max-phase 1 \
  --skip kl_sweep_1110001 kl_sweep_001 kl_sweep_005 kl_sweep_01 \
         contact_weight_sweep_03 contact_weight_sweep_07 \
         contact_weight_sweep_10 contact_weight_sweep_20
```

### Phase 2 Durumu

| Deney | Epoch | Checkpoint | Eval | Durum |
|-------|-------|------------|------|-------|
| `full` | 50 | `checkpoints/full/phase2_best.pt` | `results/full/eval.json` | ⬜ |
| `ablation_singleframe_mlp` | 50 | `checkpoints/ablation_singleframe_mlp/phase2_best.pt` | `results/ablation_singleframe_mlp/eval.json` | ✅ |
| `ablation_gru_t1` | 50 | `checkpoints/ablation_gru_t1/phase2_best.pt` | `results/ablation_gru_t1/eval.json` | ⬜ |
| `ablation_no_obj` | 50 | `checkpoints/ablation_no_obj/phase2_best.pt` | `results/ablation_no_obj/eval.json` | ⬜ |
| `ablation_bbox_obj` | 50 | `checkpoints/ablation_bbox_obj/phase2_best.pt` | `results/ablation_bbox_obj/eval.json` | ⬜ |
| `ablation_no_attention` | 50 | `checkpoints/ablation_no_attention/phase2_best.pt` | `results/ablation_no_attention/eval.json` | ⬜ |
| `ablation_no_film` | 50 | `checkpoints/ablation_no_film/phase2_best.pt` | `results/ablation_no_film/eval.json` | ⬜ |
| `ablation_no_vel` | 50 | `checkpoints/ablation_no_vel/phase2_best.pt` | `results/ablation_no_vel/eval.json` | ✅ |
| `ablation_no_aug` | 50 | `checkpoints/ablation_no_aug/phase2_best.pt` | `results/ablation_no_aug/eval.json` | ⬜ |

> `ablation_static_only` sadece phase 1'dir, phase 2 yoktur.

> Ana deney ve mimari ablation'larda baz kayıp ağırlıkları: `contact_weight=0.05`, `penetration_weight=0.1`, `quality_weight=0.1`. Contact sweep deneyleri özellikle `contact_weight ∈ {0.3, 0.7, 1.0, 2.0}` değerlerini test eder.

```bash
python src/training/orchestrate.py \
  --skip kl_sweep_0001 kl_sweep_001 kl_sweep_005 kl_sweep_01 \
         contact_weight_sweep_03 contact_weight_sweep_07 \
         contact_weight_sweep_10 contact_weight_sweep_20
```

### Sonuç Tabloları (phase 2 tamamlandıktan sonra doldur)

**Temporal ablation:**

| Model | Encoder | Window | Geodesic (°) | Contact Ratio | Jitter |
|-------|---------|--------|--------------|---------------|--------|
| ablation_static_only | GRU | 1 (p1 only) | — | — | — |
| ablation_gru_t1 | GRU | 1 | — | — | — |
| ablation_singleframe_mlp | MLP | 1 | — | — | — |
| **full** | GRU | 16 | — | — | — |

**Geometri ablation:**

| Model | Obj Encoder | Geodesic (°) | Contact Ratio |
|-------|-------------|--------------|---------------|
| ablation_no_obj | none | — | — |
| ablation_bbox_obj | bbox | — | — |
| **full** | PointNet | — | — |

**Mimari ablation:**

| Model | Attention | FiLM | Vel Loss | Geodesic (°) | Joint Limit Viol. |
|-------|-----------|------|----------|--------------|-------------------|
| ablation_no_attention | ✗ | ✓ | ✓ | — | — |
| ablation_no_film | ✓ | ✗ | ✓ | — | — |
| ablation_no_vel | ✓ | ✓ | ✗ | — | — |
| ablation_no_aug | ✓ | ✓ | ✓ (no aug) | — | — |
| **full** | ✓ | ✓ | ✓ | — | — |

---

## AŞAMA 2 — KL Sweep (sadece phase 1)

| Deney | kl_weight | Epoch | Checkpoint | Eval | Durum |
|-------|-----------|-------|------------|------|-------|
| `kl_sweep_0001` | 0.001 | 50 | `checkpoints/kl_sweep_0001/phase1_best.pt` | `results/kl_sweep_0001/eval.json` | ⬜ |
| `kl_sweep_001` | 0.01 | 50 | `checkpoints/kl_sweep_001/phase1_best.pt` | `results/kl_sweep_001/eval.json` | ⬜ |
| `kl_sweep_005` | 0.05 | 50 | `checkpoints/kl_sweep_005/phase1_best.pt` | `results/kl_sweep_005/eval.json` | ⬜ |
| `kl_sweep_01` | 0.1 | 50 | `checkpoints/kl_sweep_01/phase1_best.pt` | `results/kl_sweep_01/eval.json` | ⬜ |

```bash
python src/training/orchestrate.py --only kl_sweep_0001
python src/training/orchestrate.py --only kl_sweep_001
python src/training/orchestrate.py --only kl_sweep_005
python src/training/orchestrate.py --only kl_sweep_01
```

> `--only` tek deney adı alır; paralel çalıştıracaksan her komutu ayrı terminalde başlat.

**KL sweep sonuç tablosu:**

| kl_weight | Diversity Score | Oracle-K Geo (°) | Pose Error (°) | Joint Limit Viol. |
|-----------|-----------------|------------------|----------------|-------------------|
| 0.001 | — | — | — | — |
| 0.01 | — | — | — | — |
| 0.05 | — | — | — | — |
| 0.1 | — | — | — | — |

```bash
python src/evaluation/compare_results.py --source oakink --metric diversity_score
```

---

## AŞAMA 3 — Contact Weight Sweep (phase 1 + 2)

### Phase 1 Durumu

| Deney | contact_weight | Epoch | Checkpoint | Durum |
|-------|----------------|-------|------------|-------|
| `contact_weight_sweep_03` | 0.3 | 50 | `checkpoints/contact_weight_sweep_03/phase1_best.pt` | ⬜ |
| `contact_weight_sweep_07` | 0.7 | 50 | `checkpoints/contact_weight_sweep_07/phase1_best.pt` | ⬜ |
| `contact_weight_sweep_10` | 1.0 | 50 | `checkpoints/contact_weight_sweep_10/phase1_best.pt` | ⬜ |
| `contact_weight_sweep_20` | 2.0 | 50 | `checkpoints/contact_weight_sweep_20/phase1_best.pt` | ⬜ |

### Phase 2 Durumu

| Deney | Checkpoint | Eval | Durum |
|-------|------------|------|-------|
| `contact_weight_sweep_03` | `checkpoints/contact_weight_sweep_03/phase2_best.pt` | `results/contact_weight_sweep_03/eval.json` | ⬜ |
| `contact_weight_sweep_07` | `checkpoints/contact_weight_sweep_07/phase2_best.pt` | `results/contact_weight_sweep_07/eval.json` | ⬜ |
| `contact_weight_sweep_10` | `checkpoints/contact_weight_sweep_10/phase2_best.pt` | `results/contact_weight_sweep_10/eval.json` | ⬜ |
| `contact_weight_sweep_20` | `checkpoints/contact_weight_sweep_20/phase2_best.pt` | `results/contact_weight_sweep_20/eval.json` | ⬜ |

**Contact sweep sonuç tablosu:**

| contact_weight | Contact Ratio | Pose Error (°) | Penetration (mm) |
|----------------|---------------|----------------|-----------------|
| 0.3 | — | — | — |
| 0.7 | — | — | — |
| 1.0 | — | — | — |
| 2.0 | — | — | — |

---

## AŞAMA 4 — Multi-Seed (en sona bırak)

En iyi mimari varyantı belirledikten sonra çalıştır.

### Phase 1 Durumu

| Deney | Checkpoint | Durum |
|-------|------------|-------|
| `full_seed42` | `checkpoints/full_seed42/phase1_best.pt` | ⬜ |
| `full_seed123` | `checkpoints/full_seed123/phase1_best.pt` | ⬜ |
| `full_seed456` | `checkpoints/full_seed456/phase1_best.pt` | ⬜ |

### Phase 2 Durumu

| Deney | Checkpoint | Eval | Durum |
|-------|------------|------|-------|
| `full_seed42` | `checkpoints/full_seed42/phase2_best.pt` | `results/full_seed42/eval.json` | ⬜ |
| `full_seed123` | `checkpoints/full_seed123/phase2_best.pt` | `results/full_seed123/eval.json` | ⬜ |
| `full_seed456` | `checkpoints/full_seed456/phase2_best.pt` | `results/full_seed456/eval.json` | ⬜ |

```bash
python src/training/orchestrate.py --only full --seeds 42 123 456
```

**Multi-seed sonuç tablosu:**

| Model | Geodesic (°) | Contact Ratio | MPJPE (mm) |
|-------|--------------|---------------|------------|
| full_seed42 | — | — | — |
| full_seed123 | — | — | — |
| full_seed456 | — | — | — |
| **mean ± std** | — ± — | — ± — | — ± — |

```bash
python src/evaluation/compare_results.py --source hot3d --multi-seed
```

---

## AŞAMA 5 — Değerlendirme & Sunum

### 5-A: Seen vs Unseen Test (B2)

| Eval | Komut | Çıktı | Durum |
|------|-------|-------|-------|
| seen_test | `python src/evaluation/evaluate.py --checkpoint checkpoints/full/phase2_best.pt --split seen_test --source oakink --out results/full_seen_test/eval.json` | `results/full_seen_test/eval.json` | ⬜ |
| unseen_test | `python src/evaluation/evaluate.py --checkpoint checkpoints/full/phase2_best.pt --split unseen_test --source oakink --out results/full_unseen_test/eval.json` | `results/full_unseen_test/eval.json` | ⬜ |

| Split | Geodesic (°) | Contact Ratio | MPJPE (mm) |
|-------|--------------|---------------|------------|
| seen_test (üst sınır) | — | — | — |
| unseen_test (gerçek genelleme) | — | — | — |

### 5-B: Latency Benchmark (C3)

| K | Komut | Çıktı | Durum |
|---|-------|-------|-------|
| k=1 | `python src/evaluation/benchmark_latency.py --checkpoint checkpoints/full/phase2_best.pt --device mps --k 1` | `results/latency_k1_mps.json` | ⬜ |
| k=5 | `python src/evaluation/benchmark_latency.py --checkpoint checkpoints/full/phase2_best.pt --device mps --k 5` | `results/latency_k5_mps.json` | ⬜ |

### 5-C: Görselleştirme (C1)

| Tür | Komut | Çıktı | Durum |
|-----|-------|-------|-------|
| PNG seen | `python src/evaluation/visualize_poses.py --checkpoint checkpoints/full/phase2_best.pt --split seen_test --n 50` | `results/visualizations/seen_test_*.png` | ⬜ |
| PNG unseen | `python src/evaluation/visualize_poses.py --checkpoint checkpoints/full/phase2_best.pt --split unseen_test --n 50` | `results/visualizations/unseen_test_*.png` | ⬜ |
| GIF | `python src/evaluation/make_temporal_gif.py --checkpoint checkpoints/full/phase2_best.pt --format gif --n-seqs 5` | `results/animations/*.gif` | ⬜ |

### 5-D: Genel Karşılaştırma Tablosu

```bash
python src/evaluation/compare_results.py --source hot3d
python src/evaluation/compare_results.py --source oakink
python src/evaluation/compare_results.py --source hot3d --csv > results/ablation_table.csv
```

---

## Hızlı Durum Kontrolü

```bash
# Tamamlanan checkpoint'ler
find checkpoints -name "phase*_best.pt" | sort

# Tamamlanan eval'lar
find results -name "eval.json" | sort

# Tüm sonuçları karşılaştır
python src/evaluation/compare_results.py --source hot3d
```

---

## Paralel Çalıştırma Önerisi

```
Terminal 1: full, ablation_no_vel, ablation_no_aug
Terminal 2: ablation_singleframe_mlp, ablation_gru_t1, ablation_static_only
Terminal 3: ablation_no_obj, ablation_bbox_obj, ablation_no_attention, ablation_no_film
Terminal 4: kl_sweep_* (phase 1 only, hızlı)
Terminal 5: contact_weight_sweep_* (phase 1 + 2)
```

Her deney kendi klasörüne yazdığı için çakışma yoktur.
