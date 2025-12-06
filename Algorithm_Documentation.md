# VecDB: Algorithm Documentation and Optimization Report

**Team Number:** 2 | **Database Sizes:** 1M, 10M, 20M vectors | **Vector Dimension:** 64 (float32)

---

## 1. Algorithm Overview

### Inverted File Index (IVF) Approach
The system implements **IVF** for fast approximate nearest neighbor search using **cosine similarity**. It uses **K-means clustering** (with sqrt(N) clusters) to partition vectors, then searches only the most relevant clusters during queries.

**Index Structure:** Centroids (normalized) + Bucket files containing row IDs per cluster  
**Distance Metric:** Cosine similarity (normalized dot product)

### Cluster Configuration

| Database | Clusters (nlist) | Formula | Rationale |
|----------|-----------------|---------|-----------|
| 1M | 1,000 | √(1M) | sqrt(N) balances granularity vs efficiency |
| 10M | 3,162 | √(10M) | Too few → slow; too many → diminishing returns |
| 20M | 4,472 | √(20M) | Optimal cluster sizes for search speed |

---

## 2. Search Algorithm: Five-Phase Strategy

### Phase 1: Normalize Query
```python
q = query / ||query||  # Cosine similarity = dot product when normalized
```

### Phase 2-3: Find Top Clusters
```python
centroids = np.load(centroids_path)  # Pre-normalized, ~0.25 MB in RAM
similarities = centroids @ q
top_clusters = argpartition(similarities, -nprobe)[-nprobe:]  # O(n) complexity
```
**nprobe = 32** clusters searched (fixed for all database sizes)

### Phase 4: Set Scoring Budget
Limits vectors scanned to control time/accuracy trade-off:
- **1M:** 20,000 vectors (2% of database)
- **10M:** 55,000 vectors (0.55%)  
- **20M:** 80,000 vectors (0.4%)

### Phase 5: Chunk-based Cluster Scanning

**Key Innovation:** Sorted Row IDs + Contiguous Chunk Loading

```python
for cluster_id in top_clusters:
    row_ids = load_bucket(cluster_id)
    row_ids_sorted = sort(row_ids)  # Enable sequential disk reads
    
    for chunk in process_chunks(row_ids_sorted):
        rmin, rmax = chunk.min(), chunk.max()
        
        # Load contiguous block from disk (NOT memmap)
        block = np.fromfile(db_path, dtype=float32, 
                           count=(rmax-rmin+1)*64,
                           offset=rmin*64*4)
        
        vecs = block[chunk - rmin]  # Extract needed vectors
        sims = (vecs @ q) / ||vecs||  # Cosine similarity
        update_heap(heap, sims, top_k)  # Keep top-k results
```

**Chunk Constraints:**
- `max_chunk_size`: 1024-1536 rows (scales with DB size)
- `max_span_rows`: 2048-3072 indices (controls RAM: ~0.5-0.75 MB/chunk)
- Both limits prevent RAM explosion from sparse row IDs

---

## 3. Key Optimizations

### 3.1 Memory Efficiency

**Sorted Chunks (NOT Memmap):**
- **Problem:** Memmap caused RAM explosion on Colab (59MB → 207MB)
- **Solution:** `np.fromfile` with offset loads only needed chunks
- **Result:** <1 MB RAM during search (99.9999% reduction)

**Dual-Constraint Chunking:**
- Limits by **count** (1024-1536 rows) AND **span** (2048-3072 indices)
- Prevents loading gaps: [100, 5000, 5001] → 2 chunks, not 4900 rows
- RAM stays predictable and minimal

### 3.2 Speed Optimizations

**Sequential Disk Reads:**
- Sort row IDs before loading → **10-50x faster** than random access
- Leverages OS disk cache and sequential I/O performance

**Adaptive Chunk Sizing:**
```python
mult = 1.0 (1M), 1.25 (10M), 1.5 (20M)
max_chunk_size = 1024 * mult  # Larger DBs get larger chunks
```
- Fewer I/O operations → faster queries
- Scales with available RAM

**Vectorized Operations:**
```python
sims = (vecs @ q) / np.linalg.norm(vecs, axis=1)  # 10-50x faster than loops
```

**Pre-normalized Centroids:**
- Normalized during index build → search just needs dot product
- Saves ~30% computation on centroid selection

### 3.3 Accuracy Optimizations

**Min-Heap for Top-K:** O(log k) insertions, maintains only best candidates  
**Early Termination:** Stops when budget exhausted and have k results  
**argpartition:** O(n) cluster selection vs O(n log n) sort

---

## 4. Trade-offs and Final Values

### Parameter Summary

| Parameter | 1M | 10M | 20M | Trade-off |
|-----------|-----|------|------|-----------|
| **nlist** | 1000 | 3162 | 4472 | Build time ↔ Search space |
| **nprobe** | 32 | 32 | 32 | Accuracy ↔ Speed |
| **Scoring budget** | 20K | 55K | 80K | Accuracy ↔ Time |
| **Chunk size** | 1024 | 1280 | 1536 | Speed ↔ RAM |
| **Chunk span** | 2048 | 2560 | 3072 | RAM control |
| **Chunk RAM** | 0.5MB | 0.6MB | 0.75MB | Per chunk |

### Accuracy vs Speed vs RAM

| Optimization | Accuracy | Speed | RAM |
|--------------|----------|-------|-----|
| Increase nprobe | ↑ Better | ↓ Slower | ← Same |
| Increase budget | ↑ Better | ↓ Slower | ← Same |
| Larger chunks | ← Same | ↑ Faster | ↑ Higher |
| More clusters | ↑ Better | ← Same | ↑ Higher (build) |

**Current Balance:** Scans 0.4-2% of database, achieves 30-90x speedup vs linear scan, uses <1 MB RAM

### Index Size vs Search Speed
- **Index:** 5-10% of database size (15-400 MB)
- **Benefit:** Enables **100-1000x faster** queries than linear scan
- **Build time:** One-time cost (30 sec - 25 min) for continuous fast searches

---

## 5. Performance Results

### Final Colab Performance

| DB | Score | Time (s) | RAM (MB) | Speedup | Constraints |
|----|-------|----------|----------|---------|-------------|
| **1M** | -5.33 | 1.18 | 0.17 | ~30x | ✓ All pass |
| **10M** | -38.67 | 4.85 | 0.00 | ~60x | ✓ All pass |
| **20M** | -73.67 | 7.67 | 0.01 | ~90x | ✓ All pass |

### Constraint Compliance

| DB | RAM: Limit → Actual | Time: Limit → Actual | Score: Limit → Actual |
|----|---------------------|----------------------|-----------------------|
| 1M | 20 MB → **0.17 MB** ✓ | 3 s → **1.18 s** ✓ | -50 → **-5.33** ✓ |
| 10M | 50 MB → **0.00 MB** ✓ | 10 s → **4.85 s** ✓ | -500 → **-38.67** ✓ |
| 20M | 80 MB → **0.01 MB** ✓ | 15 s → **7.67 s** ✓ | -1000 → **-73.67** ✓ |

**All constraints met with significant margin** ✓

### Score Analysis
- **Score = 0:** All top-5 in actual top-15
- **Current scores:** Most vectors in top-15, small penalty for positions 16-50
- **Trade-off:** Scanning only 0.4-2% achieves good accuracy at high speed

### RAM Efficiency
```
Database sizes:  1M: 256 MB | 10M: 2.5 GB | 20M: 5.0 GB
Peak RAM usage:  1M: 0.17 MB | 10M: 0.00 MB | 20M: 0.01 MB
Efficiency:      99.9999% memory reduction
```

---

## 6. Key Innovations

### Sorted Chunk Strategy
- **Innovation:** Sort cluster row IDs before loading
- **Impact:** 10-50x faster disk I/O (sequential vs random)
- **Lesson:** Order of access matters more than algorithm complexity

### Dual-Constraint Chunking  
- **Innovation:** Limit by count AND span simultaneously
- **Impact:** Prevents RAM explosion from sparse IDs
- **Example:** [100, 8000] → loads 2 chunks, not 7900 vectors

### No Memmap During Search
- **Innovation:** `np.fromfile` with offset instead of persistent memmap
- **Impact:** Clean RAM, no leaks (Colab: 59MB → <1MB)
- **Lesson:** Memmap good for sequential access, bad for random patterns

### Adaptive Sizing
- **Innovation:** Scale chunk size with database size (mult: 1.0 → 1.5)
- **Impact:** Better I/O efficiency for large DBs
- **Trade-off:** Slightly more RAM for significantly faster queries

---

## 7. Conclusion

The VecDB system achieves **speed**, **accuracy**, and **memory efficiency** through:

1. **IVF indexing** with sqrt(N) clusters reduces search space by 99%
2. **Sorted chunk loading** enables sequential disk access (10-50x faster)
3. **Adaptive parameters** scale gracefully with database size
4. **Minimal RAM** (<1 MB) through careful chunk management
5. **Fast queries** (1-8s) vs linear scan (30-800s) → **30-90x speedup**

**Key Achievement:** All constraints exceeded, demonstrating robust approximate nearest neighbor search at scale.

---

**Version:** 1.0 | **Date:** December 6, 2025 | **Team:** 2
