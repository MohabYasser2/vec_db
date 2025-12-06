"""
IVF search functionality for retrieving nearest neighbors.
Uses chunk-based loading with np.fromfile for minimal RAM usage.
"""
import os
import pickle
import heapq
import numpy as np
from typing import List, Tuple, Annotated
from ivf_config import get_nprobe_high_recall


def search_ivf(db_path: str, index_dir: str, buckets_dir: str, centroids_path: str,
               query: np.ndarray, top_k: int, num_records: int, DIMENSION: int, 
               ELEMENT_SIZE: int) -> List[int]:
    """
    IVF retrieve with:
      - very low RAM usage
      - good speed
      - compatible with existing index (raw DB, bucket pkl, normalized centroids)
    Strategy:
      - use nprobe from ivf_config
      - for each probed cluster:
          * sort its row_ids
          * process in small chunks
          * each chunk loads a small contiguous span from disk via np.fromfile
    
    Args:
        db_path: Path to database file
        index_dir: Path to index directory
        buckets_dir: Path to buckets directory
        centroids_path: Path to centroids file
        query: Query vector of shape (1, D)
        top_k: Number of results to return
        num_records: Total number of vectors in database
        DIMENSION: Vector dimension
        ELEMENT_SIZE: Size of float32 element
        
    Returns:
        List of vector IDs (row numbers) for top-k most similar vectors
    """
    # -------------------------
    # 1) Normalize the query
    # -------------------------
    q = query.reshape(-1).astype(np.float32)
    q_norm = np.linalg.norm(q)
    if q_norm > 1e-12:
        q /= q_norm

    # -------------------------
    # 2) Basic stats + centroids
    # -------------------------
    if num_records == 0:
        return []

    centroids = np.load(centroids_path)  # normalized by _build_index
    K = centroids.shape[0]

    nprobe = min(get_nprobe_high_recall(num_records), K)

    # -------------------------
    # 3) Find top-nprobe centroids
    # -------------------------
    centroid_sims = centroids @ q
    top_centroid_ids = np.argpartition(centroid_sims, -nprobe)[-nprobe:]
    top_centroid_ids = top_centroid_ids[np.argsort(-centroid_sims[top_centroid_ids])]

    # -------------------------
    # 4) Scoring budget
    # -------------------------
    if num_records <= 1_500_000:
        max_vectors = 30_000
    elif num_records <= 12_000_000:
        max_vectors = 50_000
    else:
        max_vectors = 70_000

    # Each block: at most max_span_rows * DIM * 4 bytes
    # 2048 * 64 * 4 ≈ 0.5 MB per block
    max_chunk_size = 1024       # max rows per chunk
    max_span_rows = 2048        # max index span per chunk

    heap: list[tuple[float, int]] = []
    scored = 0

    # -------------------------
    # 5) Scan probed clusters
    # -------------------------
    for cid in top_centroid_ids:
        if scored >= max_vectors and len(heap) >= top_k:
            break

        bucket_path = os.path.join(buckets_dir, f"cluster_{cid}.pkl")
        if not os.path.exists(bucket_path):
            continue

        with open(bucket_path, "rb") as f:
            row_ids = pickle.load(f)

        if not row_ids:
            continue

        # Respect budget
        remaining = max_vectors - scored
        if remaining <= 0:
            break
        if len(row_ids) > remaining:
            row_ids = row_ids[:remaining]

        row_ids_np = np.asarray(row_ids, dtype=np.int64)
        # Sort so nearby ids are processed in the same chunk
        row_ids_sorted = np.sort(row_ids_np)

        start = 0
        n_ids = len(row_ids_sorted)

        while start < n_ids:
            if scored >= max_vectors and len(heap) >= top_k:
                break

            # Build a chunk limited by count and span
            first_row = int(row_ids_sorted[start])
            end = start + 1
            while end < n_ids:
                if end - start >= max_chunk_size:
                    break
                if int(row_ids_sorted[end]) - first_row > max_span_rows:
                    break
                end += 1

            chunk_ids = row_ids_sorted[start:end]

            rmin = int(chunk_ids.min())
            rmax = int(chunk_ids.max())
            length = rmax - rmin + 1

            # Safety guard (should be redundant with max_span_rows)
            if length <= 0:
                start = end
                continue

            # -----------------------------
            # Load block for this chunk
            # -----------------------------
            block = np.fromfile(
                db_path,
                dtype=np.float32,
                count=length * DIMENSION,
                offset=rmin * DIMENSION * ELEMENT_SIZE,
            )
            if block.size != length * DIMENSION:
                # Corrupted or truncated file; skip defensively
                start = end
                continue

            block = block.reshape(length, DIMENSION)

            # Gather the chunk's vectors
            local_idx = chunk_ids - rmin
            vecs = block[local_idx]

            # Cosine similarity = (v·q) / ||v||
            dots = vecs @ q
            norms = np.linalg.norm(vecs, axis=1)
            norms = np.where(norms > 1e-12, norms, 1.0)
            sims = dots / norms

            # Update heap
            for rid, sim in zip(chunk_ids, sims):
                sim = float(sim)
                rid = int(rid)
                if len(heap) < top_k:
                    heapq.heappush(heap, (sim, rid))
                else:
                    if sim > heap[0][0]:
                        heapq.heapreplace(heap, (sim, rid))

            scored += len(chunk_ids)
            start = end

    if not heap:
        return []

    heap.sort(key=lambda x: -x[0])
    return [rid for _, rid in heap[:top_k]]
