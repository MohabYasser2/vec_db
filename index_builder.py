"""
IVF (Inverted File) index building functionality.
Handles index creation, storage, and cluster management.
"""
import os
import pickle
import numpy as np
from typing import Tuple, List
from utils import DB_SEED_NUMBER, read_vector, log
from kmeans import kmeans_cosine
from ivf_config import get_nlist


def build_ivf_index(db_path: str, index_dir: str, num_records: int) -> None:
    """
    Build IVF index with bucket-based storage.
    
    Args:
        db_path: Path to database file
        index_dir: Directory to store index files
        num_records: Total number of vectors in database
    """
    K = get_nlist(num_records)
    buckets_dir = os.path.join(index_dir, "buckets")
    
    print(f"[Index] Building IVF index: {num_records:,} vectors → {K} clusters")
    
    # Create directory structure
    os.makedirs(buckets_dir, exist_ok=True)
    log(f"Created buckets directory: {buckets_dir}")
    
    # Sample vectors for k-means
    sample_size = min(300000, num_records)
    sample_vectors = _sample_vectors(db_path, num_records, sample_size)
    log(f"Sampled {sample_size:,} vectors for k-means training")
    
    # Run k-means clustering
    centroids = kmeans_cosine(sample_vectors, K)
    
    # Assign all vectors to clusters
    cluster_buckets = _assign_vectors_to_clusters(db_path, num_records, centroids)
    
    # Save cluster buckets
    _save_cluster_buckets(buckets_dir, cluster_buckets, K)
    
    # Save centroids and metadata
    _save_index_metadata(index_dir, centroids, cluster_buckets)
    
    print(f"[Index] Build complete: {K} clusters, avg size = {num_records/K:.1f}")


def _sample_vectors(db_path: str, num_records: int, sample_size: int) -> np.ndarray:
    """
    Sample random vectors from database for k-means training.
    
    Args:
        db_path: Path to database file
        num_records: Total number of vectors
        sample_size: Number of vectors to sample
        
    Returns:
        Matrix of sampled vectors
    """
    rng = np.random.default_rng(DB_SEED_NUMBER)
    sample_indices = rng.choice(num_records, size=sample_size, replace=False)
    
    print(f"[Index] Loading {sample_size:,} sample vectors from disk...")
    sample_vectors = np.zeros((sample_size, 64), dtype=np.float32)
    
    for i, idx in enumerate(sample_indices):
        sample_vectors[i] = read_vector(db_path, idx)
        if (i + 1) % 100000 == 0:
            log(f"Loaded {i + 1:,} / {sample_size:,} samples")
    
    return sample_vectors


def _assign_vectors_to_clusters(db_path: str, num_records: int, 
                                centroids: np.ndarray) -> List[List[int]]:
    """
    Assign all vectors to their nearest cluster.
    
    Args:
        db_path: Path to database file
        num_records: Total number of vectors
        centroids: Cluster centroids
        
    Returns:
        List of lists, where each inner list contains vector IDs for a cluster
    """
    K = len(centroids)
    cluster_buckets = [[] for _ in range(K)]
    batch_size = 50000
    
    print(f"[Index] Assigning {num_records:,} vectors to {K} clusters (batch_size={batch_size:,})...")
    
    for batch_start in range(0, num_records, batch_size):
        batch_end = min(batch_start + batch_size, num_records)
        batch_size_actual = batch_end - batch_start
        
        # Load batch
        batch_vectors = np.zeros((batch_size_actual, 64), dtype=np.float32)
        for i in range(batch_start, batch_end):
            batch_vectors[i - batch_start] = read_vector(db_path, i)
        
        # Assign to clusters
        similarities = np.dot(batch_vectors, centroids.T)
        nearest_clusters = np.argmax(similarities, axis=1)
        
        for i, cluster_id in enumerate(nearest_clusters):
            cluster_buckets[cluster_id].append(batch_start + i)
        
        if batch_end % 1000000 == 0 or batch_end == num_records:
            print(f"[Index] Assigned {batch_end:,} / {num_records:,} vectors")
    
    return cluster_buckets


def _save_cluster_buckets(buckets_dir: str, cluster_buckets: List[List[int]], K: int) -> None:
    """
    Save cluster buckets to individual pickle files.
    
    Args:
        buckets_dir: Directory to store bucket files
        cluster_buckets: List of vector ID lists for each cluster
        K: Number of clusters
    """
    print(f"[Index] Saving {K} cluster buckets to {buckets_dir}...")
    
    for cluster_id, bucket in enumerate(cluster_buckets):
        bucket_path = os.path.join(buckets_dir, f"cluster_{cluster_id}.pkl")
        with open(bucket_path, 'wb') as f:
            pickle.dump(bucket, f)
        
        if (cluster_id + 1) % 500 == 0:
            log(f"Saved {cluster_id + 1:,} / {K} buckets")
    
    log(f"All {K} buckets saved")


def _save_index_metadata(index_dir: str, centroids: np.ndarray, 
                        cluster_buckets: List[List[int]]) -> None:
    """
    Save index metadata (centroids and cluster sizes).
    
    Args:
        index_dir: Directory to store metadata
        centroids: Cluster centroids
        cluster_buckets: List of vector ID lists for each cluster
    """
    centroids_path = os.path.join(index_dir, "centroids.npy")
    cluster_sizes_path = os.path.join(index_dir, "cluster_sizes.npy")
    
    # Save centroids
    np.save(centroids_path, centroids.astype(np.float32))
    log(f"Saved centroids to {centroids_path}")
    
    # Save cluster sizes
    cluster_sizes = np.array([len(bucket) for bucket in cluster_buckets], dtype=np.int32)
    np.save(cluster_sizes_path, cluster_sizes)
    log(f"Saved cluster sizes to {cluster_sizes_path}")
    
    # Log statistics
    avg_size = np.mean(cluster_sizes)
    min_size = np.min(cluster_sizes)
    max_size = np.max(cluster_sizes)
    print(f"[Index] Cluster stats: avg={avg_size:.1f}, min={min_size}, max={max_size}")
