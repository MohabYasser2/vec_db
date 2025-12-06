"""
K-means clustering implementation for IVF index building.
Uses cosine similarity as distance metric.
"""
import numpy as np
from utils import DB_SEED_NUMBER, normalize_batch


def kmeans_cosine(vectors: np.ndarray, K: int, max_iter: int = 200) -> np.ndarray:
    """
    Perform K-means clustering with cosine similarity.
    
    Args:
        vectors: Matrix of shape (N, D) containing vectors to cluster
        K: Number of clusters
        max_iter: Maximum number of iterations
        
    Returns:
        Centroids of shape (K, D)
    """
    N = len(vectors)
    print(f"[K-means] Starting clustering: {N:,} vectors → {K} clusters (max_iter={max_iter})")
    
    rng = np.random.default_rng(DB_SEED_NUMBER)
    
    # Initialize centroids from random samples
    indices = rng.choice(N, size=K, replace=False)
    centroids = vectors[indices].copy()
    centroids = normalize_batch(centroids)
    print(f"[K-means] Initialized {K} centroids from random samples")
    
    prev_assignments = None
    
    for iteration in range(max_iter):
        # Assign vectors to nearest centroids (using cosine similarity)
        similarities = np.dot(vectors, centroids.T)
        assignments = np.argmax(similarities, axis=1)
        
        # Check convergence
        if prev_assignments is not None and np.array_equal(assignments, prev_assignments):
            print(f"[K-means] Converged at iteration {iteration}")
            break
        prev_assignments = assignments.copy()
        
        # Update centroids
        new_centroids = np.zeros_like(centroids)
        for k in range(K):
            mask = assignments == k
            if np.sum(mask) > 0:
                new_centroids[k] = np.mean(vectors[mask], axis=0)
            else:
                # Reinitialize dead centroid
                new_centroids[k] = vectors[rng.integers(0, N)]
        
        # Normalize centroids
        centroids = normalize_batch(new_centroids)
        
        if iteration % 20 == 0 and iteration > 0:
            print(f"[K-means] Iteration {iteration}/{max_iter}")
    
    print(f"[K-means] Clustering complete with {K} clusters")
    return centroids
