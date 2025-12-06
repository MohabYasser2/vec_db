"""
Utility functions for VecDB operations.
Handles logging, normalization, and basic vector operations.
"""
import os
import numpy as np

# Constants
DB_SEED_NUMBER = 42
ELEMENT_SIZE = np.dtype(np.float32).itemsize
DIMENSION = 64

# Logging control
ENABLE_LOGGING = os.environ.get('VECDB_DEBUG', 'false').lower() == 'true'


def log(message: str) -> None:
    """
    Conditional logging based on VECDB_DEBUG environment variable.
    
    Args:
        message: Log message to print
    """
    if ENABLE_LOGGING:
        print(f"[VecDB] {message}")


def normalize_vector(vec: np.ndarray) -> np.ndarray:
    """
    Normalize a vector for cosine similarity.
    
    Args:
        vec: Input vector
        
    Returns:
        Normalized vector (L2 norm = 1)
    """
    norm = np.linalg.norm(vec)
    if norm < 1e-10:
        return vec
    return vec / norm


def normalize_batch(vectors: np.ndarray) -> np.ndarray:
    """
    Normalize a batch of vectors for cosine similarity.
    
    Args:
        vectors: Matrix of shape (N, D) containing N vectors
        
    Returns:
        Normalized vectors (each with L2 norm = 1)
    """
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.where(norms == 0, 1, norms)


def get_num_records(db_path: str) -> int:
    """
    Get total number of vectors in database file.
    
    Args:
        db_path: Path to database file
        
    Returns:
        Number of vectors in database
    """
    return os.path.getsize(db_path) // (DIMENSION * ELEMENT_SIZE)


def read_vector(db_path: str, row_num: int) -> np.ndarray:
    """
    Read a single vector from database using offset-based memmap.
    
    Args:
        db_path: Path to database file
        row_num: Index of vector to read
        
    Returns:
        Vector as numpy array
    """
    offset = row_num * DIMENSION * ELEMENT_SIZE
    mmap_vector = np.memmap(db_path, dtype=np.float32, mode='r', 
                           shape=(1, DIMENSION), offset=offset)
    result = np.array(mmap_vector[0])
    del mmap_vector
    return result
