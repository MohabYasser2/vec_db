"""
VecDB: High-performance vector database with IVF indexing.
Supports cosine similarity search with memory-efficient operations.
"""
from typing import Annotated
import numpy as np
import os

# Import utilities and components
from utils import DB_SEED_NUMBER, DIMENSION, ELEMENT_SIZE, normalize_batch, get_num_records, read_vector
from index_builder import build_ivf_index
from search import search_ivf


class VecDB:
    """
    Vector database with IVF (Inverted File) indexing for fast similarity search.
    """
    
    def __init__(self, database_file_path="saved_db.dat", index_file_path="index_dir", 
                 new_db=True, db_size=None) -> None:
        """
        Initialize VecDB instance.
        
        Args:
            database_file_path: Path to store/load database file
            index_file_path: Path to store/load index directory
            new_db: If True, create new database; if False, load existing
            db_size: Number of vectors (required if new_db=True)
        """
        print(f"[VecDB] Initializing: db={database_file_path}, index={index_file_path}, new={new_db}")
        
        self.db_path = database_file_path
        self.index_dir = index_file_path
        self.centroids_path = os.path.join(self.index_dir, "centroids.npy")
        self.cluster_sizes_path = os.path.join(self.index_dir, "cluster_sizes.npy")
        self.buckets_dir = os.path.join(self.index_dir, "buckets")
        
        # Backward compatibility
        self.posting_list_path = self.cluster_sizes_path
        self.cluster_ptrs_path = self.cluster_sizes_path
        
        if new_db:
            print(f"[VecDB] Creating new database")
            if db_size is None:
                raise ValueError("db_size required for new database")
            if os.path.exists(self.db_path):
                print(f"[VecDB] Removing existing database file")
                os.remove(self.db_path)
            self.generate_database(db_size)
        else:
            print(f"[VecDB] Loading existing database")
            if not os.path.exists(self.db_path):
                raise FileNotFoundError(f"Database not found: {self.db_path}")
            num_records = get_num_records(self.db_path)
            print(f"[VecDB] Database loaded: {num_records:,} vectors")
    
    def generate_database(self, size: int) -> None:
        """
        Generate random normalized vectors and build IVF index.
        
        Args:
            size: Number of vectors to generate
        """
        print(f"[VecDB] Generating {size:,} random vectors (seed={DB_SEED_NUMBER})")
        rng = np.random.default_rng(DB_SEED_NUMBER)
        vectors = rng.random((size, DIMENSION), dtype=np.float32)
        
        # Normalize all vectors for cosine similarity
        print(f"[VecDB] Normalizing vectors...")
        vectors = normalize_batch(vectors)
        
        self._write_vectors_to_file(vectors)
        self._build_index()
    
    def _write_vectors_to_file(self, vectors: np.ndarray) -> None:
        """
        Write vectors to binary database file.
        
        Args:
            vectors: Matrix of vectors to write
        """
        print(f"[VecDB] Writing {len(vectors):,} vectors to {self.db_path}")
        mmap_vectors = np.memmap(self.db_path, dtype=np.float32, mode='w+', shape=vectors.shape)
        mmap_vectors[:] = vectors[:]
        mmap_vectors.flush()
        del mmap_vectors
        
        file_size_mb = os.path.getsize(self.db_path) / (1024 * 1024)
        print(f"[VecDB] Wrote {file_size_mb:.2f} MB to disk")
    
    def _get_num_records(self) -> int:
        """Get total number of vectors in database."""
        return get_num_records(self.db_path)
    
    def insert_records(self, rows: Annotated[np.ndarray, (int, 64)]):
        """
        Insert new records and rebuild index.
        
        Args:
            rows: Matrix of new vectors to insert
        """
        print(f"[VecDB] Inserting {len(rows):,} new vectors")
        
        # Normalize new vectors
        rows = normalize_batch(rows)
        
        # Append to database file
        num_old = self._get_num_records()
        num_new = len(rows)
        full_shape = (num_old + num_new, DIMENSION)
        
        mmap_vectors = np.memmap(self.db_path, dtype=np.float32, mode='r+', shape=full_shape)
        mmap_vectors[num_old:] = rows
        mmap_vectors.flush()
        del mmap_vectors
        
        print(f"[VecDB] Rebuilding index after insertion")
        self._build_index()
    
    def get_one_row(self, row_num: int) -> np.ndarray:
        """
        Load a single vector from database.
        
        Args:
            row_num: Index of vector to load
            
        Returns:
            Vector as numpy array
        """
        return read_vector(self.db_path, row_num)
    
    def _build_index(self):
        """Build IVF index for the database."""
        num_records = self._get_num_records()
        build_ivf_index(self.db_path, self.index_dir, num_records)
    
    def retrieve(self, query: Annotated[np.ndarray, (1, DIMENSION)], top_k: int = 5):
        """
        Retrieve top-k most similar vectors using IVF index.
        
        Strategy:
          - Use nprobe from ivf_config
          - For each probed cluster, sort row_ids and process in small chunks
          - Each chunk loads a contiguous span from disk via np.fromfile
          - Very low RAM usage with good speed
        
        Args:
            query: Query vector of shape (1, D)
            top_k: Number of results to return
            
        Returns:
            List of vector IDs (row numbers) for top-k most similar vectors
        """
        
        return search_ivf(
            db_path=self.db_path,
            index_dir=self.index_dir,
            buckets_dir=self.buckets_dir,
            centroids_path=self.centroids_path,
            query=query,
            top_k=top_k,
            num_records=self._get_num_records(),
            DIMENSION=DIMENSION,
            ELEMENT_SIZE=ELEMENT_SIZE
        )
