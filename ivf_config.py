"""
IVFFLAT Configuration for different database sizes
Optimized parameters based on database size
Using sqrt(N) heuristic for nlist
"""
import math

# Configuration lookup table based on number of records
IVF_CONFIG = {
    # 1M Database: sqrt(1M) = 1000
    1_000_000: {
        'nlist': 1000,
        'nprobe_fast': 31,
        'nprobe_high_recall': 32
    },
    # 10M Database: sqrt(10M) ≈ 3162
    10_000_000: {
        'nlist': 3162,
        'nprobe_fast': 56,
        'nprobe_high_recall': 32
    },
    # 20M Database: sqrt(20M) ≈ 4472
    20_000_000: {
        'nlist': 4472,
        'nprobe_fast': 66,
        'nprobe_high_recall': 32
    }
}

def get_ivf_config(num_records):
    """
    Get IVF configuration based on number of records.
    Returns the configuration for the closest database size.
    
    Args:
        num_records: Number of records in the database
        
    Returns:
        dict: Configuration with nlist, nprobe_fast, nprobe_high_recall
    """
    # Find the closest configuration
    if num_records <= 1_000_000:
        return IVF_CONFIG[1_000_000]
    elif num_records <= 10_000_000:
        return IVF_CONFIG[10_000_000]
    else:
        return IVF_CONFIG[20_000_000]

def get_nlist(num_records):
    """Get number of clusters (nlist) for given database size using sqrt(N) heuristic"""
    config = get_ivf_config(num_records)
    return config['nlist']

def get_nprobe_fast(num_records):
    """Get nprobe for fast retrieval (prioritizes speed)"""
    config = get_ivf_config(num_records)
    return config['nprobe_fast']

def get_nprobe_high_recall(num_records):
    """Get nprobe for high recall retrieval (prioritizes accuracy)"""
    config = get_ivf_config(num_records)
    return config['nprobe_high_recall']
