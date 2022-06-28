from pathlib import Path
from joblib import Memory

cur_file = Path(__file__).parent.absolute()


def _cache_dir():
    return cur_file / '../lm_model_cache'


def get_disk_cache() -> Memory:
    diskcache = Memory(_cache_dir(), verbose=0)
    return diskcache


def clear_cache_dir():
    _cache_dir().rmdir()

