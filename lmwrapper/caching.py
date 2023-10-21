from pathlib import Path

import diskcache
from joblib import Memory

cur_file = Path(__file__).parent.absolute()

_set_cache_dir = None


def set_cache_dir(path: Path):
    """
    Sets the caching directory. Note it does not affect any already constructed
    models
    """
    global _set_cache_dir
    if path.exists():
        _verify_looks_like_cache_dir(path)
    path.mkdir(parents=True, exist_ok=True)
    _set_cache_dir = path


def _verify_looks_like_cache_dir(path: Path):
    for file_or_directory in path.rglob("*"):
        if file_or_directory.is_file():
            if file_or_directory.suffix in (".py", ".txt", ".md"):
                msg = (
                    "Attempting to set cache directory to what appears to be a "
                    "source directory. Instead set it to its own subdirectory inside"
                    "the source repository."
                )
                raise ValueError(msg)


def _cache_dir() -> Path:
    if _set_cache_dir is not None:
        return _set_cache_dir
    return Path.cwd() / ".lmwrapper_cache"


def _get_disk_cache_joblib() -> Memory:
    return Memory(_cache_dir(), verbose=0)


def _get_disk_cache_diskcache() -> diskcache.FanoutCache:
    return diskcache.FanoutCache(
        str(_cache_dir()),
        timeout=int(9e9),
        size_limit=50e9,
        shards=4,
        eviction_policy="none",
    )


def get_disk_cache():
    # TODO: some kind of version number in the cache so can have migrations
    return _get_disk_cache_diskcache()


def clear_cache_dir():
    import shutil

    shutil.rmtree(_cache_dir())
