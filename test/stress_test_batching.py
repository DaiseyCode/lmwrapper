from lmwrapper.caching import clear_cache_dir
from lmwrapper.openai_wrapper import get_open_ai_lm
from lmwrapper.openai_wrapper.batching import OpenAiBatchManager
from lmwrapper.sqlcache import SqlBackedCache
from lmwrapper.structs import LmPrompt


def over50k():
    clear_cache_dir()
    cache = SqlBackedCache(lm=get_open_ai_lm())
    batching_manager = OpenAiBatchManager(
        [
            LmPrompt("a", cache=True, max_tokens=1)
            for _ in range(60000)
        ],
        cache=cache,
    )
    batching_manager.start_batch()
    print(batching_manager)


if __name__ == "__main__":
    over50k()