import random
import string
import time

from lmwrapper.abstract_predictor import get_mock_predictor
from lmwrapper.caching import clear_cache_dir
from lmwrapper.sqlcache import SqlBackedCache, create_tables
from lmwrapper.structs import LmPrediction, LmPrompt


# Helper function to generate random text
def _profile_helper_generate_random_text(length):
    return "".join(random.choices(string.ascii_letters + string.digits, k=length))


# Function to create cache tables
def _profile_helper_create_cache():
    create_tables()


# Function to delete and recreate cache (used in setup)
def _profile_helper_setup_create_cache():
    clear_cache_dir()
    _profile_helper_create_cache()


# Function to add predictions to the cache
def _profile_helper_add_predictions(cache, predictions):
    start_time = time.time()
    for pred in predictions:
        cache.add_or_set(prediction=pred)
    end_time = time.time()
    return end_time - start_time


# Function to get predictions from the cache
def _profile_helper_get_predictions(cache, prompts):
    start_time = time.time()
    for prompt in prompts:
        cache.get(prompt)
    end_time = time.time()
    return end_time - start_time


def _profile_cache(heavy_objs=False):
    clear_cache_dir()
    metad_size = 100_000 if heavy_objs else 10
    lm = get_mock_predictor(
        lambda prompt: LmPrediction(
            prompt.get_text_as_string_default_form(),
            prompt,
            metad={k: v for k, v in zip(range(100_000), range(100_000), strict=False)},
        ),
    )
    cache = SqlBackedCache(lm)

    # Create a list of 1000 random prompts for testing
    prompts = [
        LmPrompt(_profile_helper_generate_random_text(50 if not heavy_objs else 5000))
        for _ in range(1000)
    ]

    # Generate predictions once
    predictions = [
        lm.predict(prompt.get_text_as_string_default_form()) for prompt in prompts
    ]

    print("Profiling cache creation and usage...")

    # Measure cache creation time
    create_times = []
    for _ in range(5):
        clear_cache_dir()
        start_time = time.time()
        _profile_helper_create_cache()
        end_time = time.time()
        create_times.append(end_time - start_time)
    create_time = sum(create_times) / len(create_times)
    print(f"Cache creation time: {create_time:.4f} seconds")

    # Measure time to add predictions to the cache
    num_prompts = 10000
    add_times = []
    for _ in range(5):
        clear_cache_dir()
        add_time = _profile_helper_add_predictions(cache, predictions)
        add_times.append(add_time)
    add_time_avg = sum(add_times) / len(add_times)
    print(f"Adding predictions time: {add_time_avg / num_prompts:.4f} seconds/prompt")

    # Measure time to get predictions from the cache
    get_times = []
    for _ in range(5):
        clear_cache_dir()
        get_time = _profile_helper_get_predictions(cache, prompts)
        get_times.append(get_time)
    get_time_avg = sum(get_times) / len(get_times)
    print(f"Getting predictions time: {get_time_avg / num_prompts:.4f} seconds/prompt")


def main():
    _profile_cache(heavy_objs=True)


if __name__ == "__main__":
    main()
