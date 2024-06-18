from lmwrapper.utils import StrEnum


class CompletionWindow(StrEnum):
    """The"""

    ASAP = "asap"
    BATCH_ANY = "batch_any"
    """Uses the batch api willing to accept any latency.
    What this means might depend on the API. For example,
    OpenAI provides a 24hr target guarantee. However, if
    the number of inputs exceeds the user's daily queue limit,
    then this might have to be split over multiple days
    (more usage limit can be purchased by adding credits).
    Thus, this completion window doesn't make any guarantees,
    but it is often moderately fast.
    """
