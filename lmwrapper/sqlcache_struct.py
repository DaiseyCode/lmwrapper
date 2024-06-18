from dataclasses import dataclass


@dataclass
class BatchPredictionPlaceholder:
    """
    Used to represent a prediction that got batched
    but has not had its results completed or fetched yet
    """

    batch_id: str
    text_and_sample_hash: str
    api_id: str
    api_category: str
    status: str
    waiting_for_a_result: bool
    batch_total_inputs: int
