import logging

import torch
from humanize import naturalsize


def log_cuda_mem():
    if torch.cuda.is_available():
        logging.debug(
            "Allocated/Reserved: %s / %s",
            naturalsize(torch.cuda.memory_allocated()),
            naturalsize(torch.cuda.memory_reserved()),
        )
