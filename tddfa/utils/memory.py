import torch
import gc

def empty_cache_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()