import torch


def release_memory():
    """Release all GPU memory cache."""
    if torch.cuda.is_available():
        with torch.cuda.device('cuda:0'):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
