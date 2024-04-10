import torch
from torch import backends

def get_device() -> torch.device:
    if backends.mps.is_available():
        print(f"Device: METAL")
        return torch.device("mps")
    elif torch.cuda.is_available():
        print(f"Device: CUDA")
        return torch.device("cuda")
    else:
        print(f"Device: CPU")
        return torch.device("cpu")