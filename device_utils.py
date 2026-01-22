"""Device configuration utilities for PyTorch NIF"""

import torch


def get_device(device_str: str = "cuda") -> torch.device:
    """Get PyTorch device for computation.

    Args:
        device_str: Either 'cuda' or 'cpu'

    Returns:
        torch.device configured for the requested target
    """
    if device_str.lower() == "cuda":
        if not torch.cuda.is_available():
            print("WARNING: CUDA requested but not available, falling back to CPU")
            return torch.device("cpu")
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
        return torch.device("cuda")
    elif device_str.lower() == "cpu":
        print("Using CPU device")
        return torch.device("cpu")
    else:
        raise ValueError(f"Unknown device: {device_str}. Use 'cuda' or 'cpu'")
