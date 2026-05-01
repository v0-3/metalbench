import platform

import torch


def is_mps_available() -> bool:
    return bool(torch.backends.mps.is_available())


def require_mps() -> None:
    if not is_mps_available():
        raise RuntimeError("metalbench is MPS-only and requires Apple Metal MPS to be available.")


def describe_environment() -> dict[str, str | bool]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "mps_available": is_mps_available(),
        "mps_built": bool(torch.backends.mps.is_built()),
        "device": "mps",
    }
