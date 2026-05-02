"""GPU selection utilities."""
import os
import subprocess


def free_device(min_free_mib: int = 300) -> str:
    """Return the lowest-index CUDA device with at least *min_free_mib* MiB free.

    Falls back to ``"cpu"`` when CUDA is unavailable or no GPU meets the
    threshold.  Respects ``CUDA_VISIBLE_DEVICES`` when set.
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return "cpu"
    except ImportError:
        return "cpu"

    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "cpu"

    free_mib = [int(x.strip()) for x in out.strip().splitlines()]

    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if visible:
        physical_indices = [int(i) for i in visible.split(",") if i.strip().isdigit()]
    else:
        physical_indices = list(range(len(free_mib)))

    for virtual_idx, physical_idx in enumerate(physical_indices):
        if physical_idx < len(free_mib) and free_mib[physical_idx] >= min_free_mib:
            return f"cuda:{virtual_idx}"

    return "cpu"


def free_gpu_id(min_free_mib: int = 300) -> int:
    """Return the physical GPU index with at least *min_free_mib* MiB free.

    Unlike ``free_device()``, returns the raw integer suitable for passing
    to RecBole's ``gpu_id`` config key, which sets ``CUDA_VISIBLE_DEVICES``
    directly to that integer.
    """
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return 0

    free_mib = [int(x.strip()) for x in out.strip().splitlines()]

    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if visible:
        physical_indices = [int(i) for i in visible.split(",") if i.strip().isdigit()]
    else:
        physical_indices = list(range(len(free_mib)))

    for physical_idx in physical_indices:
        if physical_idx < len(free_mib) and free_mib[physical_idx] >= min_free_mib:
            return physical_idx

    return physical_indices[0] if physical_indices else 0
