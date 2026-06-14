"""
utils/seed.py
-------------
Global reproducibility helper.

Usage
-----
    from src.utils.seed import seed_everything
    seed_everything(42)

Sets seeds for Python random, NumPy, PyTorch (CPU + CUDA), and configures
PyTorch determinism flags.
"""

import os
import random
import numpy as np
import torch
from src.utils.logger import get_logger

log = get_logger(__name__)


def seed_everything(seed: int = 42, deterministic: bool = True) -> None:
    """
    Seed all relevant RNG sources for reproducibility.

    Parameters
    ----------
    seed         : Integer seed value.
    deterministic: If True, set CUDA deterministic algorithms (may slow
                   training slightly on some ops).
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # multi-GPU

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Available in PyTorch >= 1.8
        try:
            torch.use_deterministic_algorithms(True)
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        except AttributeError:
            pass  # older PyTorch — skip

    log.info(f"Global seed set to {seed} | deterministic={deterministic}")
