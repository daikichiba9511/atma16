import contextlib
import math
import os
import random
import time

import numpy as np
import psutil


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True  # type: ignore
    # torch.backends.cudnn.benchmark = False  # type: ignore
    # torch.autograd.anomaly_mode.set_detect_anomaly(False)


@contextlib.contextmanager
def trace(title: str):
    t0 = time.time()
    p = psutil.Process(os.getpid())
    m0 = p.memory_info().rss / 2.0**30
    yield
    m1 = p.memory_info().rss / 2.0**30
    delta_mem = m1 - m0
    sign = "+" if delta_mem >= 0 else "-"
    delta_mem = math.fabs(delta_mem)
    print(f"{title}: {m1:.2f}GB({sign}{delta_mem:.2f}GB):{time.time() - t0:.3f}s")
