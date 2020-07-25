import os
import random

import numpy as np
import torch

def fix_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def create_dirs(base):
    if len(base) == 0:
        base = '.'
    log_dir = f'{base}/logs'
    model_dir = f'{base}/models'
    data_dir = f'{base}/data'
    for dir in [log_dir, model_dir, data_dir]:
        if not os.path.exists(dir):
            os.makedirs(dir)
