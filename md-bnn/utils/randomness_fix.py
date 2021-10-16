import random
import numpy as np
import torch
RANDOM_SEED = 123456


##### for fixing the randomness to aid reproducibility
def seed_torch():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def init_fn(worker_id):
    random.seed(RANDOM_SEED + worker_id)
    np.random.seed(RANDOM_SEED + worker_id)
