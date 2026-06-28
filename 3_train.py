import time
import torch
import random
from experiments import *

torch.manual_seed(42)
random.seed(42)

if __name__ == '__main__':
    start_time = time.time()

    exp = Experiment()
    exp.run_experiment()
    exp.print_fn(f'Took {time.time() - start_time}')