"""Process-level parallelism for sweeping (seed, model, constraint) configurations."""

import os
import copy
import random
import itertools
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import torch

from tmo.data import split_dataset
from tmo.env import M4A1_Env
from tmo.trainer import process_model

try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    # The start method has already been set elsewhere (e.g., in a parent process).
    pass


def run_one(gpu_id, seed, model_key, model_cls, dataset, weights,
            local_device, cloud_server, latency_budget, usage_budget,
            resource_constraint, time_span, constraint_distribution,
            num_modalities=3, task_vocab=None):
    """Train + evaluate a single (model, seed) configuration on ``gpu_id``."""
    random.seed(seed)
    np.random.seed(seed)
    train_dataset, test_dataset = split_dataset(dataset, test_ratio=0.2)
    Train = copy.deepcopy({"rewards": None, "nn": None})
    train_env = M4A1_Env(train_dataset, weights, local_device, cloud_server,
                         latency_budget, usage_budget, resource_constraint, time_span,
                         Train=True, constraint_distribution=constraint_distribution,
                         num_modalities=num_modalities, task_vocab=task_vocab)
    Train['rewards'] = train_env.rewards
    Train['nn'] = train_env.nn
    test_env = M4A1_Env(test_dataset, weights, local_device, cloud_server,
                        latency_budget, usage_budget, resource_constraint, time_span,
                        Train, constraint_distribution=constraint_distribution,
                        num_modalities=num_modalities, task_vocab=task_vocab)

    device = "cuda:" + str(gpu_id) if torch.cuda.is_available() else "cpu"
    local_results = {}
    process_model(model_key, model_cls, train_env, test_env,
                  latency_budget, usage_budget, local_results, device=device)
    return local_results


def run_parallel(task_args_list, fn, num_gpus=None):
    """Submit ``fn(gpu_id, *args)`` for every entry in ``task_args_list`` in parallel."""
    results = {}
    if num_gpus is None:
        num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    if num_gpus < 1:
        num_gpus = 1
    gpu_cycle = itertools.cycle(range(num_gpus))
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as ex:
        futures = []
        for task_args in task_args_list:
            model_cls = task_args[2] if len(task_args) >= 3 else None
            gpu_id = next(gpu_cycle) if model_cls is not None else 0
            futures.append(ex.submit(fn, gpu_id, *task_args))
        for f in as_completed(futures):
            r = f.result()
            for k, v in r.items():
                results.setdefault(k, []).extend(v)
    return results
