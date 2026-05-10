"""Reproduce the main results table from the paper.

This script sweeps across (seed, resource_constraint, model) configurations
and dispatches them in parallel through ``tmo.run_parallel``.

Typical usage from the repository root::

    python scripts/run_main_experiment.py --num_gpus 4 --repeat 3

The pickled results are written to ``<results_dir>/Main_Results.pkl``.
"""

import json
import os
import pickle
import random
import warnings

from stable_baselines3 import PPO, A2C, DQN

from tmo import args_parser, run_one, run_parallel

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)


def build_task_args(dataset, args, seeds, models, resource_constraints):
    """Materialise the cartesian product (seed, constraint, model) into task args."""
    weights = [args.alpha, args.beta_association, args.beta_latency, args.beta_usage]
    task_args_list = []
    for seed in seeds:
        for resource_constraint in resource_constraints:
            for model_name, model_cls in models.items():
                if model_name in ('Random', 'Local', 'Cloud') and resource_constraint:
                    continue
                model_key = (resource_constraint, model_name, model_cls)
                task_args_list.append((
                    seed, model_key, model_cls, dataset, weights,
                    args.local_device, args.cloud_server,
                    args.latency_budget, args.usage_budget,
                    resource_constraint, args.time_span,
                    args.constraint_distribution,
                    args.num_modalities,
                ))
    return task_args_list


def main():
    args = args_parser()

    with open(args.dataset_path, 'r') as f:
        dataset = json.load(f)

    models = {'Random': None, 'Local': None, 'Cloud': None,
              'PPO': PPO, 'A2C': A2C, 'DQN': DQN}

    seeds = [random.randint(1, 10000) for _ in range(args.repeat)]
    resource_constraints = [False, True]

    task_args_list = build_task_args(dataset, args, seeds, models, resource_constraints)
    results = run_parallel(task_args_list, run_one, num_gpus=args.num_gpus)

    os.makedirs(args.results_dir, exist_ok=True)
    out_path = os.path.join(args.results_dir, 'Main_Results.pkl')
    with open(out_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"Wrote {len(results)} configurations to {out_path}.")


if __name__ == '__main__':
    main()
