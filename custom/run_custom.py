"""Custom-scenario sweep for TMO across different configurations.

The dataset is synthetic (response score, association score, latency, and
monetary cost are all set explicitly in this file -- see ``custom/README.md``)
so the demo runs without the M4A1 dataset.

Usage::
    python custom/run_custom.py --policy A2C --num_gpus 4 --repeat 3
"""

import argparse
import os
import pickle
import random
import warnings

import numpy as np
import torch
from stable_baselines3 import PPO, A2C, DQN

from tmo import M4A1_Env, evaluate, run_parallel

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

DEFAULT_RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')

POLICIES = {'PPO': PPO, 'A2C': A2C, 'DQN': DQN}

# ===========================================================================
# Configurations exercised by the demo
# ===========================================================================

# Each entry is (num_modalities, num_tasks, num_turns, label).
CONFIGS = [
    (3,  4,  5,  "M4A1-shape"),
    (2,  2,  2,  "Small"),
    (4,  4,  4,  "Medium"),
    (6,  6,  6,  "Large"),
    (10, 3,  3,  "More Modalities"),
    (3,  10, 3,  "More Tasks"),
    (3,  3,  10, "Long History"),
]

BASELINES = ['Random', 'Local', 'Cloud']
RC_OPTIONS = [False, True]


# ===========================================================================
# Synthetic dataset / cloud profile builders
# ===========================================================================

def _synth_episode(rng, num_modalities, num_tasks, num_turns, task_vocab,
                   cloud_prob=0.5, upload_prob=0.7):
    interactions = []
    association_score = []
    cum_uploaded = 0
    for _ in range(num_turns):
        task_cat = task_vocab[int(rng.integers(num_tasks))]

        if rng.random() < cloud_prob:
            action = 1
            if rng.random() < upload_prob:
                k = int(rng.integers(1, num_modalities + 1))
                image_index = sorted(
                    rng.choice(num_modalities, size=k, replace=False).tolist()
                )
            else:
                image_index = None
        else:
            action = 0
            image_index = None

        if image_index is not None:
            cum_uploaded += len(image_index)

        if action == 0:
            score_mean = 0.7
        else:
            score_mean = 1.0 + 0.1 * cum_uploaded
        score = float(np.clip(rng.normal(loc=score_mean, scale=0.3), 0.0, 2.0))
        interactions.append({
            "task_cat": task_cat,
            "action": action,
            "image_index": image_index,
            "answer": "Synthetic response.",
            "score": score,
        })
        association_score.append(
            [float(rng.uniform(0.1, 0.5)) for _ in range(num_modalities)]
        )
    return {"interactions": interactions, "association_score": association_score}


def synth_dataset(num_episodes, num_turns, num_modalities, num_tasks, rng):
    task_vocab = [f"task_{i:02d}" for i in range(num_tasks)]
    dataset = []
    for _ in range(num_episodes):
        dataset.append(_synth_episode(rng, num_modalities, num_tasks,
                                      num_turns, task_vocab))
    return dataset, task_vocab


def synth_cloud_profile(num_modalities,
                        base_latency=0.5, latency_per_modality=1.5,
                        base_cost=0.0005, cost_per_modality=0.004):
    cloud_time = [base_latency + i * latency_per_modality
                  for i in range(num_modalities + 1)]
    cloud_usage_cost = [base_cost + i * cost_per_modality
                        for i in range(num_modalities + 1)]
    return {
        "Synth": {
            "cloud_time": cloud_time,
            "cloud_usage_cost": cloud_usage_cost,
        }
    }


# ===========================================================================
# Per-(config, seed, RC, model) worker -- mirrors tmo.run_one's signature so
# it can be dispatched through tmo.run_parallel.
# ===========================================================================

def run_one_custom(gpu_id, seed, model_key, model_cls,
                   weights, num_episodes, time_span,
                   latency_budget, usage_budget, resource_constraint):
    num_modalities, num_tasks, num_turns = model_key[0]
    random.seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    dataset, task_vocab = synth_dataset(num_episodes, num_turns,
                                        num_modalities, num_tasks, rng)
    cloud_registry = synth_cloud_profile(num_modalities)

    common_kwargs = dict(
        weights=weights,
        local_device='Jetson TX2',
        cloud_server='Synth',
        latency_budget=latency_budget,
        usage_budget=usage_budget,
        Resource_Constraint=resource_constraint,
        time_span=time_span,
        num_modalities=num_modalities,
        task_vocab=task_vocab,
        cloud_registry=cloud_registry,
    )

    train_env = M4A1_Env(dataset, Train=True, **common_kwargs)
    Train = {'rewards': train_env.rewards, 'nn': train_env.nn}
    test_env = M4A1_Env(dataset, Train=Train, **common_kwargs)

    model_name = model_key[2]
    if model_cls is None:
        out = evaluate(test_env, latency_budget=latency_budget,
                       usage_budget=usage_budget, name=model_name)
    else:
        device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
        model = model_cls('MlpPolicy', train_env, device=device, verbose=0,
                          seed=seed)
        model.learn(total_timesteps=30000)
        out = evaluate(test_env, latency_budget=latency_budget,
                       usage_budget=usage_budget, model=model, name=model_name)

    return {model_key: [out]}


# ===========================================================================
# Build the (config x RC x seed x model) task grid
# ===========================================================================

def build_task_args(args, seeds):
    """Build the ``(config x seed x model x RC)`` task grid."""
    weights = [args.alpha, args.beta_association,
               args.beta_latency, args.beta_usage]
    policy_cls = POLICIES[args.policy]
    task_args_list = []
    for num_modalities, num_tasks, num_turns, _label in CONFIGS:
        config = (num_modalities, num_tasks, num_turns)
        for seed in seeds:
            for name in BASELINES:
                model_key = (config, False, name)
                task_args_list.append((
                    seed, model_key, None,
                    weights, args.num_episodes, args.time_span,
                    args.latency_budget, args.usage_budget,
                    False,
                ))
            for resource_constraint in RC_OPTIONS:
                model_key = (config, resource_constraint, args.policy)
                task_args_list.append((
                    seed, model_key, policy_cls,
                    weights, args.num_episodes, args.time_span,
                    args.latency_budget, args.usage_budget,
                    resource_constraint,
                ))
    return task_args_list

# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Custom (num_modalities, num_tasks, num_turns) scaling demo for TMO."
    )

    # -- RL policy -------------------------------------------------------
    parser.add_argument('--policy', type=str, default='A2C',
                        choices=list(POLICIES.keys()),
                        help='RL algorithm trained in every cell.')

    # -- runtime ----------------------------------------------------------
    parser.add_argument('--repeat', type=int, default=3)
    parser.add_argument('--num_gpus', type=int, default=1)

    # -- dataset ----------------------------------------------------------
    parser.add_argument('--num_episodes', type=int, default=10000)
    parser.add_argument('--time_span', type=int, default=5)
    parser.add_argument('--latency_budget', type=float, default=30.0)
    parser.add_argument('--usage_budget', type=float, default=0.05)

    # -- reward weights ---------------------------------------------------
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--beta_association', type=float, default=1/3)
    parser.add_argument('--beta_latency', type=float, default=1/3)
    parser.add_argument('--beta_usage', type=float, default=1/3)

    # -- I/O --------------------------------------------------------------
    parser.add_argument('--results_dir', type=str, default=DEFAULT_RESULTS_DIR)
    parser.add_argument('--results_filename', type=str, default='Custom_Scaling.pkl')
    args = parser.parse_args()

    out_path = os.path.join(args.results_dir, args.results_filename)

    seeds = np.random.SeedSequence().generate_state(args.repeat).tolist()
    task_args_list = build_task_args(args, seeds)
    n_tasks = len(task_args_list)
    n_rl = sum(1 for t in task_args_list if t[2] is not None)
    print(f"Running {n_tasks} tasks ({n_tasks - n_rl} baselines + "
            f"{n_rl} {args.policy}) across num_gpus={args.num_gpus} "
            f"(both RC=False and RC=True) ...")
    results = run_parallel(task_args_list, run_one_custom,
                            num_gpus=args.num_gpus)

    os.makedirs(args.results_dir, exist_ok=True)
    with open(out_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"Wrote {len(results)} (config, RC, model) cells to {out_path}.")


if __name__ == '__main__':
    main()
