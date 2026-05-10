"""Command-line configuration for the TMO experiment scripts."""

import argparse


def args_parser():
    parser = argparse.ArgumentParser(description="TMO experiment configuration.")

    # -- runtime ----------------------------------------------------------
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Torch device used when running a single, non-parallel experiment.')
    parser.add_argument('--num_gpus', type=int, default=1,
                        help='Number of GPUs to round-robin across in run_parallel.')
    parser.add_argument('--repeat', type=int, default=3,
                        help='Number of independent seeds per (model, resource_constraint).')

    # -- environment ------------------------------------------------------
    parser.add_argument('--time_span', type=int, default=5,
                        help='Length of the sliding context window (number of past dialogues).')
    parser.add_argument('--num_modalities', type=int, default=3,
                        help='Number of upload-able modalities (X). The action space has '
                             'size 1 + 2**X (1 local + 2**X cloud upload subsets).')
    parser.add_argument('--nearest_neighbors', type=int, default=5)
    parser.add_argument('--local_device', type=str, default='Jetson TX2')
    parser.add_argument('--cloud_server', type=str, default='Wired')
    parser.add_argument('--latency_budget', type=float, default=30,
                        help='Per-episode latency budget (seconds) used at evaluation time.')
    parser.add_argument('--usage_budget', type=float, default=0.05,
                        help='Per-episode monetary usage budget (USD) used at evaluation time.')
    parser.add_argument('--constraint_distribution', type=str, default='normal',
                        choices=['uniform', 'normal', 'extreme_aware'],
                        help='Distribution used to sample per-episode initial budgets during training.')

    # -- reward weights: alpha * response + beta_a * assoc - beta_l * lat - beta_u * usage
    parser.add_argument('--alpha', type=float, default=1)
    parser.add_argument('--beta_association', type=float, default=1/3)
    parser.add_argument('--beta_latency', type=float, default=1/3)
    parser.add_argument('--beta_usage', type=float, default=1/3)

    # -- I/O --------------------------------------------------------------
    parser.add_argument('--dataset_path', type=str, default='dataset/M4A1.json',
                        help='Path to the M4A1 dataset JSON file.')
    parser.add_argument('--results_dir', type=str, default='results',
                        help='Directory in which to write the pickled results.')

    args = parser.parse_args()
    return args
