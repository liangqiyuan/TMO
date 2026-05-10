"""TMO: Local-Cloud Inference Offloading for Multi-Modal Multi-Task LLMs.

Public API:

- ``M4A1_Env``                              -- Gymnasium environment.
- ``preprocess_data`` / ``create_long_samples`` / ``split_dataset`` /
  ``infer_task_vocab``                      -- dataset utilities.
- ``compute_local_costs`` / ``get_cloud_costs`` /
  ``LOCAL_DEVICES`` / ``CLOUD_SERVERS``     -- hardware & network registries.
- ``evaluate``                              -- rollout an env / policy and collect metrics.
- ``process_model``                         -- single-experiment training+evaluation helper.
- ``run_one`` / ``run_parallel``            -- launch experiments concurrently across GPUs.
- ``args_parser``                           -- the default CLI arguments used by the scripts.
"""

from tmo.data import (
    preprocess_data,
    create_long_samples,
    split_dataset,
    infer_task_vocab,
)
from tmo.devices import (
    compute_local_costs,
    get_cloud_costs,
    LOCAL_DEVICES,
    CLOUD_SERVERS,
)
from tmo.env import M4A1_Env
from tmo.evaluator import evaluate
from tmo.trainer import process_model
from tmo.parallel import run_one, run_parallel
from tmo.config import args_parser

__all__ = [
    "preprocess_data",
    "create_long_samples",
    "split_dataset",
    "infer_task_vocab",
    "compute_local_costs",
    "get_cloud_costs",
    "LOCAL_DEVICES",
    "CLOUD_SERVERS",
    "M4A1_Env",
    "evaluate",
    "process_model",
    "run_one",
    "run_parallel",
    "args_parser",
]

__version__ = "0.3.0"
