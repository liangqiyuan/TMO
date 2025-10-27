import json
from stable_baselines3 import PPO, A2C, DQN
import pickle
from models import RC_PPO, RC_A2C, RC_DQN

from options import args_parser
from utils import *

import warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)


if __name__ == '__main__':
    args = args_parser()

    with open('../dataset/M4A1.json', 'r') as f:
        dataset = json.load(f)

    weights = [args.alpha, args.beta_association, args.beta_latency, args.beta_usage]
    latency_budget = args.latency_budget
    usage_budget = args.usage_budget
    local_device = args.local_device
    cloud_server = args.cloud_server

    time_span = args.time_span
    repeat = args.repeat

    Resource_Constraints = [False, True]
    models = {'Random': None, 'Local': None, 'Cloud': None, 
            'PPO': PPO, 'A2C': A2C, 'DQN': DQN,
            'RC_PPO': RC_PPO, 'RC_A2C': RC_A2C, 'RC_DQN': RC_DQN}
    
    results = {}
    Train = {}

    for _ in range(repeat):
        train_dataset, test_dataset = split_dataset(dataset, test_ratio=0.2)
        for resource_constraint in Resource_Constraints:
            train_env = M4A1_Env(train_dataset, weights, local_device, cloud_server, latency_budget, usage_budget, resource_constraint, time_span, Train=True)
            Train['rewards'] = train_env.rewards; Train['nn'] = train_env.nn
            test_env = M4A1_Env(test_dataset, weights, local_device, cloud_server, latency_budget, usage_budget, resource_constraint, time_span, Train)
            for model_name, model_cls in models.items():
                model_key = (resource_constraint, model_name, model_cls)
                if should_process_model(model_name, resource_constraint):
                    process_model(model_key, model_cls, train_env, test_env, latency_budget, usage_budget, results)

    with open('results/Main_Results.pkl', 'wb') as f:
        pickle.dump(results, f)

