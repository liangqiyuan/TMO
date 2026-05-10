"""Evaluation loop that rolls out a (env, policy) pair and returns metrics."""

import random
import numpy as np


def evaluate(env, latency_budget, usage_budget, model=None, name=None):
    """Roll out ``env`` for ``len(env.dataset)`` episodes and aggregate metrics."""
    total_rewards = []; total_response_scores = []; total_association_scores = []
    total_latencys = []; total_usages = []; total_actions = []
    latency_out_budget = []; usage_out_budget = []

    if model is not None:
        model.policy.eval()

    num_actions = getattr(env, 'num_actions', env.action_space.n)
    cloud_actions = list(range(1, num_actions))

    for _ in range(len(env.dataset)):
        obs, _ = env.reset()
        done = False
        total_response_score = []; total_association_score = []; total_reward = []
        while not done:
            if model:
                action, _ = model.predict(obs, deterministic=True)
            else:
                if name == 'Random':
                    cloud_count = max(num_actions - 1, 1)
                    weights = [0.5] + [0.5 / cloud_count] * cloud_count
                    action = random.choices(list(range(num_actions)), weights=weights, k=1)[0]
                elif name == 'Local':
                    action = 0
                elif name == 'Cloud':
                    action = random.choice(cloud_actions)

            obs, response_score, association_score, total_latency, total_usage, reward, done = env.step_eval(action, obs)
            total_actions.append(action)
            total_response_score.append(response_score)
            total_association_score.append(association_score)
            total_reward.append(reward)

        if total_latency > latency_budget:
            latency_out_budget.append(total_latency - latency_budget)
        if total_usage > usage_budget:
            usage_out_budget.append(total_usage - usage_budget)
        latency_out_budget = [0] if latency_out_budget == [] else latency_out_budget
        usage_out_budget = [0] if usage_out_budget == [] else usage_out_budget
        total_response_scores.append(np.mean(total_response_score))
        total_association_scores.append(np.mean(total_association_score))
        total_latencys.append(total_latency)
        total_usages.append(total_usage)
        total_rewards.append(np.mean(total_reward))

    avg_response_score = np.mean(total_response_scores)
    avg_association_score = np.mean(total_association_scores)
    avg_latency = np.mean(total_latencys)
    avg_usage = np.mean(total_usages)
    avg_reward = np.mean(total_rewards)
    avg_latency_out_budget = np.mean(latency_out_budget)
    avg_usage_out_budget = np.mean(usage_out_budget)
    return (total_actions, avg_response_score, avg_association_score, avg_latency,
            avg_usage, avg_reward, avg_latency_out_budget, avg_usage_out_budget)
