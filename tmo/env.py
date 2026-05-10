"""Gymnasium environment for the local-cloud LLM offloading task.

The environment is parameterised by three scenario knobs::

    num_modalities   X   # how many modality artefacts can be uploaded
    num_tasks        Y   # vocabulary size for the task-category feature
    time_span        Z   # length of the sliding history window

Per-step state layout (length ``X + 2``)::

    [prev_action, used_m_0, ..., used_m_{X-1}, task_id]

Full observation (length ``(X + 2) * Z``, optionally + 2 budget features)::

    concat(state_{t-Z+1}, ..., state_{t}) [+ remaining_latency, remaining_usage]

Action layout (``Discrete(1 + 2**X)``)::

    0          : run on the local LLM (no upload)
    1 + i      : run on the cloud LLM, uploading the i-th modality subset

Modality subsets are enumerated in (size, lexicographic-index) order, so for
``X = 3`` the legacy 9-action layout is recovered exactly:

    {0: local,         1: cloud-text-only,
     2: cloud + m0,    3: cloud + m1,        4: cloud + m2,
     5: cloud + m0,m1, 6: cloud + m0,m2,     7: cloud + m1,m2,
     8: cloud + m0,m1,m2}

There is a single local LLM and a single cloud LLM; only the *upload subset*
is part of the decision space.
"""

import itertools
import random

import numpy as np
import gymnasium as gym
from sklearn.neighbors import NearestNeighbors

from tmo.data import preprocess_data, create_long_samples, infer_task_vocab
from tmo.devices import compute_local_costs, get_cloud_costs


class M4A1_Env(gym.Env):
    """Local-cloud offloading environment.

    Parameters
    ----------
    dataset : list of dict
        Raw dataset (see :mod:`tmo.data`).
    weights : sequence of 4 floats
        Reward weights ``[alpha, beta_assoc, beta_lat, beta_usage]``.
    local_device, cloud_server : str
        Profile names looked up in :mod:`tmo.devices`.
    latency_budget, usage_budget : float
        Per-episode resource budgets used at evaluation time.
    Resource_Constraint : bool
        If ``True``, the observation is augmented with the remaining budget
        and over-budget steps receive zero reward during training.
    time_span : int
        Length of the sliding history window (``Z``).
    Train : bool or dict
        ``True`` to fit the KNN reward simulator from this dataset; otherwise
        a dict ``{'rewards': ..., 'nn': ...}`` produced by an earlier train
        instance (used for the eval env).
    constraint_distribution : {"normal", "uniform", "extreme_aware"}
        Distribution from which initial budgets are sampled during training.
    num_modalities : int, default 3
        Number of upload-able modalities (``X``).
    task_vocab : list of str, optional
        Ordered task vocabulary. ``None`` → inferred from ``dataset``.
    local_registry, cloud_registry : dict, optional
        Override registries forwarded to :func:`tmo.devices.compute_local_costs`
        and :func:`tmo.devices.get_cloud_costs`.
    """

    def __init__(self, dataset, weights, local_device, cloud_server,
                 latency_budget, usage_budget, Resource_Constraint, time_span,
                 Train, constraint_distribution='normal',
                 num_modalities=3, task_vocab=None,
                 local_registry=None, cloud_registry=None):
        super(M4A1_Env, self).__init__()
        self.num_modalities = int(num_modalities)
        self.task_vocab = list(task_vocab) if task_vocab is not None else infer_task_vocab(dataset)
        self.num_tasks = len(self.task_vocab)
        self.state_dim_per_step = self.num_modalities + 2

        self.dataset = preprocess_data(dataset,
                                       num_modalities=self.num_modalities,
                                       task_vocab=self.task_vocab)
        self.episodes = create_long_samples(self.dataset,
                                            time_span=time_span,
                                            state_dim_per_step=self.state_dim_per_step)
        self.weights = weights
        self.local_device = local_device
        self.cloud_server = cloud_server
        self.latency_budget = latency_budget
        self.usage_budget = usage_budget
        self.Resource_Constraint = Resource_Constraint
        self.time_span = time_span
        self.constraint_distribution = constraint_distribution
        self.is_training = (Train is True)

        self.modality_subsets = []
        for size in range(self.num_modalities + 1):
            for combo in itertools.combinations(range(self.num_modalities), size):
                self.modality_subsets.append(list(combo))
        self.num_actions = 1 + 2 ** self.num_modalities
        self.action_to_modality_indices = {0: []}
        for i, subset in enumerate(self.modality_subsets):
            self.action_to_modality_indices[1 + i] = subset

        self.action_space = gym.spaces.Discrete(self.num_actions)
        per_step_low = [0] * self.state_dim_per_step
        per_step_high = [1] * (self.num_modalities + 1) + [max(self.num_tasks - 1, 0)]
        lows = per_step_low * self.time_span
        highs = per_step_high * self.time_span
        if self.Resource_Constraint:
            lows += [0, 0]
            highs += [float('inf'), float('inf')]
        self.observation_space = gym.spaces.Box(
            low=np.array(lows, dtype=np.float32),
            high=np.array(highs, dtype=np.float32),
            shape=(len(lows),), dtype=np.float32,
        )

        self.current_episode = 0
        self.current_step = 0
        self._eval_episode_cursor = 0
        (self.current_states, self.current_actions,
         self.current_rewards, self.association_scores) = self.episodes[self.current_episode]

        self.local_time, self.local_usage_cost = compute_local_costs(
            self.local_device, registry=local_registry)
        self.cloud_time, self.cloud_usage_cost = get_cloud_costs(
            self.cloud_server, self.num_modalities, registry=cloud_registry)
        self.latency_costs = [self.local_time] + self.cloud_time
        self.usage_costs = [self.local_usage_cost] + self.cloud_usage_cost

        if Train is True:
            self.nn = NearestNeighbors(algorithm='kd_tree', n_neighbors=5, metric='euclidean')
            all_states = np.vstack([ep[0] for ep in self.episodes])
            all_actions = np.vstack([ep[1] for ep in self.episodes])
            all_rewards = np.concatenate([ep[2] for ep in self.episodes])
            sa_pairs = np.hstack([all_states, all_actions])
            order = np.lexsort(sa_pairs.T)
            self.state_action_pairs = sa_pairs[order]
            self.rewards = all_rewards[order]
            self.nn.fit(self.state_action_pairs)
        else:
            self.rewards = Train['rewards']
            self.nn = Train['nn']

        self.remaining_latency = self.latency_budget
        self.remaining_usage = self.usage_budget

    # ------------------------------------------------------------------
    # Resource-constraint sampling
    # ------------------------------------------------------------------

    def _sample_remaining_resources(self):
        def _clip(v, lo, hi):
            return float(np.clip(v, lo, hi))

        def _uniform(lo, hi):
            return float(np.random.uniform(lo, hi))

        def _normal(lo, hi):
            mu = 0.5 * (lo + hi)
            sigma = max((hi - lo) / 6.0, 1e-8)
            return _clip(np.random.normal(mu, sigma), lo, hi)

        def _edge_sample(lo, hi, band_ratio=0.05, side='random'):
            span = hi - lo
            band = max(band_ratio * span, 1e-8)
            if side == 'random':
                side = 'min' if np.random.rand() < 0.5 else 'max'
            if side == 'min':
                val = lo + np.random.exponential(scale=band)
            else:
                val = hi - np.random.exponential(scale=band)
            return _clip(val, lo, hi)

        min_latency = self.local_time * self.time_span
        max_latency = self.cloud_time[self.num_modalities] * self.time_span
        min_usage = self.local_usage_cost * self.time_span
        max_usage = self.cloud_usage_cost[self.num_modalities] * self.time_span

        if self.constraint_distribution == "uniform":
            self.remaining_latency = _uniform(min_latency, max_latency)
            self.remaining_usage = _uniform(min_usage, max_usage)

        elif self.constraint_distribution == "normal":
            self.remaining_latency = _normal(min_latency, max_latency)
            self.remaining_usage = _normal(min_usage, max_usage)

        elif self.constraint_distribution == "extreme_aware":
            r = np.random.rand()
            if r < 0.50:
                self.remaining_latency = _normal(min_latency, max_latency)
                self.remaining_usage = _normal(min_usage, max_usage)
            elif r < 0.80:
                self.remaining_latency = _uniform(min_latency, max_latency)
                self.remaining_usage = _uniform(min_usage, max_usage)
            elif r < 0.90:
                if np.random.rand() < 0.5:
                    self.remaining_latency = _edge_sample(min_latency, max_latency, band_ratio=0.1, side='random')
                    self.remaining_usage = _uniform(min_usage, max_usage)
                else:
                    self.remaining_usage = _edge_sample(min_usage, max_usage, band_ratio=0.1, side='random')
                    self.remaining_latency = _uniform(min_latency, max_latency)
            else:
                self.remaining_latency = _edge_sample(min_latency, max_latency, band_ratio=0.1, side='random')
                self.remaining_usage = _edge_sample(min_usage, max_usage, band_ratio=0.1, side='random')
        else:
            self.remaining_latency = _uniform(min_latency, max_latency)
            self.remaining_usage = _uniform(min_usage, max_usage)

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        if self.is_training:
            self.current_episode = random.randint(0, len(self.episodes) - 1)
        else:
            self.current_episode = self._eval_episode_cursor % len(self.episodes)
            self._eval_episode_cursor += 1
        self.current_step = 0
        (self.current_states, self.current_actions,
         self.current_rewards, self.association_scores) = self.episodes[self.current_episode]

        if self.is_training:
            self._sample_remaining_resources()
        else:
            self.remaining_latency = self.latency_budget
            self.remaining_usage = self.usage_budget

        obs_list = list(self.current_states[self.current_step])
        if self.Resource_Constraint:
            obs_list += [self.remaining_latency, self.remaining_usage]
        return np.array(obs_list, dtype=np.float32), {}

    def normalization(self, values):
        min_value = min(values); max_value = max(values)
        values = [(x - min_value) / (max_value - min_value) for x in values]
        return values

    def Ass_transform(self, Ass):
        """Per-modality association scores -> per-action association scores.

        Length of the result equals :attr:`num_actions`.  Index 0 corresponds
        to the local action and is always 0.  Index ``1 + i`` corresponds to
        the cloud call uploading the i-th modality subset; its value is the
        sum of the modality-wise association scores in that subset (the empty
        subset, i.e. cloud-text-only, is therefore also 0).
        """
        new_Ass = [0]
        for subset in self.modality_subsets:
            new_Ass.append(sum(Ass[m] for m in subset))
        return new_Ass

    def to_one_hot(self, action):
        action = action.item() if isinstance(action, np.ndarray) else action
        if action == 0:
            one_hot = [0] + [0] * self.num_modalities
        else:
            one_hot = [1] + [0] * self.num_modalities
            for index in self.action_to_modality_indices[action]:
                one_hot[index + 1] = 1
        return one_hot

    def step(self, action):
        real_action = self.to_one_hot(action)
        action = action.item() if isinstance(action, np.ndarray) else action
        modality_indices = self.action_to_modality_indices[action]
        association_score = self.normalization(self.Ass_transform(self.association_scores[self.current_step]))[action]

        if action == 0:
            norm_latency_cost = self.normalization(self.latency_costs)[0]
            norm_usage_cost = self.normalization(self.usage_costs)[0]
            latency_cost = self.local_time
            usage_cost = self.local_usage_cost
        else:
            norm_latency_cost = self.normalization(self.latency_costs)[len(modality_indices) + 1]
            norm_usage_cost = self.normalization(self.usage_costs)[len(modality_indices) + 1]
            latency_cost = self.cloud_time[len(modality_indices)]
            usage_cost = self.cloud_usage_cost[len(modality_indices)]

        state_action = np.hstack([self.current_states[self.current_step], real_action])
        distances, indices = self.nn.kneighbors([state_action])
        weight = 1 / (distances + 1e-6)
        response_score = np.average(self.rewards[indices], weights=weight)

        reward = (self.weights[0] * response_score
                  + self.weights[1] * association_score
                  - self.weights[2] * norm_latency_cost
                  - self.weights[3] * norm_usage_cost)

        if self.Resource_Constraint:
            self.remaining_latency -= latency_cost
            self.remaining_usage -= usage_cost
            if self.remaining_latency < 0 or self.remaining_usage < 0:
                reward = 0

        self.current_step += 1
        done = self.current_step >= len(self.current_states)
        if not done:
            obs_list = list(self.current_states[self.current_step])
            if self.Resource_Constraint:
                obs_list += [self.remaining_latency, self.remaining_usage]
            next_state = np.array(obs_list, dtype=np.float32)
        else:
            extra_len = 2 if self.Resource_Constraint else 0
            next_state = np.zeros(self.state_dim_per_step * self.time_span + extra_len, dtype=np.float32) - 1

        return next_state, reward, done, False, {}

    def step_eval(self, action, state):
        real_action = self.to_one_hot(action)
        action = action.item() if isinstance(action, np.ndarray) else action
        modality_indices = self.action_to_modality_indices[action]
        association_score = self.Ass_transform(self.association_scores[self.current_step])[action]
        norm_association_score = self.normalization(self.Ass_transform(self.association_scores[self.current_step]))[action]

        step_dim = self.state_dim_per_step
        total_latency = 0; total_usage = 0
        for i in range(self.time_span):
            base = i * step_dim
            if state[base] == 0:
                total_latency += self.local_time
                total_usage += self.local_usage_cost
            elif state[base] == 1:
                modalities_from_state = int(sum(state[base + 1:base + 1 + self.num_modalities]))
                total_latency += self.cloud_time[modalities_from_state]
                total_usage += self.cloud_usage_cost[modalities_from_state]

        if action == 0:
            total_latency += self.local_time
            total_usage += self.local_usage_cost
            norm_latency_cost = self.normalization(self.latency_costs)[0]
            norm_usage_cost = self.normalization(self.usage_costs)[0]
            latency_cost = self.local_time
            usage_cost = self.local_usage_cost
        else:
            total_latency += self.cloud_time[len(modality_indices)]
            total_usage += self.cloud_usage_cost[len(modality_indices)]
            norm_latency_cost = self.normalization(self.latency_costs)[len(modality_indices) + 1]
            norm_usage_cost = self.normalization(self.usage_costs)[len(modality_indices) + 1]
            latency_cost = self.cloud_time[len(modality_indices)]
            usage_cost = self.cloud_usage_cost[len(modality_indices)]

        base_without_extras = state[:step_dim * self.time_span]
        state_action = np.hstack([base_without_extras, real_action])
        distances, indices = self.nn.kneighbors([state_action])
        weights = 1 / (distances + 1e-6)
        response_score = np.average(self.rewards[indices], weights=weights)

        reward = (self.weights[0] * response_score
                  + self.weights[1] * norm_association_score
                  - self.weights[2] * norm_latency_cost
                  - self.weights[3] * norm_usage_cost)

        self.remaining_latency -= latency_cost
        self.remaining_usage -= usage_cost

        self.current_step += 1
        done = self.current_step >= len(self.current_states)
        if not done:
            obs_list = (list(base_without_extras[step_dim:])
                        + real_action
                        + list(self.current_states[self.current_step][-1:]))
            if self.Resource_Constraint:
                obs_list += [self.remaining_latency, self.remaining_usage]
            next_state = np.array(obs_list, dtype=np.float32)
        else:
            extra_len = 2 if self.Resource_Constraint else 0
            next_state = np.zeros(step_dim * self.time_span + extra_len, dtype=np.float32) - 1
        return next_state, response_score, association_score, total_latency, total_usage, reward, done
