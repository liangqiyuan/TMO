import numpy as np
import gymnasium as gym
import random
from sklearn.neighbors import NearestNeighbors

from options import args_parser
args = args_parser()

def preprocess_data(dataset):
    episodes = [] 
    task_to_index = {'Assistive System': 0, 'Message Editing': 1, 'Query': 2, 'Recommendation': 3}
    for episode in dataset:
        states, actions, rewards = [], [], []
        previous_item = {'action': -1, 'image_index': None}
        for item in episode['interactions']:
            used_images = [0, 0, 0] 
            if previous_item['action'] == 1 and previous_item['image_index'] is not None:
                for image_index in previous_item['image_index']:
                    used_images[image_index] = 1  
            state = [previous_item['action']] + used_images + [task_to_index[item['task_cat']]]

            if item['action'] == 0:
                action = [0, 0, 0, 0]
            elif item['action'] == 1 and item['image_index'] == None:
                action = [1, 0, 0, 0]
            else:
                action = [1, 0, 0, 0]
                modality = item['image_index']
                for index in modality:
                    action[index + 1] = 1

            key_words = ["sorry", "I don't have", "can't"]
            if any(keyword in item['answer'] for keyword in key_words):
                reward = 0 
            else:
                reward = item['score']

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            previous_item = item
        association_score = episode['association_score']
        episodes.append((states, actions, rewards, association_score))
    return episodes
    
def create_long_samples(episodes):
    long_samples = []
    for episode in episodes:
        states, actions, rewards, association_scores = episode
        long_states = []
        for state in states:
            long_states.append(state)
                
        extended_states = []
        for i in range(len(long_states)):
            history = [long_states[j] if j >= 0 else [-1, -1, -1, -1, -1] for j in range(i - 4, i)]
            extended_state = [item for sublist in history for item in sublist] + long_states[i]
            extended_states.append(extended_state)

        long_samples.append((extended_states, actions, rewards, association_scores))
    return long_samples
    
def split_dataset(data, test_ratio=0.2):
    shuffled_data = data[:]
    random.shuffle(shuffled_data)
    test_size = int(len(data) * test_ratio)
    test_set = shuffled_data[:test_size]
    train_set = shuffled_data[test_size:]
    return train_set, test_set

class M4A1_Env(gym.Env):
    def __init__(self, dataset, weights, local_device, cloud_server, latency_budget, usage_budget, Resource_Constraint, time_span, Train):
        super(M4A1_Env, self).__init__()
        self.dataset = preprocess_data(dataset)
        self.episodes = create_long_samples(self.dataset)
        self.weights = weights
        self.local_device = local_device
        self.cloud_server = cloud_server
        self.latency_budget = latency_budget
        self.usage_budget = usage_budget
        self.Resource_Constraint = Resource_Constraint
        self.time_span = time_span
        
        self.action_space = gym.spaces.Discrete(9)
        self.observation_space = gym.spaces.Box(low=np.array([0, 0, 0, 0, 0]*self.time_span), high=np.array([1, 1, 1, 1, 3]*self.time_span), shape=(5*self.time_span,), dtype=np.float32)
        self.current_episode = 0
        self.current_step = 0
        self.current_states, self.current_actions, self.current_rewards, self.association_scores = self.episodes[self.current_episode]

        self.calculate_local_device()
        self.calculate_cloud_server()
        self.latency_costs = [self.local_time] + self.cloud_time
        self.usage_costs = [self.local_usage_cost] + self.cloud_usage_cost
        self.action_to_modality_indices = {0: [], 1: [], 2: [0], 3: [1], 4: [2], 5: [0, 1], 6: [0, 2], 7: [1, 2], 8: [0, 1, 2]}

        if Train:
            self.nn = NearestNeighbors(algorithm='kd_tree',n_neighbors=5, metric='euclidean')
            all_states = np.vstack([ep[0] for ep in self.episodes])
            all_actions = np.vstack([ep[1] for ep in self.episodes])
            all_rewards = np.concatenate([ep[2] for ep in self.episodes])
            self.state_action_pairs = np.hstack([all_states, all_actions])
            self.rewards = all_rewards
            self.nn.fit(self.state_action_pairs)
        else:
            self.rewards = Train['rewards']
            self.nn = Train['nn']

    def calculate_local_device(self):
        if self.local_device == "Raspberry Pi-4B":
            GF_peak = 13.5; P_max = 8
        elif self.local_device == "Raspberry Pi-5":
            GF_peak = 31.4; P_max = 12
        elif self.local_device == "Jetson Nano":
            GF_peak = 472; P_max = 10
        elif self.local_device == "Jetson TX2":
            GF_peak = 1.33e3; P_max = 15
        elif self.local_device == "Jetson Xavier NX":
            GF_peak = 21e3; P_max = 20
        elif self.local_device == "Jetson Orin NX":
            GF_peak = 100e3; P_max = 25
        elif self.local_device == "Jetson AGX Orin":
            GF_peak = 275e3; P_max = 60
        elif self.local_device == "iPhone 15 Pro":
            GF_peak = 35e3; P_max = 15

        num_parameter = 3.8e9; L_base = 1024; L_total = 2048
        cents_per_kWh = 16.68; cost_per_joule = (cents_per_kWh / 100) / (3600 * 1000)
        
        flops_per_token = 2 * num_parameter / L_base
        total_flops = flops_per_token * L_total
        self.local_time = total_flops / (GF_peak * 1e9)
        energy_consumption = P_max * self.local_time
        self.local_usage_cost = energy_consumption * cost_per_joule

    def calculate_cloud_server(self):
        if self.cloud_server == "Wired":
            self.cloud_time = [6.46030, 4.76134, 6.24569, 6.98211]
        elif self.cloud_server == "WiFi":
            self.cloud_time = [0.5, 1.898, 2.016, 2.238]
        elif self.cloud_server == "5G":
            self.cloud_time = [0.5, 6.037, 6.412, 7.118]
        elif self.cloud_server == "4G":
            self.cloud_time = [0.5, 16.603, 17.636, 19.573]

        self.cloud_usage_cost = [0.00049, 0.00500, 0.00945, 0.01368]

    def reset(self, seed=None, options=None):
        self.current_episode = random.randint(0, len(self.episodes) - 1)
        self.current_step = 0
        self.current_states, self.current_actions, self.current_rewards, self.association_scores = self.episodes[self.current_episode]
        return np.array(self.current_states[self.current_step], dtype=np.float32), {}
        
    def normalization(self, values):
        min_value = min(values); max_value = max(values)
        values = [(x - min_value) / (max_value - min_value) for x in values]
        return values

    def Ass_transform(self, Ass):
        new_Ass = [0, 0]
        new_Ass.extend(Ass[:3])
        new_Ass.extend([Ass[0] + Ass[1], Ass[0] + Ass[2], Ass[1] + Ass[2], Ass[0] + Ass[1] + Ass[2]])
        return new_Ass

    def to_one_hot(self, action):
        action = action.item() if isinstance(action, np.ndarray) else action
        if action == 0:
            one_hot = [0, 0, 0, 0]
        else:
            one_hot = [1, 0, 0, 0]

            modality = self.action_to_modality_indices[action]
            for index in modality:
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
        else:
            norm_latency_cost = self.normalization(self.latency_costs)[len(modality_indices)+1]
            norm_usage_cost = self.normalization(self.usage_costs)[len(modality_indices)+1]

        state_action = np.hstack([self.current_states[self.current_step], real_action])
        distances, indices = self.nn.kneighbors([state_action])
        weight = 1 / (distances + 1e-6)
        response_score = np.average(self.rewards[indices], weights=weight)
        
        reward = self.weights[0] * response_score + self.weights[1] * association_score - self.weights[2] * norm_latency_cost - self.weights[3] * norm_usage_cost

        self.current_step += 1
        done = self.current_step >= len(self.current_states)
        next_state = np.array(self.current_states[self.current_step], dtype=int) if not done else np.zeros(5*self.time_span) - 1
            
        return next_state, reward, done, False, {}

    def step_eval(self, action, state):
        real_action = self.to_one_hot(action)
        action = action.item() if isinstance(action, np.ndarray) else action
        modality_indices = self.action_to_modality_indices[action]
        association_score = self.Ass_transform(self.association_scores[self.current_step])[action]
        norm_association_score = self.normalization(self.Ass_transform(self.association_scores[self.current_step]))[action]
        
        total_latency = 0; total_usage = 0
        for i in range(self.time_span):
            if state[i*5] == 0:
                total_latency += self.local_time
                total_usage += self.local_usage_cost
            elif state[i*5] == 1:                
                modalities_from_state = sum(state[i*5+1:i*5+4])
                total_latency += self.cloud_time[modalities_from_state]
                total_usage += self.cloud_usage_cost[modalities_from_state]    
        
        if action == 0:
            total_latency += self.local_time
            total_usage += self.local_usage_cost
            norm_latency_cost = self.normalization(self.latency_costs)[0]
            norm_usage_cost = self.normalization(self.usage_costs)[0]
        else:
            total_latency += self.cloud_time[len(modality_indices)]
            total_usage += self.cloud_usage_cost[len(modality_indices)]
            norm_latency_cost = self.normalization(self.latency_costs)[len(modality_indices)+1]
            norm_usage_cost = self.normalization(self.usage_costs)[len(modality_indices)+1]
        
        state_action = np.hstack([state, real_action])
        distances, indices = self.nn.kneighbors([state_action])
        weights = 1 / (distances + 1e-6)
        response_score = np.average(self.rewards[indices], weights=weights)
        
        reward = self.weights[0] * response_score + self.weights[1] * association_score - self.weights[2] * norm_latency_cost - self.weights[3] * norm_usage_cost

        self.current_step += 1
        done = self.current_step >= len(self.current_states)
        next_state = np.array(list(state[5:]) + real_action + self.current_states[self.current_step][-1:], dtype=int) if not done else np.zeros(5*self.time_span) - 1
        return next_state, response_score, association_score, total_latency, total_usage, reward, done


def evaluate(env, latency_budget, usage_budget, model=None, name=None):
    total_rewards = []; total_response_scores = []; total_association_scores = []; total_latencys = []; total_usages = []; total_actions = []
    latency_out_budget = []; usage_out_budget = []
    for _ in range(len(env.dataset)):
        obs, _ = env.reset()
        done = False
        total_response_score = []; total_association_score = []; total_reward = []
        while not done:
            if model:
                action, _ = model.predict(obs, deterministic=True)
            else:
                if name == 'Random':
                    action = random.choices([0, 1, 2, 3, 4, 5, 6, 7, 8], weights=[0.5] + [0.5/8] * 8, k=1)[0]
                elif name == 'Local':
                    action = 0
                elif name == 'Cloud':
                    action = random.choice([1, 2, 3, 4, 5, 6, 7, 8])

            obs, response_score, association_score, total_latency, total_usage, reward, done = env.step_eval(action, obs)
            total_actions.append(action); total_response_score.append(response_score); total_association_score.append(association_score); total_reward.append(reward)

        if total_latency > latency_budget:
            latency_out_budget.append(total_latency - latency_budget)
        if total_usage > usage_budget:
            usage_out_budget.append(total_usage - usage_budget)
        latency_out_budget = [0] if latency_out_budget == [] else latency_out_budget; usage_out_budget = [0] if usage_out_budget == [] else usage_out_budget; 
        total_response_scores.append(np.mean(total_response_score)); total_association_scores.append(np.mean(total_association_score)); total_latencys.append(total_latency); total_usages.append(total_usage); total_rewards.append(np.mean(total_reward))
    avg_response_score = np.mean(total_response_scores); avg_association_score = np.mean(total_association_scores); avg_latency = np.mean(total_latencys); avg_usage = np.mean(total_usages); avg_reward = np.mean(total_rewards); avg_latency_out_budget = np.mean(latency_out_budget); avg_usage_out_budget = np.mean(usage_out_budget)
    return total_actions, avg_response_score, avg_association_score, avg_latency, avg_usage, avg_reward, avg_latency_out_budget, avg_usage_out_budget

def process_model(model_key, model_cls, train_env, test_env, latency_budget, usage_budget, results):
    if model_cls is None:
        results.setdefault(model_key, []).append(evaluate(env=test_env, latency_budget=latency_budget, usage_budget=usage_budget, name=model_key[1]))
    else:
        model = model_cls('MlpPolicy', train_env, device=args.device)
        model.learn(total_timesteps=30000)
        results.setdefault(model_key, []).append(evaluate(env=test_env, latency_budget=latency_budget, usage_budget=usage_budget, model=model, name=model_key[1]))

def should_process_model(model_name, resource_constraint):
    if 'RC' not in model_name and resource_constraint == False:
        return True
    elif 'RC' in model_name:
        return resource_constraint
