"""Dataset loading and preprocessing helpers for TMO dialogue datasets.

The expected dataset schema is::

    [
        {
            "interactions": [
                {
                    "task_cat":    <str>,                  # task category name
                    "action":      0 | 1,                  # 0 = local, 1 = cloud
                    "image_index": None | List[int],       # uploaded modalities (cloud)
                    "answer":      <str>,
                    "score":       <float>,
                },
                ...
            ],
            "association_score": <List[List[float]]>,      # T x num_modalities table
        },
        ...
    ]
"""

import random


def infer_task_vocab(dataset):
    """Return a deterministic task vocabulary inferred from ``dataset``"""
    seen = set()
    for episode in dataset:
        for item in episode['interactions']:
            seen.add(item['task_cat'])
    return sorted(seen)


def preprocess_data(dataset, num_modalities=3, task_vocab=None):
    """Convert raw episodes into ``(states, actions, rewards, association_score)"""
    if task_vocab is None:
        task_vocab = infer_task_vocab(dataset)
    task_to_index = {name: i for i, name in enumerate(task_vocab)}

    episodes = []
    for episode in dataset:
        states, actions, rewards = [], [], []
        previous_item = {'action': -1, 'image_index': None}
        for item in episode['interactions']:
            used_modalities = [0] * num_modalities
            if previous_item['action'] == 1 and previous_item['image_index'] is not None:
                for modality_index in previous_item['image_index']:
                    used_modalities[modality_index] = 1
            state = ([previous_item['action']]
                     + used_modalities
                     + [task_to_index[item['task_cat']]])

            if item['action'] == 0:
                action = [0] + [0] * num_modalities
            elif item['action'] == 1 and item['image_index'] is None:
                action = [1] + [0] * num_modalities
            else:
                action = [1] + [0] * num_modalities
                for modality_index in item['image_index']:
                    action[modality_index + 1] = 1

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


def create_long_samples(episodes, time_span=5, state_dim_per_step=5):
    """Frame-stack each per-step state with the previous ``time_span - 1`` states"""
    pad = [-1] * state_dim_per_step
    history_length = time_span - 1
    long_samples = []
    for episode in episodes:
        states, actions, rewards, association_scores = episode
        long_states = list(states)
        extended_states = []
        for i in range(len(long_states)):
            history = [long_states[j] if j >= 0 else pad
                       for j in range(i - history_length, i)]
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
