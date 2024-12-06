from collections import namedtuple
import random
import numpy as np
import global_variables
import torch

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))  # FIXME: unused


class ReplayMemory:
    def __init__(self, capacity):
        # self.memory = deque([], maxlen=capacity)
        self.memory = []
        self.capacity = capacity
        self.position = 0
        self.priorities = []
        self.priorities_alpha = 0.6

        self.last_done_pos = 0
        self.done_pos = []
        self.last_sampled_pos = 0

    def push(self, sample):
        """Save a transition"""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
            self.priorities.append(None)
        valid_priorities = [p for p in self.priorities if p is not None]  # Filter out None values before calling max
        max_priority = max(valid_priorities) if valid_priorities else 1.0
        self.memory[self.position] = sample
        self.priorities[self.position] = max_priority
        used_pos = self.position
        self.position = (self.position + 1) % self.capacity
        return used_pos

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, priority in zip(batch_indices, batch_priorities):
            self.priorities[idx] = priority

    def sample(self, batch_size, rnd=False, beta=0.4):
        if rnd:
            return random.sample(self.memory, batch_size)  # NOTE: sample a batch randomly
        else:
            if len(self.memory) == self.capacity:
                priorities = self.priorities
            else:
                priorities = self.priorities[:self.position]
            probabilities = [priority ** self.priorities_alpha for priority in priorities]
            sum_probabilities = sum(probabilities)
            probabilities = [p / sum_probabilities for p in probabilities]  # divide each element by the sum
            batch_indices = np.random.choice(len(self.memory), batch_size, p=probabilities)
            batch_indices = batch_indices.astype(int)
            batch = [self.memory[idx] for idx in batch_indices]

            # Calculate importance sampling weights
            probabilities = np.array(probabilities)
            sampling_probabilities = probabilities[batch_indices]
            weights = (len(self.memory) * sampling_probabilities) ** -beta
            weights /= weights.max()  # Normalize for stability

            return batch_indices, batch, weights

    def clear(self):
        self.memory = []
        self.position = 0

    def __len__(self):
        return len(self.memory)
