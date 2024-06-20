from collections import namedtuple
import numpy as np
import torch
import pickle
import random

Transition = namedtuple('Transition', ('state', 'vector', 'action', 'reward', 'nonterminal'))

class data:
    def __init__(self, capacity):
        self.current_index = 0
        self.capacity = capacity
        self.is_full = False
        self.buffer = np.array([None] * capacity)
        self.last_reward_index = 0

    def size(self):
        return self.capacity if self.is_full else self.current_index

    def add(self, item):
        assert not self.is_full
        self.buffer[self.current_index] = item
        self.current_index += 1
        self.is_full = self.current_index == self.capacity

    def get(self, index):
        return self.buffer[index % self.capacity]

    def update_last_reward_index(self):
        assert not self.is_full
        self.last_reward_index = self.current_index

    def remove_new_data(self):
        assert not self.is_full
        removed_indices = list(range(self.last_reward_index, self.current_index))
        removed_count = len(removed_indices)
        self.current_index = self.last_reward_index
        return removed_count, removed_indices

class ExperienceDataset:
    def __init__(self, device, capacity, state_shape, vector_shape, state_processor, action_processor, normalize_rewards=True):
        self.device = device
        self.capacity = capacity
        self.state_shape = state_shape
        self.vector_shape = vector_shape
        self.state_processor = state_processor
        self.action_processor = action_processor
        self.normalize_rewards = normalize_rewards

        self.empty_transition = Transition(
            torch.zeros(state_shape, dtype=torch.uint8),
            torch.zeros(vector_shape) if vector_shape else None,
            None, 0, False
        )

        self.buffer = CircularBuffer(capacity)
        self.discount_factor = 1.0
        self.multi_step = 1
        self.log_sample_indices = []

    def reshape_reward(self, reward):
        if self.normalize_rewards:
            return 1.0 if reward != 0.0 else 0.0
        return reward

    def add_sample(self, sample, log_sample=False, replace_sample=False):
        state, action, reward, done = sample[0], sample[1], sample[2], sample[4]
        image, vector = self.state_processor.get_image_vector(state)

        if log_sample and not replace_sample:
            self.log_sample_indices.append(self.buffer.current_index)

        torch_vector = (
            self.buffer.buffer[random.choice(self.log_sample_indices)].vector.clone()
            if replace_sample else
            torch.tensor(vector)
        )

        action_id = self.action_processor.get_id(action)
        torch_image = torch.from_numpy(image).permute(2, 0, 1)
        self.buffer.add(Transition(torch_image, torch_vector, action_id, reward, not done))

    def update_last_reward_index(self):
        self.buffer.update_last_reward_index()

    def remove_new_data(self):
        removed_count, removed_indices = self.buffer.remove_new_data()
        self.log_sample_indices = [idx for idx in self.log_sample_indices if idx not in removed_indices]
        return removed_count

    def save(self, path):
        with open(path, 'wb') as file:
            pickle.dump([self.buffer.current_index, self.buffer.capacity, self.buffer.is_full, self.buffer.buffer,
                         self.buffer.last_reward_index], file)

    def load(self, path):
        with open(path, 'rb') as file:
            (self.buffer.current_index, self.buffer.capacity, self.buffer.is_full, self.buffer.buffer,
             self.buffer.last_reward_index) = pickle.load(file)

    def _get_transition(self, index):
        transitions = [self.buffer.get(index)]
        for t in range(1, self.multi_step + 1):
            transitions.append(
                self.buffer.get(index + t) if transitions[t - 1].nonterminal else self.empty_transition
            )
        return transitions

    def sample_batch(self, batch_size, sequence_length):
        indices = np.random.randint(0, self.buffer.size() - sequence_length - self.multi_step, size=batch_size)
        sequence_indices = [i for idx in [list(range(i, i + sequence_length)) for i in indices] for i in idx]

        transitions = [self._get_transition(idx) for idx in sequence_indices]
        states, vectors, next_states, next_vectors, actions, returns, nonterminals = (
            [], [], [], [], [], [], []
        )

        no_vectors = False
        for transition in transitions:
            states.append(transition[0].state.to(self.device, dtype=torch.float32).div_(255))
            next_states.append(transition[self.multi_step].state.to(self.device, dtype=torch.float32).div_(255))

            if transition[0].vector is not None:
                vectors.append(transition[0].vector.to(self.device, dtype=torch.float32))
                next_vectors.append(transition[self.multi_step].vector.to(self.device, dtype=torch.float32))
            else:
                vectors.append(None)
                next_vectors.append(None)
                no_vectors = True

            actions.append(torch.tensor([transition[0].action], dtype=torch.int64, device=self.device))
            returns.append(
                torch.tensor([sum(self.discount_factor ** n * self.reshape_reward(transition[n].reward) for n in range(self.multi_step))],
                             dtype=torch.float32, device=self.device))
            nonterminals.append(torch.tensor([transition[self.multi_step].nonterminal], dtype=torch.float32, device=self.device))

        states, next_states = torch.stack(states), torch.stack(next_states)
        actions, returns, nonterminals = torch.cat(actions), torch.cat(returns), torch.stack(nonterminals)

        if not no_vectors:
            vectors, next_vectors = torch.stack(vectors), torch.stack(next_vectors)

        return states, vectors, actions, returns, next_states, next_vectors, nonterminals
