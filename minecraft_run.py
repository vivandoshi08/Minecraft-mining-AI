import numpy as np
import torch
from gym.core import Wrapper
import minerl.env.spaces as spaces
from copy import deepcopy


class TestMinecraftEnv:
    def __init__(self):
        self.state = {
            'equipped_items': {'mainhand': {'damage': 0, 'maxDamage': 0, 'type': 0}},
            'inventory': {
                'coal': 0, 'cobblestone': 0, 'crafting_table': 0, 'dirt': 0, 'furnace': 0,
                'iron_axe': 0, 'iron_ingot': 0, 'iron_ore': 0, 'iron_pickaxe': 0, 'log': 0,
                'planks': 0, 'stick': 0, 'stone': 0, 'stone_axe': 0, 'stone_pickaxe': 0,
                'torch': 0, 'wooden_axe': 0, 'wooden_pickaxe': 0
            },
            'pov': np.full([64, 64, 3], 127, dtype=np.uint8)
        }

        self.action_space = {
            'attack': spaces.Discrete(2),
            'back': spaces.Discrete(2),
            'camera': spaces.Box(np.full([2], -1), np.full([2], 1)),
            'craft': spaces.Enum('none', 'torch', 'stick', 'planks', 'crafting_table'),
            'equip': spaces.Enum('none', 'air', 'wooden_axe', 'wooden_pickaxe', 'stone_axe',
                                 'stone_pickaxe', 'iron_axe', 'iron_pickaxe'),
            'forward': spaces.Discrete(2),
            'jump': spaces.Discrete(2),
            'left': spaces.Discrete(2),
            'nearbyCraft': spaces.Enum('none', 'wooden_axe', 'wooden_pickaxe', 'stone_axe',
                                       'stone_pickaxe', 'iron_axe', 'iron_pickaxe', 'furnace'),
            'nearbySmelt': spaces.Enum('none', 'iron_ingot', 'coal'),
            'place': spaces.Enum('none', 'dirt', 'stone', 'cobblestone', 'crafting_table', 'furnace', 'torch'),
            'right': spaces.Discrete(2),
            'sneak': spaces.Discrete(2),
            'sprint': spaces.Discrete(2)
        }

        self.observation_space = {
            'equipped_items': {
                'mainhand': {
                    'damage': spaces.Box(np.full([2], -1), np.full([2], 1), dtype=np.int64),
                    'maxDamage': spaces.Box(np.full([2], -1), np.full([2], 1), dtype=np.int64),
                    'type': spaces.Enum('none', 'air', 'wooden_axe', 'wooden_pickaxe', 'stone_axe', 'stone_pickaxe',
                                        'iron_axe', 'iron_pickaxe', 'other')
                }
            },
            'inventory': {
                'coal': spaces.Box(np.full([1], -1), np.full([1], 1), dtype=np.int64),
                'cobblestone': spaces.Box(np.full([1], -1), np.full([1], 1), dtype=np.int64),
                'crafting_table': spaces.Box(np.full([1], -1), np.full([1], 1), dtype=np.int64),
                'dirt': spaces.Box(np.full([1], -1), np.full([1], 1), dtype=np.int64),
                'furnace': spaces.Box(np.full([1], -1), np.full([1], 1), dtype=np.int64),
                'iron_axe': spaces.Box(np.full([1], -1), np.full([1], 1), dtype=np.int64),
                'iron_ingot': spaces.Box(np.full([1], -1), np.full([1], 1), dtype=np.int64),
                'iron_ore': spaces.Box(np.full([1], -1), np.full([1], 1), dtype=np.int64),
                'iron_pickaxe': spaces.Box(np.full([1], -1), np.full([1], 1), dtype=np.int64),
                'log': spaces.Box(np.full([1], -1), np.full([1], 1), dtype=np.int64),
                'planks': spaces.Box(np.full([1], -1), np.full([1], 1), dtype=np.int64),
                'stick': spaces.Box(np.full([1], -1), np.full([1], 1), dtype=np.int64),
                'stone': spaces.Box(np.full([1], -1), np.full([1], 1), dtype=np.int64),
                'stone_axe': spaces.Box(np.full([1], -1), np.full([1], 1), dtype=np.int64),
                'stone_pickaxe': spaces.Box(np.full([1], -1), np.full([1], 1), dtype=np.int64),
                'torch': spaces.Box(np.full([1], -1), np.full([1], 1), dtype=np.int64),
                'wooden_axe': spaces.Box(np.full([1], -1), np.full([1], 1), dtype=np.int64),
                'wooden_pickaxe': spaces.Box(np.full([1], -1), np.full([1], 1), dtype=np.int64)
            },
            'pov': spaces.Box(np.full([64, 64, 3], -1), np.full([64, 64, 3], 1), dtype=np.uint8)
        }

        self.reward_range = (-np.inf, np.inf)
        self.metadata = {'render.modes': ['rgb_array', 'human']}
        self.timestep = 0

    def reset(self):
        self.timestep = 0
        self.state['pov'][:, :, :] = 127
        return deepcopy(self.state)

    def step(self, action):
        self.timestep += 1
        self.state['pov'][:, :, :] = 127
        done = self.timestep >= 1000
        reward = 0.1
        return deepcopy(self.state), reward, done, {}

    def close(self):
        pass

    def seed(self, seed=None):
        pass


class CustomEnvWrapper(Wrapper):
    def __init__(self, env, state_manager, action_manager):
        super().__init__(env)
        self.state_manager = state_manager
        self.action_manager = action_manager
        self.done = False
        self.last_observation = None

    def _process_observation(self, observation):
        img, vec = self.state_manager.extract_image_and_vector(observation)
        torch_img, torch_vec = self.state_manager.convert_to_torch_tensor([img], [vec])
        return torch_img, torch_vec

    def reset(self):
        observation = self.env.reset()
        self.last_observation = observation
        self.done = False
        return self._process_observation(observation)

    def step(self, action_id):
        assert not self.done, "Environment is done, reset required."

        action = self.action_manager.get_action(action_id)
        if 'craft' in action:
            if all(action[key] == 0 for key in ['attack', 'craft', 'nearbyCraft', 'nearbySmelt', 'place']):
                action['jump'] = 1
        else:
            if action['attack'] == 0:
                action['jump'] = 1

        observation, reward, self.done, info = super().step(action)
        self.last_observation = observation
        torch_img, torch_vec = self._process_observation(observation)
        return torch_img, torch_vec, reward, self.done


def evaluate_policy(writer, wrapped_env, policy, initial_img, initial_vec, episodes=100):
    rewards = []
    episode_lengths = []
    save_inventory_episodes = 30

    for episode in range(episodes):
        img, vec = (initial_img, initial_vec) if episode == 0 else wrapped_env.reset()
        writer.add_scalar('episode/start', episode, 0)

        total_reward = 0.0
        steps = 0
        done = False

        while not done:
            action_id = policy(img, vec)
            img, vec, reward, done = wrapped_env.step(action_id)
            total_reward += reward
            steps += 1

            if 'inventory' in wrapped_env.last_observation:
                inventory = deepcopy(wrapped_env.last_observation['inventory'])
                if episode < save_inventory_episodes:
                    writer.add_scalars(f"episode {episode} inventory", inventory, steps)

        rewards.append(total_reward)
        episode_lengths.append(steps)
        writer.add_scalar("reward", total_reward, episode)
        writer.add_scalar("steps", steps, episode)
        writer.flush()

        print(f"Episode {episode} - Reward: {total_reward}, Steps: {steps}")

    avg_reward = np.mean(rewards)
    avg_steps = np.mean(episode_lengths)

    print(f"Average Reward: {avg_reward}")
    writer.add_scalar("avg_reward", avg_reward, 0)
    writer.add_scalar("avg_steps", avg_steps, 0)
    writer.flush()
    writer.close()
