import torch
import numpy as np
from collections import OrderedDict
from itertools import product
import copy

class GameStateManager:
    def __init__(self, device):
        self.device = device
        self.item_types = ['none', 'wooden_pickaxe', 'stone_pickaxe', 'iron_pickaxe']
        self.float_inventory_defaults = OrderedDict([('dirt', 5.0), ('cobblestone', 100.0), ('stone', 15.0)])
        self.inventory_defaults = OrderedDict([
            ('coal', 16), ('crafting_table', 3), ('furnace', 3), ('cobblestone', 16),
            ('iron_ingot', 8), ('iron_ore', 8), ('iron_pickaxe', 3), ('log', 32),
            ('planks', 64), ('stick', 32), ('stone_pickaxe', 4), ('torch', 16),
            ('wooden_pickaxe', 4)
        ])

    def extract_image_and_vector(self, state):
        img = state['pov']
        item_type = state['equipped_items']['mainhand']['type']
        item_id = self.item_types.index(item_type) if item_type in self.item_types else 0
        vec = [0.0] * len(self.item_types)
        vec[item_id] = 1.0

        for k, v in state['inventory'].items():
            if k in self.float_inventory_defaults:
                avg = self.float_inventory_defaults[k]
                vec += [np.clip(float(v) / avg, 0.0, 5.0 * avg)]
            if k in self.inventory_defaults:
                vec += self._get_item_vector(v, self.inventory_defaults[k])

        return img, vec

    def _get_item_vector(self, amount, max_amount):
        return [1.0 if i < amount else 0.0 for i in range(max_amount)]

    def convert_to_torch_tensor(self, img_list, vec_list):
        img_tensor = torch.tensor(img_list, dtype=torch.float32, device=self.device).div_(255).permute(0, 3, 1, 2)
        vec_tensor = torch.tensor(vec_list, dtype=torch.float32, device=self.device)
        return img_tensor, vec_tensor

class ActionHandler:
    def __init__(self, device, camera_action_magnitude=22.5):
        self.device = device
        self.camera_action_magnitude = camera_action_magnitude

        self.default_action = OrderedDict([
            ('attack', 0), ('back', 0), ('camera', np.array([0.0, 0.0])), ('craft', 0),
            ('equip', 0), ('forward', 0), ('jump', 0), ('left', 0), ('nearbyCraft', 0),
            ('nearbySmelt', 0), ('place', 0), ('right', 0), ('sneak', 0), ('sprint', 0)
        ])

        self.complex_actions = OrderedDict([
            ('craft', [1, 2, 3, 4]), ('equip', [1, 3, 5, 7]),
            ('nearbyCraft', [2, 4, 6, 7]), ('nearbySmelt', [1, 2]), ('place', [1, 4, 5, 6])
        ])

        self.complex_action_keys = list(self.complex_actions.keys())
        self.complex_action_values = list(self.complex_actions.values())

        self.complex_action_names = OrderedDict([
            ('craft', ["none", "torch", "stick", "planks", "crafting_table"]),
            ('equip', ["none", "air", "wooden_axe", "wooden_pickaxe", "stone_axe", "stone_pickaxe", "iron_axe", "iron_pickaxe"]),
            ('nearbyCraft', ["none", "wooden_axe", "wooden_pickaxe", "stone_axe", "stone_pickaxe", "iron_axe", "iron_pickaxe", "furnace"]),
            ('nearbySmelt', ["none", "iron_ingot", "coal"]),
            ('place', ["none", "dirt", "stone", "cobblestone", "crafting_table", "furnace", "torch"])
        ])

        self.camera_actions = OrderedDict([
            ('turn_up', np.array([-camera_action_magnitude, 0.0])),
            ('turn_down', np.array([camera_action_magnitude, 0.0])),
            ('turn_left', np.array([0.0, -camera_action_magnitude])),
            ('turn_right', np.array([0.0, camera_action_magnitude]))
        ])

        self.discrete_actions = ['attack', 'back', 'forward', 'jump', 'left', 'right', 'sprint']
        self.all_actions = self.discrete_actions + list(self.camera_actions.keys())

        self.mutually_exclusive = [
            ('forward', 'back'), ('left', 'right'), ('attack', 'jump'),
            ('turn_up', 'turn_down', 'turn_left', 'turn_right')
        ]

        self.dependent_actions = [('sprint', 'forward')]

        self.max_active_actions = 3

        self.priority_removal = [
            'sprint', 'left', 'right', 'back', 'turn_up', 'turn_down',
            'turn_left', 'turn_right', 'attack', 'jump', 'forward'
        ]

        self.all_action_combinations = list(product(range(2), repeat=len(self.all_actions)))

        self._filter_invalid_actions()

        self.action_list = []
        for combination in self.all_action_combinations:
            action = copy.deepcopy(self.default_action)
            for key, value in zip(self.all_actions, combination):
                if key in self.camera_actions:
                    if value:
                        action['camera'] = self.camera_actions[key]
                else:
                    action[key] = value
            self.action_list.append(action)

        self.complex_action_id_mapping = OrderedDict()
        for i, key in enumerate(self.complex_action_keys):
            self.complex_action_id_mapping[key] = OrderedDict()
            for id_ in self.complex_action_values[i]:
                action = copy.deepcopy(self.default_action)
                action[key] = id_
                self.complex_action_id_mapping[key][id_] = len(self.action_list)
                self.action_list.append(action)

        self.num_action_ids = [len(self.action_list)]
        self.continuous_action_size = 0

    def _filter_invalid_actions(self):
        invalid_actions = []
        for combo in self.all_action_combinations:
            for exclusive in self.mutually_exclusive:
                if sum([combo[self.all_actions.index(action)] for action in exclusive]) > 1:
                    invalid_actions.append(combo)
            for action_a, action_b in self.dependent_actions:
                if combo[self.all_actions.index(action_a)] == 1 and combo[self.all_actions.index(action_b)] == 0:
                    invalid_actions.append(combo)
            if sum(combo) > self.max_active_actions:
                invalid_actions.append(combo)
        for action in invalid_actions:
            self.all_action_combinations.remove(action)

    def get_action(self, action_id):
        action = copy.deepcopy(self.action_list[int(action_id)])
        action['camera'] += np.random.normal(0.0, 0.5, 2)
        return action

    def print_action(self, action_id):
        action = copy.deepcopy(self.action_list[int(action_id)])
        action_str = ""
        for k, v in action.items():
            if k != 'camera':
                if v != 0:
                    if k in self.complex_action_names:
                        action_str += f'{k} {self.complex_action_names[k][v]} '
                    else:
                        action_str += f'{k} '
            else:
                if (v != np.zeros(2)).any():
                    action_str += k
        print(action_str)

    def get_action_id(self, action):
        for key in self.complex_action_keys:
            if action[key] != 0:
                if action[key] in self.complex_action_id_mapping[key]:
                    return self.complex_action_id_mapping[key][action[key]]

        action = copy.deepcopy(action)
        camera = action['camera']
        camera_movement = 0

        if -self.camera_action_magnitude / 2.0 < camera[0] < self.camera_action_magnitude / 2.0:
            action['camera'][0] = 0.0
            if -self.camera_action_magnitude / 2.0 < camera[1] < self.camera_action_magnitude / 2.0:
                action['camera'][1] = 0.0
            else:
                camera_movement = 1
                action['camera'][1] = self.camera_action_magnitude * np.sign(camera[1])
        else:
            camera_movement = 1
            action['camera'][0] = self.camera_action_magnitude * np.sign(camera[0])
            action['camera'][1] = 0.0

        for a, b in self.mutually_exclusive:
            if len(a) == 2 and action[a] and action[b]:
                action[b] = 0
        for a, b in self.dependent_actions:
            if not action[b] and action[a]:
                action[a] = 0

        for priority_action in self.priority_removal:
            if sum([action[key] for key in self.discrete_actions]) > (self.max

_active_actions - camera_movement):
                if priority_action in self.camera_actions:
                    action['camera'] = np.array([0.0, 0.0])
                    camera_movement = 0
                else:
                    action[priority_action] = 0
            else:
                break

        for key in self.camera_actions:
            action[key] = 0
        for key, val in self.camera_actions.items():
            if (action['camera'] == val).all():
                action[key] = 1
                break

        non_complex_values = tuple(action[key] for key in self.all_actions)
        return self.all_action_combinations.index(non_complex_values)

    def get_torch_action(self, action_id_batch):
        action_tensors = [torch.tensor(action_id, dtype=torch.int64, device=self.device) for action_id in action_id_batch]
        return action_tensors

    def get_reversed_mapping(self):
        reversed_mapping = []
        for action in self.action_list:
            reversed_action = copy.deepcopy(action)
            if action['left'] == 1:
                reversed_action['left'] = 0
                reversed_action['right'] = 1
            if action['right'] == 1:
                reversed_action['right'] = 0
                reversed_action['left'] = 1
            if (action['camera'] == [0, -22.5]).all():
                reversed_action['camera'][1] = 22.5
            if (action['camera'] == [0, 22.5]).all():
                reversed_action['camera'][1] = -22.5
            reversed_action_id = self.get_action_id(reversed_action)
            reversed_mapping.append(reversed_action_id)
        return reversed_mapping
