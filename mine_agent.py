import numpy as np
import torch
from torch import optim
import torch.nn.functional as F
from model import Network, ImpalaResNetCNN, FixupResNetCNN
from os.path import join as p_join

class DQNAgent:
    def __init__(self, num_actions, image_channels, vector_size, logger, network_type, batch_size, augment_flip, hidden_size, dueling, learning_rate, adam_eps, device):
        self.num_actions = num_actions
        self.logger = logger
        self.batch_size = batch_size
        self.augment_flip = augment_flip

        self.flip_action_map = None

        if self.augment_flip:
            self.flip_action_map = [0, 2, 1, 3, 4, 10, 12, 11, 13, 14, 5, 7, 6, 8, 9, 15, 17, 16, 18, 19,
                                    25, 27, 26, 28, 29, 20, 22, 21, 23, 24, 30, 32, 31, 33, 34, 35, 37,
                                    36, 38, 39, 46, 48, 47, 49, 50, 51, 40, 42, 41, 43, 44, 45, 52, 54,
                                    53, 55, 56, 57, 59, 58, 60, 62, 61, 63, 64, 70, 72, 71, 73, 74, 65,
                                    67, 66, 68, 69, 75, 77, 76, 78, 79, 81, 80, 82, 84, 83, 85, 86, 92,
                                    94, 93, 95, 96, 87, 89, 88, 90, 91, 97, 99, 98, 100, 101, 102, 104,
                                    103, 105, 107, 106, 108, 109, 111, 110, 112, 113, 114, 115, 116, 117,
                                    118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129]

        cnn_module = self._select_network(network_type)
        self.network = Network(num_actions, image_channels, vector_size, cnn_module, hidden_size,
                               dueling=dueling, double_channels=(network_type == 'double_deep_resnet')).to(device=device)

        self.network.train()
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate, eps=adam_eps, weight_decay=1e-5)

    def _select_network(self, network_type):
        if network_type == 'resnet':
            return ImpalaResNetCNN
        elif network_type == 'd_resnet':
            return FixupResNetCNN
        elif network_type == 'dd_resnet':
            return lambda x: FixupResNetCNN(x, double_channels=True)
        else:
            raise ValueError("Unknown network type")

    def select_action(self, img, vec):
        with torch.no_grad():
            logits = self.network(img, vec)
            probabilities = F.softmax(logits, dim=1).cpu().numpy()
            actions = [np.random.choice(len(p), p=p) for p in probabilities]
            assert len(actions) == 1
            return actions[0]

    def update(self, step, dataset, log=False):
        states, vecs, actions, returns, next_states, next_vecs, nonterminals = dataset.sample_line(self.batch_size, 1)

        if self.augment_flip and np.random.binomial(n=1, p=0.5):
            states = torch.flip(states, (3,))
            actions = np.array([self.flip_action_map[action] for action in actions])

        logits = self.network(states, vecs)
        loss = F.cross_entropy(logits, actions)

        if log and self.logger is not None:
            self.logger.add_scalar('loss/cross_entropy', loss.cpu().item(), step)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save_model(self, path, identifier=None):
        model_filename = f'model_{identifier}.pth' if identifier else 'model.pth'
        state_filename = f'state_{identifier}.pth' if identifier else 'state.pth'
        
        torch.save(self.network.state_dict(), p_join(path, model_filename))
        torch.save({'optimizer': self.optimizer.state_dict()}, p_join(path, state_filename))

    def load_model(self, path, identifier=None):
        model_filename = f'model_{identifier}.pth' if identifier else 'model.pth'
        state_filename = f'state_{identifier}.pth' if identifier else 'state.pth'
        
        self.network.load_state_dict(torch.load(p_join(path, model_filename)))
        optimizer_state = torch.load(p_join(path, state_filename))
        self.optimizer.load_state_dict(optimizer_state['optimizer'])

    def set_train_mode(self):
        self.network.train()

    def set_eval_mode(self):
        self.network.eval()
