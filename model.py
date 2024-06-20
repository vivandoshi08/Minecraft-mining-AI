import torch
from torch import nn
from torch.nn import functional as F
import math

class Network(nn.Module):
    def __init__(self, num_actions, image_channels, vector_size, cnn_module, hidden_size=256,
                 dueling=True, double_channels=False):
        super().__init__()
        self.num_actions = num_actions
        self.dueling = dueling

        self.cnn = cnn_module(image_channels)
        self.conv_output_size = self.cnn.output_size

        self.fc_image = nn.Linear(self.conv_output_size, hidden_size)
        self.fc_vector = nn.Linear(vector_size, 256 if double_channels else 128)
        self.fc_hidden_a = nn.Linear(hidden_size + (256 if double_channels else 128), hidden_size)
        self.fc_actions = nn.Linear(hidden_size, num_actions)

        if self.dueling:
            self.fc_hidden_v = nn.Linear(hidden_size + (256 if double_channels else 128), hidden_size)
            self.fc_value = nn.Linear(hidden_size, 1)

    def forward(self, image, vector):
        image_features = self.cnn(image)
        image_features = image_features.view(-1, self.conv_output_size)
        image_features = self.fc_image(image_features)
        
        vector_features = self.fc_vector(vector)
        combined_features = F.relu(torch.cat((image_features, vector_features), dim=1))

        action_advantages = self.fc_actions(F.relu(self.fc_hidden_a(combined_features)))
        
        if self.dueling:
            state_value = self.fc_value(F.relu(self.fc_hidden_v(combined_features)))
            return state_value + action_advantages - action_advantages.mean(dim=1, keepdim=True)

        return action_advantages


class ImpalaResNetCNN(nn.Module):
    class ResidualBlock(nn.Module):
        def __init__(self, channels):
            super().__init__()
            self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
            self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

        def forward(self, x):
            residual = x
            out = F.relu(self.conv1(x))
            out = self.conv2(out)
            return F.relu(out + residual)

    def __init__(self, input_channels):
        super().__init__()
        layers = []
        channels = input_channels

        for out_channels in [32, 64, 64]:
            layers.extend([
                nn.Conv2d(channels, out_channels, 3, padding=1),
                nn.MaxPool2d(3, stride=2, padding=1),
                self.ResidualBlock(out_channels),
                self.ResidualBlock(out_channels),
            ])
            channels = out_channels

        self.conv_layers = nn.Sequential(*layers, nn.ReLU())
        self.output_size = math.ceil(64 / 8) ** 2 * channels

    def forward(self, x):
        return self.conv_layers(x)

class FixupResNetCNN(nn.Module):
    class FixupResidualBlock(nn.Module):
        def __init__(self, channels, num_residuals):
            super().__init__()
            self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
            self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
            self.bias1 = nn.Parameter(torch.zeros([channels, 1, 1]))
            self.bias2 = nn.Parameter(torch.zeros([channels, 1, 1]))
            self.bias3 = nn.Parameter(torch.zeros([channels, 1, 1]))
            self.bias4 = nn.Parameter(torch.zeros([channels, 1, 1]))
            self.scale = nn.Parameter(torch.ones([channels, 1, 1]))

            for param in self.conv1.parameters():
                param.data.mul_(1 / math.sqrt(num_residuals))
            for param in self.conv2.parameters():
                param.data.zero_()

        def forward(self, x):
            out = F.relu(x + self.bias1)
            out = self.conv1(out) + self.bias2
            out = F.relu(out + self.bias3)
            out = self.conv2(out) * self.scale + self.bias4
            return out + x

    def __init__(self, input_channels, double_channels=False):
        super().__init__()
        channels = input_channels
        layers = []

        channel_sizes = [64, 128, 128] if double_channels else [32, 64, 64]
        for out_channels in channel_sizes:
            layers.extend([
                nn.Conv2d(channels, out_channels, 3, padding=1),
                nn.MaxPool2d(3, stride=2, padding=1),
                self.FixupResidualBlock(out_channels, 8),
                self.FixupResidualBlock(out_channels, 8),
            ])
            channels = out_channels

        layers.extend([
            self.FixupResidualBlock(channels, 8),
            self.FixupResidualBlock(channels, 8),
        ])

        self.conv_layers = nn.Sequential(*layers, nn.ReLU())
        self.output_size = math.ceil(64 / 8) ** 2 * channels

    def forward(self, x):
        return self.conv_layers(x)
