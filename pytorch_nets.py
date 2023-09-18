import torch
import torch.nn as nn


class TanhMultiplier(nn.Module):
    def __init__(self):
        super(TanhMultiplier, self).__init__()
        self.multiplier = nn.Parameter(torch.ones(1))

    def forward(self, inputs):
        exp_multiplier = torch.exp(self.multiplier)
        return torch.tanh(inputs / exp_multiplier) * exp_multiplier


class ForwardModel(nn.Module):
    def __init__(
        self,
        input_shape,
        activations=("relu", "relu"),
        hidden_size=2048,
        final_tanh=False,
    ):
        """Neural network used as surrogate of the objective function.
        """
        super(ForwardModel, self).__init__()
        layers = [nn.Linear(input_shape, hidden_size)]
        for act in activations:
            if act == "leaky_relu":
                layers.extend([nn.Linear(hidden_size, hidden_size), nn.LeakyReLU()])
            elif isinstance(act, str):
                layers.extend([nn.Linear(hidden_size, hidden_size), nn.ReLU()])
            else:
                layers.extend([nn.Linear(hidden_size, hidden_size), act])
        layers.extend([nn.Linear(hidden_size, 1)])
        if final_tanh:
            layers.extend([TanhMultiplier()])
        self.model = nn.Sequential(*layers)

    def forward(self, inputs):
        return self.model(inputs)
