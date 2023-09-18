import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy

def freeze(model):
    assert isinstance(model, nn.Module)
    for p in model.parameters():
        p.requires_grad = False

def unfreeze(model):
    assert isinstance(model, nn.Module)
    for p in model.parameters():
        p.requires_grad = True

class ConservativeObjectiveModel:
    def __init__(
        self,
        forward_model,
        forward_model_opt=optim.Adam,
        forward_model_lr=0.001,
        alpha=1.0,
        alpha_opt=optim.Adam,
        alpha_lr=0.01,
        overestimation_limit=0.5,
        particle_lr=0.05,
        particle_gradient_steps=50,
        entropy_coefficient=0.0,
        noise_std=0.0,
    ):
        super().__init__()
        self.forward_model = forward_model
        self.forward_model_opt = forward_model_opt(
            self.forward_model.parameters(), lr=forward_model_lr
        )
        # lagrangian dual descent variable (controls the conservatism of the model)
        self.log_alpha = nn.Parameter(torch.tensor(np.log(alpha).astype(np.float64)))
        self.alpha_opt = alpha_opt([self.log_alpha], lr=alpha_lr)

        self.overestimation_limit = overestimation_limit
        self.particle_lr = particle_lr
        self.particle_gradient_steps = particle_gradient_steps
        self.entropy_coefficient = entropy_coefficient
        self.noise_std = noise_std

    def optimize(self, x, steps, allow_grad=False, **kwargs):
        """Generate adversarial examples from starting point x
        used later for adversarial training. 
        """
        xt = nn.Parameter(deepcopy(x).to(x.device), requires_grad=True)
        for _ in range(steps):
            # shuffle the designs for calculating entropy
            idx = torch.randperm(xt.shape[0])
            shuffled_xt = xt[idx]

            # entropy using the gaussian kernel
            entropy = torch.mean((xt - shuffled_xt) ** 2)
            score = self.forward_model(xt)  

            loss = self.entropy_coefficient * entropy + score
            (grad,) = torch.autograd.grad(torch.sum(loss), xt)
            xt = xt + self.particle_lr * grad
        return xt.clone().detach()

    def train_step(self, x, y):
        # (potentially) corrupt the inputs with noise
        # depends on the hyper-parameter `noise_std`
        x = x + self.noise_std * torch.randn_like(x)

        statistics = dict()

        # freeze the model to not update the weights while generating 
        # adversarial examples
        freeze(self.forward_model)
        # calculate negative samples starting from the dataset
        x_neg = self.optimize(x, self.particle_gradient_steps)
        unfreeze(self.forward_model)
 
        d_pos = self.forward_model(x)
        loss_fn = nn.MSELoss()
        mse = loss_fn(d_pos, y)
        statistics[f"train/mse"] = mse.detach().clone()

        d_neg = self.forward_model(x_neg) 
        overestimation = d_neg[:, 0] - d_pos[:, 0]
        statistics[f"train/overestimation"] = torch.mean(overestimation.detach().clone())
        # if statistics[f"train/overestimation"] >= 1:
        #     print(d_neg.squeeze()[:5])
        #     print(d_pos.squeeze()[:5])

        alpha_loss = (
            self.log_alpha.exp() * self.overestimation_limit
            - self.log_alpha.exp() * overestimation
        )
        statistics[f"train/alpha"] = self.log_alpha.exp().detach().clone()

        model_loss = mse + self.log_alpha.exp().detach().clone() * overestimation
        total_loss = torch.mean(model_loss)

        alpha_loss = torch.mean(alpha_loss)
        statistics[f"train/alpha_loss"] = alpha_loss

        self.alpha_opt.zero_grad()
        alpha_loss.backward(retain_graph=True)


        self.forward_model_opt.zero_grad()
        total_loss.backward()

        self.alpha_opt.step()
        self.forward_model_opt.step()

        return statistics

    def validate_step(self, x, y):
        statistics = dict()
        with torch.no_grad():
            # calculate the prediction error and accuracy of the model
            d_pos = self.forward_model(x)  
            mse = torch.nn.functional.mse_loss(d_pos, y)
            statistics[f"validate/mse"] = mse
        return statistics

    def train(self, dataset):
        statistics = {}
        for x, y in dataset:
            for name, tensor in self.train_step(x, y).items():
                if name not in statistics.keys():
                    statistics[name] = [tensor]
                else:
                    statistics[name].append(tensor)
        for name in statistics.keys():
            statistics[name] = torch.cat([x.unsqueeze(0) for x in statistics[name]])
        return statistics

    def validate(self, dataset):
        statistics = {}
        for x, y in dataset:
            for name, tensor in self.validate_step(x, y).items():
                if name not in statistics.keys():
                    statistics[name] = [tensor]
                else:
                    statistics[name].append(tensor)
        for name in statistics.keys():
            statistics[name] = torch.cat([x.unsqueeze(0) for x in statistics[name]])
        return statistics

    def launch(self, train_data, validate_data, epochs, logger=None):
        """Entry point for training the model.
        """
        for e in range(epochs):
            print(f'epoch {e}')
            for name, value in self.train(train_data).items():
                if name == "train/alpha":
                    alpha = value
                    print(f"alpha min {alpha.min()}, max {alpha.max()}")
                if name == "train/overestimation":
                    print(f"overestimation min {value.min()}, max {value.max()}")
                if name == "train/alpha_loss":
                    print(f"alpha loss: {value.mean()}")
                if name == "train/mse":
                    print(f"train loss: {value.mean()}", end="  |  ")
            for name, loss in self.validate(validate_data).items():
                print(f"validation loss: {loss.mean()}")
        return alpha
