import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal

from .util import mlp


LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0


class GaussianPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=256, n_hidden=2, max_action=None):
        super().__init__()
        self.net = mlp([obs_dim, *([hidden_dim] * n_hidden), act_dim])
        self.log_std = nn.Parameter(torch.zeros(act_dim, dtype=torch.float32))
        #### None (default, matches original D4RL usage) = unbounded mean, exactly as before.
        #### When set, bounds the mean itself via tanh so it can never leave
        #### [-max_action, max_action] no matter how far obs is from the training
        #### distribution -- without this, a state that's slightly out-of-distribution can
        #### make the net extrapolate to an arbitrarily large action (observed: >20 m/s
        #### commands from a dataset whose actions never exceeded ~2 m/s), which then
        #### drives the closed-loop rollout further out-of-distribution and compounds.
        self.max_action = max_action

    def forward(self, obs):
        mean = self.net(obs)
        if self.max_action is not None:
            mean = self.max_action * torch.tanh(mean / self.max_action)
        std = torch.exp(self.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX))
        scale_tril = torch.diag(std)
        return MultivariateNormal(mean, scale_tril=scale_tril)
        # if mean.ndim > 1:
        #     batch_size = len(obs)
        #     return MultivariateNormal(mean, scale_tril=scale_tril.repeat(batch_size, 1, 1))
        # else:
        #     return MultivariateNormal(mean, scale_tril=scale_tril)

    def act(self, obs, deterministic=False, enable_grad=False):
        with torch.set_grad_enabled(enable_grad):
            dist = self(obs)
            return dist.mean if deterministic else dist.sample()


class DeterministicPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=256, n_hidden=2, max_action=None):
        super().__init__()
        self.net = mlp([obs_dim, *([hidden_dim] * n_hidden), act_dim],
                       output_activation=nn.Tanh)
        #### net's Tanh already bounds to [-1, 1] (the original D4RL assumption -- default
        #### None leaves that as-is); set max_action to rescale to the dataset's real range.
        self.max_action = max_action

    def forward(self, obs):
        out = self.net(obs)
        return out if self.max_action is None else self.max_action * out

    def act(self, obs, deterministic=False, enable_grad=False):
        with torch.set_grad_enabled(enable_grad):
            return self(obs)