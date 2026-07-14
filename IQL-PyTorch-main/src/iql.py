import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

from . import util  # reference util.DEFAULT_DEVICE dynamically so main.py's --device override is honored
from .util import compute_batched, update_exponential_moving_average


EXP_ADV_MAX = 100.


def asymmetric_l2_loss(u, tau):
    return torch.mean(torch.abs(tau - (u < 0).float()) * u**2)


class ImplicitQLearning(nn.Module):
    def __init__(self, qf, vf, policy, optimizer_factory, max_steps,
                 tau, beta, discount=0.99, alpha=0.005, smoothness_coef=0.0):
        super().__init__()
        self.qf = qf.to(util.DEFAULT_DEVICE)
        self.q_target = copy.deepcopy(qf).requires_grad_(False).to(util.DEFAULT_DEVICE)
        self.vf = vf.to(util.DEFAULT_DEVICE)
        self.policy = policy.to(util.DEFAULT_DEVICE)
        self.v_optimizer = optimizer_factory(self.vf.parameters())
        self.q_optimizer = optimizer_factory(self.qf.parameters())
        self.policy_optimizer = optimizer_factory(self.policy.parameters())
        self.policy_lr_schedule = CosineAnnealingLR(self.policy_optimizer, max_steps)
        self.tau = tau
        self.beta = beta
        self.discount = discount
        self.alpha = alpha
        #### Penalizes the policy's mean-action jump between consecutive states in the
        #### same trajectory (observations -> next_observations), independent of the
        #### expert actions in the dataset -- discourages a jittery learned policy even
        #### if the demonstrations themselves are smooth. 0 = original IQL (no penalty).
        self.smoothness_coef = smoothness_coef

    def update(self, observations, actions, next_observations, rewards, terminals):
        with torch.no_grad():
            target_q = self.q_target(observations, actions)
            next_v = self.vf(next_observations)

        # v, next_v = compute_batched(self.vf, [observations, next_observations])

        # Update value function
        v = self.vf(observations)
        adv = target_q - v
        v_loss = asymmetric_l2_loss(adv, self.tau)
        self.v_optimizer.zero_grad(set_to_none=True)
        v_loss.backward()
        self.v_optimizer.step()

        # Update Q function
        targets = rewards + (1. - terminals.float()) * self.discount * next_v.detach()
        qs = self.qf.both(observations, actions)
        q_loss = sum(F.mse_loss(q, targets) for q in qs) / len(qs)
        self.q_optimizer.zero_grad(set_to_none=True)
        q_loss.backward()
        self.q_optimizer.step()

        # Update target Q network
        update_exponential_moving_average(self.q_target, self.qf, self.alpha)

        # Update policy
        exp_adv = torch.exp(self.beta * adv.detach()).clamp(max=EXP_ADV_MAX)
        policy_out = self.policy(observations)
        if isinstance(policy_out, torch.distributions.Distribution):
            bc_losses = -policy_out.log_prob(actions)
            mean_action = policy_out.mean
        elif torch.is_tensor(policy_out):
            assert policy_out.shape == actions.shape
            bc_losses = torch.sum((policy_out - actions)**2, dim=1)
            mean_action = policy_out
        else:
            raise NotImplementedError
        policy_loss = torch.mean(exp_adv * bc_losses)

        if self.smoothness_coef > 0:
            next_policy_out = self.policy(next_observations)
            next_mean_action = next_policy_out.mean if isinstance(
                next_policy_out, torch.distributions.Distribution) else next_policy_out
            smoothness_loss = torch.mean(torch.sum((mean_action - next_mean_action)**2, dim=1))
            policy_loss = policy_loss + self.smoothness_coef * smoothness_loss

        self.policy_optimizer.zero_grad(set_to_none=True)
        policy_loss.backward()
        self.policy_optimizer.step()
        self.policy_lr_schedule.step()

        return {
            'v_loss': v_loss.item(),
            'q_loss': q_loss.item(),
            'policy_loss': policy_loss.item(),
        }