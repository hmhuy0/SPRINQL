import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch import distributions as pyd
from torch.autograd import Variable, grad

import Sources.utils.utils as utils

EPS = 1e-7

def orthogonal_init_(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)

class DoubleQCritic(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim, hidden_depth, args):
        """
        Initialize a DoubleQCritic.

        Args:
            obs_dim (int): Dimension of the observation space.
            action_dim (int): Dimension of the action space.
            hidden_dim (int): Dimension of the hidden layers.
            hidden_depth (int): Number of hidden layers.
            args: Additional arguments.

        """

        super(DoubleQCritic, self).__init__()
        self.obs_dim = obs_dim

        self.action_dim = action_dim
        self.args = args

        # Q1 architecture

        self.Q1 = utils.mlp(obs_dim + action_dim, hidden_dim, 1, hidden_depth)
        # Q2 architecture
        self.Q2 = utils.mlp(obs_dim + action_dim, hidden_dim, 1, hidden_depth)
        self.apply(orthogonal_init_)

    def forward(self, obs, action, both=False):
        """

        Forward pass of the DoubleQCritic.

        Args:
            obs (torch.Tensor): Observation input.
            action (torch.Tensor): Action input.
            both (bool): Whether to return both Q values.

        Returns:

            torch.Tensor: Q value(s).

        """
        assert obs.size(0) == action.size(0)
        obs_action = torch.cat([obs, action], dim=-1)
        q1 = self.Q1(obs_action)

        q2 = self.Q2(obs_action)
        if both:
            return q1, q2
        else:
            return torch.min(q1, q2)

class SingleQCritic(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim, hidden_depth, args):
        """
        Initialize a SingleQCritic.

        Args:
            obs_dim (int): Dimension of the observation space.
            action_dim (int): Dimension of the action space.
            hidden_dim (int): Dimension of the hidden layers.
            hidden_depth (int): Number of hidden layers.
            args: Additional arguments.

        """

        super(SingleQCritic, self).__init__()
        self.obs_dim = obs_dim

        self.action_dim = action_dim

        self.args = args
        # Q architecture
        self.Q = utils.mlp(obs_dim + action_dim, hidden_dim, 1, hidden_depth)
        self.apply(orthogonal_init_)

    def forward(self, obs, action):
        """

        Forward pass of the SingleQCritic.

        Args:

            obs (torch.Tensor): Observation input.
            action (torch.Tensor): Action input.

        Returns:

            torch.Tensor: Q value.

        """
        assert obs.size(0) == action.size(0)
        obs_action = torch.cat([obs, action], dim=-1)
        q = self.Q(obs_action)
        return q

class RewardFunction(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim, hidden_depth, args):
        """
        Initialize a RewardFunction.

        Args:
            obs_dim (int): Dimension of the observation space.
            action_dim (int): Dimension of the action space.
            hidden_dim (int): Dimension of the hidden layers.
            hidden_depth (int): Number of hidden layers.
            args: Additional arguments.

        """
        super(RewardFunction, self).__init__()
        self.obs_dim = obs_dim

        self.action_dim = action_dim

        self.reward_lower_bound = args.train.reward_lower_bound

        self.reward_scale = args.train.reward_scale
        self.args = args
        self.Q = utils.mlp(obs_dim + action_dim, hidden_dim, 1, hidden_depth)
        self.apply(orthogonal_init_)

    def forward(self, obs, action):
        """

        Forward pass of the RewardFunction.

        Args:
            obs (torch.Tensor): Observation input.
            action (torch.Tensor): Action input.

        Returns:

            torch.Tensor: Reward value.

        """
        assert obs.size(0) == action.size(0)
        obs_action = torch.cat([obs, action], dim=-1)
        q = self.Q(obs_action)
        return self.reward_lower_bound + torch.sigmoid(q)

    def get_reward(self,obs,action):
        """
        Compute the reward.

        Args:
            obs (torch.Tensor): Observation input.
            action (torch.Tensor): Action input.

        Returns:
            torch.Tensor: Reward value.

        """
        reward = self.forward(obs,action)*self.reward_scale
        return reward