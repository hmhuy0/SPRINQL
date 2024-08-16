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
    """
    Custom weight initialization for Conv2D and Linear layers.
    
    Args:
        m (nn.Module): The module to initialize.
    """

    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)

class TanhTransform(pyd.transforms.Transform):
    """
    Transform class for the Tanh function.
    """
    domain = pyd.constraints.real

    codomain = pyd.constraints.interval(-1.0, 1.0)
    bijective = True

    sign = +1
    
    def __init__(self, cache_size=1):
        super().__init__(cache_size=cache_size)

    
    @staticmethod
    def atanh(x):

        """
        Inverse hyperbolic tangent function.
        
        Args:
            x (torch.Tensor): Input tensor.
        
        Returns:
            torch.Tensor: Output tensor.
        """
        return 0.5 * (x.log1p() - (-x).log1p())
    
    def __eq__(self, other):
        """

        Check if two TanhTransform objects are equal.
        
        Args:
            other: The other object to compare.
        
        Returns:
            bool: True if the objects are equal, False otherwise.
        """
        return isinstance(other, TanhTransform)
    
    def _call(self, x):
        """
        Forward transformation using the Tanh function.
        
        Args:

            x (torch.Tensor): Input tensor.
        
        Returns:
            torch.Tensor: Output tensor.
        """
        return x.tanh()
    
    def _inverse(self, y):
        """
        Inverse transformation using the inverse hyperbolic tangent function.
        
        Args:

            y (torch.Tensor): Input tensor.
        
        Returns:
            torch.Tensor: Output tensor.
        """
        return self.atanh(y)
    
    def log_abs_det_jacobian(self, x, y):
        """
        Compute the logarithm of the absolute value of the determinant of the Jacobian matrix.
        
        Args:
            x (torch.Tensor): Input tensor.
            y (torch.Tensor): Output tensor.
        
        Returns:
            torch.Tensor: Logarithm of the absolute value of the determinant of the Jacobian matrix.
        """
        return 2. * (math.log(2.) - x - F.softplus(-2. * x))

class SquashedNormal(pyd.transformed_distribution.TransformedDistribution):
    """
    Transformed distribution class for a squashed normal distribution.
    """

    def __init__(self, loc, scale):
        """
        Initialize a SquashedNormal object.

        
        Args:
            loc (torch.Tensor): Mean of the distribution.
            scale (torch.Tensor): Standard deviation of the distribution.
        """
        self.loc = loc
        self.scale = scale
        self.base_dist = pyd.Normal(loc, scale)
        transforms = [TanhTransform()]
        super().__init__(self.base_dist, transforms)
    
    @property
    def mean(self):
        """
        Compute the mean of the distribution.
        
        Returns:
            torch.Tensor: Mean of the distribution.
        """

        mu = self.loc
        for tr in self.transforms:
            mu = tr(mu)
        return mu

class DiagGaussianActor(nn.Module):
    """

    Implementation of a diagonal Gaussian policy using torch.distributions.
    """
    def __init__(self, obs_dim, action_dim, hidden_dim, hidden_depth, log_std_bounds):

        """
        Initialize a DiagGaussianActor object.
        
        Args:

            obs_dim (int): Dimensionality of the observation space.
            action_dim (int): Dimensionality of the action space.
            hidden_dim (int): Dimensionality of the hidden layers.
            hidden_depth (int): Number of hidden layers.
            log_std_bounds (tuple): Bounds for the logarithm of the standard deviation.
        """
        super().__init__()
        self.log_std_bounds = log_std_bounds

        self.trunk = utils.mlp(obs_dim, hidden_dim, 2 * action_dim, hidden_depth)
        self.outputs = dict()
        self.dist_func = SquashedNormal
        self.apply(orthogonal_init_)
    
    def forward(self, obs):
        """

        Forward pass of the DiagGaussianActor.
        
        Args:
            obs (torch.Tensor): Input observation tensor.
        
        Returns:
            torch.distributions.TransformedDistribution: Distribution over actions.
        """
        mu, log_std = self.trunk(obs).chunk(2, dim=-1)

        log_std = torch.tanh(log_std)
        log_std_min, log_std_max = self.log_std_bounds

        log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std + 1)
        std = log_std.exp()
        dist = self.dist_func(mu, std)
        return dist
    
    def sample(self, obs):
        """
        Sample an action from the policy.
        
        Args:

            obs (torch.Tensor): Input observation tensor.
        
        Returns:

            torch.Tensor: Sampled action.
            torch.Tensor: Log probability of the action.
            torch.Tensor: Mean of the distribution.
        """
        dist = self.forward(obs)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        return action, log_prob, dist.mean
    
    def get_log_prob(self, obs, action):
        """
        Compute the log probability of an action given an observation.
        
        Args:
            obs (torch.Tensor): Input observation tensor.
            action (torch.Tensor): Input action tensor.
        
        Returns:
            torch.Tensor: Log probability of the action.
        """
        dist = self.forward(obs)
        action_clip = torch.clip(action, -1. + EPS, 1. - EPS)
        log_prob = dist.log_prob(action_clip).sum(-1, keepdim=True)
        return log_prob