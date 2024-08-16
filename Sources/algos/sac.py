import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
import hydra

from Sources.utils.utils import soft_update


class SAC(object):
    """
    Soft Actor-Critic (SAC) algorithm implementation.

    Args:
        obs_dim (int): Dimensionality of the observation space.
        action_dim (int): Dimensionality of the action space.
        batch_size (int): Size of the mini-batch for training.
        args (Namespace): Command-line arguments.

    Attributes:
        gamma (float): Discount factor.
        batch_size (int): Size of the mini-batch for training.
        device (torch.device): Device to run the algorithm on.
        args (Namespace): Command-line arguments.
        first_log (bool): Flag to indicate if it's the first log.
        temp_actor (None or object): Temporary actor object.
        critic_tau (float): Interpolation factor for updating the target critic network.
        learn_temp (bool): Flag to indicate if the temperature parameter is learned.
        actor_update_frequency (int): Frequency of updating the actor network.
        critic_target_update_frequency (int): Frequency of updating the target critic network.
        critic (object): Critic network.
        critic_target (object): Target critic network.
        actor (object): Actor network.
        log_alpha (torch.Tensor): Logarithm of the temperature parameter.
        actor_optimizer (torch.optim.Adam): Optimizer for the actor network.
        critic_optimizer (torch.optim.Adam): Optimizer for the critic network.
        log_alpha_optimizer (torch.optim.Adam): Optimizer for the temperature parameter.
        training (bool): Flag to indicate if the algorithm is in training mode.

    """

    def __init__(self, obs_dim, action_dim, batch_size, args):
        self.gamma = args.gamma
        self.batch_size = batch_size
        self.device = torch.device(args.device)
        self.args = args
        agent_cfg = args.agent
        self.first_log = True
        self.temp_actor = None
        
        self.critic_tau = agent_cfg.critic_tau
        self.learn_temp = agent_cfg.learn_temp
        self.actor_update_frequency = agent_cfg.actor_update_frequency
        self.critic_target_update_frequency = agent_cfg.critic_target_update_frequency

        self.critic = hydra.utils.instantiate(agent_cfg.critic_cfg, args=args).to(self.device)

        self.critic_target = hydra.utils.instantiate(agent_cfg.critic_cfg, args=args).to(
            self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor = hydra.utils.instantiate(agent_cfg.actor_cfg).to(self.device)

        self.log_alpha = torch.tensor(np.log(agent_cfg.init_temp)).to(self.device)
        self.log_alpha.requires_grad = True
        # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
        self.target_entropy = -action_dim

        # optimizers
        self.actor_optimizer = Adam(self.actor.parameters(),
                                    lr=agent_cfg.actor_lr,
                                    betas=agent_cfg.actor_betas,
                                    weight_decay=0.001)
        self.critic_optimizer = Adam(self.critic.parameters(),
                                     lr=agent_cfg.critic_lr,
                                     betas=agent_cfg.critic_betas,
                                     weight_decay=0.001)
        self.log_alpha_optimizer = Adam([self.log_alpha],
                                        lr=agent_cfg.alpha_lr,
                                        betas=agent_cfg.alpha_betas)
        self.train()
        self.critic_target.train()

    def train(self, training=True):
        """
        Set the training mode of the algorithm.

        Args:
            training (bool, optional): Flag to indicate if the algorithm is in training mode. 
                Defaults to True.

        """
        self.training = training
        self.actor.train(training)
        self.critic.train(training)

    @property
    def alpha(self):
        """
        Get the temperature parameter.

        Returns:
            torch.Tensor: Temperature parameter.

        """
        return self.log_alpha.exp()

    @property
    def critic_net(self):
        """
        Get the critic network.

        Returns:
            object: Critic network.

        """
        return self.critic

    @property
    def critic_target_net(self):
        """
        Get the target critic network.

        Returns:
            object: Target critic network.

        """
        return self.critic_target

    def choose_action(self, state, sample=False):
        """
        Choose an action based on the given state.

        Args:
            state (numpy.ndarray): Current state.
            sample (bool, optional): Flag to indicate if the action should be sampled from the 
                distribution or not. Defaults to False.

        Returns:
            numpy.ndarray: Chosen action.

        """
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        dist = self.actor(state)
        action = dist.sample() if sample else dist.mean
        return action.detach().cpu().numpy()[0]

    def getV(self, obs):
        """
        Get the value function for the given observations.

        Args:
            obs (numpy.ndarray): Observations.

        Returns:
            torch.Tensor: Value function.

        """
        action, log_prob, _ = self.actor.sample(obs)
        current_Q = self.critic(obs, action)
        current_V = current_Q - self.alpha.detach() * log_prob
        return current_V

    def get_targetV(self, obs):
        """
        Get the target value function for the given observations.

        Args:
            obs (numpy.ndarray): Observations.

        Returns:
            torch.Tensor: Target value function.

        """
        action, log_prob, _ = self.actor.sample(obs)
        target_Q = self.critic_target(obs, action)
        target_V = target_Q - self.alpha.detach() * log_prob
        return target_V

    def update(self, replay_buffer, logger, step):
        """
        Update the actor and critic networks.

        Args:
            replay_buffer (object): Replay buffer.
            logger (object): Logger for logging the training progress.
            step (int): Current training step.

        Returns:
            dict: Dictionary of losses.

        """
        obs, next_obs, action, reward, done = replay_buffer.get_samples(
            self.batch_size, self.device)

        losses = self.update_critic(obs, action, reward, next_obs, done,
                                    logger, step)

        if step % self.actor_update_frequency == 0:
            actor_alpha_losses = self.update_actor_and_alpha(obs, logger, step)
            losses.update(actor_alpha_losses)

        if step % self.critic_target_update_frequency == 0:
            soft_update(self.critic, self.critic_target,
                        self.critic_tau)

        return losses

    def update_actor(self, obs):
        """
        Update the actor network.

        Args:
            obs (numpy.ndarray): Observations.

        Returns:
            dict: Dictionary of losses.

        """
        action, log_prob, _ = self.actor.sample(obs)
        actor_Q = self.critic(obs, action)

        actor_loss = (self.alpha.detach() * log_prob - actor_Q).mean()

        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        losses = {
            'actor_loss/total_loss': actor_loss.item(),
            'update/log_prob': log_prob.mean().item(),
            'update/log_alpha': self.log_alpha.item(),
            }

        if self.learn_temp:
            self.log_alpha_optimizer.zero_grad()
            alpha_loss = (self.alpha *
                          (-log_prob - self.target_entropy).detach()).mean()

            alpha_loss.backward()
            self.log_alpha_optimizer.step()

            losses.update({
                'alpha_loss/loss': alpha_loss.item(),
                'alpha_loss/value': self.alpha.item(),
            })
        return losses

    def save(self, path, suffix=""):
        """
        Save the model parameters.

        Args:
            path (str): Path to save the model parameters.
            suffix (str, optional): Suffix to add to the saved file names. Defaults to "".

        """
        actor_path = f"{path}/actor.pth"
        critic_path = f"{path}/critic.pth"
        critic_target_path = f"{path}/critic_target.pth"

        torch.save(self.actor.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)
        torch.save(self.critic_target.state_dict(), critic_target_path)

    def load(self, path, suffix=""):
        """
        Load the model parameters.

        Args:
            path (str): Path to load the model parameters.
            suffix (str, optional): Suffix added to the saved file names. Defaults to "".

        """
        self.actor.load_state_dict(
            torch.load(f'{path}/actor.pth', 
                       map_location=self.device))
        self.critic.load_state_dict(
            torch.load(f'{path}/critic.pth', 
                       map_location=self.device))
        self.critic_target.load_state_dict(
            torch.load(f'{path}/critic_target.pth', 
                       map_location=self.device))

    def sample_actions(self, obs, num_actions):
        """
        Sample multiple actions for the given observations.

        Args:
            obs (numpy.ndarray): Observations.
            num_actions (int): Number of actions to sample.

        Returns:
            tuple: Tuple containing the sampled actions and their log probabilities.

        """
        obs_temp = obs.unsqueeze(1).repeat(1, num_actions, 1).view(
            obs.shape[0] * num_actions, obs.shape[1])
        action, log_prob, _ = self.actor.sample(obs_temp)
        return action, log_prob.view(obs.shape[0], num_actions, 1)

    def _get_tensor_values(self, obs, actions, network=None):
        """
        Get the tensor values for CQL style training.

        Args:
            obs (numpy.ndarray): Observations.
            actions (numpy.ndarray): Actions.
            network (object, optional): Network to compute the tensor values. Defaults to None.

        Returns:
            torch.Tensor: Tensor values.

        """
        action_shape = actions.shape[0]
        obs_shape = obs.shape[0]
        num_repeat = int(action_shape / obs_shape)
        obs_temp = obs.unsqueeze(1).repeat(1, num_repeat, 1).view(
            obs.shape[0] * num_repeat, obs.shape[1])
        inputs = torch.cat([obs_temp, actions], dim=-1)
        preds = network(inputs)
        preds = preds.view(obs.shape[0], num_repeat, 1)
        return preds

    def cqlV(self, obs, network, num_random=10):
        """
        Compute the CQL value for the given observations.

        Args:
            obs (numpy.ndarray): Observations.
            network (object): Network to compute the CQL value.
            num_random (int, optional): Number of random actions to sample. Defaults to 10.

        Returns:
            torch.Tensor: CQL value.

        """
        action, log_prob = self.sample_actions(obs, num_random)
        current_Q = self._get_tensor_values(obs, action, network)

        random_action = torch.FloatTensor(
            obs.shape[0] * num_random, action.shape[-1]).uniform_(-1, 1).to(self.device)

        random_density = np.log(0.5 ** action.shape[-1])
        rand_Q = self._get_tensor_values(obs, random_action, network)
        alpha = self.alpha.detach()

        cat_Q = torch.cat(
            [rand_Q - alpha * random_density, current_Q - alpha * log_prob.detach()], 1
        )
        cql_V = torch.logsumexp(cat_Q / alpha, dim=1).mean() * alpha
        return cql_V
