import numpy as np
import torch
from torch import nn

class eval_mode(object):
    """
    A context manager that temporarily sets the training mode of one or more models to False.

    Usage:
    ------
    with eval_mode(model1, model2, ...):
        # Code block where models are in evaluation mode

    Parameters:
    -----------
    *models : object
        One or more model objects to be set in evaluation mode.

    Notes:
    ------
    - The `eval_mode` context manager is used to temporarily disable the training mode of one or more models.
    - It is typically used during the evaluation phase of a machine learning model, where the models should not be updated.
    - The `__enter__` method sets the training mode of each model to False.
    - The `__exit__` method restores the original training mode of each model.

    Example:
    --------
    model1 = Model()
    model2 = Model()
    
    with eval_mode(model1, model2):
        # Code block where models are in evaluation mode
    """
    
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(False)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False
    
def evaluate(actor, env, shift, scale, num_episodes=10):
    """
    Evaluates the performance of an actor in an environment over multiple episodes.

    Args:
        actor: The actor model to evaluate.
        env: The environment to evaluate the actor in.
        shift: The shift value to apply to the state before scaling.
        scale: The scale value to apply to the state after shifting.
        num_episodes (optional): The number of episodes to evaluate the actor (default is 10).

    Returns:
        total_returns: A list of the total returns obtained in each episode.

    """
    total_returns = []

    while len(total_returns) < num_episodes:
        state, _ = env.reset()
        done = False
        total_return = 0
        with eval_mode(actor):
            while True:
                state = (state + shift) * scale
                action = actor.choose_action(state, sample=False)
                next_state, reward, done, trunc, info = env.step(action)
                total_return += reward
                state = next_state
                if done or trunc:
                    total_returns.append(total_return)
                    break

    return total_returns

def evaluate_actor(actor, env, shift, scale, num_episodes=10):
    """
    Evaluates the performance of an actor in a given environment.

    Args:
        actor (torch.nn.Module): The actor model to evaluate.
        env (gym.Env): The environment to evaluate the actor in.
        shift (float): The shift value to apply to the state before feeding it to the actor.
        scale (float): The scale value to apply to the state before feeding it to the actor.
        num_episodes (int, optional): The number of episodes to run for evaluation. Defaults to 10.

    Returns:
        list: A list of total returns obtained in each episode.

    """
    total_returns = []

    while len(total_returns) < num_episodes:
        state, _ = env.reset()
        done = False
        total_return = 0
        with eval_mode(actor):
            while True:
                state = (state + shift) * scale
                input_state = torch.FloatTensor(state).cuda().unsqueeze(0)
                action = actor(input_state).mean.detach().cpu().numpy()[0]
                next_state, reward, done, trunc, info = env.step(action)
                total_return += reward
                state = next_state
                if done or trunc:
                    total_returns.append(total_return)
                    break

    return total_returns

def soft_update(net, target_net, tau):
    """
    Performs a soft update of the target network parameters using the current network parameters.

    Args:
        net (torch.nn.Module): The current network.
        target_net (torch.nn.Module): The target network.
        tau (float): The interpolation parameter for the update. The target network parameters are updated
                     as tau * current_network_parameters + (1 - tau) * target_network_parameters.

    Returns:
        None
    """
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data +
                                (1 - tau) * target_param.data)

def hard_update(source, target):
    """
    Hard updates the target model with the parameters from the source model.

    Args:
        source (torch.nn.Module): The source model from which to copy the parameters.
        target (torch.nn.Module): The target model to which the parameters will be copied.

    Returns:
        None
    """
    for param, target_param in zip(source.parameters(), target.parameters()):
        target_param.data.copy_(param.data)

def weight_init(m):
    """
    Initializes the weights of a linear layer using orthogonal initialization.
    
    Args:
        m (nn.Linear): The linear layer to initialize.
        
    Returns:
        None
    """
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)

class MLP(nn.Module):
    """
    A multi-layer perceptron (MLP) neural network model.

    Args:
        input_dim (int): The number of input features.
        hidden_dim (int): The number of units in each hidden layer.
        output_dim (int): The number of output units.
        hidden_depth (int): The number of hidden layers.
        output_mod (optional): The output modification function.

    Attributes:
        trunk (nn.Module): The MLP trunk module.
    
    Methods:
        forward(x): Performs a forward pass through the MLP.

    """

    def __init__(self,
                 input_dim,
                 hidden_dim,
                 output_dim,
                 hidden_depth,
                 output_mod=None):
        super().__init__()
        self.trunk = mlp(input_dim, hidden_dim, output_dim, hidden_depth,
                         output_mod)
        self.apply(weight_init)

    def forward(self, x):
        """
        Performs a forward pass through the MLP.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.

        """
        return self.trunk(x)

def concat_data(add_batches, args):
    """
    Concatenates the data from multiple batches into a single batch.

    Parameters:
    - add_batches (list): A list of batches containing data to be concatenated.
    - args: Additional arguments.

    Returns:
    - batch_state (torch.Tensor): The concatenated batch of states.
    - batch_next_state (torch.Tensor): The concatenated batch of next states.
    - batch_action (torch.Tensor): The concatenated batch of actions.
    - batch_reward (torch.Tensor): The concatenated batch of rewards.
    - batch_done (torch.Tensor): The concatenated batch of done flags.
    - ls_size (list): A list containing the sizes of each concatenated batch.
    """
    batch_state = []
    batch_next_state = []
    batch_action = []
    batch_reward  =[]
    batch_done = []
    
    ls_size = [0]
    for reward,batch in zip(args.expert.reward_arr,add_batches):
        add_batch_state, add_batch_next_state, add_batch_action, add_batch_reward, add_batch_done = batch
        ls_size.append(ls_size[-1]+add_batch_state.shape[0])
        batch_state.append(add_batch_state)
        batch_next_state.append(add_batch_next_state)
        batch_action.append(add_batch_action)
        batch_reward.append(torch.full_like(add_batch_reward,reward))
        batch_done.append(add_batch_done)
    batch_state = torch.cat(batch_state, dim=0)
    batch_next_state = torch.cat(batch_next_state, dim=0)
    batch_action = torch.cat(batch_action, dim=0)
    batch_reward = torch.cat(batch_reward, dim=0)
    batch_done = torch.cat(batch_done, dim=0)
    return batch_state, batch_next_state, batch_action, batch_reward, batch_done,ls_size

def mlp(input_dim, hidden_dim, output_dim, hidden_depth, output_mod=None):
    """
    Constructs a multi-layer perceptron (MLP) neural network.

    Args:
        input_dim (int): The dimensionality of the input data.
        hidden_dim (int): The number of units in each hidden layer.
        output_dim (int): The dimensionality of the output data.
        hidden_depth (int): The number of hidden layers in the network.
        output_mod (torch.nn.Module, optional): An optional module to apply to the output layer.

    Returns:
        torch.nn.Sequential: The constructed MLP neural network.

    """
    
    if hidden_depth == 0:
        mods = [nn.Linear(input_dim, output_dim)]
    else:
        mods = [nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True)]
        for i in range(hidden_depth - 1):
            mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)]
        mods.append(nn.Linear(hidden_dim, output_dim))
    if output_mod is not None:
        mods.append(output_mod)
    trunk = nn.Sequential(*mods)
    return trunk