import types
import hydra
import os
import torch

from Sources.algos.sac import SAC
from Sources.algos.sprinql import update, update_critic, update_actor, update_single_critic

def make_agent(env, device, args):
    """
    Create an agent for reinforcement learning based on the given environment, device, and arguments.

    Args:
        env (gym.Env): The environment for the agent to interact with.
        device (torch.device): The device to run the agent on.
        args (OmegaConf): The configuration arguments for creating the agent.

    Returns:
        agent (SAC): The created agent for reinforcement learning.

    Raises:
        NotImplementedError: If no trained reward function is found for the given seed and environment.

    """

    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    args.agent.obs_dim = obs_dim
    args.agent.action_dim = action_dim
    
    agent = SAC(obs_dim, action_dim, args.train.batch, args)
    agent.update = types.MethodType(update, agent)
    if (args.env.name == 'Hopper-v3'):
        agent.update_critic = types.MethodType(update_single_critic, agent)
    else:
        agent.update_critic = types.MethodType(update_critic, agent)
    agent.update_actor = types.MethodType(update_actor, agent)
    
    
    if (args.train.use_reward_function):
        load_dir = hydra.utils.to_absolute_path(f'ref_reward/new_{args.seed}_{args.env.name}')
        if not os.path.exists(load_dir):
                load_dir = hydra.utils.to_absolute_path(f'ref_reward/trained_{args.seed}_{args.env.name}')
                if not os.path.exists(load_dir):
                    raise NotImplementedError(f'No trained reward function found for seed {args.seed} and env {args.env.name}')
                
        from Sources.algos.critic import RewardFunction
        agent.reward_function = RewardFunction(obs_dim=obs_dim,
                                        action_dim=action_dim,
                                        hidden_dim=256,
                                        hidden_depth=1,
                                        args=args).to(device)
        reward_f_path = f'{load_dir}/best.pth'
        agent.reward_function.load_state_dict(torch.load(reward_f_path))
        print('load reward function from:',reward_f_path)
    else:
        reward_arr = args.expert.reward_arr
        print(f'fixed rewards for datasets: {reward_arr}')  
    
    return agent