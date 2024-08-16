import sys
sys.path.append('.')
sys.path.append('../')

import os
import random

import hydra
from copy import deepcopy
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
import wandb

from Sources.utils.make_envs import make_env
from Sources.utils.make_agent import make_agent
from Sources.dataset.memory import Memory
from Sources.utils.utils import evaluate,soft_update,evaluate_actor

def get_args(cfg: DictConfig):
    """
    Get the arguments for the function.

    Args:
        cfg (DictConfig): The configuration dictionary.

    Returns:
        DictConfig: The updated configuration dictionary.
    """
    cfg.device = "cuda"
    cfg.hydra_base_dir = os.getcwd()
    return cfg

def save(agent, path):
    """
    Save the agent's state to the specified path.

    Args:
        agent: The agent object to be saved.
        path (str): The path where the agent's state will be saved.

    Returns:
        None
    """
    os.makedirs(path,exist_ok=True)
    print(f'save at path {path}')
    agent.save(path)
    
def get_dataset(args):
    """
    Loads and preprocesses the dataset for training.

    Args:
        args: An object containing the arguments for loading the dataset.

    Returns:
        expert_buffer: A list of Memory objects containing the loaded expert data.
        shift: The shift value used for normalizing the observations.
        scale: The scale value used for normalizing the observations.
    """
    
    expert_buffer = []
    obs_arr = []
    for id,(dir,num) in enumerate(zip(args.env.sub_optimal_demo,args.env.num_sub_optimal_demo)):
        add_memory_replay = Memory(1, args.seed)
        obs_arr.append(add_memory_replay.load(hydra.utils.to_absolute_path(f'experts/{args.env.name}/{dir}.hdf5'),
                                num_trajs=num,sample_freq=1,seed=args.seed))
        expert_buffer.append(add_memory_replay)
        print(f'--> Add memory {id} size: {add_memory_replay.size()}')
        
    obs_arr = np.concatenate(obs_arr,axis=0)
    shift = -np.mean(obs_arr, 0)
    scale = 1.0 / (np.std(obs_arr, 0) + 1e-3)
    print(f'normalize observation: shift = {np.mean(shift)}, scale = {np.mean(scale)}')
    for buffer in expert_buffer:
        buffer.shift = shift
        buffer.scale = scale
        
    return expert_buffer, shift, scale

@hydra.main(config_path="../parameters", config_name="config")
def main(cfg: DictConfig):
    """
    Main function for training the SPRINQL model.

    Args:
        cfg (DictConfig): Configuration dictionary.

    Returns:
        None
    """
    args = get_args(cfg)
    print(OmegaConf.to_yaml(args))

    # begin ----------------- initialize and seeding ---------------------------------
    run_name = f'SPRINQL'
    for expert_dir,num_expert in zip(args.env.sub_optimal_demo,args.env.num_sub_optimal_demo):
        run_name += f'-{expert_dir}({int(int(num_expert)/1000)}k)'
    print(run_name)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == 'cuda' and torch.cuda.is_available() and args.cuda_deterministic:
        print('CUDA deterministic')
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    eval_env = make_env(args)
    eval_env.reset(seed=args.seed)
    LEARN_STEPS = int(args.env.learn_steps)
    
    agent = make_agent(eval_env,device, args)
    expert_buffer, shift, scale = get_dataset(args)

    # end ----------------- initialize and seeding ---------------------------------
    
    best_eval_returns = -np.inf
    learn_steps = 0
    agent.temp_actor = deepcopy(agent.actor)
    agent.temp_actor.eval()
    
    for iter in range(LEARN_STEPS+1):
        info = {}
        if learn_steps == LEARN_STEPS:
            print('Finished!')
            return
        train_info = agent.update(expert_buffer, learn_steps)
        info.update(train_info)
        soft_update(agent.actor, agent.temp_actor,3e-5)
        
        if learn_steps % args.env.eval_interval == 0:
            eval_returns = evaluate(agent, eval_env,shift,scale, num_episodes=10)
            mean_value = np.mean(eval_returns)
            std_value = np.std(eval_returns)            
            if mean_value > best_eval_returns:
                best_eval_returns = mean_value
                best_learn_steps = learn_steps
                print(f'Best eval: {best_eval_returns:.2f} ± {std_value:.2f}, step={best_learn_steps}')
                save(agent,f'{hydra.utils.to_absolute_path(f".")}/trained_agents/{args.env.name}/best')
            
            temp_mean = evaluate_actor(agent.temp_actor, eval_env,shift,scale, num_episodes=10)
            info['Train/temp_mean'] = np.mean(temp_mean)
            
            info['Eval/mean'] = mean_value
            info['Eval/best_eval'] = best_eval_returns
            try:
                wandb.log(info,step = learn_steps)
            except:
                print(f'Step {learn_steps}:',info)
        learn_steps += 1
  
if __name__ == '__main__':
    main()