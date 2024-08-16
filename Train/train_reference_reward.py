import sys
sys.path.append('.')
sys.path.append('../')

import os
import random

import hydra
import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from omegaconf import DictConfig

from Sources.utils.make_envs import make_env
from Sources.dataset.memory import Memory
from Sources.dataset.expert_dataset import Traj_dataset
def get_args(cfg: DictConfig):
    cfg.device = "cuda"
    cfg.hydra_base_dir = os.getcwd()
    return cfg

@hydra.main(config_path="../parameters", config_name="config")
def main(cfg: DictConfig):
    """
    Main function for training the reference reward model.

    Args:
        cfg (DictConfig): Configuration dictionary.

    Returns:
        None
    """

    args = get_args(cfg)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    if device.type == 'cuda' and torch.cuda.is_available() and args.cuda_deterministic:
        print('CUDA deterministic')
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    print('device = ',device)
    env = make_env(args)
    from Sources.algos.critic import RewardFunction
    reward_function = RewardFunction(obs_dim=env.observation_space.shape[0],
                                     action_dim=env.action_space.shape[0],
                                     hidden_dim=256,
                                     hidden_depth=1,
                                     args=args).to(device)

    reward_optimizer = Adam(reward_function.parameters(),
                                     lr=5e-3)

    expert_buffer = []
    max_size = 0
    obs_arr = []
    for id,(dir,num) in enumerate(zip(args.env.sub_optimal_demo,args.env.num_sub_optimal_demo)):
        add_memory_replay = Memory(1, args.seed)
        obs_arr.append(add_memory_replay.load(hydra.utils.to_absolute_path(f'experts/{args.env.name}/{dir}.hdf5'),
                                num_trajs=num,
                                sample_freq=1,
                                seed=args.seed))
        expert_buffer.append(add_memory_replay)
        print(f'--> Add memory {id} size: {add_memory_replay.size()}')
        max_size = max(max_size,len(add_memory_replay.full_trajs['states']))
    obs_arr = np.concatenate(obs_arr,axis=0)
    shift = -np.mean(obs_arr, 0)
    scale = 1.0 / (np.std(obs_arr, 0) + 1e-3)
    for buffer in expert_buffer:
        buffer.shift = shift
        buffer.scale = scale
    partial_len = 50
    datasets = []
    for id,expert in enumerate(expert_buffer):
        state = expert.full_trajs['states']
        action = expert.full_trajs['actions']
        traj_dataset = Traj_dataset(states=state,actions=action,
                                    shift=shift,scale=scale,
                                    label=id,dup=int(max_size//len(state))-1,partial_len=partial_len)
        datasets.append(traj_dataset)

    total_iter = 10000
    save_dir = hydra.utils.to_absolute_path(f'ref_reward/new_{args.env.name}')
    os.makedirs(save_dir,exist_ok=True)
    
    for iter in range(total_iter+1):
        loaders = []
        for data in datasets:
            loaders.append(DataLoader(data, batch_size=32, shuffle=True))
        for batches in zip(*loaders):
            states_batch,actions_batch,labels_batch = [],[],[]
            for loader_idx, batch in enumerate(batches):
                _states,_actions,_labels = batch
                states_batch.append(_states)
                actions_batch.append(_actions)
                labels_batch.append(_labels)
        states_batch = torch.cat(states_batch,dim=0).to(device)
        actions_batch = torch.cat(actions_batch,dim=0).to(device)
        labels_batch = torch.cat(labels_batch,dim=0).unsqueeze(-1).to(device)
        
        reward = reward_function(states_batch,actions_batch)
        reshaped_reward = reward.view(reward.shape[0], 1, partial_len, 1).expand(-1, partial_len, -1, -1)
        
        sum_reward = torch.mean(reward.squeeze(-1),dim=-1,keepdim=True)
        
        pair_wise_sum = sum_reward - sum_reward.view(sum_reward.shape[1],sum_reward.shape[0])
        pair_wise_label = torch.exp(
            (labels_batch - labels_batch.view(labels_batch.shape[1],labels_batch.shape[0])).clamp(min=0))
        global_loss = (-pair_wise_sum*pair_wise_label).exp().mean()
        local_loss = torch.square(reshaped_reward - reshaped_reward.permute(0, 2, 1, 3)).mean()
        reward_optimizer.zero_grad()
        (global_loss+local_loss).backward()
        reward_optimizer.step()
        
        if (iter %500 ==0):
            reward_Q1,reward_Q2,reward_Q3,reward_mean = [],[],[],[]
            max_arr,min_arr = [],[]
            for buffer in expert_buffer:
                obs, _, action,_,_ = buffer.get_samples(5000, device)
                rewards = reward_function.get_reward(obs,action).detach().cpu().numpy()
                reward_Q1.append(int(np.percentile(rewards, 25)*100)/100)
                reward_Q2.append(int(np.percentile(rewards, 50)*100)/100)
                reward_Q3.append(int(np.percentile(rewards, 75)*100)/100)
                max_arr.append(int(np.max(rewards)*100)/100)
                min_arr.append(int(np.min(rewards)*100)/100)
                reward_mean.append(int(np.mean(rewards)*100)/100)
            print(f'---iter {args.env.name} {iter}/{total_iter}:')
            print(f'mean: {reward_mean}')
            print(f'Q1: {reward_Q1}')
            print(f'Q2: {reward_Q2}')
            print(f'Q3: {reward_Q3}')
            print(f'max: {max_arr}')
            print(f'min: {min_arr}')
            torch.save(reward_function.state_dict(), f'{save_dir}/best.pth')
    
if __name__ == '__main__':
    main()
