from typing import Any, Dict, IO, List, Tuple

import numpy as np
import pickle
import torch
from torch.utils.data import Dataset
import os
import random
import h5py

class Traj_dataset(Dataset):
    def __init__(self,
                 states,actions,shift,scale,
                 label,dup,
                 partial_len: int = 50):
        """
        Dataset class for trajectory data.

        Args:
            states: A list of states.
            actions: A list of actions.
            shift: A shift value.
            scale: A scale value.
            label: A label value.
            dup: A duplication value.
            partial_len: The length of partial trajectories.

        """
        self.shift = shift
        self.scale = scale
        self.partial_len = partial_len
        tmp_states,tmp_actions = [],[]
        for s,a in zip(states,actions):
            if (len(s)<self.partial_len):
                continue
            tmp_states.append(s)
            tmp_actions.append(a)
        if (dup>1):
            self.states = []
            self.actions = []
            for _ in range(dup):
                self.states += tmp_states
                self.actions += tmp_actions
        else:
            self.states = tmp_states
            self.actions = tmp_actions
        
        self.labels = [label for _ in range(len(self.states))]


    def __len__(self) -> int:
        """
        Returns the length of the dataset.

        Returns:
            The length of the dataset.

        """
        return len(self.states)

    def __getitem__(self, i):
        """
        Returns the item at the given index.

        Args:
            i: The index of the item.

        Returns:
            A tuple containing the states, actions, and label at the given index.

        """
        idx = random.randint(0, len(self.states[i]) - self.partial_len)
        states = np.array([self.states[i][j] for j in range(idx,idx+self.partial_len)])
        actions = np.array([self.actions[i][j] for j in range(idx,idx+self.partial_len)])
        states = (states + self.shift) * self.scale
        states = states + 0.1*states*np.random.randn(*states.shape).astype(np.float32)
        actions = actions + 0.1*np.random.randn(*actions.shape).astype(np.float32)
        return (states,
                actions,
                self.labels[i])
        


class ExpertDataset(Dataset):
    """
    Dataset for expert trajectories.

    Assumes expert dataset is a dict with keys {states, actions, rewards, lengths} with values containing a list of
    expert attributes of given shapes below. Each trajectory can be of different length.

    Expert rewards are not required but can be useful for evaluation.

        shapes:
            expert["states"]  =  [num_experts, traj_length, state_space]
            expert["actions"] =  [num_experts, traj_length, action_space]
            expert["rewards"] =  [num_experts, traj_length]
            expert["lengths"] =  [num_experts]
    """

    def __init__(self,
                 expert_location: str,
                 num_trajectories: int = 4,
                 subsample_frequency: int = 20,
                 seed: int = 0):
        """
        Initializes the ExpertDataset.

        Args:
            expert_location: Location of saved expert trajectories.
            num_trajectories: Number of expert trajectories to sample (randomized).
            subsample_frequency: Subsamples each trajectory at specified frequency of steps.
            seed: The seed value for random number generation.

        """
        self.all_trajectories,self.full_trajectories = load_trajectories(expert_location, num_trajectories, seed)
        self.length = self.all_trajectories['states'].shape[0]

    def __len__(self) -> int:
        """
        Returns the length of the dataset.

        Returns:
            The length of the dataset.

        """
        return self.length

    def __getitem__(self, i):
        """
        Returns the item at the given index.

        Args:
            i: The index of the item.

        Returns:
            A tuple containing the states, next states, actions, rewards, and dones at the given index.

        """
        return (self.all_trajectories['states'][i],
                self.all_trajectories['next_states'][i],
                self.all_trajectories['actions'][i],
                self.all_trajectories['rewards'][i],
                self.all_trajectories['dones'][i])


def load_trajectories(expert_location: str,
                      num_trajectories: int = 10,
                      seed: int = 0) -> Dict[str, Any]:
    """
    Load expert trajectories.

    Args:
        expert_location: Location of saved expert trajectories.
        num_trajectories: Number of expert trajectories to sample (randomized).
        seed: The seed value for random number generation.

    Returns:
        A dictionary containing the expert data attributes.

    """
    assert os.path.isfile(expert_location)
    
    with open(expert_location, 'rb') as f:
        hdf_trajs = h5py.File(expert_location, 'r')
    starts_timeout = np.where(np.array(hdf_trajs['timeouts'])>0)[0].tolist()
    starts_done = np.where(np.array(hdf_trajs['terminals'])>0)[0].tolist()
    starts = [-1]+starts_timeout+starts_done
    starts = list(dict.fromkeys(starts))
    starts.sort()
    
    rng = np.random.RandomState(seed)
    perm = np.arange(len(starts)-1)
    perm = rng.permutation(perm)
    
    total_length = 0
    for num_traj in range(len(perm)):
        total_length += (starts[perm[num_traj]+1]+1) - (starts[perm[num_traj]]+1)
        if (total_length>=num_trajectories):
            break
    num_traj += 1
    idx = perm[:num_traj]
    trajs = {}
    
    trajs['dones'] = [np.array(hdf_trajs['terminals'][starts[idx[i]]+1:starts[idx[i]+1]+1])
                        for i in range(len(idx))]
    trajs['states'] = [np.array(hdf_trajs['observations'][starts[idx[i]]+1:starts[idx[i]+1]+1])
                        for i in range(len(idx))]
    trajs['initial_states'] = np.array([hdf_trajs['observations'][starts[idx[i]]+1]
                        for i in range(len(idx))])
    trajs['next_states'] = [np.array(hdf_trajs['next_observations'][starts[idx[i]]+1:starts[idx[i]+1]+1])
                        for i in range(len(idx))]
    trajs['actions'] = [np.array(hdf_trajs['actions'][starts[idx[i]]+1:starts[idx[i]+1]+1])
                        for i in range(len(idx))]
    trajs['rewards'] = [hdf_trajs['rewards'][starts[idx[i]]+1:starts[idx[i]+1]+1]
                            for i in range(len(idx))]
    traj_full = {}
    traj_full['states'] = [np.array(hdf_trajs['observations'][starts[idx[i]]+1:starts[idx[i]+1]+1])
                        for i in range(len(idx))]
    traj_full['actions'] = [np.array(hdf_trajs['actions'][starts[idx[i]]+1:starts[idx[i]+1]+1])
                        for i in range(len(idx))]
    traj_full['next_states'] = [np.array(hdf_trajs['next_observations'][starts[idx[i]]+1:starts[idx[i]+1]+1])
                        for i in range(len(idx))]
    traj_full['rewards'] = [np.array(hdf_trajs['rewards'][starts[idx[i]]+1:starts[idx[i]+1]+1])
                        for i in range(len(idx))]
    traj_full['dones'] = [np.array(hdf_trajs['terminals'][starts[idx[i]]+1:starts[idx[i]+1]+1])
                        for i in range(len(idx))]

    trajs['dones'] = np.concatenate(trajs['dones'],axis=0)[:num_trajectories]
    trajs['states'] = np.concatenate(trajs['states'],axis=0)[:num_trajectories]
    trajs['actions'] = np.concatenate(trajs['actions'],axis=0)[:num_trajectories]
    trajs['next_states'] = np.concatenate(trajs['next_states'],axis=0)[:num_trajectories]
    reward_arr = [np.sum(trajs['rewards'][i]) for i in range(len(trajs['rewards']))]
    
    trajs['rewards'] = np.concatenate(trajs['rewards'],axis=0)[:num_trajectories]
    print(f'expert: {expert_location}, {len(idx)}/{len(perm)} trajectories')
    print(f'return: {np.mean(reward_arr):.2f} ± {np.std(reward_arr):.2f}, Q1 = {np.percentile(reward_arr, 25):.2f}'+
          f', Q2 = {np.percentile(reward_arr, 50):.2f}, Q3 = {np.percentile(reward_arr, 75):.2f},'+
          f' min = {np.min(reward_arr):.2f}, max = {np.max(reward_arr):.2f}')
    return trajs,traj_full