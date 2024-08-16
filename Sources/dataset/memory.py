from collections import deque
import numpy as np
import random
import torch

from Sources.dataset.expert_dataset import ExpertDataset


class Memory(object):
    """
    A class representing a memory buffer for storing experiences.

    Args:
        memory_size (int): The maximum size of the memory buffer.
        seed (int, optional): The seed value for random number generation. Defaults to 0.

    Attributes:
        memory_size (int): The maximum size of the memory buffer.
        buffer (deque): A deque object representing the memory buffer.

    Methods:
        add(experience): Adds an experience to the memory buffer.
        size(): Returns the current size of the memory buffer.
        sample(batch_size, continuous): Samples a batch of experiences from the memory buffer.
        save(path): Saves the contents of the memory buffer to a file.
        load(path, num_trajs, sample_freq, seed): Loads experiences from a file into the memory buffer.
        get_samples(batch_size, device): Retrieves a batch of samples from the memory buffer.

    """

    def __init__(self, memory_size: int, seed: int = 0) -> None:
        random.seed(seed)
        self.memory_size = memory_size
        self.buffer = deque(maxlen=self.memory_size)

    def add(self, experience) -> None:
        """
        Adds an experience to the memory buffer.

        Args:
            experience: The experience to be added to the memory buffer.

        Returns:
            None

        """
        self.buffer.append(experience)

    def size(self):
        """
        Returns the current size of the memory buffer.

        Returns:
            int: The current size of the memory buffer.

        """
        return len(self.buffer)

    def sample(self, batch_size: int, continuous: bool = True):
        """
        Samples a batch of experiences from the memory buffer.

        Args:
            batch_size (int): The size of the batch to be sampled.
            continuous (bool, optional): Specifies whether the batch should be continuous or not. Defaults to True.

        Returns:
            list: A list of experiences sampled from the memory buffer.

        """
        if batch_size > len(self.buffer):
            batch_size = len(self.buffer)
        if continuous:
            rand = random.randint(0, len(self.buffer) - batch_size)
            return [self.buffer[i] for i in range(rand, rand + batch_size)]
        else:
            indexes = np.random.choice(np.arange(len(self.buffer)), size=batch_size, replace=False)
            return [self.buffer[i] for i in indexes]

    def save(self, path):
        """
        Saves the contents of the memory buffer to a file.

        Args:
            path (str): The path to the file where the memory buffer will be saved.

        Returns:
            None

        """
        b = np.asarray(self.buffer)
        print(b.shape)
        np.save(path, b)

    def load(self, path, num_trajs, sample_freq, seed):
        """
        Loads experiences from a file into the memory buffer.

        Args:
            path (str): The path to the file containing the experiences.
            num_trajs: The number of trajectories to load from the file.
            sample_freq: The frequency at which to sample experiences from the trajectories.
            seed: The seed value for random number generation.

        Returns:
            np.array: An array of observations from the loaded experiences.

        """
        if not (path.endswith("pkl") or path.endswith("hdf5")):
            path += '.npy'
        data = ExpertDataset(path, num_trajs, sample_freq, seed)
        self.full_trajs = data.full_trajectories
        self.memory_size = data.__len__()
        self.buffer = deque(maxlen=self.memory_size)
        obs_arr = []
        for i in range(len(data)):
            self.add(data[i])
            obs_arr.append(data[i][0])
        return np.array(obs_arr)

    def get_samples(self, batch_size, device):
        """
        Retrieves a batch of samples from the memory buffer.

        Args:
            batch_size (int): The size of the batch to be retrieved.
            device: The device on which the samples will be stored.

        Returns:
            tuple: A tuple containing the batch of samples.

        """
        batch = self.sample(batch_size, False)
        
        batch_state, batch_next_state, batch_action, batch_reward, batch_done = zip(*batch)
        batch_state = np.array(batch_state)
        batch_next_state = np.array(batch_next_state)
        batch_action = np.array(batch_action)
        batch_state = (batch_state + self.shift) * self.scale
        batch_next_state = (batch_next_state + self.shift) * self.scale
        
        batch_state = torch.as_tensor(batch_state, dtype=torch.float, device=device)
        batch_next_state = torch.as_tensor(batch_next_state, dtype=torch.float, device=device)
        batch_action = torch.as_tensor(batch_action, dtype=torch.float, device=device)
        if batch_action.ndim == 1:
            batch_action = batch_action.unsqueeze(1)
        batch_reward = torch.as_tensor(batch_reward, dtype=torch.float, device=device).unsqueeze(1)
        batch_done = torch.as_tensor(batch_done, dtype=torch.float, device=device).unsqueeze(1)

        return batch_state, batch_next_state, batch_action, batch_reward, batch_done
