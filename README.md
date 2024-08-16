<h1>SPRINQL: Sub-optimal Demonstrations driven Offline Imitation Learning</h1>

## Introduction
We focus on offline imitation learning (IL), which aims to mimic the expert’s behavior from its demonstration without any interactions with the environment. One of the main challenges in offline IL is dealing with the limited support of expert demonstrations that cover only a small fraction of the state-action spaces.  While it is not feasible to obtain many expert demonstrations, it is always feasible to get a larger set of sub-optimal demonstrations. In this paper, we provide an offline IL approach that is able to exploit the larger set of sub-optimal demonstrations while imitating the expert trajectories. Existing offline IL approaches based on behavior cloning or distribution matching often suffer from either over fitting to the small set of expert demonstrations or imitating sub-optimal trajectories in the larger set.  To that end, our approach based on inverse soft-Q learning learns from both expert and sub-optimal demonstrations, but it gives more importance (through learned weights) to alignment with expert demonstrations and less importance to alignment with sub-optimal demonstrations. A key contribution of our approach, referred to as SPRINQL, is converting the offline IL problem to a convex optimization over the space of Q functions. Through a thorough experimental evaluation, we are able to show that SPRINQL algorithm achieves SOTA performance on offline IL benchmarks.

1. Clone the repos:
```
.
```

1. Create and activate conda environment
```
conda create -n sprinql python=3.9
conda activate sprinql
```

2. Install packages
```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
conda install -y -c conda-forge glew
conda install -y -c conda-forge mesalib
conda install -y -c menpo glfw3
export CPATH=$CONDA_PREFIX/include
pip install patchelf
```

## Directory Structure
```
.
├── experts                                 # expert data
├── parameters                              # predefined parameters
│   ├── agent
│   ├── config.yaml
│   └── env
├── README.md
├── ref_reward                              # trained reward reference network
├── requirements.txt
├── Sources
│   ├── algos                               # algorithm implementation
│   │   ├── actor.py
│   │   ├── critic.py
│   │   ├── sac.py
│   │   └── sprinql.py
│   ├── dataset                             # dataset class
│   │   ├── expert_dataset.py
│   │   ├── memory.py
│   └── utils                               # utility functions
│       ├── make_agent.py
│       ├── make_envs.py
│       └── utils.py
├── Train                                   # training scripts
│   ├── run.sh
│   ├── train.py
│   └── train_reference_reward.py
└── trained_agents                          # trained agents
```
## Usages

Download dataset from [anonymized link](https://drive.google.com/drive/folders/1b_-ajbeseonjh5hX-G8ucNDfRUaGVlgE?usp=sharing) and unzip in the main folder ```SPRINQL```.

Replace ```env = [cheetah, ant, walker, hopper, humanoid]``` for different tasks.
The scripts belows are for three datasets scenario with:

- 25000 transitions for level 2
- 10000 transitions for level 1
- 1000 transitions for level 0 (expert level).

1. run SPRINQL:
```
python -u Train/train.py env=cheetah \
env.sub_optimal_demo=[2,1,0] \
env.num_sub_optimal_demo=[25000,10000,1000] \
seed=0 
```
 

2. run training reference reward function from scratch:
```
python -u Train/train_reference_reward.py env=cheetah \
env.sub_optimal_demo=[2,1,0] \
env.num_sub_optimal_demo=[25000,10000,1000] \
seed=0 
```

## Conclusion

We have developed SPRINQL,  a novel non-adversarial inverse soft-Q learning algorithm for offline imitation learning from expert and sub-optimal demonstrations. 
We have demonstrated that our algorithm possesses several favorable properties, contributing to its well-behaved, stable, and scalable nature. Additionally, we have devised a preference-based loss function to automate the estimation of reward reference values. We have provided extensive experiments based on several benchmark tasks, demonstrating the ability of our \textit{SPRINQL} algorithm to leverage both expert and non-expert data to achieve superior performance compared to state-of-the-art algorithms. 

## Citation

```
.
```