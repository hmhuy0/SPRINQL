
import torch
import torch.nn.functional as F

from Sources.utils.utils import soft_update,concat_data
  
  
def update(self, buffers, step):
    """
    Updates the model by performing a training step.

    Args:
        buffers (list): A list of buffers containing samples for training.
        step (int): The current training step.

    Returns:
        dict: A dictionary containing information about the update process.

    """
    add_batches = []
    for id, add_buffer in enumerate(buffers):
        batch = add_buffer.get_samples(self.batch_size, self.device)
        add_batches.append(batch)
    update_info = self.update_critic(add_batches, step)

    add_obs = []
    for id, batch in enumerate(add_batches):
        add_obs.append(batch[0])
    add_obs = torch.cat(add_obs, dim=0)
    obs = add_obs
    actor_loss = self.update_actor(obs, step)
    update_info.update(actor_loss)

    soft_update(self.critic_net, self.critic_target_net, self.critic_tau)
    self.first_log = False
    return update_info

def update_actor(self, obs, step):
    """
    Updates the actor network based on the given observations and training step.

    Args:
        obs (torch.Tensor): The input observations.
        step (int): The current training step.

    Returns:
        dict: A dictionary containing the losses and other metrics.

    """
    action, log_prob, _ = self.actor.sample(obs)
    actor_Q = self.critic(obs, action)
    actor_loss = (self.alpha.detach() * log_prob - actor_Q).mean()
    losses = {}

    if (step > self.args.train.KL_start and self.temp_actor is not None):
        with torch.no_grad():
            best_action = self.temp_actor(obs).mean
            best_log_prob = self.temp_actor.get_log_prob(obs, best_action)
        pi_log_prob = self.actor.get_log_prob(obs, best_action)
        KL_loss = 0.001 * (best_log_prob - pi_log_prob).mean()
        actor_loss += KL_loss
        losses['actor_loss/KL_loss'] = KL_loss.item()

    losses.update({
        'actor_loss/total_loss': actor_loss.item(),
        'update/log_prob': log_prob.mean().item(),
        'actor_loss/actor_Q': actor_Q.mean().item(),
    })
    # optimize the actor
    self.actor_optimizer.zero_grad()
    actor_loss.backward()
    self.actor_optimizer.step()

    return losses

def update_critic(self, add_batches, step):
    """
    Updates the critic network based on the given additional batches of data.

    Args:
        add_batches (list): A list of additional batches of data.
        step (int): The current step number.

    Returns:
        dict: A dictionary containing the loss values for different components of the critic network.

    Raises:
        NotImplementedError: If the loss method specified in the arguments is not implemented.

    """
    
    args = self.args
    batch = concat_data(add_batches, args)
    obs, next_obs, action,env_reward,done,ls_size =batch
    if (args.train.use_reward_function):
        reward = self.reward_function.get_reward(obs, action)
    else:
        reward = env_reward
    
    weights = []
    for idx in range(len(ls_size)-1):
        w = torch.ones_like(reward[ls_size[idx]:ls_size[idx+1]])\
                *reward[ls_size[idx]:ls_size[idx+1]].mean()
        weights.append(w)
    weights = torch.concat(weights)
    
    with torch.no_grad():
        next_action, log_prob, _ = self.actor.sample(next_obs)
        target_next_V = self.critic_target(next_obs, next_action)  - self.alpha.detach() * log_prob
        y_next_V = (1 - done) * self.gamma * target_next_V.clip(min=-100,max=100)
    current_Q1,current_Q2 = self.critic(obs, action,both=True)
    current_V = self.getV(obs)
    
    pred_reward_1 = current_Q1 - y_next_V
    pred_reward_2 = current_Q2 - y_next_V
    
    reward_loss_1 = (-weights * pred_reward_1 + 1/2 * (pred_reward_1**2)).mean()
    reward_loss_2 = (-weights * pred_reward_2 + 1/2 * (pred_reward_2**2)).mean()
    DM_loss = (reward_loss_1 + reward_loss_2)/2
    
    if (args.method.loss=='strict_value'):
        if (self.first_log):
            print('[Critic]: use strict_value loss')
        value_dif = current_V - y_next_V
        value_loss = (value_dif + 1/2*value_dif**2).mean()    
    elif (args.method.loss=='value'):
        if (self.first_log):
            print('[Critic]: use value loss')
        value_loss = (current_V - y_next_V).mean()
    elif (args.method.loss=='v0'):
        if (self.first_log):
            print('[Critic]: use v0 loss')
        value_loss = (1-self.gamma) * current_V.mean()
    else:
        raise NotImplementedError
    
    RR_loss = ( ((current_Q1 - reward)**2 + y_next_V**2 
                 + 2*(reward - current_Q1) * y_next_V).mean()+ 
                ((current_Q2 - reward)**2 + y_next_V**2 
                 + 2*(reward - current_Q2) * y_next_V).mean() 
                )/2
    Q_loss = (
        value_loss 
        + DM_loss
        + args.agent.rr_coef * RR_loss
    )
    
    num_random = args.agent.num_random
    if (self.first_log):
        print(f'[Critic]: use CQL*{args.agent.cql_coef} ({num_random} randoms) loss')
    cql_loss_1 = self.cqlV(obs, self.critic.Q1,num_random) - current_Q1.mean()
    cql_loss_2 = self.cqlV(obs, self.critic.Q2,num_random) - current_Q2.mean()
    cql_loss = args.agent.cql_coef*(cql_loss_1+cql_loss_2)/2    
    critic_loss = Q_loss + cql_loss

    loss_dict  ={
        'loss/critic_loss':critic_loss.item(),
        'loss/cql_loss':cql_loss.item(),
        'loss/Q_loss':Q_loss.item(),
        'loss/RR_loss':RR_loss.item(),
        'loss/DM_loss':DM_loss.item(),
    }
    
    if (step%args.env.eval_interval == 0):
        expert_probs = []
        with torch.no_grad():
            for id,batch in enumerate(add_batches):
                b_obs,b_next_obs,b_action,b_env_r,b_done = batch
                b_next_action, b_log_prob, _ = self.actor.sample(b_next_obs)
                b_next_target_V = self.critic_target(b_next_obs, b_next_action)  - self.alpha.detach() * b_log_prob
                b_Q1,b_Q2 = self.critic(b_obs, b_action,both=True)
                b_Q = (b_Q1+b_Q2)/2
                b_reward = b_Q - (1 - b_done) * self.gamma * b_next_target_V
                if (args.train.use_reward_function):
                    b_ref_reward = self.reward_function.get_reward(b_obs,b_action)
                else:
                    b_ref_reward = b_env_r
                
                pi_action, pi_prob, _ = self.actor.sample(b_obs)
                pi_Q1,pi_Q2 = self.critic(b_obs,pi_action,both=True)
                pi_Q = (pi_Q1+pi_Q2)/2
                pi_reward = pi_Q - (1 - b_done) * self.gamma * b_next_target_V
                
                exp_log_prob = self.actor.get_log_prob(b_obs, b_action).mean().item()
                expert_probs.append(exp_log_prob)
                loss_dict[f'value/pi_Q_{id}'] = pi_Q.mean().item()
                loss_dict[f'reward/pi_reward_{id}'] = pi_reward.mean().item()
                loss_dict[f'reward/ref_reward_{id}'] = b_ref_reward.mean().item()
                loss_dict[f'log_prob/log_prob_{id}'] = exp_log_prob
                loss_dict[f'value/Q_{id}'] = b_Q.mean().item()
                loss_dict[f'reward/reward_{id}'] = b_reward.mean().item()
        
    self.critic_optimizer.zero_grad()
    critic_loss.backward()
    self.critic_optimizer.step()
    return loss_dict

def update_single_critic(self, add_batches, step):
    """
    Updates the single critic model based on the given additional batches and step.

    Args:
        add_batches (list): A list of additional batches containing observation, next observation, action, environment reward, done, and ls_size.
        step (int): The current step.

    Returns:
        dict: A dictionary containing the loss values for different components of the critic model.

    Raises:
        NotImplementedError: If the loss method is not implemented.

    """
    
    args = self.args
    batch = concat_data(add_batches, args)
    obs, next_obs, action,env_reward,done,ls_size =batch
    if (args.train.use_reward_function):
        reward = self.reward_function.get_reward(obs, action)
    else:
        reward = env_reward
    
    weights = []
    for idx in range(len(ls_size)-1):
        w = torch.ones_like(reward[ls_size[idx]:ls_size[idx+1]])\
                *reward[ls_size[idx]:ls_size[idx+1]].mean()
        weights.append(w)
    weights = torch.concat(weights)
    
    with torch.no_grad():
        next_action, log_prob, _ = self.actor.sample(next_obs)
        target_next_V = self.critic_target(next_obs, next_action)  - self.alpha.detach() * log_prob
        y_next_V = ((1 - done) * self.gamma * target_next_V).clip(min=-100,max=100)
        target_Q = reward + y_next_V
    current_Q = self.critic(obs, action)
    current_V = self.getV(obs)
    pred_reward = current_Q - y_next_V
    
    DM_loss = (-weights * pred_reward + 1/2 * (pred_reward**2)).mean()
    
  
    if (args.method.loss=='value'):
        if (self.first_log):
            print('[Critic]: use value loss')
        value_loss = (current_V - y_next_V).mean()
    elif (args.method.loss=='v0'):
        if (self.first_log):
            print('[Critic]: use v0 loss')
        value_loss = (1-self.gamma) * current_V.mean()
    else:
        raise NotImplementedError
    
    RR_loss =  ((current_Q - reward)**2 + y_next_V**2 
                 + 2*(reward - current_Q) * y_next_V).mean()

    Q_loss = (
        value_loss 
        + DM_loss
        + args.agent.rr_coef * RR_loss
    )
    
    num_random = args.agent.num_random
    if (self.first_log):
        print(f'[Critic]: use CQL*{args.agent.cql_coef} ({num_random} randoms) loss')
    cql_loss = args.agent.cql_coef*(self.cqlV(obs, self.critic.Q,num_random) - current_Q.mean())
    critic_loss = Q_loss + cql_loss

    loss_dict  ={
        'loss/critic_loss':critic_loss.item(),
        'loss/cql_loss':cql_loss.item(),
        'loss/Q_loss':Q_loss.item(),
        'loss/RR_loss':RR_loss.item(),
        'loss/DM_loss':DM_loss.item(),
    }
    
    if (step%args.env.eval_interval == 0):
        expert_probs = []
        with torch.no_grad():
            for id,(batch,b_env_r) in enumerate(zip(add_batches,args.expert.reward_arr)):
                b_obs,b_next_obs,b_action,_,b_done = batch
                b_next_action, b_log_prob, _ = self.actor.sample(b_next_obs)
                b_next_target_V = self.critic_target(b_next_obs, b_next_action)  - self.alpha.detach() * b_log_prob
                b_Q = self.critic(b_obs, b_action)
                b_reward = b_Q - (1 - b_done) * self.gamma * b_next_target_V
                if (args.train.use_reward_function):
                    b_ref_reward = self.reward_function.get_reward(b_obs,b_action).mean().item()
                else:
                    b_ref_reward = b_env_r
                
                pi_action, pi_prob, _ = self.actor.sample(b_obs)
                pi_Q = self.critic(b_obs,pi_action)
                pi_reward = pi_Q - (1 - b_done) * self.gamma * b_next_target_V
                
                exp_log_prob = self.actor.get_log_prob(b_obs, b_action).mean().item()
                expert_probs.append(exp_log_prob)
                loss_dict[f'value/pi_Q_{id}'] = pi_Q.mean().item()
                loss_dict[f'reward/pi_reward_{id}'] = pi_reward.mean().item()
                loss_dict[f'reward/ref_reward_{id}'] = b_ref_reward
                loss_dict[f'log_prob/log_prob_{id}'] = exp_log_prob
                loss_dict[f'value/Q_{id}'] = b_Q.mean().item()
                loss_dict[f'reward/reward_{id}'] = b_reward.mean().item()
        
    self.critic_optimizer.zero_grad()
    critic_loss.backward()
    self.critic_optimizer.step()
    return loss_dict
