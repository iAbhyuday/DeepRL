import gym
import json
import torch
import random
import numpy as np
import torch.nn as nn
from rollout_buffer import RolloutBuffer
from torch.nn.functional import mse_loss
from collections import namedtuple, deque
from torch.utils.tensorboard import SummaryWriter


class PPOAgent(nn.Module):
    def __init__(self, env: gym.vector.AsyncVectorEnv, **config) -> None:
        super(PPOAgent, self).__init__()
        
        self.env = env

        self.is_continuous = (
            True if type(self.env.single_action_space) == gym.spaces.Box else False
        )

        self.config = config

        self.gamma = self.config["gamma"]
        self.tau = self.config["tau"]
        self.clip = self.config["clip"]
        self.hidden_units = self.config["hidden_units"]
        self.decay_std_rate = self.config["action_std_decay_rate"] 
        self.action_std_min = self.config["action_std_min"]
        self.std_decay = self.config['std_decay']

        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() and self.config["cuda"] else "cpu"
        )

        self.obs_space = self.env.single_observation_space.shape
        self.action_space = self.env.single_action_space.shape

        self.buffer = RolloutBuffer(
            obs_shape=self.obs_space,
            action_shape=self.action_space,
            device=self.device, 
            **self.config
        )

        self.time_steps = 0.0
        self.writer = SummaryWriter(log_dir=f"{self.config['tensorboard_dir']}/{self.config['experiment_name']}")

        
        if self.is_continuous:
            self.actor = nn.Sequential(
                        nn.Linear(np.array(self.obs_space).prod(),self.hidden_units),
                        nn.Tanh(),
                        nn.Linear(self.hidden_units,2*self.hidden_units),
                        nn.Tanh(),
                        nn.Linear(2*self.hidden_units,self.hidden_units),
                        nn.Tanh(),
                        nn.Linear(self.hidden_units,np.array(self.action_space).prod()), 
                        nn.Tanh()
                        ).to(self.device)    

        else:

            self.actor = nn.Sequential(
                        nn.Linear(np.array(self.obs_space).prod(),self.hidden_units),
                        nn.Tanh(),
                        nn.Linear(self.hidden_units,2*self.hidden_units),
                        nn.Tanh(),
                        nn.Linear(2*self.hidden_units,self.hidden_units),
                        nn.Tanh(),
                        nn.Linear(self.hidden_units,np.array(self.action_space).prod()), 
                        nn.Softmax()
                        ).to(self.device)

        self.critic_local = nn.Sequential(
                        nn.Linear(np.array(self.obs_space).prod(),2*self.hidden_units),
                        nn.ReLU(),
                        nn.Linear(2*self.hidden_units,self.hidden_units),
                        nn.ReLU(),
                        nn.Linear(self.hidden_units,1)
                        ).to(self.device) 

        self.critic_target = nn.Sequential(
                        nn.Linear(np.array(self.obs_space).prod(),2*self.hidden_units),
                        nn.ReLU(),
                        nn.Linear(2*self.hidden_units,self.hidden_units),
                        nn.ReLU(),
                        nn.Linear(self.hidden_units,1)
                        ).to(self.device) 

        if self.is_continuous:
            self.action_std = self.config["action_std_max"]
            self.set_action_var()

        
        self.opt = torch.optim.Adam([
            { 'params' : self.actor.parameters(), 'lr' : self.config['actor_lr'], 'eps': 1e-5},
            { 'params' : self.critic_local.parameters(), 'lr' : self.config['critic_lr'], 'eps': 1e-5}
        ])




    def set_action_var(self):
        self.action_var = torch.full((self.action_space[0],),self.action_std**2).to(self.device)



    
    def forward(self, state: torch.Tensor, mb_actions = None):
        """
        Args:
            state (torch.Tensor): The current state of the environment.

        Returns:
            torch.Tensor: The Q-value of the state
        """
        logits = self.actor(state) 
        value = self.critic_target(state)
        if self.is_continuous:
            cov_mat = torch.diag(self.action_var).unsqueeze(0).to(self.device)
            dist = torch.distributions.MultivariateNormal(logits, cov_mat)
        else:
            dist = torch.distributions.Categorical(logits=logits)
        return dist, value
        


    def eval_actor_critic(self, mb_states: torch.Tensor, mb_actions: torch.Tensor):
        
        values = self.critic_local(mb_states) 
        logits = self.actor(mb_states) 

        dist = None 
        log_probs = None

        if self.is_continuous:
            cov_mat = torch.diag(self.action_var).unsqueeze(0).to(self.device)
            dist = torch.distributions.MultivariateNormal(logits, cov_mat)
            log_probs = dist.log_prob(mb_actions)
            
        else:
            dist = torch.distributions.Categorical(logits=logits)
            log_probs = dist.log_prob(mb_actions)
        
        return values, log_probs, dist.entropy().mean()

    
    
    def decay_std(self):
        self.action_std = self.action_std - self.decay_std_rate
        if(self.action_std <= self.action_std_min):
            self.action_std = self.action_std_min
        print('New std : {}'.format(self.action_std))
        self.set_action_var()
    


    def critic_update(self):
        for target_param, local_param in zip(
            self.critic_target.parameters(), self.critic_local.parameters()
        ):
            target_param.data.copy_(
                self.tau * local_param.data + (1.0 - self.tau) * target_param.data
            )


    def update(self):

        for e in range(self.config['ppo_epochs_per_update']):

            for (
                mb_states,
                mb_actions,
                mb_log_probs,
                mb_returns,
                mb_dones,
                mb_values,
                mb_adv,
            ) in self.buffer.sample_minibatches(batch_size= self.config['batch_size']):

                new_values, new_log_probs, entropy = self.eval_actor_critic(mb_states, mb_actions)
                new_values = self.critic_local(mb_states).squeeze(-1)

                # print(f'states shape : {mb_states.shape}')
                # print(f'actions shape : {mb_actions.shape}')
                # print(f'new log_probs shape : {new_log_probs.shape}')
                # print(f'old log_probs shape : {mb_log_probs.shape}')
                # print(f'ADV shape : {mb_gae.shape}')
                # print(f'New values shape : {new_values.shape}')
                # print(f'Returns shape : {mb_returns.shape}')

                # break

                critic_loss = mse_loss(new_values, mb_returns.to(torch.float32)).mean()

                ratio = (new_log_probs - mb_log_probs).exp()

                surrogate_function = ratio * mb_adv
                cliped_surr = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * mb_adv

                actor_loss = torch.min(surrogate_function, cliped_surr).mean()

                loss = -actor_loss + 0.5 * critic_loss - 0.01 * entropy

                
                self.opt.zero_grad()

                loss.backward()

                self.opt.step()
                

            
            self.critic_update()

        self.writer.add_scalar(
            "losses/critic_loss", critic_loss.item(), self.time_steps
        )
        self.writer.add_scalar("losses/actor_loss", actor_loss.item(), self.time_steps)
        self.writer.add_scalar("losses/loss", loss.item(), self.time_steps)

    
    
    def evaluate(self):

        with torch.no_grad():
            
            next_obs = torch.Tensor(self.env.reset()).to(self.device)
            next_done = torch.zeros(self.config['num_workers']).to(self.device)

            for t in range(self.config['ppo_eval_steps']):

                self.buffer.states[t] = next_obs
                self.buffer.dones[t] = 1 - next_done

                dist, value = self.forward(next_obs)
                action = dist.sample()
                log_prob = dist.log_prob(action)
            
                value = value.flatten()

                self.buffer.actions[t] = action
                self.buffer.log_probs[t] = log_prob
                self.buffer.values[t] = value


                next_obs, reward, next_done, info = self.env.step(action.cpu().numpy())
                self.buffer.rewards[t] = torch.Tensor(reward).to(self.device).view(-1)
                
                
                next_obs, next_done = torch.Tensor(next_obs).to(self.device), torch.Tensor(next_done).to(self.device)
                self.time_steps += self.config['num_workers']

                if self.std_decay and self.time_steps%self.config['action_std_decay_freq']==0:
                    print('Decaying std ...')
                    self.decay_std()
                
                for item in info:

                    if "episode" in item.keys():
                        print(
                            f"global_step = {self.time_steps}, episodic_return = {item['episode']['r']}"
                        )
                        self.writer.add_scalar(
                            "charts/episodic_return",
                            item["episode"]["r"],
                            self.time_steps,
                        )
                        self.writer.add_scalar(
                            "charts/episodic_length",
                            item["episode"]["l"],
                            self.time_steps,
                        )

                        break

            next_value = self.critic_local(next_obs).flatten()
            next_done = 1 - next_done

            self.buffer.compute_gae(next_value, next_done)

            
        # TODO:
        # 1. Add experiences to Rollout buffer
        # 2. Complete Rollout minibatch function
        # 3. Try prioritized rollout

    def train(self):
        
        while self.time_steps <= self.config['max_time_steps']:

            self.evaluate()
            self.update()
            self.buffer.clear()
