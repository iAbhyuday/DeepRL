import gym
import json
import torch
import random
import numpy as np
import torch.nn as nn
from torch.nn.functional import mse_loss
from collections import namedtuple, deque
from torch.utils.tensorboard import SummaryWriter


class Agent(nn.Module):
    def __init__(self, env: gym.vector.AsyncVectorEnv) -> None:
        super(Agent, self).__init__()
        
        self.env = env

        self.is_continuous = (
            True if type(self.env.single_action_space) == gym.spaces.Box else False
        )

        self.config = json.load("config.json")

        self.gamma = self.config["gamma"]
        self.tau = self.config["tau"]
        self.clip = self.config["clip"]
        
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() and self.config["cuda"] else "cpu"
        )

        self.obs_space = self.env.single_observation_space.shape
        self.action_space = self.env.single_action_space.shape
        
        self.actor = Actor(self.obs_space[0], self.action_space[0]).to(self.device)
        self.critic_target = Critic(self.obs_space[0]).to(self.device)
        self.critic_local = Critic(self.obs_space[0]).to(self.device)
        
        self.opt = torch.optim.Adam([
            { 'params' : self.actor.parameters(), 'lr' : self.config['actor_lr'], 'eps': 1e-5},
            { 'params' : self.critic_local.parameters(), 'lr' : self.config['critic_lr'], 'eps': 1e-5}
        ])

        self.buffer = RolloutBuffer(
            observation_shape=self.obs_space, action_shape=self.action_space
        )
        
        self.time_steps = 0.0
        
        self.writer = SummaryWriter(log_dir=f"{self.config['tensorboard_dir']}/{self.config['experiment_name']}")



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

                dist = self.actor(mb_states)
                entropy = dist.entropy().mean()
                new_log_probs = dist.log_prob(mb_actions)
                new_log_probs = new_log_probs.sum(1)

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
                

                # print(f'\r Epoch : {e}\t loss = {loss.item()}')
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

                dist = self.actor(next_obs)
                action = dist.sample()
                log_prob = dist.log_prob(action).sum(1)
                value = self.critic_local(next_obs)
                value = value.flatten()

                self.buffer.actions[t] = action
                self.buffer.log_probs[t] = log_prob
                self.buffer.values[t] = value

                next_obs, reward, next_done, info = self.env.step(action.cpu().numpy())
                self.buffer.rewards[t] = torch.Tensor(reward).to(self.device).view(-1)
                
                
                next_obs, next_done = torch.Tensor(next_obs).to(self.device), torch.Tensor(next_done).to(self.device)
                self.time_steps += self.config['num_workers']
                
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

            #print(f"Sample collected: {self.buffer.states.shape[0]}")

        # TODO:
        # 1. Add experiences to Rollout buffer
        # 2. Complete Rollout minibatch function
        # 3. Try prioritized rollout

    def train(self):
        score = 0.0
        while self.time_steps <= self.config['max_time_steps']:

            self.evaluate()
            self.update()
            self.buffer.clear()
