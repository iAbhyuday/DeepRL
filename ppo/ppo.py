#%%
import gym
import torch 
import random 
import numpy as np
import torch.nn as nn 
from torch.nn.functional import mse_loss 
from collections import namedtuple, deque
from torch.utils.tensorboard import SummaryWriter

#%%
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
random.seed(0)
#%%
EVAL_TRAJ = 10
PPO_TRAIN_STEPS= 50
PPO_EVAL_STEPS = 2000
NUM_WORKERS = 4
LR = 2.5e-4
TAU=0.7
CLIP=0.2


gamma = 0.99 
lambda_ = 0.95
MAX_SCORE = 300.0
#%%
def asyncEnvs():
    def make_env():    
        env = gym.wrappers.RecordEpisodeStatistics(env) 
        env = gym.wrappers.ClipAction(env) 
        #env = gym.wrappers.NormalizeObservation(env)
        #env = gym.wrappers.TransformObservation(env, lambda obs:np.clip(obs,-10,10))
        #env = gym.wrappers.NormalizeReward(env) 
        #env = gym.wrappers.TransformReward(env, lambda r: np.clip(r,-10,10))

        env.seed(1)
        env.action_space.seed(1)
        env.observation_space.seed(1)

        return env
    return make_env


#%%
class RolloutBuffer(object):

    def __init__(self, max_size=8000,num_workers=4,observation_shape=None, action_shape=None) -> None:
        self.time_steps = max_size 
        self.num_workers = num_workers 
        self.observation_shape = observation_shape 
        self.action_shape = action_shape
        
        self.clear() 

    def compute_gae(self, next_value, next_done):

        gae = 0.0 
        delta = 0.0
        for t in reversed(range(self.time_steps)):
            
            if t==self.time_steps-1:
                delta = self.rewards[t] + gamma * next_done * next_value - self.values[t]
                gae = delta + gamma * lambda_ * next_done * gae
            else:

                delta = self.rewards[t] + gamma* self.dones[t+1]*self.values[t+1] - self.values[t] 

                gae = delta + gamma * lambda_ * self.dones[t+1] * gae 
        
            self.gae[t] = gae 
        
        self.returns = self.gae + self.values 
        #self.gae = (self.gae - self.gae.mean())/(self.gae.std() + 1e-5) 


    def sample_minibatches(self, batch_size=None):

        ids =torch.randperm(self.time_steps)
        start = 0 
        end = batch_size
        for i in range((self.time_steps//batch_size)):

            idx = ids[start: end]
            mb_states = self.states[idx] 
            mb_actions = self.actions[idx] 
            mb_logprobs = self.log_probs[idx]
            mb_returns = self.returns[idx]
            mb_dones = self.dones[idx]
            mb_values = self.values[idx]
            mb_gae= self.gae[idx] 

            start = end 
            end = end + batch_size
            
            mb_states = mb_states.reshape((-1,) + self.observation_shape) 
            mb_actions = mb_actions.reshape((-1,) + self.action_shape) 
            mb_logprobs = mb_logprobs.reshape(-1)
            mb_gae = mb_gae.reshape(-1) 
            mb_returns = mb_returns.reshape(-1)
            mb_values = mb_values.reshape(-1) 


            yield mb_states, mb_actions, mb_logprobs, mb_returns, mb_dones, mb_values, mb_gae  

    def clear(self):
        self.states = torch.zeros((self.time_steps, self.num_workers,) + self.observation_shape).to(device) 
        self.actions = torch.zeros((self.time_steps, self.num_workers,) + self.action_shape).to(device)
        self.log_probs = torch.zeros((self.time_steps, self.num_workers)).to(device) 
        self.rewards = torch.zeros((self.time_steps, self.num_workers)).to(device)
        self.dones = torch.zeros((self.time_steps, self.num_workers)).to(device)
        self.values = torch.zeros((self.time_steps, self.num_workers)).to(device) 

        self.gae = torch.zeros((self.time_steps, NUM_WORKERS)).to(device) 
        self.returns = torch.zeros((self.time_steps, NUM_WORKERS)).to(device)




            

            




        
#%%

class Actor(nn.Module):

    def __init__(self,obs, action,hidden=64):
        super(Actor,self).__init__()
        self.policy = nn.Sequential(
            nn.Linear(obs,hidden),
            nn.Tanh(),
            nn.Linear(hidden,2*hidden),
            nn.Tanh(),
          
            nn.Linear(2*hidden,hidden),
            nn.Tanh()
            )
        self.mu = nn.Sequential(nn.Linear(hidden,action), nn.Tanh())
        #self.log_std = nn.Linear(hidden,action)
        
        self.log_std =nn.Parameter(torch.zeros(1,action))
  
        # Create the covariance matrix
        #self.cov_mat = torch.diag(self.cov_var).to(device)

    def forward(self,state):
        logits = self.policy(state)

        mu = self.mu(logits)
        #std = self.log_std(logits) 
        std =self.log_std.exp() + 1e-5
        #std = self.log_std.expand_as(mu)

        dist = torch.distributions.Normal(mu, std) 
        return dist 
#%%
    
class Critic(nn.Module):

    def __init__(self,obs,hidden=64):

        super(Critic,self).__init__()
        self.base = nn.Sequential(
            nn.Linear(obs,2*hidden),
            nn.ReLU(),
            nn.Linear(2*hidden,hidden),
            nn.ReLU(),
            nn.Linear(hidden,1)
        
        )

    def forward(self,state):

        return self.base(state)

#%%

class Agent(object):

    def __init__(self,env):
        self.env = env
        self.obs_space = self.env.single_observation_space.shape
        self.action_space = self.env.single_action_space.shape
        self.actor = Actor(self.obs_space[0],self.action_space[0]).to(device)
        self.critic_target = Critic(self.obs_space[0]).to(device)
        self.critic_local = Critic(self.obs_space[0]).to(device)
        self.opt_act = torch.optim.Adam(self.actor.parameters(), lr=LR)
        self.opt_crt = torch.optim.Adam(self.critic_local.parameters(), lr=0.0005)
        self.buffer = RolloutBuffer(observation_shape = self.obs_space, action_shape=self.action_space)
        self.time_steps = 0.0
        self.writer = SummaryWriter(log_dir='runs/ppo')



    def critic_update(self):
        for target_param, local_param in zip(self.critic_target.parameters(), self.critic_local.parameters()):
            target_param.data.copy_(TAU*local_param.data + (1.0-TAU)*target_param.data)
    



    def update(self):
      
        for e in range(PPO_TRAIN_STEPS):
            act_loss = 0.0
            crt_loss = 0.0 
            total_loss = 0.0
            

            for mb_states, mb_actions, mb_log_probs, mb_returns, mb_dones, mb_values, mb_gae in self.buffer.sample_minibatches(batch_size=32):

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

                critic_loss = mse_loss(new_values,mb_returns.to(torch.float32)).mean()
                
                ratio = (new_log_probs-mb_log_probs).exp()

                surrogate_function = ratio*mb_gae
                cliped_surr = torch.clamp(ratio,1-CLIP,1+CLIP)*mb_gae

                actor_loss = torch.min(surrogate_function,cliped_surr).mean()
                
                
               
                
                loss = (-actor_loss + 0.5*critic_loss- 0.01*entropy) 
 
                self.opt_crt.zero_grad()
                self.opt_act.zero_grad()

                loss.backward() 

                self.opt_crt.step()
                self.opt_act.step()

                #print(f'\r Epoch : {e}\t loss = {loss.item()}')
            self.critic_update()
            
        self.writer.add_scalar('losses/critic_loss', critic_loss.item(), self.time_steps) 
        self.writer.add_scalar('losses/actor_loss', actor_loss.item(), self.time_steps) 
        self.writer.add_scalar('losses/loss', loss.item(), self.time_steps) 

            


    def evaluate(self):

        with torch.no_grad():
            score_window=[]
            next_obs = torch.Tensor(self.env.reset()).to(device) 
            next_done = torch.zeros(NUM_WORKERS).to(device) 

            for t in range(PPO_EVAL_STEPS):
                

                self.buffer.states[t] = next_obs 
                self.buffer.dones[t] = 1 - next_done

                
                dist = self.actor(next_obs)
                action = dist.sample() 
                log_prob = dist.log_prob(action).sum(1)
                value = self.critic_local(next_obs)
                value  = value.flatten()
                
                self.buffer.actions[t] = action
                self.buffer.log_probs[t] = log_prob
                self.buffer.values[t] = value



                next_obs,reward,next_done,info = self.env.step(action.cpu().numpy())
                self.buffer.rewards[t] = torch.Tensor(reward).to(device).view(-1)
                next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)
                self.time_steps += NUM_WORKERS 
                for item in info:

                    if "episode" in item.keys():
                        print(f"global_step = {self.time_steps}, episodic_return = {item['episode']['r']}")
                        self.writer.add_scalar('charts/episodic_return', item['episode']['r'], self.time_steps) 
                        self.writer.add_scalar('charts/episodic_length', item['episode']['l'], self.time_steps)

                        break
            
            next_value = self.critic_local(next_obs).flatten() 
            next_done = 1- next_done 

            self.buffer.compute_gae(next_value,next_done) 



            
            print(f'Sample collected: {self.buffer.states.shape[0]}')

        
       

        # TODO: 
        # 1. Add experiences to Rollout buffer
        # 2. Complete Rollout minibatch function 
        # 3. Try prioritized rollout 


    def train(self):
        score = 0.0
        while score<= MAX_SCORE:

            self.evaluate()
            self.update()
            self.buffer.clear() 
            
             

#%%
         

if __name__=='__main__':
    envs = [asyncEnvs() for i in range(NUM_WORKERS)]
    env = gym.vector.AsyncVectorEnv(envs)
#%%   
    agent = Agent(env)
    agent.train()

# %%
