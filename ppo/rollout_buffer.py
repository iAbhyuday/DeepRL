import torch 

class RolloutBuffer(object):

    def __init__(self,obs_shape, action_shape, device: torch.DeviceObjType, **hyperparams) -> None:

        self.time_steps = hyperparams['ppo_eval_steps'] 
        self.num_workers =hyperparams['num_workers'] 
        self.observation_shape = obs_shape  
        self.action_shape = action_shape   
        self.is_gae = hyperparams['is_gae'] 
        self.device = device
        self.lambda_ = hyperparams['lambda'] 
        self.gamma = hyperparams['gamma']


        
        self.clear() 

    def compute_gae(self, next_value, next_done):

        gae = 0.0 
        delta = 0.0
        if self.i_gae:
            for t in reversed(range(self.time_steps)):

                if t==self.time_steps-1:
                    delta = self.rewards[t] + self.gamma * next_done * next_value - self.values[t]
                    gae = delta + self.gamma * self.lambda_ * next_done * gae
                else:

                    delta = self.rewards[t] + self.gamma* self.dones[t+1]*self.values[t+1] - self.values[t] 

                    gae = delta + self.gamma * self.lambda_ * self.dones[t+1] * gae 

                self.gae[t] = gae 
        
            self.returns = self.gae + self.values 
            #self.gae = (self.gae - self.gae.mean())/(self.gae.std() + 1e-5) 

        else:
            r = 0.0
            for t in reversed(range(self.time_steps)):
                if t==self.timesteps - 1:
                    self.returns[t]= self.rewards[t]  + self.gamma * next_done * next_value
                else:
                    self.returns[t] = self.rewards[t]  + self.gamma * self.dones[t+1] * self.returns[t+1]
            
            self.gae = self.returns - self.values

                


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
        self.states = torch.zeros((self.time_steps, self.num_workers,) + self.observation_shape).to(self.device) 
        self.actions = torch.zeros((self.time_steps, self.num_workers,) + self.action_shape).to(self.device)
        self.log_probs = torch.zeros((self.time_steps, self.num_workers)).to(self.device) 
        self.rewards = torch.zeros((self.time_steps, self.num_workers)).to(self.device)
        self.dones = torch.zeros((self.time_steps, self.num_workers)).to(self.device)
        self.values = torch.zeros((self.time_steps, self.num_workers)).to(self.device) 

        self.gae = torch.zeros((self.time_steps, self.num_workers)).to(self.device) 
        self.returns = torch.zeros((self.time_steps, self.num_workers)).to(self.device)

