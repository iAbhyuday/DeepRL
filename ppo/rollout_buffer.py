import torch 

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

