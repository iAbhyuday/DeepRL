#%%
import gym 
import json
import argparse
from ppo_agent import PPOAgent 

#%%

def asyncEnvs(env_id):
    def make_env():
        env = gym.make(env_id)    
        env = gym.wrappers.RecordEpisodeStatistics(env) 
        if type(env.action_space) == gym.spaces.Box:
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
if __name__== '__main__':
    
    config = None 
    with open('config.json', 'r') as f:
        config = json.load(f)
    
    envs = [asyncEnvs(config['env_id']) for i in range(config['num_workers'])]
    
    env = gym.vector.AsyncVectorEnv(envs)
#%%
    #ppo_agent = PPOAgent(env, **config)
    #ppo_agent.train()

state = env.reset()
info = None
for i in range(1000):

    action = env.action_space.sample()
    next_state, reward, done,info = env.step(action)
    state = next_state

# %%
