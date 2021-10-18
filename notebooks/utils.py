import numpy as np
import gym
from gym import spaces

def smoothing_window(vals, radius=50):
    cumvals = np.array(vals).cumsum()
    return (cumvals[radius:] - cumvals[:-radius]) / radius

def seed(env, policy, seed=42):
    env.seed(seed)
    policy.seed(seed)
    
def space_to_shape(s):
    if isinstance(s, spaces.Tuple):
        return tuple([space_to_shape(ss) for ss in s.spaces])
    elif isinstance(s, spaces.Discrete):
        return s.n
    return None

class EnvWrapper(gym.Wrapper):
    @property
    def nA(self):
        return space_to_shape(self.env.action_space)
    
    @property
    def nS(self):
        return space_to_shape(self.env.observation_space)
    
    def legal_actions(self, state):
        if hasattr(self.env, "legal_actions"):
            return self.env.legal_actions(state)
        mask = np.ones(self.nA, dtype=bool) # allow all
        return mask