import numpy as np
from gym.utils import seeding

class EpsilonGreedyPolicy(object):
    
    def __init__(self, *Q, epsilon=0.1, reduction="sum", tie_breaking="random"):
        self.Q = Q  # note that Q is an array of estimates
        self.epsilon = epsilon
        self.reduction = reduction
        self.np_random = None
        self.break_ties_randomly = tie_breaking == "random"
        self.seed()
        
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return seed
    
    def sample_action(self, state, legal_actions, break_ties_randomly=True):
        if self.reduction == "sum":
            q = np.array(self.Q).sum(axis=0)
        elif self.reduction == "mean":
            q = np.array(self.Q).mean(axis=0)
        else:
            raise ValueError("unknown reduction function: %s" % self.reduction)
            
        if len(self.Q) == 1:
            assert (q == self.Q[0]).all()
        
        # print("q is", q.shape)
        if self.np_random.uniform() < self.epsilon:
            action = self.np_random.choice(np.arange(len(legal_actions))[legal_actions])
        
        elif self.break_ties_randomly:
            # breaks ties randomly
            # print(q[state].shape)
            # print(q[state,0].shape)
            action = self.np_random.choice(
                np.where(q[state][legal_actions] == q[state][legal_actions].max())[0]
            )
        else:
            # breaks ties deterministically
            action_q_values = q[state].copy()
            action_q_values[~legal_actions] = np.nan
            action = np.nanargmax(action_q_values)
        return action
    