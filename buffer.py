from collections import namedtuple, deque
import random
import torch
import numpy as np

Transition = namedtuple('Transition',
                        ('state', 'action','action_prob', 'next_state', 'reward',"terminated","weight","mask"))


class ReplayMemory(object):

    def __init__(self, capacity,model_num=1):
        self.memory = deque([], maxlen=capacity)
        self.Transition=Transition
        self.model_num=model_num
        self.cum_R=[]

    def push(self, *args):
        """Save a transition"""
        mask=torch.bernoulli(0.5*torch.ones(self.model_num).unsqueeze(0))
        self.memory.append(Transition(*args,mask))

    def save_cum_R(self, cum_R):
        self.cum_R.append(cum_R)
       
    def reward_detected(self):
        cum_R=torch.tensor(self.cum_R)
        if len(cum_R)<2:
            return 0
        return cum_R.var()
    def dataset_diversity_slope(self):
        cum_R=torch.tensor(self.cum_R)
        cum_R_last=torch.tensor(self.cum_R[:-1])
        return (cum_R.var()+1e-6)**0.5-(cum_R_last.var()+1e-6)**0.5

    def sample(self, batch_size):
        weights=self.Transition(*zip(*self.memory)).weight
        return random.choices(self.memory,weights=weights, k=batch_size)
    

    def __len__(self):
        return len(self.memory)