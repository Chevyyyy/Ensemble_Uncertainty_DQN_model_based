from collections import namedtuple, deque
import random
import torch

Transition = namedtuple('Transition',
                        ('state', 'action','action_prob', 'next_state', 'reward',"terminated","mask"))


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

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    

    def __len__(self):
        return len(self.memory)