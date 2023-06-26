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

    def push(self, *args):
        """Save a transition"""
        mask=torch.bernoulli(0.5*torch.ones(self.model_num).unsqueeze(0))
        self.memory.append(Transition(*args,mask))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)