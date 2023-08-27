import numpy as np
import collections
from typing import Any, Callable, Dict, List, Tuple
import matplotlib.pyplot as plt
from IPython.display import clear_output
import gym
from gym import spaces

# @title (CODE) Define the _Deep Sea_ environment


class DeepSea(gym.Env):
    def __init__(self, size: int, seed: int = None):
        self._size = size
        self.state = 0

        if seed is None:
            seed = int(np.random.randint(100))
        rng = np.random.RandomState(seed)
        self._action_mapping = rng.binomial(1, 0.5, size)

        self.action_space = spaces.Discrete(2)
        self.low = np.array([0])
        self.high = np.array([size])
        self.observation_space = spaces.Box(self.low, self.high, dtype=np.float32)

    def step(self, action: int):
        correct = action == self._action_mapping[self.state]
        self.step_num+=1
        terminated = False
        if correct:
            self.state += 1
        else:
            terminated = True
            
        if self.state == self._size:
            reward = 1.0 
        else:
            reward = 0.0
        
        if  self.step_num==self._size:
            terminated = True 

        return np.array([self.state]), reward, terminated , False, {}



    def reset(self):
        self.state = 0
        self.step_num=0
        return np.array([self.state]), {}

    @property
    def obs_shape(self) -> Tuple[int]:
        return self.reset().observation.shape

    @property
    def num_actions(self) -> int:
        return 2

    @property
    def optimal_return(self) -> float:
        return self._goal_reward - self._move_cost



