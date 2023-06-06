from networks.MLP import MLP
import torch
from buffer import ReplayMemory
import random
import math
from utilis import soft_update_model_weights
class DQN():
    def __init__(self,n_observations,n_actions,env):
        self.Q_net=MLP(n_observations,n_actions)
        self.Q_net_target=MLP(n_observations,n_actions)
        self.optimizer=torch.optim.AdamW(self.Q_net.parameters(),lr=1e-4,amsgrad=True)
        self.Q_net_target.load_state_dict(self.Q_net.state_dict())
        self.buffer=ReplayMemory(10000)
        self.env=env
        self.steps_done=0
        self.EPS_START=0.9
        self.EPS_END = 0.05
        self.EPS_DECAY = 1000
        self.BATCH_SIZE = 300
        self.GAMMA = 0.99 
        self.TAU=0.005

    def select_action(self,state):
        """select action give a state

        Args:
            state (tensor): current state 
            env (openai_env): environment

        Returns:
            tensor,bool: action,exploration or not,action_prob
        """
        sample = random.random()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
            math.exp(-1. * self.steps_done / self.EPS_DECAY)
        self.steps_done += 1

        action0=self.Q_net(state).max(1)[1].view(1, 1)
        action1=torch.tensor([[self.env.action_space.sample()]], dtype=torch.long)

        if sample > eps_threshold:
                    return action0,0,None 
        else:
            E=(action0.item()!=action1.item())
            return action1,E,None
            
    def update(self):
        if len(self.buffer) < self.BATCH_SIZE:
            return
        transitions = self.buffer.sample(self.BATCH_SIZE)
        batch = self.buffer.Transition(*zip(*transitions))

        next_states = torch.cat(batch.next_state)
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        dones = torch.cat(batch.terminated)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.Q_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.BATCH_SIZE)
        with torch.no_grad():
            next_state_values = self.Q_net_target(next_states).max(1)[0]
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.GAMMA)*(1-dones) + reward_batch

        # Compute Huber loss
        criterion = torch.nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.Q_net.parameters(), 100)
        self.optimizer.step()
        # soft update target
        soft_update_model_weights(self.Q_net,self.Q_net_target,self.TAU) 