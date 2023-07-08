import torch
from scipy.stats import norm
from buffer import ReplayMemory
import numpy as np
from networks.MLP import MLP
from torch.distributions import Normal, Categorical
from torch import nn

class PPO():
    def __init__(self,n_observations,n_actions,env,writer,prior):
        self.actor_net=MLP(n_observations,n_actions,prior)
        self.critic_net=MLP(n_observations,1,prior)
        self.actor_optimizer=torch.optim.AdamW(self.actor_net.parameters(),lr=3e-4,amsgrad=True)
        self.critic_optimizer=torch.optim.AdamW(self.critic_net.parameters(),lr=1e-3,amsgrad=True)
        self.writer=writer
        self.clip_param = 0.2
        self.max_grad_norm = 0.5

        self.n_actions=n_actions
        self.buffer=ReplayMemory(10000)
        self.steps_done=0
        self.BATCH_SIZE = 300
        self.GAMMA = 0.99 
        self.env=env

    def select_action(self,state,eval=False):
        """select action give a state

        Args:
            state (tensor): current state 
        Returns:
            tensor,bool: action,exploration or not,action_prob
        """
        state = torch.from_numpy(state.detach().numpy()).float()
        with torch.no_grad():
            action_prob = torch.softmax(self.actor_net(state),1)
        c = Categorical(action_prob)
        action = c.sample()
        action_max=np.argmax(action_prob)
        E=1
        if action.item()==action_max.item():
            E=0
            

        if eval:
            return action_max,0,1
        return torch.tensor(action).unsqueeze(0),E, action_prob[:,action.item()].item()
            
    def update(self):
        if len(self.buffer) < self.BATCH_SIZE:
            return
        transitions = self.buffer.sample(self.BATCH_SIZE)
        batch = self.buffer.Transition(*zip(*transitions))

        next_state = torch.cat(batch.next_state)
        state = torch.cat(batch.state)
        action = torch.cat(batch.action)
        reward = torch.cat(batch.reward)
        done = torch.cat(batch.terminated)
        old_action_prob=torch.cat(batch.action_prob)

        #with torch.no_grad():
        V = self.critic_net(state)
        with torch.no_grad():
            td_target=reward.unsqueeze(1)+self.GAMMA*self.critic_net(next_state)*((1-done).unsqueeze(1))
        delta = td_target - V
        advantage = delta.detach()  
        # epoch iteration, PPO core!!!
        action_prob = torch.softmax(self.actor_net(state),1).gather(1, action) # new policy

        ratio = (action_prob/old_action_prob)
        surr1 = ratio * advantage
        surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advantage

        # update actor network
        action_loss = -torch.min(surr1, surr2).mean()  # MAX->MIN desent
        self.actor_optimizer.zero_grad()
        action_loss.backward()
        nn.utils.clip_grad_norm_(self.actor_net.parameters(), self.max_grad_norm)
        self.actor_optimizer.step()

        #update critic network
        value_loss = nn.MSELoss()(V,td_target)
        self.critic_optimizer.zero_grad()
        value_loss.backward()
        nn.utils.clip_grad_norm_(self.critic_net.parameters(), self.max_grad_norm)
        self.critic_optimizer.step()