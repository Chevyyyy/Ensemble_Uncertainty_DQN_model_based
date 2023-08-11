from networks.MLP import MLP
from networks.CNN import CNN 
import torch
from buffer import ReplayMemory
import random
import math
from utilis import soft_update_model_weights
class R_uncertainty():
    def __init__(self,n_observations,n_actions,writer,env,CNN_flag=False,GAMMA=0.99,BATCH_SIZE=300,TAU=0.005,prior=0):
        if CNN_flag:
            self.Q_net=CNN(n_observations,n_actions,prior)
            self.Q_net_U=CNN(n_observations,n_actions,prior)
            self.Q_net_target=CNN(n_observations,n_actions,prior)
            self.Q_net_target_U=CNN(n_observations,n_actions,prior)
        else:
            self.Q_net=MLP(n_observations[0],n_actions,prior)
            self.Q_net_target=MLP(n_observations[0],n_actions,prior)
            self.Q_net_U=MLP(n_observations[0],n_actions,prior)
            self.Q_net_target_U=MLP(n_observations[0],n_actions,prior)
        
        self.R_net=MLP(n_observations[0]+1,1,prior)

        self.optimizer=torch.optim.AdamW(self.Q_net.parameters(),lr=1e-4,amsgrad=True)
        self.optimizer_U=torch.optim.AdamW(self.Q_net_U.parameters(),lr=1e-4,amsgrad=True)
        self.optimizer_R=torch.optim.AdamW(self.R_net.parameters(),lr=1e-4,amsgrad=True)

        self.Q_net_target.load_state_dict(self.Q_net.state_dict())
        self.buffer=ReplayMemory(10000)
        self.env=env
        self.steps_done=0
        self.EPS_START=0.9
        self.EPS_END = 0.05
        self.EPS_DECAY = 1000
        self.BATCH_SIZE =BATCH_SIZE 
        self.GAMMA = GAMMA 
        self.TAU=TAU
        self.writer=writer

    def select_action(self,state,eval=False):
        """select action give a state

        Args:
            state (tensor): current state 
            env (openai_env): environment

        Returns:
            tensor,bool: action,exploration or not,action_prob
        """
        if eval==False:
            self.steps_done += 1
        
        Q=self.Q_net(state)
        Q_U=torch.relu(self.Q_net_U(state)+1e-6)**0.5
        Q_sample=Q+torch.randn(1,Q_U.shape[-1])*Q_U

        action=Q_sample.max(1)[1].view(1, 1)
        action1=Q.max(1)[1].view(1, 1)

        self.writer.add_scalar("action/std of Q",Q_U[:,action],self.steps_done)

        return action,(action.item()!=action1.item()),1

            
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

        criterion = torch.nn.SmoothL1Loss()
        
        # update R_net
        R_pred=self.R_net(torch.cat((state_batch,action_batch),1))
        loss_R=criterion(R_pred.squeeze(),reward_batch)
        self.optimizer_R.zero_grad()
        loss_R.backward()
        self.optimizer_R.step()


        
        
        # update Q_net

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
            next_state_values,next_max_index = self.Q_net_target(next_states).max(1)
            # Compute the expected Q values
            expected_state_action_values = (next_state_values * self.GAMMA)*(1-dones) + R_pred.squeeze()

        # Compute Huber loss
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.Q_net.parameters(), 100)
        self.optimizer.step()
        # soft update target
        soft_update_model_weights(self.Q_net,self.Q_net_target,self.TAU) 

        
        # update Q_net_U
        QU_pred=self.Q_net_U(state_batch).gather(1, action_batch)
        with torch.no_grad():
            QU_target=(1-dones.unsqueeze(-1))*self.GAMMA*self.Q_net_target_U(next_states).gather(1,next_max_index.unsqueeze(-1))+((R_pred.squeeze()-reward_batch)**2).unsqueeze(-1)
        loss_U=criterion(QU_pred,QU_target)
        self.optimizer_U.zero_grad()
        loss_U.backward()
        self.optimizer_U.step()
        soft_update_model_weights(self.Q_net_U,self.Q_net_target_U,self.TAU)