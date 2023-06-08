import torch
from scipy.stats import norm
from buffer import ReplayMemory
import numpy as np
from utilis import soft_update_model_weights
from networks.deep_endemble_NN_model import GaussianMixtureMLP
class DQN_ensemble():
    def __init__(self,n_model,n_observations,n_actions,writer):
        self.Ensemble_Q_net=GaussianMixtureMLP(n_model,n_observations,n_actions)
        self.Ensemble_Q_net_target=GaussianMixtureMLP(n_model,n_observations,n_actions)
        self.optimizer=torch.optim.AdamW(self.Ensemble_Q_net.parameters(),lr=1e-4,amsgrad=True)
        self.Ensemble_Q_net_target.load_state_dict(self.Ensemble_Q_net.state_dict())
        self.writer=writer

        self.buffer=ReplayMemory(10000)
        self.steps_done=0
        self.BATCH_SIZE = 300
        self.GAMMA = 0.99 
        self.TAU=0.005

    def select_action(self,state):
        """select action give a state

        Args:
            state (tensor): current state 
        Returns:
            tensor,bool: action,exploration or not
        """
        self.steps_done+=1
        R,var=self.Ensemble_Q_net(state)
        R=R.squeeze()
        var=var.squeeze()
        R=R.detach().tolist()
        p=[]

        for i in range(len(var)):
            try:
                p.append(np.log(norm.pdf(0,0,var[i].item()**0.5)))
            except:
                p.append(0)
                print("var:",var[i])
        FE=-np.array(R)
        action1=torch.tensor(np.argmin(FE)).unsqueeze(0)

        FE=-np.array(R)+np.array(p)
        nsoftFE=np.exp(-FE)/(np.exp(-FE).sum())
        action=torch.tensor(np.argmax(np.random.multinomial(1,nsoftFE))).reshape(1,1)
        action=torch.tensor(np.argmin(FE)).unsqueeze(0)
        self.writer.add_scalar("var of Q",var[0],self.steps_done)
        self.writer.add_scalar("var of Q1",var[1],self.steps_done)
        
        E=0
        if action.item()-action1.item()!=0:
            E=1
            
        return action,E,nsoftFE[action] 
    
    def select_action(self,state):
        """select action give a state

        Args:
            state (tensor): current state 
        Returns:
            tensor,bool: action,exploration or not
        """
        self.steps_done+=1
        R,var=self.Ensemble_Q_net(state)
        R=R.squeeze()
        var=var.squeeze()
        R_var=R+torch.tensor(np.random.normal(size=2))*(var**0.5)/2
        action=torch.argmax(R_var).reshape(1,1)
        action1=torch.argmax(R).reshape(1,1)

        self.writer.add_scalar("var of Q",var[0],self.steps_done)
        
        E=0
        if action.item()-action1.item()!=0:
            E=1
            
        return action,E,1
    
    
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
        
        self.Ensemble_Q_net.optimize_replay(state_batch,next_states,action_batch,reward_batch,dones,self.GAMMA,self.Ensemble_Q_net_target)

        soft_update_model_weights(self.Ensemble_Q_net,self.Ensemble_Q_net_target,self.TAU) 