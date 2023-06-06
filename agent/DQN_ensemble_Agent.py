import torch
from scipy.stats import norm
from buffer import ReplayMemory
import numpy as np
from utilis import soft_update_model_weights
from networks.deep_endemble_NN_model import GaussianMixtureMLP
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()
class DQN_ensemble():
    def __init__(self,n_model,n_observations,n_actions):
        self.Ensemble_Q_net=GaussianMixtureMLP(n_model,n_observations,n_actions)
        self.Ensemble_Q_net_target=GaussianMixtureMLP(n_model,n_observations,n_actions)
        self.optimizer=torch.optim.AdamW(self.Ensemble_Q_net.parameters(),lr=1e-4,amsgrad=True)
        self.Ensemble_Q_net_target.load_state_dict(self.Ensemble_Q_net.state_dict())

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
        # action=torch.tensor(np.argmin(FE)).unsqueeze(0)
        writer.add_scalar("var of Q",var[0],self.steps_done)
        
        E=0
        if action.item()-action1.item()!=0:
            E=1
            
        return action,E 
            
    def update(self):
        if len(self.buffer) < self.BATCH_SIZE:
            return
        transitions = self.buffer.sample(self.BATCH_SIZE)
        batch = self.buffer.Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)

        
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        
        self.Ensemble_Q_net.optimize_replay(state_batch,batch.next_state,action_batch,reward_batch,self.GAMMA,self.Ensemble_Q_net_target)

        soft_update_model_weights(self.Ensemble_Q_net,self.Ensemble_Q_net_target,self.TAU) 