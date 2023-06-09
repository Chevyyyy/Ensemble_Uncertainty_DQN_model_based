import torch
from scipy.stats import norm
from buffer import ReplayMemory
import numpy as np
from utilis import soft_update_model_weights
from networks.deep_endemble_NN_model import GaussianMixtureMLP
from networks.MLP import MLP 
class DQN_ensemble():
    def __init__(self,n_model,n_observations,n_actions,writer,CNN_flag=False,GAMMA=0.99,BATCH_SIZE=300,TAU=0.005,bootstrap=False,prior=0,prior_noise=0):

        
        
        self.Ensemble_Q_net=GaussianMixtureMLP(n_model,n_observations,n_actions,CNN_flag,prior=prior,prior_noise=prior_noise)
        self.Ensemble_Q_net_target=GaussianMixtureMLP(n_model,n_observations,n_actions,CNN_flag,prior=prior,prior_noise=prior_noise)
        # self.value=MLP(n_observations,1)
        # self.target_value=MLP(n_observations,1)
        self.optimizer=torch.optim.AdamW(self.Ensemble_Q_net.parameters(),lr=1e-4,amsgrad=True)
        # self.optimizer_value=torch.optim.AdamW(self.value.parameters(),lr=1e-4,amsgrad=True)
        self.Ensemble_Q_net_target.load_state_dict(self.Ensemble_Q_net.state_dict())
        # self.target_value.load_state_dict(self.value.state_dict())
        self.writer=writer
        self.n_actions=n_actions
        
        self.bootstrap=bootstrap
        self.buffer=ReplayMemory(10000,n_model)
        self.steps_done=0
        self.BATCH_SIZE = BATCH_SIZE 
        self.GAMMA = GAMMA 
        self.TAU=TAU
        self.max_R=1

    def select_action1(self,state):
        """select action give a state

        Args:
            state (tensor): current state 
        Returns:
            tensor,bool: action,exploration or not
        """
        self.steps_done+=1
        R,var=self.Ensemble_Q_net(state)
        R=R.squeeze()
        delta_max=abs(R[0]-R[1])
        var=var.squeeze()
        var=torch.clamp(var,torch.tensor(0.01),delta_max**2)
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
        try:
            action=torch.tensor(np.argmax(np.random.multinomial(1,nsoftFE))).reshape(1,1)
        except:
            print(nsoftFE)
            action=torch.tensor(0).reshape(1,1)
        
        # action=torch.tensor(np.argmin(FE)).unsqueeze(0)
        self.writer.add_scalar("var of Q",var[0],self.steps_done)
        self.writer.add_scalar("var of Q1",var[1],self.steps_done)
        
        E=0
        if action.item()-action1.item()!=0:
            E=1
            
        return action,E,nsoftFE[action] 
    
    def select_action1(self,state,eval=False):
        """select action give a state

        Args:
            state (tensor): current state 
        Returns:
            tensor,bool: action,exploration or not
        """
        if eval==False:
            self.steps_done+=1
        R,var=self.Ensemble_Q_net(state)
        R=R.squeeze()
        delta_max=abs(R[0]-R[1])
        var=var.squeeze()
        # std=torch.clamp(var**0.5,torch.tensor(0),delta_max)
        std=var**0.5
        R_var=R+torch.tensor(np.random.normal(size=self.n_actions))*std
        action=torch.argmax(R_var).reshape(1,1)
        action1=torch.argmax(R).reshape(1,1)
        if eval:
            return action1,0,1

        self.writer.add_scalar("action/std of Q",var[0]**0.5,self.steps_done)
        self.writer.add_scalar("action/delta_max",delta_max,self.steps_done)
        self.writer.add_scalar("action/delta_max_std_ratio",delta_max/std[0],self.steps_done)
        
        E=0
        if action.item()-action1.item()!=0:
            E=1
        return action,E,1

    def select_action(self,state,eval=False):
        """select action give a state

        Args:
            state (tensor): current state 
        Returns:
            tensor,bool: action,exploration or not
        """
        if eval==False:
            self.steps_done+=1
        R=self.Ensemble_Q_net(state)
        R=R.squeeze()
        var_R_MSE=R.var(0)
        R_sample=R[torch.randint(0,5,(1,))].squeeze()

        action=torch.argmax(R_sample).reshape(1,1)
        action1=torch.argmax(R.mean(0)).reshape(1,1)
        if eval:
            return action1,0,1
        self.writer.add_scalar("action/std of Q",var_R_MSE[0]**0.5,self.steps_done)
        E=0
        if action.item()-action1.item()!=0:
            E=1
        return action,E,1
    
    def update(self):
        batch_UPB=self.BATCH_SIZE*(1+9*self.bootstrap)
        if len(self.buffer) < batch_UPB:
            return
        
        transitions = self.buffer.sample(batch_UPB)

        batch = self.buffer.Transition(*zip(*transitions))


        next_states = torch.cat(batch.next_state)
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        dones = torch.cat(batch.terminated)
        masks = torch.cat(batch.mask)
        

            
        
        if self.bootstrap==False:
            masks=torch.ones(masks.shape)
        self.Ensemble_Q_net.optimize_replay(state_batch,next_states,action_batch,reward_batch,dones,masks,self.GAMMA,self.Ensemble_Q_net_target,self.BATCH_SIZE)

        soft_update_model_weights(self.Ensemble_Q_net,self.Ensemble_Q_net_target,self.TAU) 
       
         
         
         
        # # optimize the value function
        # self.optimizer_value.zero_grad()
        # # current value
        # current_value=self.value(state_batch) 
        # # target value
        # with torch.no_grad():
        #     target_value=reward_batch.unsqueeze(1)+self.GAMMA*self.target_value(next_states)*(1-dones.unsqueeze(1))
        # loss=torch.nn.MSELoss()(target_value,current_value)    
        # loss.backward() 
        # self.optimizer_value.step()
        # soft_update_model_weights(self.value,self.target_value,self.TAU) 
