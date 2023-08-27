import torch
from scipy.stats import norm
from buffer import ReplayMemory
import numpy as np
from utilis import soft_update_model_weights
from networks.deep_ensemble_NN_model import GaussianMixtureMLP
from networks.MLP import MLP 
import random
from torch.distributions import Categorical
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DQN_ensemble():
    def __init__(self,env,n_model,n_observations,n_actions,writer,CNN_flag=False,GAMMA=0.99,BATCH_SIZE=300,TAU=0.005,bootstrap=False,prior=0,prior_noise=0,p_net=False,DP_init=False,real_bootstrap=False,A_change=False,var_net_flag=False,T_net=False,buffer_size=30000,lr=1e-4):

        if prior != 0 or prior_noise!=0:
            optimistic_init=True
        else:
            optimistic_init=False   
        
        self.Ensemble_Q_net_p=GaussianMixtureMLP(n_model,n_observations,n_actions,CNN_flag,prior=prior,prior_noise=prior_noise,optimistic_init=False,env=env,with_piror=p_net,var_net_flag=var_net_flag,lr=lr).to(device)
        self.Ensemble_Q_net=GaussianMixtureMLP(n_model,n_observations,n_actions,CNN_flag,prior=prior,prior_noise=prior_noise,optimistic_init=optimistic_init,env=env,with_piror=p_net,DP_init=DP_init,p_net=self.Ensemble_Q_net_p,var_net_flag=var_net_flag,lr=lr).to(device)
        self.Ensemble_Q_net_target=GaussianMixtureMLP(n_model,n_observations,n_actions,CNN_flag,prior=prior,prior_noise=prior_noise,optimistic_init=False,env=env,with_piror=p_net,var_net_flag=var_net_flag,lr=lr).to(device)

        # dynamic model
        self.Ensemble_T_net=None
        if T_net:
            self.Ensemble_T_net=GaussianMixtureMLP(n_model,n_observations,n_actions,CNN_flag,prior=prior,prior_noise=prior_noise,optimistic_init=False,env=env,with_piror=False,DP_init=False,p_net=None,var_net_flag=False,T_net=True).to(device)

        self.Ensemble_Q_net_target.load_state_dict(self.Ensemble_Q_net.state_dict())

        self.T_net=T_net
        self.writer=writer
        self.env=env
        self.n_actions=n_actions
        self.n_model=n_model
        self.with_prior=p_net
        self.var_net_flag=var_net_flag
        
        self.bootstrap=bootstrap
        self.buffer=ReplayMemory(30000,n_model)
        self.steps_done=0
        self.BATCH_SIZE = BATCH_SIZE 
        self.GAMMA = GAMMA 
        self.TAU=TAU
        self.max_R=1
        self.real_bootstrap=real_bootstrap
        self.epsiod_num=0
        self.action_model_ID=0
        self.action_model_T_ID=0
        self.A_change=A_change

    def select_action_T(self,state,eval=False):
        state.to(device=device)

        now_epsiode_num=len(self.buffer.cum_R) 
        if self.real_bootstrap:# sample each epsiode
            if now_epsiode_num>self.epsiod_num:
                self.epsiod_num=now_epsiode_num
                self.action_model_ID=torch.randint(0,self.n_model,(1,)).item()
                self.action_model_T_ID=torch.randint(0,self.n_model,(1,)).item()
        else:  #sample each step 
            self.action_model_ID=torch.randint(0,self.n_model,(1,)).item()
            self.action_model_T_ID=torch.randint(0,self.n_model,(1,)).item()

        if eval==False:
            self.steps_done+=1

        next_state_predicts=self.Ensemble_T_net(state)
        next_state_predict=next_state_predicts[self.action_model_T_ID].squeeze()
        next_state_predict=torch.mean(next_state_predicts,0)

        # compute the extpected value for the predicted next state
        R=self.Ensemble_Q_net(next_state_predict.T)+self.with_prior*self.Ensemble_Q_net_p(next_state_predict.T)
        R_sample=R[self.action_model_ID].mean(-1).squeeze() 

       
        if self.buffer.reward_detected()<0.5:
            R_sample=torch.var(next_state_predicts.squeeze(),0)
            # R_sample+=torch.var(next_state_predicts.squeeze(),0)*10
            # m = Categorical(torch.softmax(R_sample,0))
            # action=m.sample().reshape(1,1).detach()

        
        # argmax to select the action
        action=torch.argmax(R_sample).reshape(1,1).detach()
 

        action1=torch.argmax(R.mean(0).mean(-1)).reshape(1,1).detach()
        if eval:
            return action1,0,1

        self.writer.add_scalar("action/std of Q",torch.var(R.mean(-1),0)[action]**0.5,self.steps_done)
        self.writer.add_scalar("action/Q",R_sample[action],self.steps_done)
        soft_max_r_sample=torch.softmax(R_sample,0)
        self.writer.add_scalar("action/Entropy",torch.sum(-soft_max_r_sample*torch.log(soft_max_r_sample)),self.steps_done)

        E=0
        if action.item()-action1.item()!=0:
            E=1
        return action,E,1

    def select_action(self,state,eval=False,UCB=False):
        """select action give a state

        Args:
            state (tensor): current state 
        Returns:
            tensor,bool: action,exploration or not
        """
        
        # if self.T_net:
        #     return self.select_action_T(state,eval=eval)
            
        state.to(device=device)
        if eval==False:
            self.steps_done+=1
        if self.var_net_flag:
            R,var=self.Ensemble_Q_net(state)+self.Ensemble_Q_net_p(state)*self.with_prior
        else:
            R=self.Ensemble_Q_net(state)+self.Ensemble_Q_net_p(state)*self.with_prior
        R=R.squeeze().to(device=device)
        var_R_MSE=R.var(0).to(device=device)

        now_epsiode_num=len(self.buffer.cum_R) 
        if self.real_bootstrap:# sample each epsiode
            if now_epsiode_num>self.epsiod_num:
                self.epsiod_num=now_epsiode_num
                self.action_model_ID=torch.randint(0,self.n_model,(1,)).item()
            R_sample=R[self.action_model_ID].squeeze().to(device=device)

        else:# sample each step
            if self.A_change:
                if random.random()<1-np.exp(-0.03*self.buffer.reward_detected()**0.5): 
                    self.action_model_ID=torch.randint(0,self.n_model,(1,)).item()
                if now_epsiode_num>self.epsiod_num:
                    self.epsiod_num=now_epsiode_num
                    self.action_model_ID=torch.randint(0,self.n_model,(1,)).item()
                R_sample=R[self.action_model_ID].squeeze()
            else:    
                self.action_model_ID=torch.randint(0,self.n_model,(1,)).item()
                R_sample=R[self.action_model_ID].squeeze()

                # sample among all Q posterior
                # index=torch.randint(0,self.n_model,(1,R.shape[-1]))
                # R_sample=R.gather(0,index).squeeze()
            if UCB:
                R_sample=R.median(0)[0]+var_R_MSE**0.5
        if self.var_net_flag:
            R_sample=R_sample-0.1*(var[self.action_model_ID].squeeze())**0.5

        action=torch.argmax(R_sample).reshape(1,1).detach()

        action1=torch.argmax(R.mean(0)).reshape(1,1).detach()
        if eval:
            return action1,0,1
        self.writer.add_scalar("action/std of Q",var_R_MSE[action]**0.5,self.steps_done)
        self.writer.add_scalar("action/Q",R_sample[action],self.steps_done)
        soft_max_r_sample=torch.softmax(R_sample,0)
        self.writer.add_scalar("action/Entropy",torch.sum(-soft_max_r_sample*torch.log(soft_max_r_sample)),self.steps_done)

        # a1,a2=self.Ensemble_Q_net(torch.tensor([[3.0]])).mean(0).squeeze()
        # a14,a24=self.Ensemble_Q_net(torch.tensor([[4.0]])).mean(0).squeeze()
        # self.writer.add_scalar("action/s3",a1,self.steps_done)
        # self.writer.add_scalar("action/s4",a24,self.steps_done)
        # self.writer.add_scalar("action/s3_correct",a24,self.steps_done)
        # self.writer.add_scalar("action/s4_correct",a14,self.steps_done)
        # self.writer.add_scalar("action/Q4",R[3,0],self.steps_done)
        # self.writer.add_scalar("action/Q5",R[4,0],self.steps_done)
        E=0
        if action.item()-action1.item()!=0:
            E=1
        return action,E,1
    
    
    
    
    def update(self):
        batch_UPB=self.BATCH_SIZE*(1+9*self.bootstrap)
        if len(self.buffer) < self.BATCH_SIZE:
            return

        
        try:
            transitions = self.buffer.sample(batch_UPB)
        except:
            transitions = self.buffer.sample(len(self.buffer))
            
        batch = self.buffer.Transition(*zip(*transitions))


        next_states = torch.cat(batch.next_state).to(device=device)
        state_batch = torch.cat(batch.state).to(device=device)
        action_batch = torch.cat(batch.action).to(device=device)
        reward_batch = torch.cat(batch.reward).to(device=device)
        dones = torch.cat(batch.terminated).to(device=device)
        masks = torch.cat(batch.mask).to(device=device)
        

            
        
        if self.bootstrap==False:
            masks=torch.ones(masks.shape)
        reward_detected=self.buffer.reward_detected()
        dataset_diversity_slope=self.buffer.dataset_diversity_slope()
        self.writer.add_scalar("CUM_R_STD",reward_detected**0.5,self.steps_done)
        self.writer.add_scalar("CUM_R_STD_SLOPE",dataset_diversity_slope,self.steps_done)
        self.Ensemble_Q_net.optimize_replay(state_batch,next_states,action_batch,reward_batch,dones,masks,self.GAMMA,self.Ensemble_Q_net_target,self.Ensemble_Q_net_p,self.BATCH_SIZE,self.Ensemble_T_net,reward_detected)
        if self.T_net:
            self.Ensemble_T_net.optimize_replay_T(state_batch,next_states,action_batch,masks,self.BATCH_SIZE)

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
