import torch
from scipy.stats import norm
from buffer import ReplayMemory
import numpy as np
from utilis import soft_update_model_weights
from networks.deep_endemble_NN_model import GaussianMixtureMLP
class model_1_AI():
    def __init__(self,n_model,n_observations,n_actions):
        self.Ensemble_Q_net=GaussianMixtureMLP(n_model,n_observations,n_actions)
        self.Ensemble_Q_net_target=GaussianMixtureMLP(n_model,n_observations,n_actions)
        self.Ensemble_Q_net_target.load_state_dict(self.Ensemble_Q_net.state_dict())

        self.Ensemble_T_net=GaussianMixtureMLP(n_model,n_observations+1,n_observations)
        self.Ensemble_T_net_target=GaussianMixtureMLP(n_model,n_observations+1,n_observations)
        self.Ensemble_T_net_target.load_state_dict(self.Ensemble_T_net.state_dict())


        self.n_actions=n_actions
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
        action_all=torch.arange(self.n_actions).unsqueeze(1)

        mean_next_state,var=self.Ensemble_T_net(torch.cat([state.repeat(self.n_actions,1),action_all],1))


        action_value=torch.zeros((action_all.shape[0]))
        action_value_var=torch.zeros((action_all.shape[0]))

        n_particle=10
        for i in range(self.n_actions):
            particles=torch.normal(mean_next_state[i].repeat(n_particle,1),var[i].repeat(n_particle,1)**0.5)
            value,var_p=self.Ensemble_Q_net(particles)
            action_value[i]=value.mean().item()
            var_p=var_p.mean(1)
            value=value.mean(1)
            action_value_var[i]=(var_p+value.pow(2)).mean(0)-value.mean().pow(2)
        
        action_value=action_value.squeeze()
        action_value_var=action_value_var.squeeze()
        action_value=action_value.detach().tolist()
        p=[]

        for i in range(len(action_value_var)):
            try:
                p.append(np.log(norm.pdf(0,0,action_value_var[i].item()**0.5)))
            except:
                p.append(0)
                print("var:",action_value_var[i])

        FE=-np.array(action_value)
        action1=torch.tensor(np.argmin(FE)).unsqueeze(0)

        FE=-np.array(action_value)+np.array(p)
        action=torch.tensor(np.argmin(FE)).unsqueeze(0)

        E=0
        if action.item()-action1.item()!=0:
            E=1
        return action,E,1
            
    def update(self):
        if len(self.buffer) < self.BATCH_SIZE:
            return
        transitions = self.buffer.sample(self.BATCH_SIZE)
        batch = self.buffer.Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)

        
        next_states = torch.cat(batch.next_state)
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        dones = torch.cat(batch.terminated)
        

        self.Ensemble_Q_net.optimize_replay(state_batch,next_states,action_batch,reward_batch,dones,self.GAMMA,self.Ensemble_Q_net_target)
        self.Ensemble_T_net.optimize_replay_T(state_batch,next_states,action_batch)

        soft_update_model_weights(self.Ensemble_Q_net,self.Ensemble_Q_net_target,self.TAU) 
        soft_update_model_weights(self.Ensemble_T_net,self.Ensemble_T_net_target,self.TAU) 