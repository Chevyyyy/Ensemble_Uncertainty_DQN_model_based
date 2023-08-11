# define the class for ensemble DQN network
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from networks.CNN import CNN
from networks.MLP import MLP 
import numpy as np
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GaussianMultiLayerPerceptron(nn.Module):
    
    def __init__(self, input_dim, output_dim,prior=0):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.fc1 = nn.Linear(self.input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, self.output_dim)
        if prior>0:
            self.fc3.bias.data=torch.tensor([prior]).float()
            self.fc3.bias.requires_grad = False
        
    def forward(self, x):
        batch_n=x.shape[0]
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x).reshape(batch_n,2,-1)
        mean=x[:,0,:]
        variance=x[:,1,:]
        variance = F.softplus(variance) +1e-6  #Positive constraint
        return mean, variance
    
    
class GaussianMultiLayerCNN(nn.Module):
    
    def __init__(self, input_dim, output_dim,prior=0):
        super().__init__()
        self.CNN = CNN(input_dim,output_dim,prior)

    def forward(self, x):
        batch_n=x.shape[0]
        x = self.CNN(x).reshape(batch_n,2,-1)
        mean=x[:,0,:]
        variance=x[:,1,:]
        variance = F.softplus(variance) +1e-6  #Positive constraint
        return mean, variance
    
    
        
class GaussianMixtureMLP(nn.Module):
    """ Gaussian mixture MLP which outputs are mean and variance.

    Attributes:
        models (int): number of models
        inputs (int): number of inputs
        outputs (int): number of outputs
        hidden_layers (list of ints): hidden layer sizes

    """
    def __init__(self, num_models=5, inputs=1, outputs=1,CNN_flag=False,prior=0,prior_noise=0,optimistic_init=False,env=None,with_piror=False,DP_init=False,p_net=None,var_net_flag=False,T_net=False):
        super(GaussianMixtureMLP, self).__init__()
        self.num_models = num_models
        self.inputs = inputs
        self.outputs = outputs
        self.env=env
        self.with_piror=with_piror
        self.var_net_flag=var_net_flag
        if T_net:
            lr=1e-2
        else:
            lr=1e-4
        for i in range(self.num_models):
            if CNN_flag:
                model = CNN(self.inputs+T_net,self.outputs*(1+var_net_flag)*(1-T_net)+T_net*self.inputs,prior+prior_noise*np.random.normal())
            else:
                model = MLP(self.inputs[0]+T_net,self.outputs*(1+var_net_flag)*(1-T_net)+T_net*self.inputs[0],prior+prior_noise*np.random.normal())

                
            setattr(self, 'model_'+str(i), model)
            optim=torch.optim.AdamW(getattr(self, 'model_' + str(i)).parameters(),lr=lr)
            setattr(self,"optim_"+str(i),optim)

            if optimistic_init:
                p_model=getattr(p_net, 'model_' + str(i))
                self.optimistic_init(model,p_model=p_model,max_value=prior+prior_noise*np.random.normal())

        self.optim_all=torch.optim.AdamW(self.parameters(),lr=lr)
        if T_net:
            print("DP initialize the T net")
            # self.diversity_init_T()      
        if DP_init:
            print("DP initialize the Q-net prior")
            self.diversity_init(p_net)
            # make the initial parameters of all models the same
            # self.same_initailize_all_model() 
    
    
        
    def same_initailize_all_model(self):
        model = getattr(self, 'model_' + '0')
        for i in range(1,self.num_models):
            model_i = getattr(self, 'model_' + str(i))
            model_i.load_state_dict(model.state_dict())
        
    def diversity_init_T(self):
        for i in tqdm(range (100)):
            high=self.env.observation_space.high
            low=self.env.observation_space.low
            s=[]
            for i in range(len(high)):
                try:
                    s.append(torch.FloatTensor(128,1).uniform_(low[i],high[i]))
                except:
                    s.append(torch.FloatTensor(128,1).uniform_(-100,100))
            data=torch.cat(s,1)

            # forward
            n_actions=self.env.action_space.n
            current_state_all_actions=torch.cat((data,torch.randint(0,2,(128,1))),1)

            next_state_predict_var=self(current_state_all_actions).var(0).mean()

            # compute the loss    
            self.optim_all.zero_grad()
            loss=-next_state_predict_var
            
            # optimize
            loss.backward()
            self.optim_all.step()
             
    def diversity_init(self,p_net):
        """initalzie the networks with diversity to make the softmax output different in each member"""

        for i in tqdm(range (100)):
            high=self.env.observation_space.high
            low=self.env.observation_space.low
            s=[]
            for i in range(len(high)):
                try:
                    s.append(torch.FloatTensor(128,1).uniform_(low[i],high[i]))
                except:
                    s.append(torch.FloatTensor(128,1).uniform_(-100,100))
            data=torch.cat(s,1)

            model_index=torch.randint(0,self.num_models,(1,))
            model_p = getattr(p_net, 'model_' + str(model_index.item()))
            model = getattr(self, 'model_' + str(model_index.item()))

            optim=torch.optim.AdamW(model_p.parameters(),lr=0.01)
            # forward
            if self.var_net_flag:
                mean = torch.softmax(model(data)[:,:self.outputs]+model_p(data)[:,:self.outputs],dim=1)

                with torch.no_grad():
                    means=self(data)[0]+p_net(data)[0]
                    target=torch.softmax(means.median(0)[0],dim=1)

            else:
                
                mean = torch.softmax(model(data)+model_p(data),dim=1)

                with torch.no_grad():
                    means=self(data)+p_net(data)
                    target=torch.softmax(means.median(0)[0],dim=1)

            # compute the loss    
            optim.zero_grad()
            # loss=-F.mse_loss(mean,target)*100-means.var(0).mean()*100
            criterion = nn.KLDivLoss(reduction='batchmean')
            loss=-criterion(mean,target)
            
            # optimize
            loss.backward()
            optim.step()
            
        
        
        
        
        
    def optimistic_init(self,model,p_model,max_value):
        optimizer=torch.optim.AdamW(model.parameters(),lr=0.01)
        for i in tqdm(range (1000)):
            high=self.env.observation_space.high
            low=self.env.observation_space.low
            s=[]
            for i in range(len(high)):
                try:
                    s.append(torch.FloatTensor(128,1).uniform_(low[i],high[i]))
                except:
                    s.append(torch.FloatTensor(128,1).uniform_(-100,100))
            data=torch.cat(s,1)
            # forward
            mean = model(data)+p_model(data)
            with torch.no_grad():
                target=torch.ones(128,self.outputs)*max_value


            # compute the loss    
            optimizer.zero_grad()
            loss=F.mse_loss(mean,target)

            
            # optimize
            loss.backward()
            optimizer.step()

            
            
            
            

    def forward(self, x,value_net=None):
        self.eval()
        # connect layers
        means = []
        variances = []
        for i in range(self.num_models):
            model = getattr(self, 'model_' + str(i))
            mean = model(x)
            means.append(mean)
        means = torch.stack(means)
        if self.var_net_flag==False:
            return means
        else:
            means,vars=means.split(int(means.shape[-1]/2),dim=-1)
            return means,torch.relu(vars)+1e-6

        # variance = (variances + (means-means.max(2)[0].unsqueeze(-1)).pow(2)).mean(dim=0)
        # variance = (means-means.max(2)[0].unsqueeze(-1)).var(dim=0)
        # if value_net is not None:
        #     variance = (means-value_net(x)).var(0)
        # else:
        #     variance = (means-means.mean(2).unsqueeze(-1)).var(0)
        # variance=F.relu(variance)+1e-6
    
    def optimize(self,x_M,t_M):
        self.train()
        for i in range(self.num_models):
            model = getattr(self, 'model_' + str(i))
            optim = getattr(self, 'optim_' + str(i))
            # forward
            mean, var = model(x_M[i])
            # compute the loss
            optim.zero_grad()
            
            loss=F.gaussian_nll_loss(mean,t_M[i],var)
            # optimize
            loss.backward()
            optim.step()
    def KL_normal(self,mean1,var1,mean2,var2):
        """compute the KL divergence between two normal distribution"""
        return torch.log(var2**0.5/var1**0.5)+(var1+(mean1-mean2)**2)/(2*var2)-0.5

    def optimize_replay(self,current_state,next_state,action,reward,dones,masks,gamma,target_net,prior_net,batch_size,reward_detected=1):


        self.train()
        loss_all=0
        for i in range(self.num_models):
            index=masks[:,i].bool()
            model = getattr(self, 'model_' + str(i))
            optim = getattr(self, 'optim_' + str(i))
            p_net=getattr(prior_net, 'model_' + str(i))
            t_net_single=getattr(target_net, 'model_' + str(i))
            
            if self.var_net_flag:
                # forward
                n_avial=current_state[index].shape[0]
                batch_size=min(batch_size,n_avial)
                mean,var= (model(current_state[index][:batch_size])+self.with_piror*p_net(current_state[index][:batch_size])).split(int(self.outputs),dim=-1)
                # compute the loss
                optim.zero_grad()
                state_action_values = mean.gather(1, action[index][:batch_size]).squeeze().to(device=device)
                state_action_values_var = torch.relu(var.gather(1, action[index][:batch_size]).squeeze().to(device=device))+1e-6

                with torch.no_grad():
                    # use seperate target network
                    mean_next,var_next=(t_net_single(next_state[index][:batch_size])+self.with_piror*p_net(next_state[index][:batch_size]).to(device=device)).split(int(self.outputs),dim=-1)
                    next_state_values,action_index=mean_next.max(1)
                    next_state_values_var=torch.relu(var_next.gather(1, action_index.unsqueeze(-1)).squeeze().to(device=device))+1e-6
                
                loss_all+=self.KL_normal(state_action_values,state_action_values_var,(1-dones[index][:batch_size].squeeze())*gamma*next_state_values+reward[i],(1-dones[index][:batch_size].squeeze())*gamma*next_state_values_var+1e-6).mean()                   

            else:
                # forward
                mean= model(current_state[index][:batch_size])+self.with_piror*p_net(current_state[index][:batch_size]).to(device=device)
                # compute the loss
                optim.zero_grad()

                state_action_values = mean.gather(1, action[index][:batch_size]).squeeze().to(device=device)

                with torch.no_grad():
                    # add prior
                    # use unifed target network
                    # mean_next=target_net(next_state[index][:batch_size])+self.with_piror*prior_net(next_state[index][:batch_size])
                    # median_next=mean_next.median(0)[0]
                    # next_state_values,action_index = median_next.max(1)

                    # use seperate target network
                    mean_next=t_net_single(next_state[index][:batch_size])+self.with_piror*p_net(next_state[index][:batch_size]).to(device=device)
                    next_state_values,action_index=mean_next.max(1)
                    
                    
                    # mean_next=target_net(next_state[index][:batch_size])[torch.randint(0,5,(1,)).item()]
                    value_target=(1-dones[index][:batch_size].squeeze())*next_state_values*gamma+reward[i].squeeze().to(device=device)
                # loss_all+=((value_target-state_action_values)**2).mean()*(rew ard_detected**0.5+1)/100
                loss_all+=((value_target-state_action_values)**2).mean().to(device=device)
            
        # optimize
        loss_all.backward()
        torch.nn.utils.clip_grad_value_(model.parameters(), 100)
        self.optim_all.step()
            
    def optimize_replay_T(self,current_state,next_state,action,masks,batch_size):

        n_scale=(current_state.max(0)[0]-current_state.min(0)[0])**2+1e-6
        n_scale=1
        self.train()
        loss_all=0
        
        for i in range(self.num_models):
            index=masks[:,i].bool()
            model = getattr(self, 'model_' + str(i))
            optim = getattr(self, 'optim_' + str(i))
            
            # forward
            state_action=torch.concatenate((current_state[index][:batch_size],action[index][:batch_size]),axis=1)
            mean= model(state_action)
            # compute the losss
            optim.zero_grad()

            loss_all+=(((next_state[index][:batch_size]-mean)**2).mean(0)/n_scale).mean().to(device=device)
            
        # optimize
        loss_all.backward()
        torch.nn.utils.clip_grad_value_(model.parameters(), 100)
        self.optim_all.step()
