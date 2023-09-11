from networks.deep_ensemble_NN_model import GaussianMixtureMLP
from networks.MLP import MLP
from matplotlib import pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
from torch import nn
from torch.nn import functional as F
class BSDP(torch.nn.Module):
    def __init__(self):
        super(BSDP, self).__init__()
        # initalize 5 models
        self.model0 = MLP(1,2,0)
        self.model1 = MLP(1,2,0)
        self.model2 = MLP(1,2,0)
        self.model3 = MLP(1,2,0)
        self.model4 = MLP(1,2,0)
        self.model5 = MLP(1,2,0)
        self.model6 = MLP(1,2,0)
        self.model7 = MLP(1,2,0)
        self.model8 = MLP(1,2,0)
        self.model9 = MLP(1,2,0)
        # initialize 5 prior models
        self.prior0 = MLP(1,2,0)
        self.prior1 = MLP(1,2,0)
        self.prior2 = MLP(1,2,0)
        self.prior3 = MLP(1,2,0)
        self.prior4 = MLP(1,2,0)
        self.prior5 = MLP(1,2,0)
        self.prior6 = MLP(1,2,0)
        self.prior7 = MLP(1,2,0)
        self.prior8 = MLP(1,2,0)
        self.prior9 = MLP(1,2,0)
    
    def forward(self, x):
        # forward pass
        out0 = self.model0(x)
        out1 = self.model1(x)
        out2 = self.model2(x)
        out3 = self.model3(x)
        out4 = self.model4(x)
        out5 = self.model5(x)
        out6 = self.model6(x)
        out7 = self.model7(x)
        out8 = self.model8(x)
        out9 = self.model9(x)
        # prior pass
        prior0 = self.prior0(x)
        prior1 = self.prior1(x)
        prior2 = self.prior2(x)
        prior3 = self.prior3(x)
        prior4 = self.prior4(x)
        prior5 = self.prior5(x)
        prior6 = self.prior6(x)
        prior7 = self.prior7(x)
        prior8 = self.prior8(x)
        prior9 = self.prior9(x)

        # return
        return out0+prior0,out1+prior1, out2+prior2, out3+prior3, out4+prior4, out5+prior5, out6+prior6, out7+prior7, out8+prior8, out9+prior9

    def diversity_init(self):
        """initalzie the networks with diversity to make the softmax output different in each member"""

        for i in tqdm(range (500)):
            s=[]
            s.append(torch.FloatTensor(128,1).uniform_(-5,5))
            data=torch.cat(s,1)

            model_index=torch.randint(0,10,(1,))
            model_p = getattr(self, 'prior' + str(model_index.item()))
            model = getattr(self, 'model' + str(model_index.item()))

            optim=torch.optim.AdamW(model_p.parameters(),lr=0.005)
            # forward
            mean = model(data)+model_p(data)
            mean_plus=model(data+0.05)+model_p(data+0.05)
            mean_minus=model(data-0.05)+model_p(data-0.05)
            # compute the second derivate
            second_derivate=(mean_plus+mean_minus-2*mean)/0.05**2
            
            
            mean = torch.softmax(mean,-1)+1e-6
            means=self(data)
            # compute the predcitions of ensemble
            means=torch.stack(means)

            with torch.no_grad():
                target=means.detach()
                model_index=torch.randint(0,10,(1,))
                target=torch.softmax(target.median(0)[0],-1)+1e-6
                # target=target-target.mean(-1).unsqueeze(-1).repeat(1,1,2)
                # target[model_index]*=100000

            # compute the loss    
            optim.zero_grad()
            # loss=-F.mse_loss(mean,target)*100-means.var(0).mean()*100
            # min_distance=torch.abs(target-mean).min(0)[0]
            # loss=(torch.clamp(min_distance,0,0.5)**2).mean()*10000+(means**2).mean()
            # loss=F.mse_loss(means.var(0),target)+(means**2).mean()
            # loss=-F.mse_loss(mean,target)
            # loss=-torch.clamp((mean-target)**2,0,0.1).mean()+means.mean()**2+(torch.abs(means)**2).mean()
            kl_loss=self.KL(mean,target)
            nonlinearity_loss=-(second_derivate.abs()).mean()
            a,b,c=0.1,0,0.1
            loss=-torch.clamp(kl_loss,0,a).mean()+b*nonlinearity_loss+(means**2).mean()*c
            # loss=-torch.clamp(kl_loss,0,0.1).mean()+nonlinearity_loss
            # loss=nonlinearity_loss*10
            # optimize
            loss.backward()
            optim.step()
    def KL(self,a,b):
        return (a*torch.log(a/b)).sum(-1)
    def second_derivate(self,a):
        return torch.autograd.grad(a.sum(),a,retain_graph=True)[0]

            
x=torch.arange(-5,5,0.01).reshape(-1,1)


BSDP_model = BSDP()
y1=BSDP_model(x)


# plot the BSP cruve
for i in range(len(y1)):
    plt.plot(x,y1[i][:,0].detach(),label='model'+str(i+1))
    plt.ylim(-20,20)    
# plt.legend(loc=2)
plt.xlabel("state")
plt.ylabel("Q(s,a=0)")
plt.savefig("imgs/BSDP/BSP_0.png",dpi=500)
plt.close()

# plot the BSP cruve
for i in range(len(y1)):
    plt.plot(x,y1[i][:,1].detach(),label='model'+str(i+1))
    plt.ylim(-20,20)    
# plt.legend(loc=2)
plt.xlabel("state")
plt.ylabel("Q(s,a=1)")
plt.savefig("imgs/BSDP/BSP_1.png",dpi=500)
plt.close()

# plot the BSDP cruve
BSDP_model.diversity_init()

y1=BSDP_model(x)


# plot the BSP cruve
for i in range(len(y1)):
    plt.plot(x,y1[i][:,0].detach(),label='model'+str(i+1))
    plt.ylim(-20,20)    
# plt.legend(loc=2)
plt.xlabel("state")
plt.ylabel("Q(s,a=0)")
plt.savefig("imgs/BSDP/BSDP_0_kl_bound.png",dpi=500)
plt.close()

# plot the BSP cruve
for i in range(len(y1)):
    plt.plot(x,y1[i][:,1].detach(),label='model'+str(i+1))
    plt.ylim(-20,20)    
# plt.legend(loc=2)
plt.xlabel("state")
plt.ylabel("Q(s,a=1)")
plt.savefig("imgs/BSDP/BSDP_0_kl_bound.png",dpi=500)
plt.close()