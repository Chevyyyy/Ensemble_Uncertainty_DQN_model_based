from torch import nn
from torch.functional import F
class MLP(nn.Module):

    def __init__(self, n_observations, n_actions,prior):
        super(MLP, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)
        if prior>0:
            self.layer3.bias.data=prior
            self.layer3.bias.requires_grad = False


    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

