from torch import nn
import torch
from torch.functional import F

class CNN(nn.Module):

    def __init__(self, state_shape, n_actions):
        super(CNN, self).__init__()
        self.state_shape=state_shape
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=state_shape[-1], out_channels=32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        self.fc = nn.Sequential(
            nn.Linear(in_features=(state_shape[0]-6)*(state_shape[0]-6)*64 , out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=n_actions)
        )

    def forward(self, x):
        x=x.reshape(x.shape[0],x.shape[3],x.shape[2],x.shape[1]).float()
        conv_out = self.conv(x).view(x.size()[0],-1)
        return self.fc(conv_out)

