import torch
from torch import nn, Tensor

from utils.activation_functions import Swish



class MLP(nn.Module):
    def __init__(self, input_dim: int = 2, time_dim: int = 1, hidden_dim: int = 128):
        super().__init__()
        
        self.input_dim = input_dim
        self.time_dim = time_dim
        self.hidden_dim = hidden_dim

        self.main = nn.Sequential(
            nn.Linear(input_dim+input_dim+time_dim, hidden_dim),
            Swish(),
            nn.Linear(hidden_dim, hidden_dim),
            Swish(),
            nn.Linear(hidden_dim, hidden_dim),
            Swish(),
            nn.Linear(hidden_dim, hidden_dim),
            Swish(),
            nn.Linear(hidden_dim, input_dim),
            )
    

    def forward(self, x: Tensor, cond: Tensor, t: Tensor) -> Tensor:
        
        #print("INSIDE")
        sz = x.size()
        #print(x.shape)
        x = x.reshape(-1, self.input_dim)
        #print(x.shape)
        t = t.reshape(-1, self.time_dim).float()
        #print(t.shape)
        f = cond.reshape(-1, self.input_dim)

        t = t.reshape(-1, 1).expand(x.shape[0], 1)
        h = torch.cat([x, f, t], dim=1)
        output = self.main(h)
        
        return output.reshape(*sz)