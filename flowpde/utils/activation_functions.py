import torch
from torch import nn, Tensor


class Swish(nn.Module):
    """Swish activation function: f(x) = x * sigmoid(x)
    
    Also known as SiLU (Sigmoid Linear Unit).
    Self-gated activation function that has been shown to
    work better than ReLU in many deep learning applications.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor: 
        return torch.sigmoid(x) * x
