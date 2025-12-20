import torch
from torch import nn, Tensor


class Swish(nn.Module):
    """Swish activation function: $f(x) = x \cdot \sigma(x)$
    
    Also known as SiLU (Sigmoid Linear Unit).
    Self-gated activation function that has been shown to
    work better than ReLU in many deep learning applications.
    
    Where $\sigma(x) = \frac{1}{1 + e^{-x}}$ is the sigmoid function.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor: 
        return torch.sigmoid(x) * x
