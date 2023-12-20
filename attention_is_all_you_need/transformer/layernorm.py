import torch
from torch import nn

class LayerNorm(nn.Module):
    def __init__(self,d_model=512, eps=1e-12):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps
    def forward(self,x):
        mean = x.mean(-1,keepdim=True)
        var = x.var(-1, keepdim=True, unbiased=False)

        out = (x - mean)/torch.sqrt(var + self.eps)
        out = out * self.gamma + self.beta
        return out
