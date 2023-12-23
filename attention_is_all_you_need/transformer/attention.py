import torch
from torch import nn

class MultiHeadAttention(nn.Module):
    def __init__(self, heads: int = 8, d_model: int = 512, context_window = 16):
        super(MultiHeadAttention, self).__init__()
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)
        self.q = nn.Linear(d_model, d_model, bias=False)
        self.w_concat = nn.Linear(d_model, d_model)
        self.softmax = nn.Softmax(dim=-1)
        self.heads = heads
        #self.register_buffer('mask', torch.tril(torch.ones(context_window,context_window)).view(1,1,context_window,context_window))
    

    def forward(self, k: torch.tensor, v: torch.tensor, q: torch.tensor, mask: torch.tensor = None):

        k = self.split(self.k(k))
        v = self.split(self.v(v))
        q = self.split(self.q(q))



        k_t = k.transpose(-2,-1)
        # print(q.shape, k_t.shape)
        scale = (q@k_t)*(k.shape[-1]**-0.5)
        # print(scale.shape)
        if mask is not None:
            # Write a very low value (indicating -inf) to the positions where mask == 0
            scale = scale.masked_fill_(mask == 0, -1e9)
        # if mask:
        #     B, nH, T, Hs = scale.shape 
        #     scale = scale.masked_fill(mask[:,:,:T,:T] == 0, -10000)
        #     # print('scale shape:', scale.shape)
        #     # print('scale values:', scale)

        score = self.softmax(scale)

        scale_product = score @ v
        
        batch,heads, length, d_tensor = scale_product.size()
        scale_product = scale_product.transpose(1, 2).contiguous().view(batch, length, heads*d_tensor)
        # print(scale_product.shape)
        scale_product = self.w_concat(scale_product)

        return scale_product
    
    def split(self, tensor:torch.tensor):

        batch, length, d_model = tensor.size()
        d_tensor = d_model//self.heads

        new_tensor = tensor.view(batch, length, self.heads, d_tensor).transpose(1, 2)

        return new_tensor
    




    