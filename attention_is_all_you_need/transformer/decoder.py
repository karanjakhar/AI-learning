import torch
from torch import nn 
from transformer.attention import MultiHeadAttention 
from transformer.positionwise_feed_forward import PositionwiseFeedForward 
from transformer.layernorm import LayerNorm 



class Decoder(nn.Module):
    def __init__(self,n_layers=1, vocab_size=51, d_model=512,drop_prob=0.1, device='cpu') -> None:
        super(Decoder, self).__init__()

        self.positional_embedding = nn.Embedding(vocab_size, d_model)
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.attention = MultiHeadAttention()
        self.ffn = PositionwiseFeedForward()
        self.layer_norm = LayerNorm()
        self.dropout = nn.Dropout(p=drop_prob)

        self.n_layers = n_layers
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.device = device

        

    
    def forward(self, x,k,v,encoder_mask, decoder_mask):
        B, T = x.shape
        tok_emb = self.token_embedding(x) 
        pos_emb = self.positional_embedding(torch.arange(T, device=self.device))
        x = tok_emb + pos_emb


        for i in range(self.n_layers):
            _x = x
            x = self.attention(x,x,x, decoder_mask)
            x = self.dropout(x)
            x = self.layer_norm(x + _x)

            _x = x
            x = self.attention(k,v,x, encoder_mask)
            x = self.dropout(x)
            x = self.layer_norm(x + _x)

            _x = x
            x = self.ffn(x)
            x = self.dropout(x)
            x = self.layer_norm(x + _x)

            
        return x
            
