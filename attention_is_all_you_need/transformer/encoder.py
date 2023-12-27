import torch
from torch import nn 
from transformer.attention import MultiHeadAttention 
from transformer.positionwise_feed_forward import PositionwiseFeedForward 
from transformer.layernorm import LayerNorm 
from transformer.position_encoding import PositionalEncoding



class Encoder(nn.Module):
    def __init__(self,n_layers=1, drop_prob=0.1, vocab_size=51, seq_len=2000, d_model=512, device='cpu'):
        super(Encoder, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_embedding = PositionalEncoding(d_model, seq_len)
        self.attention = MultiHeadAttention()
        self.ffn = PositionwiseFeedForward()
        self.layer_norm = LayerNorm()
        self.dropout = nn.Dropout(p=drop_prob)
        self.n_layers = n_layers
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.device = device

    def forward(self, x, encoder_mask):
        B, T = x.shape
        tok_emb = self.token_embedding(x)
        x = self.positional_embedding(tok_emb)

        for i in range(self.n_layers):
            _x = x
            x = self.attention(x,x,x, mask=encoder_mask)
            x = self.dropout(x)
            x = self.layer_norm(x + _x)

            _x = x
            x = self.ffn(x)
            x = self.dropout(x)
            x = self.layer_norm(x + _x)

        return x
        