from torch import nn 
from encoder import Encoder
from decoder import Decoder


class EncoderDecoder(nn.Module):
    def __init__(self, d_model=512, vocab_size=51) -> None:
        super(EncoderDecoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self,x):
        encoder_output = self.encoder(x)
        output = self.decoder(x, encoder_output, encoder_output)
        output = self.lm_head(output)
        return output
    