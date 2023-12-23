from torch import nn 
from transformer.encoder import Encoder
from transformer.decoder import Decoder


class EncoderDecoder(nn.Module):
    def __init__(self, d_model=512, vocab_size=51, device='cpu') -> None:
        super(EncoderDecoder, self).__init__()
        self.encoder = Encoder(device=device, vocab_size=vocab_size)
        self.decoder = Decoder(device=device, vocab_size=vocab_size)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self,encoder_input, decoder_input, encoder_mask, decoder_mask):
        encoder_output = self.encoder(encoder_input, encoder_mask)
        output = self.decoder(decoder_input, encoder_output, encoder_output,encoder_mask, decoder_mask)
        output = self.lm_head(output)
        return output
    