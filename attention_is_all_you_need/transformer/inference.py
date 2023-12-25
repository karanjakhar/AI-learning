import torch
from transformer.dataset import causal_mask

def greedy_decode(model, input_text, input_mask, tokenizer, max_len, device='cpu'):
    sos_idx = tokenizer.token_to_id('[SOS]')
    eos_idx = tokenizer.token_to_id('[EOS]')

    # Compute the encoder output once
    encoder_output = model.encode(input_text, input_mask)

    # Initialize decoder input with sos token and continue decode call till max_len or get eos
    decoder_input = torch.empty((1,1)).fill_(sos_idx).type_as(input_text).to(device)
    next_word = sos_idx
    while decoder_input != max_len and next_word != eos_idx:

        # build decoder mask
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(input_mask).to(device)

        decoder_output = model.decode(decoder_input, encoder_output, input_mask, decoder_mask)
        output = model.projection_layer(decoder_output)

        _, next_word = torch.max(output[:, -1])

        decoder_input = torch.cat(
            [decoder_input, torch.empty((1,1)).type_as(input_text).fill_(next_word.item()).to(device)],
            dim=1
        )


    return decoder_input.squeeze(0)

