## every batch in data should have 
## input, label, encoder_padding_mask, decoder_padding_mask, decoder_mask


import pandas as pd 

test_df = pd.read_csv('/home/karan/kj_workspace/kj_ai/AI-learning/attention_is_all_you_need/data/cnn_dailymail/test.csv')
from torch.utils.data import Dataset
import torch

class SummaryDataset(Dataset):
    def __init__(self, ds, seq_len, tokenizer) -> None:
        super().__init__()
        self.ds = ds
        self.seq_len = seq_len
        self.tokenizer = tokenizer

        self.sos_token = torch.tensor([tokenizer.token_to_id("[SOS]")], dtype=torch.int64)
        self.eos_token = torch.tensor([tokenizer.token_to_id("[EOS]")], dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer.token_to_id("[PAD]")], dtype=torch.int64)

    def __len__(self):
        return len(self.ds)
    

    def __getitem__(self, index):
        src_text = self.ds['input'][index]
        tgt_text = self.ds['summary'][index]

        #Transform the text into tokens
        enc_input_tokens = self.tokenizer.encode(src_text).ids
        dec_input_tokens = self.tokenizer.encode(tgt_text).ids

        # Add sos (start of sentence), eos (end of sentence)
        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2 # 2 for start and end token

        # Add sos (start of sentence) to decoder input and eos (end of sentence) to label
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1 # 1 for sos for decoder input and eos for label

        # Make sure the number of padding tokens is not negative. If it is, the sentence is too long
        # number of enc_input_tokens and dec_input_tokens should be less than set sequence length (seq_len)
        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError(f"Sentence is too long. Encode tokens: {len(enc_input_tokens)} or Decode tokens: {len(dec_input_tokens)} greater than seq_len: {self.seq_len}")
        

        # Add start (sos) and end (eos) token
        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(enc_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token]*enc_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0
        )

        # Add only start token (sos)
        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(dec_input_tokens, dtype=torch.int64), 
                torch.tensor([self.pad_token]*dec_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )

        # Add only end token (eos)
        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token]*dec_num_padding_tokens, dtype=torch.int64)
            ],
            dim=0,
        )

        # Check all tensors are seq_len long
        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        return {
            "encoder_input": encoder_input, # (seq_len)
            "decoder_input": decoder_input, # (seq_len)
            "label": label, # (seq_len)
            "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(), # (1, 1, seq_len)
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int() & causal_mask(decoder_input.size(0)), # (1, 1, seq_len)
            "src_text": src_text,
            "tgt_text": tgt_text,
        }
    

def causal_mask(size):
    mask = torch.tril(torch.ones((1,size,size))).type(torch.int)
    return mask