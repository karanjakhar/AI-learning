import pandas as pd
from torch.utils.data import DataLoader
from torch import nn

from encoder_decoder_model import EncoderDecoder
from transformer.dataset import SummaryDataset
from transformer.tokenizer import get_or_build_tokenizer

def train():

    experiment_data_df = pd.read_csv('experiment_summary_data.csv')
    tokenizer = get_or_build_tokenizer('./tokenizer_exp.json', experiment_data_df)
    summary_dataset = SummaryDataset(experiment_data_df, 300, tokenizer )
    
    train_dataloader = DataLoader(summary_dataset, batch_size=2, shuffle=True)

    model = EncoderDecoder()

    # Initialize the parameters
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)