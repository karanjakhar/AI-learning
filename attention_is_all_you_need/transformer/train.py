import pandas as pd
from torch.utils.data import DataLoader
from torch import nn
import torch
from tqdm import tqdm

from encoder_decoder_model import EncoderDecoder
from transformer.dataset import SummaryDataset
from transformer.tokenizer import get_or_build_tokenizer



def validation():
    pass





def train(epochs = 5):
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.has_mps or torch.backends.mps.is_available() else "cpu"
    device = 'cpu' # for testing purposes
    experiment_data_df = pd.read_csv('experiment_summary_data.csv')
    tokenizer = get_or_build_tokenizer('./tokenizer_exp.json', experiment_data_df)
    summary_dataset = SummaryDataset(experiment_data_df, 300, tokenizer )
    
    train_dataloader = DataLoader(summary_dataset, batch_size=2, shuffle=True)

    model = EncoderDecoder(device=device, vocab_size=tokenizer.get_vocab_size())

    # Initialize the parameters
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=10**-4, eps=1e-9)

    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.token_to_id('[PAD]'), label_smoothing=0.1).to(device)


    for epoch in range(epochs):
        if device == 'cuda':
            torch.cuda.empty_cache()
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f'Processing Epoch {epoch:02d}')
        for batch in batch_iterator:

            encoder_input = batch['encoder_input'].to(device)
            decoder_input = batch['decoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)
            decoder_mask = batch['decoder_mask'].to(device)

            decoder_output = model(encoder_input, decoder_input, encoder_mask, decoder_mask)

            # decoder_output shape (Batch, seq_len, vocab_size)
            label = batch['label'].to(device) # (Batch, seq_len)

            loss = loss_fn(decoder_output.view(-1, tokenizer.get_vocab_size()), label.view(-1))
            batch_iterator.set_postfix({'loss': f'{loss.item():6.3f}'})

            loss.backward()

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        