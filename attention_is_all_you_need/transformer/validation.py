import torch
import os
from transformer.inference import greedy_decode

def run_validation(model, validation_ds, tokenizer, max_len, device, print_msg):
    model.eval()
    count = 0

    source_texts = []
    expected = []
    predicted = []


    try:
        # get the console window width to print nice separations
        with os.popen('stty size', 'r') as console:
            _, console_width = console.read().split()
            console_width = int(console_width)
            
    except:
        console_width = 80


    with torch.no_grad():
        for batch in validation_ds:
            count += 1
            encoder_input = batch['encoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)


            # check that the batch size is 1
            assert encoder_input.size(0) == 1, "Batch size must be 1 for validation"

            model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer, max_len, device)

            source_text = batch['src_text']
            target_text = batch['tgt_text']
            model_out_text = tokenizer.decode(model_out.detach().cpu().numpy())

            # source_texts.append(source_text)
            # expected.append(target_text)
            # predicted.append(model_out_text)

            # Print the source, target and model output using tqdm print
            print_msg('-'*console_width)
            # print_msg(f"{'SOURCE: ':>12}{source_text}")
            print_msg(f"{'TARGET: ':>12}{target_text}")
            print_msg(f"{'PREDICTED: ':>12}{model_out_text}")

            print_msg('-'*console_width)
