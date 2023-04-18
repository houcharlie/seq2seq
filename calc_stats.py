from datasets import load_dataset
import torch 
from torch.utils.data import DataLoader
from transformers import (BartTokenizerFast, default_data_collator, GenerationConfig, LogitsProcessorList)
from models.bart_embed_one import BartForConditionalGenerationOne
from tqdm import tqdm
import numpy as np

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
dataset = load_dataset('bookcorpus', split="train[95%:]", cache_dir='/home/ubuntu/huggingface')
tokenizer = BartTokenizerFast.from_pretrained("/home/ubuntu/bartone/final_epoch/gpt2_tokenizer")
model = BartForConditionalGenerationOne.from_pretrained("/home/ubuntu/bartone/final_epoch")

encoder = model.get_encoder()
column_names = dataset.column_names
max_seq_length = 64
text_column_name = "text" if "text" in column_names else column_names[0]

def tokenize_function(examples):
    examples[text_column_name] = [
                line for line in examples[text_column_name] if len(line) > 0 and not line.isspace()
            ]
    source_tokenized = tokenizer(
        examples[text_column_name], padding="max_length", truncation=True, max_length=max_seq_length
    )

    batch = {k: v for k, v in source_tokenized.items()}
    return batch

test_dataset = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=10,
        remove_columns=column_names,
        desc="Running bart tokenizers on train dataset",
    )
test_dataloader = DataLoader(
        test_dataset, shuffle=True, collate_fn=default_data_collator, batch_size=16
    )

model.eval()
model.to(device)
encoder = model.get_encoder()
sentence_rep_list = []
with torch.no_grad():
    for batch in tqdm(test_dataloader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        outputs = encoder(input_ids=input_ids, attention_mask=attention_mask)
        sentence_reps = outputs[0][:,0,:].cpu().numpy()
        sentence_rep_list.append(sentence_reps)

sentence_rep_vec = np.vstack(sentence_rep_list)

sentence_mean = np.mean(sentence_rep_vec, axis=0)
sentence_cov = np.cov(sentence_rep_vec, rowvar=False)
np.save('/home/ubuntu/seq2seq/90plus_mean.np', sentence_mean)
np.save('/home/ubuntu/seq2seq/90plus_cov.np', sentence_cov)
