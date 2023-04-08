from models.conditioned_gpt2 import RobertaCondGPT2
from transformers import RobertaTokenizer, GPT2Tokenizer
from datasets import load_dataset

data = load_dataset("bookcorpus", split="train[:1000]", cache_dir='/dev/shm/huggingface')
roberta_tokenizer = RobertaTokenizer.from_pretrained("roberta-base", cache_dir='/dev/shm/huggingface', padding_side='right', truncation_side='right')
gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2", cache_dir='/dev/shm/huggingface', padding_side='right', truncation_side='right')
gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token
roberta_inputs = roberta_tokenizer(data['text'][:5], padding="max_length", truncation=True, max_length=256, return_tensors='pt')

gpt2_inputs = gpt2_tokenizer(data['text'][:5], padding="max_length", truncation=True, max_length=256, return_tensors='pt')

model = RobertaCondGPT2.from_encoder_decoder_pretrained('roberta-base', 'gpt2', '/dev/shm/huggingface')

model_output = model(roberta_inputs['input_ids'].to(model.device), roberta_inputs['attention_mask'].to(model.device), gpt2_inputs['input_ids'].to(model.device), gpt2_inputs['attention_mask'].to(model.device))
print(model_output.loss)