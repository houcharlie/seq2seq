import argparse
import json
import logging
import math
import os
import random
from itertools import chain
from pathlib import Path

import datasets
import torch
from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset
from huggingface_hub import Repository, create_repo
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from tabulate import tabulate
import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    SchedulerType,
    get_scheduler,
    GPT2TokenizerFast,
    RobertaTokenizerFast,
    default_data_collator,
    GenerationConfig
)
from transformers.utils import check_min_version, get_full_repo_name, send_example_telemetry
from transformers.utils.versions import require_version
from models.conditioned_gpt2 import RobertaCondGPT2
logger = get_logger(__name__)

dataset_name = 'bookcorpus'

generation_config = GenerationConfig.from_pretrained("gpt2")
generation_config.max_new_tokens = 512
roberta_tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base", cache_dir='/dev/shm/huggingface', padding_side='right', truncation_side='right')
gpt2_tokenizer = GPT2TokenizerFast.from_pretrained("gpt2", cache_dir='/dev/shm/huggingface', padding_side='right', truncation_side='right')
gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token
model = RobertaCondGPT2.from_encoder_decoder_pretrained('roberta-base', 'gpt2', '/dev/shm/huggingface')
max_seq_length = 512
dataset = load_dataset(dataset_name, split="train[:1000]", cache_dir='/dev/shm/huggingface')
train_data_txt, validation_data_txt = dataset.train_test_split(test_size=0.1).values()
def generate_summary(test_samples, model):
    inputs = roberta_tokenizer(
        test_samples["text"],
        padding="max_length",
        truncation=True,
        max_length=max_seq_length,
        return_tensors="pt",
    )
    input_ids = inputs.input_ids.to(model.device)
    attention_mask = inputs.attention_mask.to(model.device)
    outputs = model.generate(input_ids, attention_mask=attention_mask, generation_config=generation_config, max_new_tokens=512)
    output_str = gpt2_tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return outputs, output_str


test_samples = validation_data_txt.select(range(16))
train_samples = train_data_txt.select(range(16))

summaries_after_tuning = generate_summary(test_samples, model)[1]


print(
    tabulate(
        zip(
            range(len(summaries_after_tuning)),
            test_samples["text"],
            summaries_after_tuning,

        ),
        headers=["Id", "Text before [test]", "Text after [test]"],
    )
)

summaries_after_tuning = generate_summary(train_samples, model)[1]


print(
    tabulate(
        zip(
            range(len(summaries_after_tuning)),
            train_samples["text"],
            summaries_after_tuning,
        ),
        headers=["Id", "Text before [train]", "Text after [train]"],
    )
)