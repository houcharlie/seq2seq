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
from accelerate import Accelerator, DistributedType, DistributedDataParallelKwargs
from accelerate.logging import get_logger
from accelerate.utils import set_seed, ProjectConfiguration
from datasets import load_dataset
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
    BartTokenizerFast,
    default_data_collator,
    GenerationConfig
)
from transformers.utils import check_min_version, get_full_repo_name, send_example_telemetry
from transformers.utils.versions import require_version
from models.bart_embed_one import BartForConditionalGenerationOneNoise

logger = get_logger(__name__)

## parameters
batch_noise = 4
gradient_accumulation_steps = 8
output_dir = '/home/ubuntu/bartone/noise0.5'
max_seq_length = 64
data_num_workers = 51
dataset_name = 'bookcorpus'
per_device_train_batch_size = 16
per_device_eval_batch_size = 8
weight_decay = 0.1
learning_rate = 5e-5
max_train_steps = None
num_train_epochs = 10
num_warmup_steps = 500
checkpointing_steps_set = 'epoch'
log_steps = 1
report_to = 'wandb'
lr_scheduler_type = 'linear'
args = {}
args['gradient_accumulation_steps'] = gradient_accumulation_steps
args['output_dir'] = output_dir
args['max_seq_len'] = max_seq_length
args['dataset_name'] = dataset_name
args['per_device_train_batch_size'] = per_device_train_batch_size
args['weight_decay'] = weight_decay
args['learning_rate'] = learning_rate
args['epochs'] = num_train_epochs
args['num_warmup_steps'] = num_warmup_steps
args['lr_scheduler_type'] = 'linear'


## script
accelerator_log_kwargs = {}


accelerator_log_kwargs["log_with"] = report_to
accelerator_log_kwargs["logging_dir"] = output_dir
kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
accelerator = Accelerator(gradient_accumulation_steps=gradient_accumulation_steps, kwargs_handlers=[kwargs], **accelerator_log_kwargs)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger.info(accelerator.state, main_process_only=False)
if accelerator.is_local_main_process:
    datasets.utils.logging.set_verbosity_warning()
    transformers.utils.logging.set_verbosity_info()
else:
    datasets.utils.logging.set_verbosity_error()
    transformers.utils.logging.set_verbosity_error()

if accelerator.is_main_process:
    os.makedirs(output_dir, exist_ok=True)

accelerator.wait_for_everyone()

dataset = load_dataset(dataset_name, split="train[50%:51%]", cache_dir='/home/ubuntu/huggingface')
train_data_txt, validation_data_txt = dataset.train_test_split(test_size=0.1).values()
# if "validation" not in raw_datasets.keys():


column_names = train_data_txt.column_names
text_column_name = "text" if "text" in column_names else column_names[0]
tokenizer = BartTokenizerFast.from_pretrained("/home/ubuntu/bartone/reconstruction/no_noise/bart_tokenizer")
model = BartForConditionalGenerationOneNoise.from_pretrained("/home/ubuntu/bartone/reconstruction/no_noise")


def tokenize_function(examples):
    examples[text_column_name] = [
                line for line in examples[text_column_name] if len(line) > 0 and not line.isspace()
            ]
    source_tokenized = tokenizer(
        examples[text_column_name], padding="max_length", truncation=True, max_length=max_seq_length
    )

    batch = {k: v for k, v in source_tokenized.items()}
    
    batch["labels"] = [
        [-100 if token == tokenizer.pad_token_id else token for token in l]
        for l in source_tokenized["input_ids"]
    ]
    return batch

with accelerator.main_process_first():
    train_dataset = train_data_txt.map(
        tokenize_function,
        batched=True,
        num_proc=data_num_workers,
        remove_columns=column_names,
        desc="Running bart tokenizers on train dataset",
    )
    eval_dataset = validation_data_txt.map(
        tokenize_function,
        batched=True,
        num_proc=data_num_workers,
        remove_columns=column_names,
        desc="Running bart tokenizers on validationdataset",
    )


train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=default_data_collator, batch_size=per_device_train_batch_size
    )
eval_dataloader = DataLoader(
    eval_dataset, collate_fn=default_data_collator, batch_size=per_device_train_batch_size
)

no_decay = ["bias", "layer_norm.weight"]
optimizer_grouped_parameters = [
    {
        "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
        "weight_decay": weight_decay,
    },
    {
        "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
        "weight_decay": 0.0,
    },
]
optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=learning_rate)

overrode_max_train_steps = False
num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
if max_train_steps is None:
    max_train_steps = num_train_epochs * num_update_steps_per_epoch
    overrode_max_train_steps = True

lr_scheduler = get_scheduler(
    name=lr_scheduler_type,
    optimizer=optimizer,
    num_warmup_steps=num_warmup_steps * gradient_accumulation_steps,
    num_training_steps=max_train_steps * gradient_accumulation_steps,
)

# Prepare everything with our `accelerator`.
model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
)

num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
if overrode_max_train_steps:
    max_train_steps = num_train_epochs * num_update_steps_per_epoch
# Afterwards we recalculate our number of training epochs
num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

# Figure out how many steps we should save the Accelerator states
checkpointing_steps = checkpointing_steps_set
if checkpointing_steps is not None and checkpointing_steps.isdigit():
    checkpointing_steps = int(checkpointing_steps)
experiment_config = args
# TensorBoard cannot log Enums, need the raw value
accelerator.init_trackers("clm_no_trainer", experiment_config)
total_batch_size = per_device_train_batch_size * accelerator.num_processes * gradient_accumulation_steps

logger.info("***** Running training *****")
logger.info(f"  Num examples = {len(train_dataset)}")
logger.info(f"  Num Epochs = {num_train_epochs}")
logger.info(f"  Instantaneous batch size per device = {per_device_train_batch_size}")
logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
logger.info(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
logger.info(f"  Total optimization steps = {max_train_steps}")

completed_steps = 0
starting_epoch = 0

for epoch in range(starting_epoch, num_train_epochs):
    model.train()
    total_loss = 0
    for step, batch in enumerate(train_dataloader):
        with accelerator.accumulate(model):
            outputs = model(**batch, batch_noise=batch_noise) 
            loss = outputs.loss
            total_loss += loss.detach().float()
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        # Checks if the accelerator has performed an optimization step behind the scenes
        if accelerator.sync_gradients:
            completed_steps += 1
        
        # if isinstance(checkpointing_steps, int):
        #     if completed_steps % checkpointing_steps == 0:
        #         output_dir = f"step_{completed_steps }"
        #         if output_dir is not None:
        #             output_dir = os.path.join(output_dir, output_dir)
        #         accelerator.save_state(output_dir)
        
        if accelerator.is_main_process:
            if step % 100 == 0:
                print('Epoch', epoch, 'step', step, 'loss', loss.detach().float().item(), 'percent done', completed_steps/num_update_steps_per_epoch)

        if completed_steps >= max_train_steps:
            break

    model.eval()
    losses = []
    for step, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            outputs = model(**batch, batch_noise=batch_noise)

        loss = outputs.loss
        losses.append(accelerator.gather_for_metrics(loss.repeat(per_device_eval_batch_size)))

    losses = torch.cat(losses)
    try:
        eval_loss = torch.mean(losses)
        perplexity = math.exp(eval_loss)
    except OverflowError:
        perplexity = float("inf")
    if accelerator.is_main_process:
        logger.info(f"epoch {epoch}: perplexity: {perplexity} eval_loss: {eval_loss}")
    accelerator.log(
                {
                    "perplexity": perplexity,
                    "eval_loss": eval_loss,
                    "train_loss": total_loss.item() / len(train_dataloader),
                    "epoch": epoch,
                    "step": completed_steps,
                },
                step=completed_steps,
            )
    if checkpointing_steps == "epoch":
        curr_output_dir = f"latest"
        if output_dir is not None:
            curr_output_dir = os.path.join(output_dir, curr_output_dir)
        # accelerator.save_state(curr_output_dir)
        if accelerator.is_main_process:
            unwrapped_model = accelerator.unwrap_model(model)
            def generate_summary(test_samples, model):
                inputs = tokenizer(
                    test_samples["text"],
                    padding="max_length",
                    truncation=True,
                    max_length=max_seq_length,
                    return_tensors="pt",
                )
                input_ids = inputs.input_ids.to(model.device)
                attention_mask = inputs.attention_mask.to(model.device)
                outputs = unwrapped_model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=max_seq_length)
                output_str = tokenizer.batch_decode(outputs, skip_special_tokens=True)
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
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(
                curr_output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
            )
            tokenizer.save_pretrained(os.path.join(curr_output_dir, 'bart_tokenizer'))
            with open(os.path.join(curr_output_dir, "all_results.json"), "w") as f:
                json.dump({"perplexity": perplexity}, f)


if output_dir is not None:
    curr_output_dir = os.path.join(output_dir, 'final_epoch')
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(
        curr_output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
    )
    if accelerator.is_main_process:
        tokenizer.save_pretrained(os.path.join(curr_output_dir, 'bart_tokenizer'))
        with open(os.path.join(output_dir, "all_results.json"), "w") as f:
            json.dump({"perplexity": perplexity}, f)





