from models.bart_embed_one import BartForConditionalGenerationOne
from datasets import load_dataset
import datasets
import numpy as np
import torch
from tabulate import tabulate
import nltk
import wandb
from transformers import (
    BartTokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
)
from datetime import datetime
model = BartForConditionalGenerationOne.from_pretrained("facebook/bart-base", cache_dir='/dev/shm/huggingface')
tokenizer = BartTokenizer.from_pretrained("facebook/bart-base", cache_dir='/dev/shm/huggingface')

print('available cuda devices', torch.cuda.device_count())

data = load_dataset("bookcorpus", split="train[:200000]", cache_dir='/dev/shm/huggingface')
encoder_max_length = 256
decoder_max_length = 256
print(data['text'][0])

def list2samples(example):
    srcs = []
    tgts = []
    for sample in example['text']:
        srcs.append(sample)
        tgts.append(sample)
    return {"sources": srcs, "targets": tgts}

dataset = data.map(list2samples, batched=True)

print(dataset['sources'][1])
print(dataset['targets'][1])
train_data_txt, validation_data_txt = dataset.train_test_split(test_size=0.1).values()

def batch_tokenize_preprocess(batch, tokenizer, max_source_length, max_target_length):
    source, target = batch["sources"], batch["targets"]
    source_tokenized = tokenizer(
        source, padding="max_length", truncation=True, max_length=max_source_length
    )
    target_tokenized = tokenizer(
        target, padding="max_length", truncation=True, max_length=max_target_length
    )

    batch = {k: v for k, v in source_tokenized.items()}
    # Ignore padding in the loss
    batch["labels"] = [
        [-100 if token == tokenizer.pad_token_id else token for token in l]
        for l in target_tokenized["input_ids"]
    ]
    return batch

train_data = train_data_txt.map(
    lambda batch: batch_tokenize_preprocess(
        batch, tokenizer, encoder_max_length, decoder_max_length
    ),
    batched=True,
    remove_columns=train_data_txt.column_names,
)

validation_data = validation_data_txt.map(
    lambda batch: batch_tokenize_preprocess(
        batch, tokenizer, encoder_max_length, decoder_max_length
    ),
    batched=True,
    remove_columns=validation_data_txt.column_names,
)

nltk.download("punkt", quiet=True)

metric = datasets.load_metric("rouge")


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

    return preds, labels


def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = metric.compute(
        predictions=decoded_preds, references=decoded_labels, use_stemmer=True
    )
    # Extract a few results from ROUGE
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}

    prediction_lens = [
        np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds
    ]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result

training_args = Seq2SeqTrainingArguments(
    output_dir="results",
    num_train_epochs=200,  # demo
    do_train=True,
    do_eval=True,
    per_device_train_batch_size=16,  # demo
    per_device_eval_batch_size=4,
    # learning_rate=3e-05,
    warmup_steps=500,
    weight_decay=0.1,
    label_smoothing_factor=0.1,
    predict_with_generate=True,
    logging_dir="logs",
    logging_steps=50,
    save_total_limit=3,
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_data,
    eval_dataset=validation_data,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)
trainer.args._n_gpu = 1
wandb_run = wandb.init(
    project="bart_book_corpus",
    config={
        "per_device_train_batch_size": training_args.per_device_train_batch_size,
        "learning_rate": training_args.learning_rate,
        "dataset": "book_corpus_100000",
    },
)

now = datetime.now()
current_time = now.strftime("%H%M%S")
wandb_run.name = "run_" + current_time
print(trainer.evaluate())
trainer.train()
print(trainer.evaluate())
wandb_run.finish()
def generate_summary(test_samples, model):
    inputs = tokenizer(
        test_samples["sources"],
        padding="max_length",
        truncation=True,
        max_length=encoder_max_length,
        return_tensors="pt",
    )
    input_ids = inputs.input_ids.to(model.device)
    attention_mask = inputs.attention_mask.to(model.device)
    outputs = model.generate(input_ids, attention_mask=attention_mask)
    output_str = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return outputs, output_str


test_samples = validation_data_txt.select(range(16))
train_samples = train_data_txt.select(range(16))

summaries_after_tuning = generate_summary(test_samples, model)[1]


print(
    tabulate(
        zip(
            range(len(summaries_after_tuning)),
            summaries_after_tuning,
            test_samples["targets"],
        ),
        headers=["Id", "Text after [test]", "Text before [test]"],
    )
)



print("TRAIN RESULTS")

summaries_after_tuning = generate_summary(train_samples, model)[1]


print(
    tabulate(
        zip(
            range(len(summaries_after_tuning)),
            summaries_after_tuning,
            train_samples["targets"],
        ),
        headers=["Id", "Text after [train]", "Text before [train]"],
    )
)

# Take a look at the data
# for k, v in data["article"][0].items():
#     print(k)
#     print(v)

# def flatten(example):
#     return {
#         "document": example["article"]["document"],
#         "summary": example["article"]["summary"],
#     }


# def list2samples(example):
#     documents = []
#     summaries = []
#     for sample in zip(example["document"], example["summary"]):
#         if len(sample[0]) > 0:
#             documents += sample[0]
#             summaries += sample[1]
#     return {"document": documents, "summary": summaries}


# dataset = data.map(flatten, remove_columns=["article", "url"])
# dataset = dataset.map(list2samples, batched=True)

# train_data_txt, validation_data_txt = dataset.train_test_split(test_size=0.1).values()
