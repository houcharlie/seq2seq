from models.bart_embed_one import BartForConditionalGenerationOne
from transformers import (BartTokenizerFast, default_data_collator, GenerationConfig, LogitsProcessorList)
from transformers.generation.logits_process import (
    EncoderNoRepeatNGramLogitsProcessor,
    EncoderRepetitionPenaltyLogitsProcessor,
    EpsilonLogitsWarper,
    EtaLogitsWarper,
    ExponentialDecayLengthPenalty,
    ForcedBOSTokenLogitsProcessor,
    ForcedEOSTokenLogitsProcessor,
    ForceTokensLogitsProcessor,
    HammingDiversityLogitsProcessor,
    InfNanRemoveLogitsProcessor,
    LogitNormalization,
    LogitsProcessorList,
    MinLengthLogitsProcessor,
    MinNewTokensLengthLogitsProcessor,
    NoBadWordsLogitsProcessor,
    NoRepeatNGramLogitsProcessor,
    PrefixConstrainedLogitsProcessor,
    RepetitionPenaltyLogitsProcessor,
    SuppressTokensAtBeginLogitsProcessor,
    SuppressTokensLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
    TypicalLogitsWarper,
)
from transformers.generation.stopping_criteria import (
    MaxLengthCriteria,
    MaxTimeCriteria,
    StoppingCriteria,
    StoppingCriteriaList,
    validate_stopping_criteria,
)
from datasets import load_dataset
from torch.utils.data import DataLoader
import torch 
from typing import Callable, List, Optional, Union

def _merge_criteria_processor_list(
        default_list: Union[LogitsProcessorList, StoppingCriteriaList],
        custom_list: Union[LogitsProcessorList, StoppingCriteriaList],
    ) -> Union[LogitsProcessorList, StoppingCriteriaList]:
        if len(custom_list) == 0:
            return default_list
        for default in default_list:
            for custom in custom_list:
                if type(custom) is type(default):
                    object_type = "stopping criteria" if isinstance(custom, StoppingCriteria) else "logits processor"
                    raise ValueError(
                        f"A custom {object_type} of type {type(custom)} with values {custom} has been passed to"
                        f" `generate`, but it has already been created with the values {default}. {default} has been"
                        " created by passing the corresponding arguments to generate or by the model's config default"
                        f" values. If you just want to change the default values of {object_type} consider passing"
                        f" them as arguments to `generate` instead of using a custom {object_type}."
                    )
        default_list.extend(custom_list)
        return default_list
def _get_logits_processor(
        generation_config: GenerationConfig,
        input_ids_seq_length: int,
        encoder_input_ids: torch.LongTensor,
        prefix_allowed_tokens_fn: Callable[[int, torch.Tensor], List[int]],
        logits_processor: Optional[LogitsProcessorList],
    ) -> LogitsProcessorList:
        """
        This class returns a [`LogitsProcessorList`] list object that contains all relevant [`LogitsProcessor`]
        instances used to modify the scores of the language model head.
        """
        # instantiate processors list
        processors = LogitsProcessorList()

        # the following idea is largely copied from this PR: https://github.com/huggingface/transformers/pull/5420/files
        # all samplers can be found in `generation_utils_samplers.py`
        if generation_config.diversity_penalty is not None and generation_config.diversity_penalty > 0.0:
            processors.append(
                HammingDiversityLogitsProcessor(
                    diversity_penalty=generation_config.diversity_penalty,
                    num_beams=generation_config.num_beams,
                    num_beam_groups=generation_config.num_beam_groups,
                )
            )
        if (
            generation_config.encoder_repetition_penalty is not None
            and generation_config.encoder_repetition_penalty != 1.0
        ):
            processors.append(
                EncoderRepetitionPenaltyLogitsProcessor(
                    penalty=generation_config.encoder_repetition_penalty, encoder_input_ids=encoder_input_ids
                )
            )
        if generation_config.repetition_penalty is not None and generation_config.repetition_penalty != 1.0:
            processors.append(RepetitionPenaltyLogitsProcessor(penalty=generation_config.repetition_penalty))
        if generation_config.no_repeat_ngram_size is not None and generation_config.no_repeat_ngram_size > 0:
            processors.append(NoRepeatNGramLogitsProcessor(generation_config.no_repeat_ngram_size))
        if generation_config.bad_words_ids is not None:
            processors.append(
                NoBadWordsLogitsProcessor(generation_config.bad_words_ids, generation_config.eos_token_id)
            )
        if (
            generation_config.min_length is not None
            and generation_config.eos_token_id is not None
            and generation_config.min_length > 0
        ):
            processors.append(MinLengthLogitsProcessor(generation_config.min_length, generation_config.eos_token_id))
        if (
            generation_config.min_new_tokens is not None
            and generation_config.eos_token_id is not None
            and generation_config.min_new_tokens > 0
        ):
            processors.append(
                MinNewTokensLengthLogitsProcessor(
                    input_ids_seq_length, generation_config.min_new_tokens, generation_config.eos_token_id
                )
            )
        if prefix_allowed_tokens_fn is not None:
            processors.append(
                PrefixConstrainedLogitsProcessor(
                    prefix_allowed_tokens_fn, generation_config.num_beams // generation_config.num_beam_groups
                )
            )
        if generation_config.forced_bos_token_id is not None:
            processors.append(ForcedBOSTokenLogitsProcessor(generation_config.forced_bos_token_id))
        if generation_config.forced_eos_token_id is not None:
            processors.append(
                ForcedEOSTokenLogitsProcessor(generation_config.max_length, generation_config.forced_eos_token_id)
            )
        if generation_config.remove_invalid_values is True:
            processors.append(InfNanRemoveLogitsProcessor())
        if generation_config.exponential_decay_length_penalty is not None:
            processors.append(
                ExponentialDecayLengthPenalty(
                    generation_config.exponential_decay_length_penalty,
                    generation_config.eos_token_id,
                    input_ids_seq_length,
                )
            )
        if generation_config.suppress_tokens is not None:
            processors.append(SuppressTokensLogitsProcessor(generation_config.suppress_tokens))
        if generation_config.begin_suppress_tokens is not None:
            begin_index = input_ids_seq_length
            begin_index = (
                begin_index
                if (input_ids_seq_length > 1 or generation_config.forced_bos_token_id is None)
                else begin_index + 1
            )
            if generation_config.forced_decoder_ids is not None:
                # generation starts after the last token that is forced
                begin_index += generation_config.forced_decoder_ids[-1][0]
            processors.append(
                SuppressTokensAtBeginLogitsProcessor(generation_config.begin_suppress_tokens, begin_index)
            )
        if generation_config.forced_decoder_ids is not None:
            processors.append(ForceTokensLogitsProcessor(generation_config.forced_decoder_ids))
        processors = _merge_criteria_processor_list(processors, logits_processor)
        # `LogitNormalization` should always be the last logit processor, when present
        if generation_config.renormalize_logits is True:
            processors.append(LogitNormalization())
        return processors

dataset = load_dataset('bookcorpus', split="train[74004000:]", cache_dir='/home/ubuntu/huggingface')

# tokenizer = BartTokenizerFast.from_pretrained("/home/ubuntu/bartone/noise0.5/latest/bart_tokenizer")
# model = BartForConditionalGenerationOne.from_pretrained("/home/ubuntu/bartone/noise0.5/latest")
tokenizer = BartTokenizerFast.from_pretrained("/home/ubuntu/bartone/reconstruction/no_noise/bart_tokenizer")
model = BartForConditionalGenerationOne.from_pretrained("/home/ubuntu/bartone/reconstruction/no_noise")
# tokenizer = BartTokenizerFast.from_pretrained("/home/ubuntu/bartone/reconstruction/no_noise/bart_tokenizer")
# model = BartForConditionalGenerationOne.from_pretrained("/home/ubuntu/bartone/noiseunk")
# tokenizer = BartTokenizerFast.from_pretrained("/home/ubuntu/bartone/reconstruction/no_noise/bart_tokenizer")
# model = BartForConditionalGenerationOne.from_pretrained("/home/ubuntu/bartone/noise0.1/epoch_4")
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

# train_dataloader = DataLoader(
#         train_dataset, shuffle=True, collate_fn=default_data_collator, batch_size=16
#     )
generation_config = GenerationConfig.from_model_config(model.config)
bos_token = generation_config.bos_token_id
input_ids = torch.ones((1, 1), dtype=torch.long) * bos_token
input_ids_seq_length = input_ids.shape[-1]


eos_token_id = [generation_config.eos_token_id]
pad_token_id = generation_config.pad_token_id
logits_processor = _get_logits_processor(generation_config=generation_config,
                                         input_ids_seq_length=input_ids_seq_length,
                                         encoder_input_ids=test_dataset['input_ids'][0],
                                         prefix_allowed_tokens_fn=None,
                                         logits_processor=LogitsProcessorList())
model.eval()
example = {'input_ids': torch.tensor(test_dataset['input_ids'][0])[None,:], 'attention_mask': torch.tensor(test_dataset['attention_mask'][0])[None,:]}
unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long)
with torch.no_grad():
    encoder_outputs = encoder(**example)
    encoder_outputs['last_hidden_state'] += torch.randn_like(encoder_outputs['last_hidden_state']) * 0.1
    input_ids = torch.ones((1, 1), dtype=torch.long) * bos_token
    eos_token_id_tensor = torch.tensor(eos_token_id).to(input_ids.device) if eos_token_id is not None else None
    while True:
        model_inputs = model.prepare_inputs_for_generation(input_ids, encoder_outputs=encoder_outputs)
        outputs = model(**model_inputs, return_dict=True, output_attentions=generation_config.output_attentions, output_hidden_states=generation_config.output_hidden_states)
        next_token_logits=outputs.logits[:,-1,:]
        next_tokens_scores = logits_processor(input_ids, next_token_logits)
        next_tokens = torch.argmax(next_tokens_scores, dim=-1)
        if eos_token_id is not None:
            if pad_token_id is None:
                raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        if eos_token_id_tensor is not None:
            unfinished_sequences = unfinished_sequences.mul(
                next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
            )
        if eos_token_id_tensor is not None:
            unfinished_sequences = unfinished_sequences.mul(
                next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
            )
        if unfinished_sequences.max() == 0 or input_ids.shape[1] > max_seq_length:
            break
    done_input_ids = input_ids
    output_str = tokenizer.batch_decode(done_input_ids, skip_special_tokens=True)


    print('Input\n', dataset['text'][0])
    print('Output\n', output_str[0])


# for step, batch in enumerate(train_dataloader):
#     with torch.no_grad():
#         encoder_outputs = encoder(**batch)
#         import ipdb; ipdb.set_trace()