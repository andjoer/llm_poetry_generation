
from transformers import pipeline
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import os

from parameters import gtp2_model

os.environ["TOKENIZERS_PARALLELISM"] = "false"



generator = pipeline('text-generation', model=gtp2_model,
                 tokenizer=gtp2_model, framework = 'pt')





def gpt2(input_text,max_length= 10, num_return_sequences=5):
    tokenizer = GPT2Tokenizer.from_pretrained(gtp2_model)
    max_length += tokenizer.encode(input_text,return_tensors='pt').size(1)
    generated = generator(input_text, max_length=max_length,return_full_text = False, num_return_sequences=num_return_sequences)
    
    return [item['generated_text'] for item in generated]



def gpt2_top_p(input_text,max_length = 10,num_return_sequences=5):
    tokenizer = GPT2Tokenizer.from_pretrained(gtp2_model)
    model = GPT2LMHeadModel.from_pretrained(gtp2_model,pad_token_id = tokenizer.eos_token_id)
    input_ids = tokenizer.encode(input_text,return_tensors='pt')
    max_length += input_ids.size(1)
    start = input_ids.size()[1]
    output = model.generate(
        input_ids,
        do_sample = True,
        max_length = 200,
        top_p = 0.92,
        top_k = 0,
        num_return_sequences = num_return_sequences,
        early_stopping = True,
        num_repeat_ngram_size = 2
    )

    return [tokenizer.decode(sample_output[start:],skip_special_tokens=True) for sample_output in output]

def gpt2_beam(input_text,max_length = 10,num_return_sequences=5):
    tokenizer = GPT2Tokenizer.from_pretrained(gtp2_model)
    model = GPT2LMHeadModel.from_pretrained(gtp2_model,pad_token_id = tokenizer.eos_token_id)
    input_ids = tokenizer.encode(input_text,return_tensors='pt')
    max_length += input_ids.size(1)
    start = input_ids.size()[1]
    output = model.generate(
        input_ids,
        do_sample = True,
        max_length = max_length,
        num_beams = 5,
        num_return_sequences = num_return_sequences,
        early_stopping = True,
        num_repeat_ngram_size = 2
    )

    return [tokenizer.decode(sample_output[start:],skip_special_tokens=True) for sample_output in output]


