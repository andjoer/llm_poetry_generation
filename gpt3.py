
import os

import openai

def gpt3(input_text,_,max_length=25,num_return_sequences=15,stop=['#'],repetition_penalty = 0.5,top_p = 1,temperature = 0.8, block_linebreak = False):

    repetition_penalty = (repetition_penalty-1)*2.5

    openai.api_key = os.environ.get('OPENAI_API_KEY')
    openai.organization = os.environ.get('OPENAI_API_ID')
    responses = openai.Completion.create(
    engine="text-davinci-002",
    prompt=input_text,
    temperature=temperature,
    max_tokens=max_length,
    top_p=top_p,
    frequency_penalty=repetition_penalty,
    presence_penalty=0.0,
    stop=stop,
    n=num_return_sequences
    )
    output = []
    for response in responses['choices']:
        output.append(response['text'])

    return output
