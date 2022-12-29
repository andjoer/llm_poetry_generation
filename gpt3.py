
import os

import openai

def gpt3(input_text,LLM,max_length=25,num_return_sequences=15,stop=['#']):
    openai.api_key = os.environ.get('OPENAI_API_KEY')
    openai.organization = os.environ.get('OPENAI_API_ID')
    responses = openai.Completion.create(
    engine="text-davinci-002",
    prompt=input_text,
    temperature=0.9,
    max_tokens=max_length,
    top_p=1.0,
    frequency_penalty=0.5,
    presence_penalty=0.0,
    stop=stop,
    n=num_return_sequences
    )
    output = []
    for response in responses['choices']:
        output.append(response['text'])

    return output
