
import os

import openai
import pandas as pd
import numpy as np
import pickle 

def gpt3(input_text,temperature = 0.8,max_length=25,num_return_sequences=10):
    openai.api_key = os.environ.get('OPENAI_API_KEY')
    openai.organization = os.environ.get('OPENAI_API_ID')
    responses = openai.Completion.create(
    engine="text-davinci-002",
    prompt=input_text,
    temperature=temperature,
    max_tokens=max_length,
    top_p=1.0,
    frequency_penalty=0.0,
    presence_penalty=0.0,
    stop=["#"],
    n=num_return_sequences
    )
    output = []
    for response in responses['choices']:
        output.append(response['text'])

    return output

prompts = ['''Bewerte folgendes Gedicht anhand der Kriterien Verständlichkeit, Kreativität, Sprachqualität und dem, was du beim Lesen fühlst. Begründe! Sei sehr anspruchsvoll! Es könnte schlecht oder gut sein.\n''',
'''Bewerte folgendes Gedicht anhand der Kriterien Verständlichkeit, Kreativität, Sprachqualität und dem, was du beim Lesen fühlst. Begründe! Sei sehr anspruchsvoll!''']

poem_df =pd.read_csv('data/generated_poems.csv')



save_every = 10
temperature = 0.3
prompt_id = 0

counter = 0
for temperature in [0.3,0.8]:
    for prompt_id in [0,1]:
        column_name = 'temp_' + str(temperature)+'_prompt'+str(prompt_id+1)

        poem_df[column_name] = np.empty((len(poem_df), 0)).tolist()

        for idx, row in poem_df.iterrows():
            prompt = prompts[prompt_id] +'\"'+ row['poem']+'\"'
            poem_df.at[idx,column_name] = gpt3(prompt,temperature=temperature,max_length = 160)
            counter += 1
            if counter % save_every == 0:
                fname = 'data/checkpoints/poem_reviews_' + str(counter) +'.pkl'
                poem_df.to_pickle(fname)
    


poem_df.to_pickle('data/poem_reviews.pkl')
poem_df.to_csv('data/poem_reviews.csv')
print(poem_df.head())