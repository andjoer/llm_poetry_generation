
import os, time
from tqdm import tqdm

import openai
import pandas as pd
import numpy as np
import pickle 

def gpt3(input_text,temperature = 0.8,max_length=700,num_return_sequences=10):

    openai.api_key = os.environ.get('OPENAI_API_KEY')
    openai.organization = os.environ.get('OPENAI_API_ID')
    while True: 
        try:
            responses = openai.Completion.create(
            engine="text-davinci-003",
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
        except:
            print('failed to connect with api')
            time.sleep(10)

prompts = ['''Bewerte folgendes Gedicht anhand der Kriterien Aussage, Verständlichkeit, Kreativität, emotionaler Wirkung und Sprachqualität. Begründe! Sei sehr anspruchsvoll! Es könnte schlecht oder gut sein. 
Nur weil es viele Metaphern enthält, muss es nicht zwangsläufig kreativ sein. Denn die Metaphern können schlechte, unverständliche, aber auch gute und kreative Metaphern sein! Es gibt auch andere Kriterien für Kreativität als Metaphern. 
Wenn Du die Aussage nicht verstehst, sag es! Sage am Schluss, wie Du das Gedicht insgesamt findest!\n''',
'''Bewerte folgendes Gedicht anhand der Kriterien Verständlichkeit, Kreativität, Sprachqualität und dem, was du beim Lesen fühlst. Begründe! Sei sehr anspruchsvoll!''']

poem_df =pd.read_csv('data/generated_poems.csv')
len_inp = len(poem_df.columns)
def rate_poems(poem_df,
                prompts,
                prompt_ids,
                temperatures,
                save_every = 50,
                temperature = 0.3,
                prompt_id = 0):
    counter = 0
    for temperature in temperatures:
        for prompt_id in prompt_ids:
            column_name = 'temp_' + str(temperature)+'_prompt'+str(prompt_id+1)

            poem_df[column_name] = np.empty((len(poem_df), 0)).tolist()

            for idx, row in tqdm(poem_df.iterrows(),total=poem_df.shape[0]):
                prompt = prompts[prompt_id] +'\"'+ row['poem']+'\"'
                poem_df.at[idx,column_name] = gpt3(prompt,temperature=temperature,max_length = 750)
                counter += 1
                if counter % save_every == 0:
                    fname = 'data/checkpoints/poem_reviews_' + str(counter) +'.pkl'
                    poem_df.to_pickle(fname)
                    fname = 'data/checkpoints/poem_reviews_' + str(counter) +'.csv'
                    poem_df.to_csv(fname)
    
    return poem_df

temperatures = [0.8]
prompt_ids = [0]

poem_df_out = rate_poems(poem_df,
                        prompts,
                        prompt_ids,
                        temperatures)

poem_df_out_csv = poem_df_out.copy()
for coll in list(poem_df_out.columns[4:]):
    poem_df_out_csv[coll] = poem_df_out_csv[coll].apply(lambda x: ' review \n'.join(x))

poem_df_out.to_pickle('data/poem_reviews.pkl')
poem_df_out_csv.to_csv('data/poem_reviews.csv',index = False)
print(poem_df_out_csv.head())