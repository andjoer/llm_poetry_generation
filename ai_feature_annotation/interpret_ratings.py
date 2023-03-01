from germansentiment import SentimentModel
import pandas as pd
import numpy as np
import openai
import os
import re

from tqdm import tqdm
tqdm.pandas()
import time
model = SentimentModel()

sent_dict = {'neutral':0,'negative':-1,'positive':1}

def gpt3(input_text,temperature = 0,max_length=700,num_return_sequences=1):
    
    openai.api_key = os.environ.get('OPENAI_API_KEY_aj')
    #openai.organization = os.environ.get('OPENAI_API_ID_aj')
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
            time.sleep(30)

def interprete_creative(reviews):
    ratings = []
   
    for review in reviews: 
        try:
            prompt = '\"'+review+'''\"
            Beurteile wie diese Bewertung das Gedicht in den Aspekten Sprache, Emotion, Verständlichkeit und Kreativität sowie gesamt bewertet. Antworte mit Zahlen in einem dictionary: 1 für gut, 0 für neutral, -1 für schlecht. Beispiel: {'Verständlichkeit': 0,'Kreativität': 0, 'Sprache': 0, 'Emotion':0, 'gesamt':0}!
            '''
            ratings.append(gpt3(prompt)[0])
        except:
            ratings.append['']

    return ratings

def interprete_review(reviews):       
    results = model.predict_sentiment(reviews)
    results = [sent_dict[result] for result in results]
    return results

def isolate_aspect(reviews,aspect):
    reviews_aspect = []
    for review in reviews:
        review_lst = review.split('\n\n')
        if len(review_lst) < 5:
            review_lst = review.split('\n')
        review_lst = [item for item in review_lst if item.split()]
        

        for asp_review in review_lst:

            if aspect.lower() in asp_review.split()[0].lower():

                reviews_aspect.append(asp_review)
           

    return reviews_aspect

def interprete_aspect(reviews,aspect):
    aspect_isolated = isolate_aspect(reviews,aspect)

    if aspect_isolated:
        return interprete_review(aspect_isolated)
    else:
        return [np.nan]

review_df = pd.read_pickle('data/reviews/poem_reviews.pkl')[99:]
print(len(review_df))
len_inp = len(review_df.columns)


save_every = 10
cnt = 120
reviews_creative = []
fname = 'poem_reviews_interpreted_v4_krea'
for coll in list(review_df.columns[4:]):
    numeric_coll = coll + '_num'
    review_df[numeric_coll] = review_df[coll].apply(lambda x: interprete_review(x))
    rating_coll = coll + '_rating'
    review_df[rating_coll] = review_df[numeric_coll].apply(lambda x: sum(x))
  
for idx, review in tqdm(review_df.iterrows()):

    reviews_creative.append(interprete_creative(review[coll]))
    cnt += 1
    if cnt % save_every == 0:
        print('saved checkpoint')
        ckp = review_df[:len(reviews_creative)]
        ckp['dict'] = reviews_creative
        #ckp['rating_creative'] = ckp['num_creative'].apply(lambda x: sum(x))
        ckp_name = fname + str(cnt)
        ckp.to_csv('data/reviews/checkpoints/'+ckp_name+'.csv')
        ckp.to_pickle('data/reviews/checkpoints/'+ckp_name+'.pkl')


review_df['dict'] = reviews_creative

#review_df['rating_creative'] = review_df['num_creative'].apply(lambda x: sum(x))

review_df_csv = review_df.copy()
#for coll in list(review_df.columns[4:len_inp]):
    #review_df_csv[coll] = review_df_csv[coll].apply(lambda x: ' end review \n'.join(x))
review_df_csv.to_csv('data/reviews/poem_reviews_interpreted_v4_krea.csv')
review_df.to_pickle('data/reviews/poem_reviews_interpreted_v4_krea.pkl')