import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

import pandas as pd 
import tqdm

from rythm_utils import verse_cl
from preprocess_annotation import preprocess_ano

def annotate_rythm(poem_df):
    rythm_lst = []
    rythm_tok_lst = []
    pos_lst = []
    ipa_lst = []
    for index, row in tqdm.tqdm(poem_df.iterrows(), total=poem_df.shape[0]):
        verse_rythm = []
        verse_rythm_tok = []
        verse_pos = []
        verse_ipa = []
       
        for line in row['text']:
            
            verse = verse_cl(line)
            verse_rythm.append(verse.rythm)
            verse_rythm_tok.append(verse.rythm_tokens)
            verse_pos.append(verse.token_pos)
            verse_ipa.append(verse.ipa)
       
        rythm_lst.append(verse_rythm)
        rythm_tok_lst.append(verse_rythm_tok)
        pos_lst.append(verse_pos)
        ipa_lst.append(verse_ipa)
    text = poem_df['text'].tolist()
    annotated_df = pd.DataFrame(list(zip(text, rythm_lst, rythm_tok_lst, ipa_lst, pos_lst)),
               columns =['text', 'rythm','rythm_tok','ipa','pos'])
    return annotated_df


if __name__ == "__main__":  
    path = 'data'
    fname = 'rhyme_df.pkl'
    fname_save = 'rythm_df_pred.csv'
    fname_save_pkl = 'rythm_df_pred.pkl'

    poem_df_in = pd.read_pickle(os.path.join(path, fname))[:10]


    poem_df = preprocess_ano(poem_df_in )

    
    print(poem_df.head())

    annotated_df = annotate_rythm(poem_df)
    annotated_df.to_csv(os.path.join(path, fname_save))
    annotated_df.to_pickle(os.path.join(path, fname_save_pkl))

    print(annotated_df.head())
