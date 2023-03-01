import pandas as pd
import os
import re
import numpy as np
import ast

def preprocess_ano(df,col_text = 'text',col_rhyme='rhyme'):
    
    try: 
        string_list = type(ast.literal_eval(df[col_text][1])) == list
    except: 
        string_list = False
    if  string_list:
        df[col_text] = df[col_text].apply(lambda x: ast.literal_eval(x))

    elif type(df[col_text][1]) == str:
        df[col_text] = df[col_text].apply(lambda x: str(x).lower().split('\n'))
        

    df[col_text] = df[col_text].apply(lambda x: [' '.join(re.sub(r'[^a-zäöüß ]', ' ', line).split()).strip() for line in x if not (line.isspace() or not line or 'titel:' in line)])
    df[col_text] = df[col_text].apply(lambda x: x if len(x) > 1 else np.nan)


    poem_df = df # poem_df.rename(columns={col_rhyme: "rhyme"})
    
    if col_rhyme in poem_df.columns:
        poem_df[col_rhyme].replace('', np.nan, inplace=True)
        poem_df.dropna(subset=[col_rhyme], inplace=True)  
    poem_df.dropna(subset=[col_text], inplace=True)     
    poem_df['endings'] = poem_df[col_text].apply(lambda x: [line.split()[-1] for line in x if not (line.isspace() or not line)])

    return(poem_df)
