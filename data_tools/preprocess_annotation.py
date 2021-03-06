import pandas as pd
import os
import re
import numpy as np


def preprocess_ano(df,col_text = 'vers',col_rhyme='Reimschema'):

    ''' Preprocesses an annotated table of poetry strophes with their corresponding rhyme schemes (for exampel abab)

        Args: 
            df (Dataframe): Pandas Dataframe of the annotated Data
            col_text (str): The name of the colum that contains the strophes 
            col_rhyme (str): The name of the column that contains the annotation

    '''
    df[col_text] = df[col_text].apply(lambda x: x.lower().split('\r\n'))
    df[col_text] = df[col_text].apply(lambda x: [re.sub(r'[^a-zäöüß ]', '', line) for line in x if not (line.isspace() or not line)])

    poem_df = df[[col_text,col_rhyme]]
    poem_df = poem_df.rename(columns={col_text: "text", col_rhyme: "rhyme"})
    poem_df['rhyme'].replace('', np.nan, inplace=True)
    poem_df.dropna(subset=['rhyme'], inplace=True)
    poem_df['endings'] = poem_df['text'].apply(lambda x: [line.split()[-1] for line in x])

    return(poem_df)


