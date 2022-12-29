import os
import pandas as pd
import re

from ortho_to_ipa import ortho_to_ipa

custom_dict = {'seine': 'ˈzaɪ̯nə','be':'bɛ'}
dirname = os.path.dirname(__file__)

otoi_path = os.path.join(dirname, 'data/de_ortho_ipa.csv')
otoi_df = pd.read_csv(otoi_path)
otoi_df['word'] = otoi_df['word'].str.lower()

custom_dict = {'seine': 'ˈzaɪ̯nə','be':'bɛ'}

otoi = ortho_to_ipa(load = True)

def convert_to_ipa(ortho):
    word_ortho = ortho.lower()
    word_ortho = re.sub(r'[^a-zäöüß]', '', word_ortho)

    if word_ortho in custom_dict.keys():
            ipa = custom_dict[word_ortho]
    else:
        try:
            ipa = (otoi_df.loc[otoi_df['word'] == ortho]['ipa']).values[0]
            
        except:
            
            ipa = otoi.translate(ortho)
        
        
    return ipa
