import os
import sys
import inspect
import re
import argparse
import glob

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

import pandas as pd 
import tqdm

from rythm_utils import verse_cl

def preprocess_ano_rythm(df,col_text = 'text'):
    if type(df[col_text][0]) == str:
        df[col_text] = df[col_text].apply(lambda x: str(x).lower().split('\n'))
    df[col_text] = df[col_text].apply(lambda x: [re.sub(r'[^a-zäöüß ]', '', line.strip()) for line in x if not (line.isspace() or not line)])

    return(df)

class Annotate_rythm():
    def __init__(self,args):
        self.args = args
        self.rythm_lst = []
        self.rythm_tok_lst = []
        self.pos_lst = []
        self.ipa_lst = []

        input_file = os.path.join(self.args.data_dir, self.args.fname_input)

        try: 
            self.poem_df = pd.read_csv(input_file)
        except: 
            print('file not found or not a csv file')
            raise Exception

        self.poem_df = preprocess_ano_rythm(self.poem_df,col_text = args.text_column)

        files = glob.glob(args.data_dir+'/'+args.fname_output+'*')
        max_idx = 1
        if files: 
            for file in files: 
                try: 
                    max_idx = max(int(re.findall(r'\d+', file)[0]),max_idx)    # find the number of the last file
                except: 
                    pass
        
            self.args.fname_output = args.fname_output +'_' + str(max_idx +1)

    def annotate_rythm(self):
        
        for index, row in tqdm.tqdm(self.poem_df.iterrows(), total=self.poem_df.shape[0]):
            verse_rythm = []
            verse_rythm_tok = []
            verse_pos = []
            verse_ipa = []
        
            for line in row[self.args.text_column]:
                
                verse = verse_cl(line)
                verse_rythm.append(verse.rythm)
                verse_rythm_tok.append(verse.rythm_tokens)
                verse_pos.append(verse.token_pos)
                verse_ipa.append(verse.ipa)
        
            self.rythm_lst.append(verse_rythm)
            self.rythm_tok_lst.append(verse_rythm_tok)
            self.pos_lst.append(verse_pos)
            self.ipa_lst.append(verse_ipa)

            if (index+1)%self.args.save_every == 0: 
                self.save_ckp()

        annotated_df = self.get_dataframe()
        self.save_df(self.args.data_dir,self.args.fname_output, annotated_df)

    def get_dataframe(self):
        text = self.poem_df['text'].tolist()[:len(self.rythm_lst)]
        annotated_df = pd.DataFrame(list(zip(text, self.rythm_lst, self.rythm_tok_lst, self.ipa_lst, self.pos_lst)),
                columns =['text', 'rythm','rythm_tok','ipa','pos'])
        return annotated_df

    def save_ckp(self):
        annotated_dataframe = self.get_dataframe()
        fname = self.args.fname_output +'_ckp_' + str(len(self.rythm_lst))
        self.save_df(args.data_dir+'/checkpoints',fname,annotated_dataframe)

    def save_df(self,path,fname,dataframe):
        dataframe.to_csv(os.path.join(path, fname +'.csv'))
        dataframe.to_pickle(os.path.join(path, fname +'.pkl'))
        print('saved file: '+ os.path.join(path, fname))


if __name__ == "__main__":  
    path = 'data'
    fname = 'rhyme_df.pkl'
    fname_save = 'rythm_df_pred.csv'
    fname_save_pkl = 'rythm_df_pred.pkl'

    parser = argparse.ArgumentParser()
    

    parser.add_argument("--data_dir", type=str,default='data',help="subdirectory for the input and output data")
    parser.add_argument("--fname_output", type=str,default='gutenberg_rythm_ano',help="filename of the output file")
    parser.add_argument("--fname_input", type=str,default='gutenberg.csv',help="filename of the created file")
    parser.add_argument("--text_column", type=str,default='text',help="name of the column that contains the texts")
    parser.add_argument("--save_every", type=int,default=10000,help="number of verses after which a checkpoint will be saved")

    args = parser.parse_args()

    annotation = Annotate_rythm(args)
    annotation.annotate_rythm()
