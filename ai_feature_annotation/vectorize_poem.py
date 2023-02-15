import glob,re 
import pandas as pd
from tqdm import tqdm
import argparse

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import torch
tqdm.pandas()

if __name__ == "__main__":  
    
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str,default='data',help="relative path to the input files")
    parser.add_argument("--path_out", type=str,default='data',help="relative path to output file")
    parser.add_argument("--fname_input", type=str,default='gutenberg.csv',help="filename of the input file")
    parser.add_argument("--fname_output", type=str,default='gutenberg_strophe_vectors',help="filename of the output file")
    parser.add_argument("--sent_transformer", type=str,default='T-Systems-onsite/cross-en-de-roberta-sentence-transformer',help="name of the sentence transformer model")
    args = parser.parse_args()

    sentence_model = SentenceTransformer(args.sent_transformer).to('cuda')
    args.fname_output = args.fname_output.split('.')[0]

    file_path_out = args.path_out+'/'+args.fname_output 
    file_path_in = args.path+'/'+args.fname_input

    files = glob.glob(file_path_out + '*')

    max_idx = 1
    if files: 
        for file in files: 
            try: 
                max_idx = max(int(re.findall(r'\d+', file)[0]),max_idx)    # find the number of the last file
            except: 
                pass
    
        file_path_out += '_' + str(max_idx +1)

    if args.fname_input.split('.')[1] == 'pkl':
        poem_df = pd.read_pickle(file_path_in)
    else:
        poem_df = pd.read_csv(file_path_in)

    poem_df['vector'] = poem_df['text'].progress_apply(lambda x: sentence_model.encode(str(x)))


    poem_df.to_pickle(file_path_out+'.pkl')