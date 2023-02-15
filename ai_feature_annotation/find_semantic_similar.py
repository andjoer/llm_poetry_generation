import glob,re 
import pandas as pd
from tqdm import tqdm
import argparse

from sentence_transformers import SentenceTransformer
import numpy as np
import torch


def find_similar(sents,vector_df,model, metric = 'cdist'):

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    if metric == 'cos':
        measure = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    else:
        measure = torch.cdist
    sentence_citations = []
    sentence_distances = []
    reff_array = np.vstack(vector_df['vector'].values.tolist())
    reff_tensors = torch.tensor(reff_array).to(device)
    for sent in tqdm(sents): 
        sent = sent.strip()
        sentence_vector = torch.tensor((model.encode(sent))).to(device)
        sentence_vector = torch.reshape(sentence_vector,(1,-1))           # bs, vector dim
        distances = measure(sentence_vector,reff_tensors)      
        best_idx = torch.argmin(distances).item()
        best_dist = torch.min(distances).item()
        sentence_citations.append(vector_df.iloc[best_idx]['text'])
        sentence_distances.append(best_dist)

    return sentence_citations, sentence_distances

if __name__ == "__main__":  
    
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str,default='data',help="relative path to the input files")
    parser.add_argument("--path_out", type=str,default='data',help="relative path to output file")
    parser.add_argument("--fname_input", type=str,default='generated_poems.csv',help="filename of the input file")
    parser.add_argument("--fname_comp", type=str,default='gutenberg_strophe_vectors.pkl',help="filename of the input file to look for similar texts")
    parser.add_argument("--fname_output", type=str,default='generated_verse_compare',help="filename of the output file")
    parser.add_argument("--sent_transformer", type=str,default='T-Systems-onsite/cross-en-de-roberta-sentence-transformer',help="name of the sentence transformer model")
    args = parser.parse_args()

    sentence_model = SentenceTransformer(args.sent_transformer).to('cuda')
    args.fname_output = args.fname_output.split('.')[0]

    file_path_out = args.path_out+'/'+args.fname_output 
    file_path_in = args.path+'/'+args.fname_input
    file_path_comp = args.path+'/'+args.fname_comp

    files = glob.glob(file_path_out + '*')

    max_idx = 1
    if files: 
        for file in files: 
            try: 
                max_idx = max(int(re.findall(r'\d+', file)[0]),max_idx)    # find the number of the last file
            except: 
                pass
    
        file_path_out += '_' + str(max_idx +1)

    poem_df_comp = pd.read_pickle(file_path_comp)
    poem_df = pd.read_csv(file_path_in)

    sents = list(poem_df['poem'])
    sentence_citations, sentence_distances = find_similar(sents,poem_df_comp,sentence_model)

    poem_df['close_text'] = sentence_citations
    poem_df['semantic_dist']  = sentence_distances

    poem_df.to_csv(file_path_out+'.csv',index=False)
   