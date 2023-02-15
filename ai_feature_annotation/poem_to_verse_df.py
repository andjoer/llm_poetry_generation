import glob,re 
import pandas as pd
from tqdm import tqdm
import argparse
if __name__ == "__main__":  
    
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str,default='data',help="relative path to the input files")
    parser.add_argument("--path_out", type=str,default='data',help="relative path to output file")
    parser.add_argument("--fname_input", type=str,default='gutenberg.csv',help="filename of the input file")
    parser.add_argument("--fname_output", type=str,default='gutenberg_verse',help="filename of the output file")
  

    args = parser.parse_args()
    
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


    authors = []
    titles = []
    verses = []
    for index, row in tqdm(poem_df.iterrows(), total=poem_df.shape[0]):
        text_lst = str(row['text']).strip().split('\n')
        text_lst = [text.strip() for text in text_lst]
        verses += text_lst
        authors += [row['author']]*len(text_lst)
        titles += [row['title']]*len(text_lst)

    verse_df = pd.DataFrame(list(zip(authors, titles, verses)),
                columns =['author', 'title','text'])

    verse_df.to_csv(file_path_out+'.csv',index=False)

