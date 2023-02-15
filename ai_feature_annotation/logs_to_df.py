import glob
import re
import os
import pandas as pd
import argparse

def output_file(file,start_gen = 4,factor = 1):
   
    with open(file) as f:
        lines = f.readlines()

    poem_start = False
    poem = ''
    LLM = ''
    rating = 0
    for idx, line in enumerate(lines):
        if 'result by:' in line: 
            LLM = line.split()[-1]
            
        if poem_start and 'rating' not in line and '***' not in line: 
            poem += line
        elif poem_start and 'rating' in line and '***' not in line:

            if 'pending' not in line.split(':')[-1]: 
                rating = float(line.split(':')[-1].strip())
            else: 
                rating = 0

        if '*** final output ***' in line:
            poem_start = True


    poem = poem.strip()
    if len(poem.split('\n')) > 3: 
    
        return LLM, poem, rating
    else: 
        return '','',0
        
    
   

    

if __name__ == "__main__":  
    
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str,default='logs',help="relative path to the log files")
    parser.add_argument("--path_out", type=str,default='data',help="relative path to output file")
    parser.add_argument("--name_out", type=str,default='generated_poems.csv',help="filename of the output file")
    args = parser.parse_args()

    files = glob.glob(args.path+"/*.log")
    indexed_files = []
    for file in files: 
        indexed_files.append([file,int(re.findall(r'\d+', file)[0])])

    LLMs = []
    poems = []
    ratings = []
    for file, index in indexed_files:
        LLM, poem,rating = output_file(file)
        if poem:
            LLMs.append(LLM)
            poems.append(poem)
            ratings.append(rating)

    poem_df = pd.DataFrame({'LLM':LLMs,'rating:':ratings,'poem':poems})

    poem_df.to_csv(args.path_out+'/'+args.name_out)


