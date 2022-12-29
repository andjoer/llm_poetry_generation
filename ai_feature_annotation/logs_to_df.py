import glob
import re
import os
import pandas as pd


def output_file(file,start_gen = 4,factor = 1):
   
    with open(file) as f:
        lines = f.readlines()

    poem_start = False
    poem = ''
    LLM = ''
    for idx, line in enumerate(lines):
        if 'result by:' in line: 
            LLM = line.split()[-1]
            
        if poem_start and 'rating' not in line and '***' not in line: 
            poem += line

        if '*** final output ***' in line:
            poem_start = True


    poem = poem.strip()
    if len(poem.split('\n')) > 3: 
    
        return LLM, poem
    else: 
        return '',''
        
    
   

    

if __name__ == "__main__":  
    
    files = glob.glob("logs/*.log")
    

    indexed_files = []
    for file in files: 
        indexed_files.append([file,int(re.findall(r'\d+', file)[0])])

    LLMs = []
    poems = []
    for file, index in indexed_files:
        LLM, poem = output_file(file)
        if poem:
            LLMs.append(LLM)
            poems.append(poem)

    poem_df = pd.DataFrame({'LLM':LLMs,'poem':poems})

    poem_df.to_csv('data/generated_poems.csv')


