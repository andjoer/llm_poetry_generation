
import glob
import re
import numpy as np 
from rhyme import find_rhyme
from rythm import fix_rythm
from rythm_utils import verse_cl, get_meter_difference

import argparse, ast

from gpt2 import LLM_class
from gpt_poet import gpt_poet,iterate_verse
from generator_utils import conv_rhyme_scheme, Logger
from perplexity import perplexity
import random

from prompts import prompts, jambus, trochee
from modify_poem_utils import parse_arguments, initialize_llms, get_LLM_name


import sys

jambus = [0,1]
trochee = [1,0]

input_poem = '''Will sich Hektor ewig von mir wenden,
Wo Achill mit den unnahbarn Händen
Dem Patroklus schrecklich Opfer bringt?
Wer wird künftig deinen Kleinen lehren
Speere werfen und die Götter ehren,
Wenn der finstre Orkus dich verschlingt?'''

def modify_poem(args,LLM_perplexity = None,LLM_rhyme = None):

    args.poem = input_poem
    rhyme_scheme = args.rhyme_scheme

    if rhyme_scheme:

        rhyme_scheme = conv_rhyme_scheme(rhyme_scheme)

    verse_lst_text = args.poem.split('\n')

    if args.fix_meter:
        verse_lst_text, args = fix_meter(args,verse_lst_text)

    if not rhyme_scheme:
        return '\n'.join(verse_lst_text)


    freq = len(rhyme_scheme)
    verse_lst = [verse_cl(verse) for verse in verse_lst_text]

    for cnt, _ in enumerate(verse_lst):
        verse_lst[cnt].context = '\n'.join(verse_lst_text[:cnt])
        if rhyme_scheme and rhyme_scheme[(cnt)%freq] != -1:  # if rhyme partner already exists 
            verse_lst = find_rhyme(args,verse_lst,int(int(cnt/freq)*freq+rhyme_scheme[cnt%freq]),cnt,LLM_perplexity,LLM=LLM_rhyme, force_rhyme = False)

        for verse in verse_lst:
            print(' '.join(verse.text))


    print_text = ''
    for verse in verse_lst:
        new_text = re.sub(r'\s([,.!?;:](?:\s|$))', r'\1', ' '.join(verse.text))
        print_text += new_text + '\n'

    return print_text

def fix_meter(args, verse_lst_text):

    rythm_lst = [jambus,trochee]

    
    max_syll_len = 0
    meter_probs = [0]*len(rythm_lst)
    for verse_text in verse_lst_text:
        verse = verse_cl(verse_text)
        if len(verse.rythm) > max_syll_len:
            max_syll_len = len(verse.rythm)

        prob_lst = []
        for rythm in rythm_lst:
            prob_lst.append(get_meter_difference(verse,rythm))

        prob_lst = np.asarray(prob_lst)
        best_meter = np.argmin(prob_lst)
        meter_probs[best_meter] +=1 

    meter_probs = np.asarray(meter_probs)
    best_meter = np.argmax(meter_probs)

    if not args.target_rythm:
        args.target_rythm = rythm_lst[best_meter]

    num_syll = max_syll_len

    for index, verse in enumerate(verse_lst_text):
        poem = '\n'.join(verse_lst_text)
        print('poem:')
        print(poem)
        print('###################')
        new_verse_text = iterate_verse(args,poem,index,num_syll)
        print('new text')
        print(new_verse_text)
        new_verse = verse_cl(new_verse_text)
        new_verse.context = '\n'.join(verse_lst_text[:index])
        new_verse.context_after = '\n' + '\n'.join(verse_lst_text[index+1:])
        new_verse = fix_rythm(args,new_verse,args.target_rythm,num_syll,LLM_perplexity)          # fix the rythm of the generated verse
        new_text = re.sub(r'\s([,.!?;:](?:\s|$))', r'\1', ' '.join(new_verse.text))
        verse_lst_text[index] = new_text
            
    return verse_lst_text, args

if __name__ == "__main__":  
    '''
    start the program
    
    '''
    args = parse_arguments()


    sys.stdout = Logger()
    files = glob.glob("logs/*.log")
    max_idx = 0
    for file in files: 
        max_idx = max(int(re.findall(r'\d+', file)[0]),max_idx)    # find the number of the last log file

    
    LLM_rhyme, LLM_perplexity = initialize_llms(args)

    start_idx = max_idx + 1
    print(start_idx)   

    print('parameters')

    print('############## begin parameters ##############')
    for arg in vars(args):
        print(str(arg) + ': ' + str(getattr(args, arg)))
    print('############## end of parameters ##############')

    print(args.poem)
    
    text = modify_poem(args,LLM_perplexity=LLM_perplexity,LLM_rhyme=LLM_rhyme)

    print('*** final output ***')
    print('\n')
    print('\n')
    print(text)
    print('\n')
    print('***')

       
    sys.stdout.save(start_idx)
     