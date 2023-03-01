
import glob
import re
import numpy as np 
from rhyme import find_rhyme, find_rhyme_independent
from rythm import fix_rythm
from rythm_utils import verse_cl

import argparse, ast

from gpt2 import LLM_class
from gpt_poet import gpt_poet
from generator_utils import conv_rhyme_scheme, Logger
from perplexity import perplexity
import random
import torch

from prompts import prompts, jambus, trochee
from poetry_generator_utils import parse_arguments, initialize_llms, get_LLM_name


import sys

prompt_new_2 = ''' Titel: Warum ist etwas und nicht nur nichts

Das ist die Frage, die sich mir stellt noch
In tiefen Nächten, wenn die Sterne fliehen
Was ist der Sinn des Lebens, frag ich mich
In dunklen Stunden, wenn die Wolken ziehen.
Ist es der Tod, der uns am Ende holt
Und alles auslöscht, was wir hier getragen
Ist es das Leben, das uns alles nimmt
Und doch nichts bleibt, als diese leere Hülle?
'''

prompt_new = ''' Titel: Warum ist etwas und nicht nur nichts

Das ist die Frage, die sich mir stellt noch
In tiefen Nächten, wenn die Sterne fliehen
Was ist der Sinn des Lebens, frag ich mich
In dunklen Stunden, wenn die Wolken ziehen.
'''

def generate_poetry(args, 
                    shots = 1, 
                    LLM=None, 
                    LLM_rhyme=None, 
                    LLM_2  = None,
                    LLM_perplexity = None):
    '''
    contains the main loop that generates the peoms
    takes: 
    prompt: the prompt that will be forwarded to the llm
    target_rythm: the target rythm (jambus or trochee)
    num_syll_lst: a list that contains the number of syllables in each verse. Once the list ends it will be read 
                 from the beginning again
    rhyme_scheme: the rhyme scheme
    shots: number of candidate verses to generate for each resulting verse
    LLM: the llm that should be used 
    LLM_rhyme: the llm that should be used to generate alternative verse endings
    use_tts: Use the mffc-feature method in order to check the results from the sia-rhyme method
    num_lines: number of verses to generate
    '''

    failed_rhyme_count_limit = args.failed_rhyme_count_limit

    prompt = args.prompt
    target_rythm = args.target_rythm
    num_syll_lst = args.num_syll_list
    rhyme_scheme = args.rhyme_scheme
    num_lines = args.generated_lines

    if args.sample_rhymes_independent:
        rhyme_finder = find_rhyme_independent
    else:
        rhyme_finder = find_rhyme
    if rhyme_scheme:
        rhyme_scheme_print = rhyme_scheme
        
        rhyme_scheme = conv_rhyme_scheme(rhyme_scheme)
        
        
    else:
        rhyme_scheme_print = 'No Rhyme'

    freq = len(rhyme_scheme)  # number of verses per strophe

    input_text = prompt
    print('input_text')
    print(input_text)
    lines = []

    verse_lst = []
    input_text_0 = input_text
    print_text = ''
    cnt = 0
    title = ''
    title_accepted = False              # feature disabled; the generator won't suggest titles; set true to enable
    offset = 0
    rating = 'pending'
    rhyme_pending = False

    failed_rhyme_count = 0
    shots = args.verse_versions

    iteration = 0
    while len(verse_lst) < num_lines or rhyme_pending:

        num_syll = num_syll_lst[len(verse_lst)%len(num_syll_lst)]

        if not rhyme_pending:
            last_state = None
            tmp_lst = []                                              # the list only gets deleted if no rhyme is pending; otherwise the rejected verses still could be the best option left
            perp_lst = []
            is_title = False
            iteration = 0
            top_k_0 = 0

        for j in range(shots):

            if last_state:
                if len(last_state) > 3: 
                    last_state.trunkate()

                if len(last_state) == 0:
                    iteration += 1
                    last_state = None
            else: 
                iteration += 1

            line, last_state, top_k_0 = gpt_poet(args,input_text,num_syll,title_accepted,LLM=LLM,LLM_2=LLM_2,last_state=last_state,top_k_0 = top_k_0,iteration=iteration)  # generate a new verse

            if line[:2] == '##':                                          # no text has been created, but maybe a new title
                offset += cnt
                cnt = 0
                input_text += line +'\n'
                title = line[2:]
                title_accepted = False
                is_title = True
                break

            elif line =='**':                                   
                if (j > 1 and not tmp_lst) or shots == 1:                         # we could try again but  we basically tried enough  -> stop           
                    return print_text, rating

            else:        
                verse_tmp = verse_cl(line)
                verse_tmp.context = input_text[-400:]
                verse_tmp = fix_rythm(args,verse_tmp,target_rythm,num_syll,LLM_perplexity)          # fix the rythm of the generated verse

                perplexity_check_text = '\n'.join(input_text.split('\n')[-2:]) + ' '.join((re.sub('[^a-zA-ZäöüÄÖÜß ]','',' '.join(verse_tmp.text)).split()))
                
                if args.check_end == True: 
                    perplexity_check_text += '.'

                perp = perplexity(perplexity_check_text,LLM_perplexity)
                perp_lst.append(perp) # measure the perplexity of the verse

                print('generated verse: ')
                print(' '.join(verse_tmp.text))
                print('perplexity: ' + str(perp))
                tmp_lst.append(verse_tmp)

        if not is_title:
            title_accepted = False
            if perp_lst:                                            
                best_idx = np.argmin(np.asarray(perp_lst))          # choose the candidate verse with the least perplexity
                verse = tmp_lst[best_idx]
                del tmp_lst[best_idx]
                del perp_lst[best_idx]

            else: 
                return print_text, rating 

            verse_lst.append(verse)
            
            cnt = len(verse_lst)-1                                 # TEMPORARLY fix when feature ready
            if rhyme_scheme and rhyme_scheme[(cnt)%freq] != -1:  # if rhyme partner already exists
                
                if failed_rhyme_count == 0:
                    verse_lst_0 = verse_lst.copy()
                if failed_rhyme_count > failed_rhyme_count_limit:
                    failed_rhyme_count = 0
                    verse_lst = verse_lst_0.copy()
                else: 
                    force_rhyme = args.force_rhyme    
                    verse_lst = rhyme_finder(args,verse_lst,offset + int(int(cnt/freq)*freq+rhyme_scheme[cnt%freq]),cnt,LLM_perplexity,LLM=LLM_rhyme,LLM2=LLM_2, force_rhyme = force_rhyme)

                if len(verse_lst)-1 < cnt:
                    rhyme_pending = True
                    failed_rhyme_count += 1

                else:
                    rhyme_pending = False
                input_text = input_text_0
                print_text = ''
                for verse in verse_lst:

                    new_text = re.sub(r'\s([,.!?;:](?:\s|$))', r'\1', ' '.join(verse.text))
                    input_text += new_text + '\n'
                    print_text += new_text + '\n'

            else:

                new_text = re.sub(r'\s([,.!?;:](?:\s|$))', r'\1', str(verse.doc).strip())
                input_text = input_text + new_text + '\n'
                print_text += new_text + ' \n'

            cnt += 1

        if print_text:
            rating = 1000/perplexity(print_text,LLM_perplexity)
        else:
            rating = 'pending'

    
        print('result by: ' + str(get_LLM_name(LLM)))
        print('rhyme scheme: ' + str(rhyme_scheme_print))
        print('rating: ' + str(rating))
        print(title)
        print(print_text)

    return print_text, rating


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

    
    start_idx = max_idx + 1
    print(start_idx)

    

    if args.target_rythm == 'jambus':
        args.target_rythm = jambus
    elif args.target_rythm == 'trochee':
        args.target_rythm = trochee
    elif args.target_rythm: 
        print('rythm not supported, using jambus')
        args.target_rythm = jambus

    prompt_0 = args.prompt
    title_0 = args.title
    rhyme_scheme_0 = args.rhyme_scheme
    num_syll_lst_0 = args.num_syll_list
    target_rythm_0 = args.target_rythm

    LLM, LLM_perplexity, LLM_rhyme, LLM_2 = initialize_llms(args)

    for i in range(args.generated_poems):  
        rhyme_schemes = ['aabb','abba','abab','']              # rhyme schemes to sample from

        args.prompt = prompt_0
         # prompt to sample from
  
        prompt = random.choice(prompts)
        num_syll = prompt[1]                       # the metric properties are defined in the prompt
        rythm = prompt[2] 
        prompt_text = prompt[0] # + ' \n Titel: Warum ist etwas und nicht nur nichts \n' # title of the created poem

        if not rhyme_scheme_0: 
            args.rhyme_scheme = random.choice(rhyme_schemes)
        
        if args.add_title_indicator:
            title_indicator = 'titel: '
        else:
            title_indicator = ''
        if (not prompt_0 and not title_0) or prompt_0 == 'ran': 
            prompt = random.choice(prompts)
            
            if title_0:
                title = title_indicator + title_0 + '\n'
            else: 
                title = ''
            args.prompt = prompt[0] + '\n' +title

            if not num_syll_lst_0:
                args.num_syll_list = prompt[1]
            if not target_rythm_0:
                args.target_rythm = prompt[2]
        else: 
            if not prompt_0 and title_0: 
                args.prompt =  title_indicator + args.title + ':\n' #removed word title
            elif title_0:
                args.prompt = title_indicator + prompt_0 +  args.title + ':\n' # removed title
            else: args.prompt += '\n'
            
            if not num_syll_lst_0:
                args.num_syll_list = [9,11]
            if not target_rythm_0:
                args.target_rythm = jambus

        if not rhyme_scheme_0: 

            if len(args.num_syll_list) > 1:
                if (args.num_syll_list[0] - args.num_syll_list[1]) % 2 == 0:
                    args.rhyme_scheme = random.choice(['aabb','abba'])
                else:
                    args.rhyme_scheme = 'abab'
            else:
                args.rhyme_scheme = random.choice(['aabb','abba','abab'])

        gpt3_prompt_2 = 'schreibe ein Gedicht auf Deutsch! Denke philosophisch tiefgründig, nicht trivial! Verpacke neue Gedanken! Verwende eine innovative, lyrische und überraschende Sprache!\n'
        gpt3_prompt_1 = 'schreibe ein Gedicht auf Deutsch! \n'
        if args.LLM == 'GPT3':
            args.prompt = gpt3_prompt_2  + args.prompt

        args.prompt = random.choice([prompt_new, prompt_new_2])
        args.rhyme_scheme = 'abab'
        args.num_syll_list = [10,11]
        print('parameters')

        print('############## begin parameters ##############')
        for arg in vars(args):
            if not str(arg) == 'vocab':
                print(str(arg) + ': ' + str(getattr(args, arg)))
            else: 
                print('vocabulary: using vocabulary')
        print('############## end of parameters ##############')


        print('LLM: ' + str(get_LLM_name(LLM)))
        print('LLM_rhyme: ' + str(get_LLM_name(LLM_rhyme)))
        print('iterations per verse: ' + str(args.verse_versions))

        if LLM_2:
            print('LLM_2: ' + LLM_2.model_name)
        print(args.prompt)
        
        text, rating = generate_poetry(args,LLM=LLM,LLM_rhyme=LLM_rhyme,LLM_2=LLM_2,LLM_perplexity=LLM_perplexity)

        print('*** final output ***')
        print('\n')
        print('rating: ' + str(rating))
        print('\n')
        print(text)
        print('\n')
        print('***')

       
        sys.stdout.save(start_idx+i)
        