
import os
import numpy as np
import re

from rythm import check_rythm

from transformers import pipeline


from rythm_utils import extend_target_rythm, verse_cl
from gpt3 import gpt3

from gpt2 import gpt2, gpt2_beam, gpt2_top_p
jambus = [0,1]





def clean_word(word):
    return re.sub('[^a-zäöüß]', '', word.lower())
    
def gpt_poet(input_text, target_rythm,num_syll,title_accepted,tollerance = 4,LLM = 'GPT2-large'):

    len_past = 450
    if LLM == 'GPT2-large':
        LLM_poet = gpt2

    elif LLM == 'GPT2-large_beam':
        LLM_poet = gpt2_beam

    elif LLM == 'GPT2-large_top_p':
        LLM_poet = gpt2_top_p

    elif LLM == 'GPT3':
        LLM_poet = gpt3
        len_past = 1000

    else:
        print('invalid LLM selection, will use gpt2')
        LLM_poet = gpt2

    new_text = ' '
    last = (num_syll-1)%len(target_rythm)

    cnt = 0

    print('start generating')
    rythm_comp = [0]
    while True:

        if cnt % 4 == 0:
            input_text_new = input_text[-len_past:]
            input_text_title = input_text[-len_past:]
            
        if cnt > 9:
            return '**'

        cnt += 1
        print(cnt)
        generated = LLM_poet(input_text_new, max_length= 20,num_return_sequences=20)
        candidates_ends = []
        candidates = []

        if num_syll - len(rythm_comp) > 2:
            min_len = 2

        for text in generated:
            
            

            lines = text.split('\n')
        
            idx_0 = -1
            line = ''
            
            for line_tmp in lines:
                if 'titel' in line_tmp.lower():
                        if title_accepted:
                            return '##'+line_tmp

                        break

                else:
                    if len(line_tmp.split()) > min_len:      
                            line = line_tmp.strip()
                            break

            if line:
                
                verse = verse_cl(new_text + line)
                rythm = np.asarray(verse.rythm)
                target_rythm_ext = np.asarray(extend_target_rythm(rythm,target_rythm))
            
                comp = np.abs((target_rythm_ext-rythm) * (rythm != 0.5))

                enter_idx = len(verse.text)

                if np.sum(comp) != 0:
                    problem_idx = np.amin(np.where(comp != 0)[0])
                    token_idx = verse.token_dict[problem_idx]

                else:
                    token_idx = 20

                if token_idx <= enter_idx and verse.token_ends[-1] < num_syll - 3:

                    candidates.append(' '.join(verse.text[:token_idx]))
                    candidates_ends.append(token_idx)
                    rythm_comp = rythm
                    
                if len(rythm) > 0:                                                                               # rythm has no [-1]
                    if np.sum(comp) == 0 and rythm[-1] == last and len(verse.rythm) <= num_syll and len(verse.rythm) >= num_syll*0.65:

                        return re.sub('[.].','',' '.join(verse.text[:token_idx])) + '\n'
               

        if candidates:
            best_idx = np.argmax(np.asarray(candidates_ends))

            new_text +=  candidates[best_idx] + ' '
            input_text_new = input_text_title + ' '+ new_text 
            
               
def gpt_synonyms(verse,target_rythm,num_remove=2,LLM='GPT2-large'):
    lines = [' '.join(verse.text)]
    '''if verse.text[-1].isalpha():
        input_text = verse.text[:-num_remove]

    else:
        input_text = verse.text[:-num_remove]'''

    for i in range(1,len(verse.text)):
        if len(clean_word(verse.text[-i])) > 1:
            input_text = verse.text[:-(i+num_remove-1)]
            last_idx = -i
            break

    input_text = ' '.join(input_text)
    input_text_cont = verse.context + '\n' + input_text
    if LLM == 'GPT2-large':
        
        generated = gpt2(input_text_cont, max_length=10,num_return_sequences=150)

    elif LLM == 'GPT3':
        generated = gpt3(input_text_cont, max_length=10,num_return_sequences=128)
    else:
        print('invalid LLM selection, will use gpt2')
        generated = gpt2(input_text_cont, max_length=10,num_return_sequences=150)

    if len(target_rythm) > 0:    

        target_rythm_ext = np.asarray(extend_target_rythm(verse.rythm,target_rythm))

  
    for text in generated:


        line = text.split('\n')[0]
        
        if len(line) > 1 and len(line) < (len(' '.join(verse.text[-num_remove:])) + 5):
            verse_tmp = verse_cl(input_text  +' '+line)
            
            if len(target_rythm) > 0:  
                rythm =  np.asarray(verse_tmp.rythm)
                
                if len(rythm) == len(target_rythm_ext):
                    comp = np.abs((target_rythm_ext-rythm) * (rythm != 0.5))

                    if np.sum(comp) == 0:
                        lines.append(input_text  + ' ' + line)
            else: 
                lines.append(input_text  + ' ' + line)

    
    return lines

