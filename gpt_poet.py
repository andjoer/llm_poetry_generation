
import os
import numpy as np
import re
from copy import copy

#from rythm import check_rythm

from rythm_utils import extend_target_rythm, verse_cl, rate_candidate_meter
from gpt3 import gpt3

from gpt2 import gpt2, gpt_sample_systematic, get_input_text, clean_word, LLM_class 

import random
jambus = [0,1]



from transformers import pipeline
from transformers import GPT2Tokenizer, GPT2LMHeadModel



def modify_verse(args,verse,LLM_2,num_remove = 1):
    '''
    replace the last token with one of the same rythm and word type
    '''
    context = verse.context

    print('modify verse')
    print(' '.join(verse.text))

    target_rythm = args.target_rythm
    top_p_dict = args.top_p_dict_replace
    top_p = args.LLM_2_top_p
    temperature = args.LLM_2_temperature

    #token_pos = verse.token_pos[-1]
    if LLM_2.sampling != 'systematic':
        num_return_sequences = 120
    else: 
        num_return_sequences = 1
    new_sent = gpt_synonyms(verse,target_rythm,num_remove=num_remove, use_pos = True,eol = False,LLM = LLM_2,num_return_sequences=num_return_sequences,top_p_dict =top_p_dict,top_p=top_p,temperature=temperature,
            replace_linebreaks=args.replace_linebreaks)

    new_words = new_sent[-1].split()[-num_remove:]
   
    verse.text = verse.text[:-num_remove]
    verse.text += new_words

    verse = verse_cl(verse.text)
    verse.context = context
    
    print('modified verse:')
    print(' '.join(verse.text))   
    return verse

def compare_meter(verse,target_rythm,num_syll):
    rythm = verse.rythm

    rythm = np.asarray(rythm)
    target_rythm_ext = np.asarray(extend_target_rythm(rythm,target_rythm))

    comp = np.abs((target_rythm_ext-rythm) * (rythm != 0.5))

    return np.sum(comp) == 0 and len(rythm) <= num_syll

def iterate_verse(args,input_text,verse_idx,num_syll):

    target_rythm = args.target_rythm
    verse_text =  input_text.split('\n')[verse_idx]
    verse = verse_cl(verse_text)
    error_rythm, error_split, splits, chosen = rate_candidate_meter(verse, target_rythm)
    candidates = [verse_text]
    candidate_errors = [error_split+error_rythm]
    if compare_meter(verse,target_rythm,num_syll):
        return verse_text

    for i in range(7):
        
        prompt = '\"' + input_text + '\"\n'
        if i < 5:
            prompt_change = f'''ersetze in obigem Gedicht den Vers \"{verse_text}\" durch eine Alternative mit ähnlicher Bedeutung, aber mit zum Teil veränderten Worten! Verändere nur diesen einen Vers! Stelle sicher, dass er zum nachfolgenden Vers passt!'''
        else: 
            prompt_change = f'''ersetze in obigem Gedicht den Vers \"{verse_text}\" durch eine Alternative mit ähnlicher Bedeutung, verändere die Stellung der Worte im Vers bei gleichbleibender Bedeutung! Ändere dann manche Worte bei ähnlicher Bedeutung! Verändere nur diesen einen Vers! Stelle sicher, dass er zum nachfolgenden Vers passt!'''
        prompt += prompt_change
        alternatives = gpt3(prompt,0)

        for alternative in alternatives:
            
            lines = alternative.strip().split('\n')
            if len(lines) == 1:
                sentence = lines[0]
                if sentence: 
                    verse_new = verse_cl(sentence.strip())
                    try:
                        error_rythm, error_split, splits, chosen = rate_candidate_meter(verse_new, target_rythm)
                    except: 
                        error_split = 100 
                    candidates.append(sentence.strip())
                    candidate_errors.append(error_split+error_rythm)
                    print('alternative verse:')
                    print(sentence.strip())
                    print('rythm of alternative verse:')
                    print(verse_new.rythm)
                    if compare_meter(verse_new,target_rythm,num_syll):
                        return sentence.strip()
    
    candidate_errors = np.asarray(candidate_errors)
    best_values = np.amin(candidate_errors)

    best_candidate = np.where(candidate_errors == best_values)[0][0]

    return (candidates[best_candidate])

def gpt_poet(args,input_text, num_syll,title_accepted, LLM = None,LLM_2=None,last_state = None,top_k_0 = 0, iteration = 0):

    '''
    generate a new verse with a matching metrum
    '''

    target_rythm = args.target_rythm
    num_syll_tollerance = args.syllable_count_toll
    stop_tokens = args.verse_stop_tokens
    LLM_2_pos = args.LLM_2_pos
    top_p = args.LLM_top_p
    trunkate_after = args.trunkate_after
    temperature = args.LLM_temperature
    random_first = args.LLM_random_first
    random_all = args.LLM_random_all
    only_alpha_after = args.verse_alpha_only_after
    dividable_rest = args.dividable_rest
    invalid_verse_ends = args.invalid_verse_ends
    repetition_penalty = args.repetition_penalty

    top_k = top_k_0 + 20
    last_state_out = None

    len_past = 450
    current_num_syll = 0
    if type(LLM) != str:
        LLM_poet = gpt2
        sampling = LLM.sampling

    elif LLM == 'GPT3':
        LLM_poet = gpt3
        len_past = 1000
        sampling = 'multinomial'

    else:
        raise Exception('invalid LLM selection')

    new_text = ' '
    last = (num_syll-1)%len(target_rythm)

    cnt = 0

    print('start generating')
    print('sampling ' + sampling)
    last_replaced = 0 
    last_stress = None
    top_p_current = top_p
    replaced_a_word = False
    while True:

        if cnt % 4 == 0:                                # if there are more then 4 tries, start again
            if input_text[-1] != '\n':
                input_text += '\n'
            input_text_new = input_text[-len_past:]
            input_text_title = input_text[-len_past:] 
            last_replaced = 0
            block_linebreak = False
            
        if sampling != 'systematic' and cnt > 9:                                     # if there are more then 10 tries, stop the complete poem
            return '**', None, top_k_0

        cnt += 1
        
        if sampling == 'systematic':

            
            pending_syllables = num_syll - current_num_syll
            rythm_shift = current_num_syll%len(target_rythm)
            target_rythm_shifted = list(np.roll(np.asarray(target_rythm),rythm_shift))
            print('syllable count tollerance: ' + str(num_syll_tollerance))
            print('top p value: ' + str(top_p_current))
            print('top k value: ' + str(top_k))
            if current_num_syll > 0:
                random_first = False

            generated, last_state_out = gpt_sample_systematic(args,input_text_new,LLM,num_return_sequences = 1,top_p = top_p_current,top_k = top_k,top_k_0 = top_k_0, temperature = temperature,random_first = random_first, random_all = random_all,stop_tokens_alpha = stop_tokens,block_non_alpha = False,
                                                num_syll=pending_syllables,target_rythm=target_rythm_shifted, last_stress=last_stress,num_syll_tollerance=num_syll_tollerance,trunkate_after = trunkate_after,dividable_rest=dividable_rest,
                                                only_alpha_after = only_alpha_after,invalid_verse_ends=invalid_verse_ends,repetition_penalty=repetition_penalty,return_last_state=True, last_state=last_state,replace_linebreaks=args.replace_linebreaks)

            if generated:
                top_p_current = top_p
                
            else: 
                top_p_current += 0.02
                top_k_0 = top_k
                last_state_out = None
                top_k += 20
                num_syll_tollerance -= 0.1
                if top_p_current > 1:
                    return '**', None, top_k_0

        else:

            generated = LLM_poet(input_text_new, LLM, max_length= 20,num_return_sequences=20,repetition_penalty = repetition_penalty,top_p = top_p,temperature = temperature,block_linebreak=block_linebreak,replace_linebreaks=args.replace_linebreaks) # generate from the prompt


        candidates_ends = []
        candidates = []

        for text in generated:
            line = None
            lines = text.strip().split('\n')         # only one verse, so cut the rest from the generation

            idx_0 = -1
            
            for line_tmp in lines:
                if 'titel' in line_tmp[:10].lower():          # if the line begins with "titel"
                        if title_accepted:
                            return '##'+line_tmp, None, top_k_0

                        break

                else:

                    line = line_tmp.strip()

                    break

            if line:                   # if a valid verse was created

                
                verse = verse_cl(new_text + line)

                if len(verse.rythm) > num_syll:
                    verse = verse_cl(new_text + (re.split('! | ? | . ', line)[0]))
                    
                rythm = np.asarray(verse.rythm)
                target_rythm_ext = np.asarray(extend_target_rythm(rythm,target_rythm))
            
                comp = np.abs((target_rythm_ext-rythm) * (rythm != 0.5))

                enter_idx = len(verse.text)

                if np.sum(comp) != 0:
                    problem_idx = np.amin(np.where(comp != 0)[0])            # where to cut the verse since the metrum gets incorrect
                    token_idx = verse.token_dict[problem_idx]

                else:
                    token_idx = 20

                need_replacement = False
                num_remove = 1

                if LLM_2:
                    count = 0
                    for last_idx, token_text in enumerate(verse.text[last_replaced:][::-1]):   # index of second alphanumerical word
                        if token_text.isalpha():
                            count += 1
                        if count == 2: 
                            break

                    for idx, token_pos in enumerate(verse.token_pos[last_replaced:]):
                        if  token_pos in LLM_2_pos:                        
                            if idx == len(verse.token_pos[last_replaced:]) - 1 - last_idx:                            # dont delete the last word when close to the end (otherwise it is difficult for LLM1 to find a matching last word)
                                num_remove = last_idx 
                            else:
                                verse.shorten(min(token_idx,last_replaced+idx+1))        # keep the token that should be replaced and cut afterwards
                                rythm = np.asarray(verse.rythm) 
                            need_replacement = True
                            break

                else: 

                    verse.shorten(token_idx)    


                #if token_idx <= enter_idx: # and token_idx > min(num_syll,3) and token_idx > last_token_idx: # and verse.token_ends[-1] < num_syll - 3:
                if verse.text and len(verse.rythm) <= num_syll:
                    candidates.append(verse)                 
                    candidates_ends.append(token_idx)

                rythm_comp = rythm         

                if len(rythm) > 0:         # rythm has [-1]

                    if np.sum(comp) == 0 and len(rythm) <= num_syll and (len(rythm) >= num_syll*num_syll_tollerance or sampling == 'systematic') and ((num_syll - len(rythm))%len(target_rythm) == 0 or not dividable_rest) and (not invalid_verse_ends or verse.token_pos not in invalid_verse_ends): # if the resulting verse is long enough: finsihed (rythm[-1] == last or rythm[-1] == 0.5)

                        if need_replacement:
                            last_state_out = None
                            verse = modify_verse(args, verse,LLM_2,num_remove=num_remove)

                        if last_state_out is not None:
                            if len(last_state_out) == 0 and not replaced_a_word: 
                                top_k_0 = top_k
                                top_k += 20
    
                        return re.sub('[.].','',' '.join(verse.text)) + '\n', last_state_out, top_k_0
               

        if candidates and (LLM != 'GPT3' or LLM_2):
            if LLM_2:
                last_state_out = None
                best_idx = random.choice(range(len(candidates_ends)))
                if candidates[best_idx].token_pos[-1] in LLM_2_pos:
                    candidates[best_idx] = modify_verse(args, candidates[best_idx],LLM_2)
                    last_replaced = len(candidates[best_idx].text)
            else:
                best_idx = np.argmax(np.asarray(candidates_ends))           # choose the candidate that is the longest

            new_text =  ' '.join(candidates[best_idx].text) + ' '
            current_num_syll = len(candidates[best_idx].rythm)
            #last_token_idx = len(candidates[best_idx].rythm_tokens)-1
            input_text_new = input_text_title + new_text 
            print('generated part: ' + new_text)
            block_linebreak = True
        
        else:                                                           # gpt3 has a problem with continuing started lines; it would be necessary to block '\n'-token at the beginning 
            block_linebreak = False
            new_text = ''
            num_syll_tollerance -= 0.05
            if num_syll_tollerance <= 0.2:
                return '**', None, top_k_0



def gpt_synonyms(args,verse,target_rythm,num_remove=2, max_length = 10, LLM=None,eol = False,use_pos = False, elastic = False,num_return_sequences=150,top_p_dict = {},top_p = 0.5,temperature = 0.9,top_k=20,
                stop_tokens=['\n','.','!','?',','],allow_pos_match=False,invalid_verse_ends=[],repetition_penalty= 1,replace_linebreaks=False):

    if type(LLM) != str:
        if LLM.sampling == 'systematic':
            sampling = 'systematic'
        else:
            sampling = 'multinomial'
    else: 
        sampling = 'multinomial'

    if sampling == 'systematic':
        verse_text, _ = get_input_text(verse,num_remove)
        if not eol:
            stop_tokens = None
        outputs = gpt_sample_systematic(args,verse,LLM,num_return_sequences=num_return_sequences, num_words_remove = num_remove,pos=use_pos,check_rythm = True, target_rythm = target_rythm,top_p_dict=top_p_dict,stop_tokens_alpha=stop_tokens,
                                            temperature=temperature,top_p=top_p,top_k = top_k,allow_pos_match=allow_pos_match,invalid_verse_ends=invalid_verse_ends,repetition_penalty=repetition_penalty, replace_linebreaks=replace_linebreaks)
        lines = [verse_text + output for output in outputs]
        if not lines: 
            lines = [' '.join(verse.text)]
    else: 
        lines = gpt_sample_synonyms(verse,target_rythm,num_remove=num_remove, max_length = max_length, LLM=LLM, eol = eol, use_pos = use_pos, elastic = False,num_return_sequences=num_return_sequences,allow_pos_match=allow_pos_match,
                                    invalid_verse_ends=invalid_verse_ends,repetition_penalty=repetition_penalty,top_p=top_p,temperature=temperature,replace_linebreaks=replace_linebreaks)

    return lines



def gpt_sample_synonyms(verse,target_rythm,num_remove=2, max_length = 10, LLM=None, eol = True, use_pos = True, elastic = False,num_return_sequences=150,allow_pos_match = False,invalid_verse_ends=[],
                        repetition_penalty=1,top_p = 1,temperature = 0.9,replace_linebreaks=False):

    '''
    create alternative Verse endings
    '''

    if use_pos or allow_pos_match:
        verse_clean = verse_cl(re.sub('[^a-zA-ZäöüÄÖÜß ]', '',' '.join(verse.text)))
        pos = verse_clean.token_pos

    lines = [' '.join(verse.text)]

    input_text = ''

    last_idx = 1


    created_lines = []
    for i in range(1,len(verse.text)):
        if len(clean_word(verse.text[-i])) > 1:
            input_text = verse.text[:-(i+num_remove-1)]
            last_idx = i
            break

    input_text = ' '.join(input_text)
    input_text_cont = verse.context + '\n' + input_text

    input_text_last = re.sub('[^a-zA-ZäöüÄÖÜß]', '', ' '.join(verse.text[-(last_idx+num_remove):]))
 
    if len(target_rythm) > 0 and not elastic:    

        target_rythm_ext = np.asarray(extend_target_rythm(verse.rythm,target_rythm))
        
    if type(LLM) != str:

        generated = gpt2(input_text_cont, LLM, max_length=10,num_return_sequences=num_return_sequences,repetition_penalty=repetition_penalty,top_p = top_p,temperature = temperature,replace_linebreaks=replace_linebreaks)

    elif LLM == 'GPT3':
        generated = gpt3(input_text_cont, LLM, max_length=max_length,num_return_sequences=128,repetition_penalty=repetition_penalty,top_p = top_p,temperature = temperature,replace_linebreaks=replace_linebreaks)

    else:
        raise Exception('invalid LLM selection')
    
    for text in generated:

        if not eol: 
            line = re.sub('[^A-Za-zäöüÄÖÜß ]', ' ',text).split()
            if line: 
                line = ' '.join(line[:num_remove])
            else: 
                line = ''

        else:
            line = text.split('\n')[0]


        next_text = re.sub('[^A-Za-zäöüÄÖÜß]', '', line)
        
        input_text = ' '.join(re.sub('[^A-Za-zäöüÄÖÜß ]', ' ', input_text).split()).strip()

        if len(line) > 1 and next_text != input_text_last and next_text not in created_lines: # and len(line) < (len(' '.join(verse.text[-num_remove:])) + 5)
            created_lines.append(next_text)
            line_clean = ' '.join(re.sub('[^A-Za-zäöüÄÖÜß ]', ' ', line).split()).strip()
            verse_tmp = verse_cl(input_text  +' '+ line_clean)

            condition = True
       
            
            if use_pos and verse_tmp.token_pos[-num_remove:] != pos[-num_remove:]:
                condition = False
                
         
            if elastic:
                    target_rythm_ext = np.asarray(extend_target_rythm(verse_tmp.rythm,target_rythm))
            

            if len(verse_tmp.rythm) > len(target_rythm_ext) and allow_pos_match and not len(target_rythm) == 0 and not use_pos and eol:
             
                verse_tmp.shorten(len(verse_clean.rythm_tokens))
     
                condition = False
                if len(verse_tmp.rythm) == len(target_rythm_ext):

                    if verse_tmp.token_pos[-num_remove:] == pos[-num_remove:]:
                        condition = True


            if invalid_verse_ends and verse_tmp.token_pos[-1] in invalid_verse_ends:
                condition = False

            rythm =  np.asarray(verse_tmp.rythm)
            if len(target_rythm) > 0 and condition:  

                if len(rythm) == len(target_rythm_ext):
                    comp = np.abs((target_rythm_ext-rythm) * (rythm != 0.5))      # check if the rythm is correct

                    if np.sum(comp) == 0:

                        lines.append(input_text  + ' ' + line)

            elif len(target_rythm) == 0 and condition: 
                lines.append(input_text  + ' ' + line)

    
    return lines




if __name__ == "__main__":  

    poem = '''Ja wer wird denn gleich verzweifeln,
        weil er klein und laut und dumm ist?
        Jedes Leben endet. Leb so,
        daß du, wenn dein Leben um ist

        von dir sagen kannst: Na wenn schon!
        Ist mein Leben jetzt auch um,
        habe ich doch was geleistet:
        ich war klein und laut und dumm.'''

    args = 0

    print(iterate_verse(args,poem,2,[0,1])  ) 