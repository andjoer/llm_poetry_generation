
#from transformers import pipeline
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
import numpy as np
import re
import os
import random
#from rythm import check_rythm
#from numba import cuda

from rythm_utils import extend_target_rythm, verse_cl

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def clean_word(word):
    return re.sub('[^a-zäöüß]', '', word.lower())

class LLM_class:
    def __init__ (self,model_name,tokenizer_name = '',sampling = 'systematic',device='cpu'):
        self.model_name = model_name
        self.device = device
        self.sampling = sampling
        '''if "cuda" in device and sampling != 'systematic':
            if device[-1].isnumeric():
                device_pipeline = int(device[-1])

            else: 
                device_pipeline = 0

        else: 
            device_pipeline = -1'''

        if not tokenizer_name: 
            self.tokenizer_name = model_name
        else: 
            self.tokenizer_name = tokenizer_name

        #if sampling == 'systematic':
        self.model = GPT2LMHeadModel.from_pretrained(model_name).to(self.device)
        #else: 
            #self.model = pipeline('text-generation', model=model_name,
                        #tokenizer=model_name, framework = 'pt',device = device_pipeline)
            
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        if sampling == 'systematic':
            self.get_block_tokens()
        else: 
            self.block_tokens = None


    def get_block_tokens(self):
        self.block_tokens = []
        self.block_tokens_num = []
        for i in range(len(self.tokenizer)):
            if not self.tokenizer.decode(i).strip().isalpha():
                self.block_tokens.append(i)
        for i in range(10):
            self.block_tokens_num.append(self.tokenizer.encode(str(i)))

class LLM_state():
    def __init__(self,tokens,logits):
        self.possible_tokens = tokens
        self.possible_logits = logits



def gpt2(input_text,LLM, max_length= 10, num_return_sequences=5,stop=['\n'],repetition_penalty = 1.15,top_p = 1,temperature = 0.8, block_linebreak = False):

    input_ids = LLM.tokenizer.encode(input_text,return_tensors='pt').to(LLM.device)
    max_length += input_ids.size(1)
    
    #generated = LLM.model(input_text, max_length=max_length,return_full_text = False, num_return_sequences=num_return_sequences,repetition_penalty=1.2)

    generated = LLM.model.generate(
                                        input_ids,
                                        do_sample=True, 
                                        max_length=max_length, 
                                        top_p=top_p, 
                                        temperature = temperature,
                                        num_return_sequences=num_return_sequences,
                                        repetition_penalty = repetition_penalty
                                    )
    
    #return [item['generated_text'] for item in generated]
    if block_linebreak:
        linebreak = LLM.tokenizer.encode('a\n')[-1]                    # due to colab issue

        return [' ' + LLM.tokenizer.decode(item[input_ids.size(1):], skip_special_tokens=True) for item in generated if item[input_ids.size(1)] != linebreak]
    else: 
        return [' ' + LLM.tokenizer.decode(item[input_ids.size(1):], skip_special_tokens=True) for item in generated]


'''
def gpt2_top_p(input_text,max_length = 10,num_return_sequences=5):
    tokenizer = GPT2Tokenizer.from_pretrained(gpt2_model)
    model = GPT2LMHeadModel.from_pretrained(gpt2_model,pad_token_id = tokenizer.eos_token_id)
    input_ids = tokenizer.encode(input_text,return_tensors='pt')
    max_length += input_ids.size(1)
    start = input_ids.size()[1]
    output = model.generate(
        input_ids,
        do_sample = True,
        max_length = max_length,
        top_p = 0.92,
        top_k = 0,
        num_return_sequences = num_return_sequences,
        early_stopping = True,
        num_repeat_ngram_size = 2
    )

    return [tokenizer.decode(sample_output[start:],skip_special_tokens=True) for sample_output in output]

def gpt2_beam(input_text,max_length = 10,num_return_sequences=5):
    tokenizer = GPT2Tokenizer.from_pretrained(gpt2_model)
    model = GPT2LMHeadModel.from_pretrained(gpt2_model,pad_token_id = tokenizer.eos_token_id)
    input_ids = tokenizer.encode(input_text,return_tensors='pt')
    max_length += input_ids.size(1)
    start = input_ids.size()[1]
    output = model.generate(
        input_ids,
        do_sample = True,
        max_length = max_length,
        num_beams = 5,
        num_return_sequences = num_return_sequences,
        early_stopping = True,
        num_repeat_ngram_size = 2
    )

    return [tokenizer.decode(sample_output[start:],skip_special_tokens=True) for sample_output in output]'''

def remove_last(possible_tokens, possible_logits,tokenizer,max_word_count):
    
    first = True
    while len(re.sub(r'[^A-Za-zÄÖÜäöüß ]', ' ',tokenizer.decode([tokens[-1]for tokens in possible_tokens])).split()) > max_word_count or first and possible_tokens:  # necessary when words consisting out of many tokens get shortened
        possible_tokens[-1] = possible_tokens[-1][:-1]
        possible_logits[-1] = possible_logits[-1][:-1]

        first = False

        while not possible_tokens[-1] and possible_tokens[0]:
            possible_tokens = possible_tokens[:-1]
            possible_tokens[-1] = possible_tokens[-1][:-1]

            possible_logits = possible_logits[:-1]
            possible_logits[-1] = possible_logits[-1][:-1]

        if not possible_tokens[0]:
            break

        

    return possible_tokens, possible_logits

def get_input_text(verse,num_words_remove):
    input_text = ''

    if num_words_remove:
        idx_out = 0
        idx = 1
        while idx_out < num_words_remove :
            try:
                if len(clean_word(verse.text[-idx])) > 1:
                    idx_out += 1
            except: 
                return '', 1
            idx += 1
        idx -= 1
        input_text = verse.text[:-idx]
        input_text = ' '.join(input_text)
    else:
        input_text = ''
        idx = 1

    return input_text,idx

def get_bigram_dict(input_tokens, max_n_bigrams = 3):
    bigram_dict = {}

    for idx, token in enumerate(input_tokens[0,:-1]):
        if token not in bigram_dict.keys():
            bigram_dict[token] = []
        bigram_dict[token] += [input_tokens[0,idx+1]]

    filtered_bigram_dict = {}
    for token in bigram_dict.keys():
        if len(bigram_dict[token]) >= max_n_bigrams:
            filtered_bigram_dict[token] = bigram_dict[token]
    return filtered_bigram_dict

def get_num_ngram(sentence, N):
    if type(sentence) != list:
        sentence = sentence.split()
    if len(sentence) < N:
        return 0
    n_grams = [sentence[i:i+N] for i in range(len(sentence)-N+1)]       
 
    return n_grams.count(n_grams[-1])


def gpt_sample_systematic(verse,LLM,num_return_sequences = 100,loop_limit = 10000, num_words_remove = None, top_p = None,top_k = 20, temperature = 0.9,random_first = False, random_all = False,stop_tokens_alpha = [],block_non_alpha = True,
                        top_p_dict = {},pos=False,check_rythm = True, target_rythm = [],num_syll = None,num_syll_tollerance = 1,last_stress = None, trunkate_after = 100,pos_alternative = False,factor_stop_token=0.2,bigram_limit=2, trigram_limit = 1,
                        dividable_rest=False, only_alpha_after = 3,allow_pos_match=False,repetition_penalty=1.2,invalid_verse_ends = [],return_last_state = False,last_state = None):

 
    if num_words_remove and type(verse) != str:
        input_text, idx = get_input_text(verse,num_words_remove)
        prompt = verse.context + '\n' + input_text
        reff_sentence = ' '.join(verse.text)
        reff_sentence = re.sub(r'[^A-Za-zÄÖÜäöüß ]', ' ',reff_sentence).strip().split()[-num_words_remove:]
        reff_verse = verse_cl(reff_sentence)
        last_verse_rythms =  [item for sublist in verse.rythm_tokens[-idx:] for item in sublist]
        if target_rythm:
            reff_verse.rythm = extend_target_rythm(verse.rythm,target_rythm)[-len(last_verse_rythms):]
        num_syll = len(reff_verse.rythm)
  
    elif type(verse) != str:
        reff_verse = None
        input_text, _ = get_input_text(verse,num_words_remove)
        prompt = verse.context + '\n' + input_text
    else: 
        reff_verse = None
        prompt = verse


    if num_words_remove and not top_p:
        top_p_vs_num_words = {1:0.8,2:0.7,3:0.6}
        if num_words_remove <=3:
            top_p = top_p_vs_num_words[num_words_remove]
        else: 
            top_p = 0.5
  
    elif not top_p:
        top_p = 0.5


    if not num_words_remove: 
        max_word_count = float('inf')
    else: 
        max_word_count = num_words_remove

    
    if not prompt: 
        prompt = '<|endoftext|>'
    tokenizer = LLM.tokenizer#GPT2Tokenizer.from_pretrained(LLM[0])
    model = LLM.model#GPT2LMHeadModel.from_pretrained(LLM[1]).to('cuda')
    stop_tokens = []
    if stop_tokens_alpha:
        if '\n' in stop_tokens_alpha:          # fix for google colab; no explanation for the bug currently
            stop_tokens_alpha.remove('\n')
            stop_tokens = [tokenizer.encode('a\n')[-1]]

        stop_tokens += [tokenizer.encode(stop_token)[0] for stop_token in stop_tokens_alpha]
 
        

   
    block_tokens_alpha = LLM.block_tokens
    for token in stop_tokens:
        try:
            block_tokens_alpha.remove(token)
        except:   # if it is not in the list
            pass 
    

    if block_non_alpha:
        block_tokens_0 = block_tokens_alpha
    else: 
        block_tokens_0 = []
    

    block_tokens = block_tokens_0
    block_tokens_num = LLM.block_tokens_num

    # linebreak = tokenizer.encode('\n')[0]
    linebreak = tokenizer.encode('a\n')[-1]                    # due to colab issue

    sm = torch.nn.Softmax(dim = 1)

    
    if last_state:
        possible_tokens = last_state.possible_tokens
        possible_logits = last_state.possible_logits
    else:
        possible_tokens = []
        possible_logits = []

    possible_combinations = []
    combination_logits = []

    inputs = tokenizer(prompt,return_tensors='pt')['input_ids']

    max_token_count = max_word_count*3  # 3 times more tokens then words


    if top_p_dict:
        top_p_dict_np = np.asarray(list(top_p_dict.keys()))

    fulfill_pos = False

    possible_end = False
    pos_match_end = False
    depth_lst = []

    last_word_start = 0   # stop building endless words
    '''if reff_verse:
        print('prompt')
        print(prompt)
        print(pos)
        print('num syll')
        print(num_syll)
        print('num remove')
        print(num_words_remove)
        print(target_rythm)
        
        print(reff_verse.rythm)'''
    with torch.no_grad():
        for i in range(loop_limit): 
    
            if len(possible_tokens) > 0:

                while not possible_logits[-1]:
                    possible_tokens = possible_tokens[:-1]
                try:
                    new_tokens =  torch.reshape(torch.IntTensor([tokens[-1] for tokens in possible_tokens]),(1,-1))
              
                except: 
                    print('possible tokens')
                    print(possible_tokens)
                    raise Exception
                input_tokens = torch.cat((inputs,new_tokens),1)

            else: 
                input_tokens = inputs

            depth_lst.append(len(possible_tokens))
         
            try:
                outputs = model(input_tokens.to(LLM.device))
            except:
                break
           
            logits = outputs.logits[:,-1,:]/temperature

            logits[:,input_tokens[0,-3:]] = -float('inf') # avoid repetition within the last 3 logits

            #blocked tokens
            logits[:,block_tokens] = -float('inf')
            logits[:,block_tokens_num] = -float('inf')
            
            #repetition panelty
            input_token_penalty_lst = input_tokens[0].tolist()
            try:
                input_token_penalty_lst.remove(linebreak)
            except: 
                pass
            for previous_token in set(input_token_penalty_lst):
                # if score < 0 then repetition penalty has to multiplied to reduce the previous token probability
                if logits[:, previous_token] < 0:
                    logits[:, previous_token] *= repetition_penalty
                else:
                    logits[:, previous_token] /= repetition_penalty

            last_token_test = 'test' + tokenizer.decode(torch.argmax(logits))                                   # check if next token contains a space

            generated = re.sub(r'[^A-Za-zÄÖÜäöüß ]', ' ',tokenizer.decode([tokens[-1]for tokens in possible_tokens]))

            complete_text = re.sub(r'[^A-Za-zÄÖÜäöüß ]', ' ',tokenizer.decode(input_tokens[0,-12:]))

            sentence = generated.split()

            if generated and (((pos or check_rythm or target_rythm) and (len(last_token_test.split()) > 1) or torch.argmax(logits) in stop_tokens or not tokenizer.decode(torch.argmax(logits)).isalpha())):
                fulfill_requirements = True
                possible_end = True
                last_word_start = len(possible_tokens)
                generated_verse = verse_cl(generated)
               
                
                if num_words_remove and generated[0] != ' ':
                    fulfill_requirements = False

                if pos and generated_verse.token_pos[:len(sentence)] != reff_verse.token_pos[:len(sentence)]:
           
                    fulfill_requirements = False
                    fulfill_pos = False

                else:
                    fulfill_pos = True

                if get_num_ngram(complete_text,2) > bigram_limit or get_num_ngram(complete_text,3) > trigram_limit:
                    fulfill_requirements = False


                if allow_pos_match and generated_verse.token_pos == reff_verse.token_pos:
                    pos_match_end = True
                else: 
                    pos_match_end = False
                if generated_verse.token_pos:
                    if invalid_verse_ends and generated_verse.token_pos[-1] in invalid_verse_ends:
                        possible_end = False
                else: possible_end = False

                if num_syll:
                    if len(generated_verse.rythm) < num_syll*num_syll_tollerance:
                        
                        possible_end = False

                    if len(generated_verse.rythm) > num_syll*num_syll_tollerance - only_alpha_after:            # some LLM end a verse with , or . and then continue in line -> prevent this
                        block_non_alpha = True
                        block_tokens = block_tokens_alpha
                    else: 
                        block_non_alpha = False
                        block_tokens = block_tokens_0

                    if len(generated_verse.rythm) > num_syll:
                       
                        fulfill_requirements = False
                        possible_end = False

                    if last_stress:
                        if generated_verse.rythm[-1] != last_stress:
                            possible_end = False

                    if dividable_rest and target_rythm:
                        if (len(generated_verse.rythm)-num_syll)%len(target_rythm) != 0:
                            possible_end = False

                if (check_rythm or target_rythm):
                    if reff_verse:
                        if reff_verse.rythm[:len(generated_verse.rythm)] != generated_verse.rythm:
                            fulfill_requirements = False

                    else:
                        target_rythm_ext = np.asarray(extend_target_rythm(generated_verse.rythm,target_rythm))
                        rythm =  np.asarray(generated_verse.rythm)
                        diff = np.sum(np.abs((target_rythm_ext-rythm) * (rythm != 0.5)) )

                        if diff != 0:
                            fulfill_requirements = False
                try:
                    if len(sentence[-1] )< 2:
                        fulfill_requirements = False
                except: 
                    fulfill_requirements = False   # sentence was []
            else:
                if len(possible_tokens) - last_word_start < 5:          # no endless compund tokens without space separation
                    fulfill_requirements = True
                else: 
                    fulfill_requirements = False
                possible_end = False

            if torch.argmax(logits) == linebreak and not possible_end:
                fulfill_requirements = False

            if fulfill_requirements:

                if not possible_end:
                    logits[:,linebreak] = -float('inf')
                    if block_non_alpha:
                        logits[:,stop_tokens] = -float('inf')

                logits_sorted,indices_sorted = torch.sort(logits, descending=True)
                logits_sorted = sm(logits_sorted)
                cum_sum = torch.cumsum(logits_sorted, dim=-1)
                cum_sum[:,0] = 0               
                if top_p_dict:
                    top_p = top_p_dict[top_p_dict_np[top_p_dict_np  <= len(possible_tokens)].max()]    
                                              
                token_inside_top_p = cum_sum <= top_p                                   # keep at least one index
                stop_token_inside_top_p = cum_sum <= top_p * factor_stop_token

                top_p_stop_token_list = [tensor.item() for tensor in indices_sorted[stop_token_inside_top_p]]

            if len(possible_tokens) >= max_token_count or not fulfill_requirements:  
            
                if len(depth_lst) > trunkate_after*2:                                     # too many repetitions with same trunk -> the trunk could be the problem
                    possible_tokens = possible_tokens[:1]
                    possible_logits = possible_logits[:1]
                    depth_lst = []

                elif len(depth_lst) > trunkate_after:
                    cut = max(int(min(depth_lst[-(trunkate_after+int(trunkate_after*0.8)):])/2),1)
                    if cut == 1:
                        depth_lst = []
                    possible_tokens = possible_tokens[:cut]
                    possible_logits = possible_logits[:cut]

                possible_tokens, possible_logits = remove_last(possible_tokens, possible_logits,tokenizer,max_word_count)
                if len(possible_tokens) == 1:
                    depth_lst = []
                                
            elif ((stop_tokens and list(set(stop_tokens) & set(top_p_stop_token_list))) or pos_match_end or len(sentence) == max_word_count or (pos_alternative and fulfill_pos)) and possible_end:
                #print(tokenizer.decode([tokens[-1].item() for tokens in possible_tokens]))
                '''print('rythm in generation function')
                print(generated_verse.text)
                print(generated_verse.rythm)
                print(generated_verse.token_pos)'''

                depth_lst = []
                possible_combinations.append([tokens[-1].item() for tokens in possible_tokens])
     
                last_logits_sum = sum([logits[-1].item() for logits in possible_logits])
                combination_logits.append(last_logits_sum)

                possible_tokens, possible_logits = remove_last(possible_tokens, possible_logits,tokenizer,max_word_count)
                
                if not possible_tokens[0] or len(possible_combinations) >= num_return_sequences:
                    break

            elif fulfill_requirements:

                if len(possible_tokens) == 0 and len(indices_sorted[token_inside_top_p])  < top_k:
                    indices_filtered = torch.flip(indices_sorted[0,:top_k],dims=[-1])    # highest probability last so it gets accessed first
                    logits_filtered = torch.flip(logits_sorted[0,:top_k],dims=[-1])           
            
                else:
                    indices_filtered = torch.flip(indices_sorted[token_inside_top_p],dims=[-1])    # highest probability last so it gets accessed first
                    logits_filtered = torch.flip(logits_sorted[token_inside_top_p],dims=[-1])           

                if random_all or (random_first and not possible_tokens):            # without randomness always the same poem would be created from the same prompt
                    all_indices_ran = torch.multinomial(logits_filtered,num_samples = len(logits_filtered))
                    logits_filtered = logits_filtered[all_indices_ran]
                    indices_filtered = indices_filtered[all_indices_ran]
                

                possible_tokens.append(list(indices_filtered))
                possible_logits.append(list(logits_filtered))
            else:
                possible_tokens, possible_logits = remove_last(possible_tokens, possible_logits,tokenizer,max_word_count)

            if not possible_tokens:
                break

            if not possible_tokens[0]:
                break

    last_state = LLM_state(possible_tokens,possible_logits)

    if return_last_state:
        return [tokenizer.decode(combination) for combination in possible_combinations], last_state
    else:
        return [tokenizer.decode(combination) for combination in possible_combinations]

if __name__ == "__main__":  
    LLM_2 = LLM_class('Anjoe/german-poetry-gpt2-large',device='cuda')

    verse = verse_cl('in einem schönen, Haus')
    verse.context = 'Wär ich doch ein Engel'

    print(gpt_sample_systematic(verse,LLM_2,num_return_sequences=5, num_words_remove = 2,pos=False,target_rythm = [0,1]))
    #print(gpt_sample_systematic('Wär ich doch ein Engel\n',LLM_2,num_return_sequences=1, stop_token='\n',num_syll=8,target_rythm=[0,1]))
