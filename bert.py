from transformers import pipeline
import re
import numpy as np
import math
import spacy
import pandas as pd
from perplexity import perplexity
import torch

from spacy.lang.de.examples import sentences 
#from sia_rhyme.siamese_rhyme import siamese_rhyme
from rythm_utils import get_rythm, rythm_comp_adaptive,stressed_list,unstressed_list, extend_target_rythm, verse_cl

#rhyme_model = siamese_rhyme()

#from parameters import bert_model
nlp = spacy.load("de_core_news_lg")
###############################################

no_verse_end = ['CCONJ','SCONJ','CONJ','DET','ADP']
###############################################

from annotate_meter.ipa_hyphenate import hyphenate_ipa


def clean_word(word):
    return re.sub('[^a-zäöüß]', '', word.lower())

def get_synonyms_cand(args,verse,tok_id,target_rythms, LLM_perplexity, adaptive=False,after = False,verse_end = False,max_cand = 50, top_k = 150):
    '''
    get synonym candidates
    takes
    verse: input verse
    tok_id: token id that should be replaced
    target_rythms: the target rythm
    adaptive: if the syllable count is fixed (False) or flexible (True)
    verse_end: if the synonym is at the end of a verse
    max_cand: the maximum count of candiates to return
    top_k: number of candidates BERT creates 
    '''
    text = ''
    if torch.cuda.device_count() > 0:
        device = 0
    else: 
        device = -1
    unmasker = pipeline('fill-mask', model = args.bidirectional_model[0], top_k = top_k,framework='pt',device = device)

    if not verse.text[-1].isalpha():
        sign = verse.text[-1]
    else: 
        sign = ''


    if tok_id > -1:
        for token in verse.doc:
            if token.i != tok_id:
                text += token.text + ' '
            elif token.i == tok_id and not after:
                text+= args.mask_tok[0] + ' '
                token_pos = token.pos_          
                morphology = token.morph
                
            else:
                text += token.text + ' '+args.mask_tok[0]+' '
                token_pos = None          # just to avoid a nested if
                morphology = None

    else:
        text = args.mask_tok[0] + ' ' + str(verse.doc)
        token_pos = None
        morphology = None


    #text = verse.context + ' ' + text
    predictions = unmasker(verse.context + ' ' + text + sign + verse.context_after)
    print(text)
    candidates = []
    candidates_perp = []
    candidates_vec = []

    found_correct = False

    
    for i in range(2):
        for prediction in predictions:

            word = prediction['token_str']
            word = re.sub(r'[^a-zäöüß]','',word.lower())
            
            chunks= text.split()
            
            mask_idx = chunks.index(args.mask_tok[0])

            chunks[mask_idx] = word
            text_pred = ' '.join(chunks)
             
            if len(word) > 1 and word != 'unk' and word not in verse.text: # if it is a valid candidate
            
                doc = nlp(text_pred)
                tok_id = [token.i for token in doc if token.text == word and abs(token.i - mask_idx) < 3][0]
                # rythm = get_rythm(word)
                #rythm,_,_ = hyphenate_ipa(word)
                rythm = verse_cl(word).rythm
                proceed = True
                if verse_end:
                    if doc[tok_id].pos_ in no_verse_end:
                        proceed = False

                if ((doc[tok_id].pos_ == token_pos and doc[tok_id].morph == morphology) or i > 0 or after) and proceed:   # the second loop is in case spacy detected for example a propn instead of an adv
                                                                                                            # check if the candidate has the same grammatical properties
                    
                    '''if list(rythm ) == [0.5]:
    
                        if doc[tok_id].pos_ in stressed_list: 
                            if not (doc[tok_id].dep_ == 'mo' and len(doc[tok_id].text)<3):
                                rythm = [1]

                        if doc[tok_id].pos_ in unstressed_list:
                            rythm = [0]'''

                    if target_rythms:
                       
                        if rythm_comp_adaptive(rythm,target_rythms,adaptive):  # the rythm of the synonym has to be correct
                          
                            candidates.append(word)
                            perp = perplexity(verse.context + ' ' + ' '.join(text_pred.split()) + verse.context_after, LLM_perplexity)
                            candidates_perp.append(perp)
                    else: 
                        candidates.append(word)
                        candidates_perp.append(perplexity(verse.context + ' ' + text_pred + verse.context_after, LLM_perplexity))
                        
        if len(candidates) > 8:
            if i == 0:
                found_correct = True
            break

    candidates = candidates[:max_cand]
    candidates_perp = candidates_perp[:max_cand]
    #print(candidates)
    return candidates, candidates_perp, found_correct
               
def get_synonym(args,verse,tok_id,target_rythms,LLM_perplexity, adaptive = False,verbose = False,after=False,verse_end = False):
    '''
    return a synonym for a given inmput
    '''

    perp_0 = perplexity(' '.join(str(verse.doc).split()),LLM_perplexity)


    candidates, candidates_perp, found_correct = get_synonyms_cand(args,verse,tok_id,target_rythms,LLM_perplexity, adaptive=adaptive,after=after,verse_end = verse_end)
  
    if candidates:
        candidates_perp = np.asarray(candidates_perp)
        best_idx = np.argmin(candidates_perp)
        best_candidate = candidates[best_idx] # choose the candidate for which gpt2 gave the best perplexity

        if not verbose:
            return best_candidate

        else: return best_candidate,candidates_perp[best_idx],found_correct

    else: 
        if not verbose:
            return []

        else: return '',float('inf'),False



def bidirectional_synonyms_single(args,verse,last_idx,context_aft, target, LLM_perplexity, num_out = 100):

    text = ''
    if torch.cuda.device_count() > 0:
        device = 0
    else: 
        device = -1

    if not verse.text[-1].isalpha():
        sign = verse.text[-1]
    else: 
        sign = ''

    predictions = []

    input_text = verse.text[:last_idx]
    
    target_pos = nlp(' '.join(verse.text[last_idx]))[0].pos_


    #target_rythm_ext = np.asarray(extend_target_rythm(verse.rythm,target_rythm))
    input_text = ' '.join(input_text)

    #text = verse.context[-100:] + '[SOV]' + input_text + '[MASK]' + ' [EOV]' + context_aft

    text = verse.context[-250:] + '\n' +  input_text + '[MASK]' + sign + '\n' + context_aft
    gpt2_text_1 = verse.context[-250:] + ' ' + input_text
    gpt2_text_2 = '\n' + context_aft
    
    for model_id, model in enumerate(args.bidirectional_model):

        unmasker_rhyme = pipeline('fill-mask', model = model, top_k = num_out,framework='pt',device=device)

        text_inp = re.sub('\[MASK\]',args.mask_tok[model_id], text)
        print(text_inp)
        predictions.append(unmasker_rhyme(text_inp))


    pred_words = []

    for prediction in predictions: 
        pred_words += [pred['token_str'] for pred in prediction]

    candidates = []
    candidates_perp = []

    if target is None:
        target = []


    for word in pred_words:

        #word = prediction['token_str']
        word = re.sub(r'[^a-zäöüß]','',word.lower())
        if len(target) > 0: 
            #rythm = get_rythm(word)
            rythm,_,_ = hyphenate_ipa(word)
        doc = nlp(word)
        found_correct = []
        if len(word) > 1 and word != 'unk' and word not in input_text and not doc[0].pos_ in no_verse_end:
            if len(target) > 0: 

                if len(rythm) == len(target): 
                    if np.sum(np.abs(rythm-target)*(rythm != 0.5)) == 0:
                        candidates.append(word)
                        candidates_perp.append(perplexity(gpt2_text_1 + ' ' + word + ' ' + gpt2_text_2, LLM_perplexity))
                        found_correct.append(doc[0].pos_ == target_pos)
            else:
        
                candidates.append(word)
                candidates_perp.append(perplexity(gpt2_text_1 + ' ' + word + ' ' + gpt2_text_2, LLM_perplexity))
                found_correct.append(doc[0].pos_ == target_pos)
               

    candidates_perp = np.asarray(candidates_perp)

    correct_idx = np.where(np.asarray(found_correct))[0]
    candidates = pd.Series(candidates)

    if len(correct_idx) > 8:
        candidates_perp = candidates_perp[correct_idx]
        candidates = candidates[correct_idx]

    #max_value = (np.sort(candidates_perp)[:-int(len(candidates_perp)/3)])[-1]                      # lower 2/3 of the values

    return candidates, candidates_perp
    


def bidirectional_synonyms(args,verse,context_aft, target_rythm, LLM_perplexity, num_out = 50):
    '''
    create alternative endings for a verse
    '''


    for i in range(1,len(verse.text)):
        if len(clean_word(verse.text[-i])) > 1:
            input_text = verse.text[:-i]
            last_idx = -i
            break

    
    if target_rythm:

        target_rythm_ext = np.asarray(extend_target_rythm(verse.rythm,target_rythm))
        target = target_rythm_ext[-len(verse.rythm_tokens[last_idx]):]

        if len(target) < 3:
            split = False
        else: 
            split = True
    else: 
        split = False
        target = None

    ################################################################################
    # one alternative for last word
    ################################################################################

    candidates_single, candidates_perp_single = bidirectional_synonyms_single(args,verse, last_idx, context_aft, target, LLM_perplexity, num_out = 150)

    #candidate_idx = np.argsort(candidates_perp_first)[:-int(len(candidates_perp)*0.6)]


    #candidates_chosen = set(list(candidates[candidate_idx]))
    #candidates_chosen = [[candidate] for candidate in candidates_chosen if candidate.isalpha()]

    if split:
        ###############################################################################
        # split word into two words
        ###############################################################################
        target_rythms = []
        min_last_syll = 1
    
        for i in range(1,len(target)-min_last_syll+1):
            target_rythms.append(target[:i])

        candidates_first, candidates_perp, _ = get_synonyms_cand(args,verse,len(verse.text)+last_idx-1,target_rythms,LLM_perplexity,after = True,max_cand = 50)
        

        candidate_idx = np.argsort(candidates_perp)
        candidates_first = pd.Series(candidates_first)
        candidates_first_chosen = list(set(list(candidates_first[candidate_idx])))[:10]
    
        candidates_tuple = []
        candidates_perp_second = []
        for candidate in candidates_first_chosen:
            if target_rythm:
                candidate_rythm = verse_cl(candidate).rythm
                length_rythm = len(candidate_rythm)
                target_tmp = target[length_rythm:]
            else: 
                target_tmp = None

            txt_tmp = ' '.join(verse.text[:last_idx]) + ' ' + candidate + ' ' + verse.text[last_idx]
            verse_tmp = verse_cl(txt_tmp)
            verse_tmp.context = verse.context
            candidates_second, candidates_perp = bidirectional_synonyms_single(args,verse_tmp, -1, context_aft, target_tmp, LLM_perplexity, num_out = 20)
            candidates_tuple += [[candidate,candidate_second] for candidate_second in candidates_second if candidate_second.isalpha()]
            candidates_perp_second += list(candidates_perp)

        '''if len(candidates_perp_second) > 3:
            cut = -int(len(candidates_perp_second)/3)

        else:
            cut = None'''
        candidates_perp = list(candidates_perp_single) + candidates_perp_second#np.asarray(candidates_perp)
        

        candidates = list(candidates_single) + list(candidates_tuple)
    else:
        candidates = list(candidates_single)
        candidates_perp = list(candidates_perp_single)

    candidates_perp = np.asarray(candidates_perp)
    candidate_idx = np.argsort(candidates_perp)
    candidates = pd.Series(candidates)
    candidates_sorted = list(candidates[candidate_idx])


    candidates_sorted = [[verse.text[last_idx]]] + candidates_sorted

    candidates_set = []
    candidates_lower = []

    for cand in candidates_sorted:
        if type(cand) == list:
            cand_str = ' '.join(cand)
        else: 
            cand_str = cand

        if cand_str.lower() not in candidates_lower:
            candidates_set.append(cand)
            candidates_lower.append(cand_str.lower())

    return candidates_set
