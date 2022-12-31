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

from parameters import bert_model
nlp = spacy.load("de_core_news_lg")
###############################################

no_verse_end = ['CCONJ','SCONJ','CONJ','DET','ADP']
###############################################

from annotate_meter.ipa_hyphenate import hyphenate_ipa


def clean_word(word):
    return re.sub('[^a-zäöüß]', '', word.lower())

def get_synonyms_cand(verse,tok_id,target_rythms, LLM_perplexity, adaptive=False,after = False,verse_end = False,max_cand = 50, top_k = 150):
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
    unmasker = pipeline('fill-mask', model = bert_model, top_k = top_k,framework='pt',device = device)
    if tok_id > -1:
        for token in verse.doc:
            if token.i != tok_id:
                text += token.text + ' '
            elif token.i == tok_id and not after:
                text+= '[MASK] '
                token_pos = token.pos_          
                morphology = token.morph
                
            else:
                text += token.text + ' [MASK] '
                token_pos = None          # just to avoid a nested if
                morphology = None

    else:
        text = '[MASK] ' + str(verse.doc)
        token_pos = None
        morphology = None


    #text = verse.context + ' ' + text
    predictions = unmasker(verse.context + ' ' + text)
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
            
            mask_idx = chunks.index('[MASK]')

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
                            perp = perplexity(' '.join(text_pred.split()), LLM_perplexity)
                            candidates_perp.append(perp)
                    else: 
                        candidates.append(word)
                        candidates_perp.append(perplexity(text_pred), LLM_perplexity)
                        
        if len(candidates) > 8:
            if i == 0:
                found_correct = True
            break

    candidates = candidates[:max_cand]
    candidates_perp = candidates_perp[:max_cand]
    #print(candidates)
    return candidates, candidates_perp, found_correct
               
def get_synonym(verse,tok_id,target_rythms,LLM_perplexity, adaptive = False,verbose = False,after=False,verse_end = False):
    '''
    return a synonym for a given inmput
    '''

    perp_0 = perplexity(' '.join(str(verse.doc).split()),LLM_perplexity)

    candidates, candidates_perp, found_correct = get_synonyms_cand(verse,tok_id,target_rythms,LLM_perplexity, adaptive=adaptive,after=after,verse_end = verse_end)
  
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



def bidirectional_synonyms_single(verse,last_idx,context_aft, target, LLM_perplexity, num_out = 100):

    unmasker_rhyme = pipeline('fill-mask', model = bert_model, top_k = num_out,framework='pt')

    input_text = verse.text[:last_idx]
    
    target_pos = nlp(' '.join(verse.text[last_idx]))[0].pos_


    #target_rythm_ext = np.asarray(extend_target_rythm(verse.rythm,target_rythm))
    input_text = ' '.join(input_text)

    text = verse.context[-100:] + '[SOV]' + input_text + '[MASK]' + ' [EOV]' + context_aft
    gpt2_text_1 = verse.context[-100:] + ' ' + input_text
    gpt2_text_2 = '\n' + context_aft

    predictions = unmasker_rhyme(text)

    candidates = []
    candidates_perp = []
    for prediction in predictions:

        word = prediction['token_str']
        word = re.sub(r'[^a-zäöü]','',word.lower())
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

  

    '''candidates_chosen = []
    for candidate in candidates:

        rythm = np.asarray(get_rythm(candidate))
        target = np.asarray(verse.rythm_tokens[last_idx])'''

        
           
    return candidates, candidates_perp
    


def bidirectional_synonyms(verse,context_aft, target_rythm, LLM_perplexity, num_out = 50):
    '''
    create alternative endings for a verse
    '''


    for i in range(1,len(verse.text)):
        if len(clean_word(verse.text[-i])) > 1:
            input_text = verse.text[:-i]
            last_idx = -i
            break

    target_rythm_ext = np.asarray(extend_target_rythm(verse.rythm,target_rythm))
    target = target_rythm_ext[-len(verse.rythm_tokens[last_idx]):]

    if len(target) < 4:
        ################################################################################
        # one alternative for last word
        ################################################################################
        candidates, candidates_perp = bidirectional_synonyms_single(verse, last_idx, context_aft, target, LLM_perplexity, num_out = 150)

        candidate_idx = np.argsort(candidates_perp)[:-int(len(candidates_perp)*0.6)]


        candidates_chosen = set(list(candidates[candidate_idx]))
        candidates_chosen = [[candidate] for candidate in candidates_chosen if candidate.isalpha()]

    else:
        ###############################################################################
        # split word into two words
        ###############################################################################
        target_rythms = []
        min_last_syll = 1
    
        for i in range(1,len(target)-min_last_syll+1):
            target_rythms.append(target[:i])

        candidates_first, candidates_perp, _ = get_synonyms_cand(verse,len(verse.text)+last_idx-1,target_rythms,LLM_perplexity,after = True,max_cand = 50)
        

        candidate_idx = np.argsort(candidates_perp)
        candidates_first = pd.Series(candidates_first)
        candidates_first_chosen = list(set(list(candidates_first[candidate_idx])))[:10]
    
        candidates_tuple = []
        perplexities_second = []
        for candidate in candidates_first_chosen:
            length_rythm = len(get_rythm(candidate))
            target_tmp = target[length_rythm:]

            txt_tmp = ' '.join(verse.text[:-last_idx-1]) + ' ' + candidate + ' ' + verse.text[last_idx]
            verse_tmp = verse_cl(txt_tmp)
            candidates_second, candidates_perp = bidirectional_synonyms_single(verse, -1, context_aft, target_tmp, LLM_perplexity, num_out = 20)
            candidates_tuple += [[candidate,candidate_second] for candidate_second in candidates_second if candidate_second.isalpha()]
            perplexities_second += list(candidates_perp)

        if len(perplexities_second) > 3:
            cut = -int(len(perplexities_second)/3)

        else:
            cut = None
        candidates_perp = np.asarray(candidates_perp)
        candidate_idx = np.argsort(perplexities_second)[:cut]

        candidates_tuple = pd.Series(candidates_tuple)
        candidates_chosen = list(candidates_tuple[candidate_idx])

    candidates_chosen.append([verse.text[last_idx]])

    return candidates_chosen
