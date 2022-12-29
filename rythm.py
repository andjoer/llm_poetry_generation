
import pandas as pd
import re
import numpy as np 
import math
import os
import pyphen
import spacy
import itertools
import random
from spacy.attrs import LOWER, POS, ENT_TYPE, IS_ALPHA, DEP, LEMMA, LOWER, IS_PUNCT, IS_DIGIT, IS_SPACE, IS_STOP
from spacy.tokens import Doc

from spacy_utils import remove_tokens_idx, get_childs_idx, is_conj_adv

from spacy.lang.de.examples import sentences 

from rythm_utils import  get_all_comb, get_single_comb, get_rythm, rythm_comp_adaptive, get_best_shift,extend_target_rythm, compare_verse_rythm, stressed_list, unstressed_list,verse_cl
from bert import get_synonym
from perplexity import perplexity


nlp = spacy.load("de_core_news_lg")



jambus = [0,1]


def token_compliance(verse, target_rythm):   # check if a word could theoretically have the correct rythm
    if len(target_rythm )== 2:
        target_rythms = [[0,1],[1,0]]
    else:
        target_rythms = [[1,0,0],[0,1,0],[0,0,1]]   

    text_tokens = []                                      
    for token in verse.doc:
        if len(token.text) > 2:       
            
            rythm = verse.rythm_tokens[token.i]
            if not rythm_comp_adaptive(rythm,target_rythms,adaptive=True):
                text_tokens.append(get_synonym(verse,token.i,target_rythms,adaptive=True))

            else: 
                text_tokens.append(token.text)

        else: text_tokens.append(token.text)


    
    verse.update(text_tokens)


def get_synonyms_shift(verse,token_idx,target_rythm_ext,shift,LLM_perplexity,force_correct = False):
    perplexities = []
    synonyms = []
    found_correct_lst = []
    shift_lst = [shift]
    if len(verse.rythm_tokens[token_idx]) > shift:
        shift_lst.append(-shift)

    for shift_ in shift_lst:

        tok_tar_rythms = [target_rythm_ext[verse.token_starts[token_idx]:verse.token_ends[token_idx]+shift_]]
        
        synonym, perp,found_correct = get_synonym(verse,token_idx,tok_tar_rythms,LLM_perplexity,verbose = True)
        perplexities.append(perp)
        synonyms.append(synonym)
        found_correct_lst.append(found_correct)

    return synonyms, perplexities, found_correct_lst

def fix_shifted_rythm(verse,target_rythm,LLM_perplexity, len_window = 6,target_len = 10):
    offset = 0

    print('start_fixing')
    print(target_rythm)
    print(str(verse.doc))
    rythm = np.asarray(verse.rythm)
    target_rythm_ext = np.asarray(extend_target_rythm(rythm,target_rythm))
    comp = np.abs((target_rythm_ext-rythm) * (rythm != 0.5))

    while np.sum(np.abs(comp)) != 0:
        print(comp)
        if len(verse.rythm) >= target_len:
            extend = False

        else:
            extend = True
            
        text = re.findall(r"[\w']+|[.,!?;]", str(verse.doc))
        problem_idx = np.amin(np.where(comp != 0)[0])
        token_idx = verse.token_dict[problem_idx]
        best_shift = get_best_shift(rythm,target_rythm,start=problem_idx,end=problem_idx+len_window)        
        tok_tar_rythms = []
        print('tok idx')
        print(token_idx)
        if verse.doc[token_idx-1].is_alpha:
            previous_idx = token_idx-1
        else:
            previous_idx = token_idx-2

        if token_idx not in verse.token_starts:            #problem is not at the word beginning -> cant be solved by something following
           
            if len(verse.rythm_tokens[token_idx]) - best_shift > 0:         # todo: works only for jambus/trochaus
                tok_tar_rythms.append(target_rythm_ext[verse.token_starts[token_idx]:verse.token_ends[token_idx]-best_shift])

            tok_tar_rythms.append(target_rythm_ext[verse.token_starts[token_idx]:verse.token_ends[token_idx]+best_shift])

            synonym = get_synonym(verse,token_idx,tok_tar_rythms,LLM_perplexity)
            text[token_idx] = synonym

        else:
            
            synonyms = []
            perplexities = []
            if best_shift == 0:
                print('shift 0 ')
                if not extend:
                   
                    #################################################

                    # shorten previous word, make the current longer
                    #################################################
                    if len(verse.rythm_tokens[token_idx-1]) > 1 and token_idx > 0:

                        
                
                        tok_tar_rythms = [target_rythm_ext[verse.token_starts[previous_idx]:verse.token_ends[previous_idx]-1]]

                        synonym = get_synonym(verse,token_idx -1,tok_tar_rythms,LLM_perplexity)
                        synonyms.append([synonym])
                        text_tmp = text.copy()
                        text_tmp[token_idx] = synonym

                        verse_tmp = verse_cl(' '.join(text_tmp))

                        tok_tar_rythms = [target_rythm_ext[verse.token_starts[token_idx]-1:verse.token_ends[token_idx]]]

                        synonym, perp,_ = get_synonym(verse_tmp,token_idx,tok_tar_rythms,LLM_perplexity,verbose = True)
                        perplexities.append(perp)

                        synonyms[-1] += [synonym]

                        

                    ################################################################
                    # shorten current word, make previous longer

                    ################################################################
                    if len(verse.rythm_tokens[token_idx]) > 1:
                        
                        tok_tar_rythms = [target_rythm_ext[verse.token_starts[previous_idx]:verse.token_ends[previous_idx]+1]]

                        synonym = get_synonym(verse,previous_idx,tok_tar_rythms,LLM_perplexity)
                        if synonym:
                            synonyms.append([synonym])
                            text_tmp = text.copy()
                            text_tmp[token_idx] = synonym
                            
                            verse_tmp = verse_cl(' '.join(text_tmp))

                            tok_tar_rythms = [target_rythm_ext[verse.token_starts[token_idx]+1:verse.token_ends[token_idx]]]

                            synonym, perp, _ = get_synonym(verse_tmp,token_idx,tok_tar_rythms,LLM_perplexity,verbose = True)
                            perplexities.append(perp)

                            synonyms[-1] += [synonym]
                    
                    


                if not synonyms:                                                                                                # worst case, not possible to make no shift
                    

                    tok_tar_rythms = [target_rythm_ext[verse.token_starts[token_idx]:verse.token_starts[token_idx]+1]]
                    print('tok tar rythms')
                    print(tok_tar_rythms)
                    synonym,perp,_ = get_synonym(verse,previous_idx,tok_tar_rythms,LLM_perplexity,after=True,verbose = True)
                    synonyms.append([synonym])
                    perplexities.append(perp)

                    tok_tar_rythms = [target_rythm_ext[verse.token_starts[token_idx]:verse.token_starts[token_idx]+3]]
                    print(tok_tar_rythms)

                    synonym, perp, _ = get_synonym(verse,token_idx,tok_tar_rythms,LLM_perplexity,verbose = True)
                    perplexities.append(perp)

                    synonyms.append([synonym])
                    perplexities.append(perp)

            
                    
            
                best_idx = perplexities.index(min(perplexities))
                text = text[:token_idx-(len(synonyms[best_idx])-1)] + synonyms[best_idx] + text[token_idx:]

            if best_shift != 0:
                print('shift 1')
                found_correct_lst = []
                synonyms_idx = []

                if not extend:
                    synonyms_tmp, perp_tmp,found_correct = get_synonyms_shift(verse,token_idx,target_rythm_ext,best_shift,LLM_perplexity)
                    synonyms += synonyms_tmp
                    perplexities += perp_tmp
                    found_correct_lst += found_correct
                    synonyms_idx += [0]*len(synonyms_tmp)
        
            
                    if token_idx > 0:
                        synonyms_tmp, perp_tmp, found_correct = get_synonyms_shift(verse,previous_idx,target_rythm_ext,best_shift,LLM_perplexity)
                        synonyms += synonyms_tmp
                        perplexities += perp_tmp
                        synonyms_idx += [1]*len(synonyms_tmp)
                        found_correct_lst += found_correct
                
                tok_tar_rythms = [target_rythm_ext[verse.token_starts[token_idx]-1:verse.token_starts[token_idx]]]

                if not tok_tar_rythms[0]:
                    tok_tar_rythms = [target_rythm[:best_shift]]
    
                synonym,perp,_ = get_synonym(verse,previous_idx,tok_tar_rythms,LLM_perplexity,after=True,verbose = True)
                synonyms.append(synonym)
                perplexities.append(perp)
                synonyms_idx += [2]
                found_correct_lst += [True]
                print(synonyms)
                if sum(found_correct_lst) > 0:
                    found_correct_arr = np.asarray(found_correct_lst)
                    poss_idx = np.where(found_correct_arr)[0]
                    perplexities_arr = np.asarray(perplexities)
                    best_val = np.amin(perplexities_arr[poss_idx])
                    best_idx = np.where(np.logical_and(perplexities_arr == best_val,found_correct_arr))[0][0]
            
                else:
                    best_idx = perplexities.index(min(perplexities))

                if synonyms_idx[best_idx] == 2:
                        text = text[:token_idx] + [synonyms[best_idx]] + text[token_idx:]

                else: 

                    text[token_idx - synonyms_idx[best_idx]] = synonyms[best_idx]

     


        
                

        #verse.text = ' '.join(text)
        print('fix_text')
        print(text)
        verse.update(text)
        print(verse.text)
        print('rythm')
        print(verse.rythm)

        rythm = np.asarray(verse.rythm)
        target_rythm_ext = np.asarray(extend_target_rythm(rythm,target_rythm))
        comp = np.abs((target_rythm_ext-rythm) * (rythm != 0.5))
         

    return verse

def extend_verse(verse,target_rythm,target_len,LLM_perplexity): # extend the verse

    while len(verse.rythm) <= (target_len-len(target_rythm)):                   # if len(rythm) is larger then target_len minus length of the target rythm no word with rythm = target_rythm would fit in                
        target_rythm_ext = extend_target_rythm(verse.rythm,target_rythm)
        fill_words = []
        perplexities = []
        text = []
        cut = None
        if not verse.doc[-1].is_alpha: # if the last token is not a letter 
            cut = -1
        for token_idx, token in enumerate(verse.doc[:cut]):  # for every token try to fill a word in front of it
    
            tok_tar_rythms = [target_rythm_ext[verse.token_starts[token_idx]:verse.token_starts[token_idx]+len(target_rythm)]] # only words with rythms % target rythm == 0 don't mess up the rythm. For simplicity only words with rythm == target_rythm are considered
            fill_word, perp, _ =  get_synonym(verse,token_idx-1,tok_tar_rythms,LLM_perplexity,after=True,verbose = True) # also get the perplexities
            fill_words.append(fill_word)
            perplexities.append(perp)
            text.append(token.text)

        best_idx = perplexities.index(min(perplexities)) # get the solution with the lowest perplexity
        text = text[:best_idx] + [fill_words[best_idx]] + text[best_idx:]

        if cut: 
            text.append(verse.doc[-1].text)

   
        #verse.text = ' '.join(text)
        verse.update(text)
        print('text:')
        print(' '.join(verse.text))
        print('rythm:')
        print(verse.rythm_tokens)
       
    if len(verse.rythm) != target_len:
        text = verse.text
        target_rythm_ext = extend_target_rythm([1]*target_len,target_rythm)

  
        if verse.doc[-1].is_alpha:
            token_idx = verse.doc[-1].i
            sign = []
        else: 
            token_idx = verse.doc[-2].i
            sign = [verse.doc[-1].text]
                
        #####################################
        #change last word
        #####################################
        

        tok_tar_rythms = [target_rythm_ext[verse.token_starts[token_idx]:]]

        fill_word_1, perp_1, found_correct = get_synonym(verse,token_idx,tok_tar_rythms,LLM_perplexity,verbose = True,verse_end=True)

        #####################################
        #add last word
        #####################################

        tok_tar_rythms = [target_rythm_ext[verse.token_ends[token_idx]:]]

        fill_word_5, perp_3, _ = get_synonym(verse,token_idx,tok_tar_rythms,LLM_perplexity,after=True,verbose = True,verse_end=True)
       

        ####################################
        # fill in syllab before last word
        ###################################

        tok_tar_rythms = [target_rythm_ext[verse.token_starts[token_idx]:verse.token_starts[token_idx]+(target_len - len(verse.rythm))]]
        fill_word_3, _, _ = get_synonym(verse,token_idx-1,tok_tar_rythms,LLM_perplexity,after=True,verbose = True)
        text = text[:token_idx] + [fill_word_3] + text[token_idx:]
        verse_tmp = verse_cl(' '.join(text))

        if verse_tmp.doc[-1].is_alpha:
            token_tmp_idx = verse_tmp.doc[-1].i
        else: token_tmp_idx = verse_tmp.doc[-2].i

        tok_tar_rythms = [target_rythm_ext[verse_tmp.token_starts[token_idx]:verse_tmp.token_ends[token_idx]]]
        fill_word_4, perp_2, _ = get_synonym(verse_tmp,token_tmp_idx,tok_tar_rythms,LLM_perplexity,verbose = True,verse_end=True)   # when the word before the last word is replaced, the last stress could be wrong
        
        
        if perp_1 < min(perp_2,perp_3):

            text = str(verse.doc).split()[:token_idx] + [fill_word_1] + sign
        
        elif perp_2 < perp_3: 

            text = str(verse.doc).split()[:token_idx] + [fill_word_3 ]+ [fill_word_4] + sign

        else: text = str(verse.doc).split()[:token_idx+1] +[fill_word_5] + sign

        #verse.text = ' '.join(text)
        verse.update(text)  
        print('text:')
        print(' '.join(verse.text))
        print('rythm:')
        print(verse.rythm_tokens)

    return verse

def rythm_diff(rythm,target):
    rythm = np.asarray(rythm)
    target = np.asarray(target)
    diff = np.sum(np.abs(rythm - target)*(rythm != 0.5) ) 
    return diff

def flatten(lst):
    return [item for sublist in lst for item in sublist]

def get_start_idx (list_lists):
    starts = []
    start = 0
    for lst in list_lists: 
        starts.append(start)
        start += len(lst)

    return starts

def compare_rythm(rythm,target_rythm,verse,mode='for_optim'):
   
    len_target = len(target_rythm)
    target_rythm_ext = np.asarray(extend_target_rythm(rythm,target_rythm)+target_rythm)
    opt_shift = 0
    comps = []
    correct_scores_lst = []
    for i in range(len(target_rythm)):

        comp = np.abs(rythm - target_rythm_ext[i:i-len_target or None])
        
       
        comp = comp * (rythm != 0.5)

        comps.append(comp)
   
        correct = np.where(comp == 0)[0]

        correct_clusters = (np.split(correct, np.where(np.diff(correct) != 1)[0]+1))

        correct_scores = np.zeros(comp.shape[0])

        for cluster in correct_clusters:
          
            correct_scores[cluster] = cluster.shape[0]

        correct_scores_lst.append(correct_scores)

    comp_0 = comps[0]
    correct_scores_0 = correct_scores_lst[0]
    chosen = np.zeros(comp_0.shape[0])
    for i, comp in enumerate(comps[1:]):
        chosen[np.logical_and(comp < comp_0, correct_scores_lst[i+1] >= correct_scores_0)] =  i + 1
        correct_scores_0[chosen == i+1] = correct_scores_lst[i+1][chosen == i+1]
        comp_0[chosen == i+1] = comp[chosen == i+1]
        
    splits = (np.where(np.diff(chosen) != 0)[0]+1)

    start_idx = np.asarray(get_start_idx(verse.rythm_tokens))
    final_output = np.zeros(comp_0.shape[0])
    if splits.size != 0: 
        idx_0 = 0
        for i in splits:
            idx = np.amin(start_idx[start_idx >= i])
            final_output[idx_0:idx] = comps[int(chosen[i-1])][idx_0:idx]
            idx_0 = idx

    else: 
        final_output = comps[int(chosen[0])]

    error_rythm = np.sum(final_output)
    error_split = splits.shape[0]

    return error_rythm, error_split, splits, chosen[0]


def remove_verse_tokens(verse,idxs):

    verb_lst = ['VERB','AUX']
    subjects = []
    subject_idx = []
    print('verse doc')
    print(str(verse.doc))
    doc = nlp(str(verse.doc))
    print(doc)

    ###########################################################################
    # fixing sentence structure 
    # 1) verb may not be first
    ###########################################################################

    subject_groups = []
    subject_roots = []
    for token in doc:
        subject_group = []
       
        if token.pos_ in verb_lst:

            child_idx = []

            for child in token.children:
                child_idx.append(child.i)
 
            remaining_children = [i for i in child_idx if i not in idxs]

            if remaining_children:

                if min(remaining_children) >= token.i:
                    for i in remaining_children:

                        if doc[i].dep_ == 'sb':

                            all_child = get_childs_idx(doc[i])

                            all_child = [item for idx, item in enumerate(all_child) if idx not in idxs]
                            all_child.append(i)
                            all_child.sort()
                            for j in all_child:                               
                                subject_group.append(doc[j].text)

                            subject_groups.append(subject_group)
                            subject_roots.append(token.text)    
                        

        text = [item for idx, item in enumerate(verse.text) if idx not in idxs]
        
        for i, subject_group in enumerate(subject_groups):
            text = [item for item in text if item not in subject_group]
            subject_idx = text.index(subject_roots[i])                          # To Do: catch the very rare case if the same verb is twice in the same shorted sentence. 
            text = text[:subject_idx] + subject_group + text[subject_idx:] 

    ###########################################################################################
    #2) AUX may only be last, if preceeded by conjunctor
    ###########################################################################################

    '''conjunctions = ['ADV','SCONJ','DET']

    doc = nlp(' '.join(text))

    AUX_lst = []
    targt_sb_lst = []
    rel_pos_lst = []
    for token in doc: 
        if token.pos_ == 'AUX':
            all_child_idx = get_childs_idx(token)
            if max(all_child_idx) <= token.i:
                if doc[token.i-1].pos_ == 'VERB' and token.i-1 in all_child_idx:
                    conj = min(all_child_idx)
                    print(doc[conj])
                    print(doc[conj+1])
                    if doc[conj].pos_ not in conjunctions:
                        AUX_lst.append(token.text)
                        if doc[conj].dep_ == 'sb':
                            targt_sb_lst.append(doc[conj].text)
                            rel_pos_lst.append(1)
                        else: 
                            for child in token.children: 
                                if child.dep_ == 'sb':
                                    targt_sb_lst.append(child.text)
                                    rel_pos_lst.append(0)

    
    for i, AUX in enumerate(AUX_lst):
        print('moving something big')
  
        text = [item for item in text if item != AUX]
        subject_idx = text.index(targt_sb_lst[i])                          # To Do: catch the very rare case if the same verb is twice in the same shorted sentence. 
        text = text[:subject_idx + rel_pos_lst[i]] + [AUX] + text[subject_idx+ rel_pos_lst[i]:] '''


    text = [item for idx, item in enumerate(text) if not (not item.isalpha() and not text[idx-1].isalpha())]
    print(text)

    #verse.text = ' '.join(text)
    verse.update(text=text)

    return verse


def find_opt_rythm(verse, combs, target_rythm,LLM_perplexity):

    n_splits = 100
    splits = []
    best_shift_0 = 0
    best_comb = []


    perplexities = []
    for comb in combs:
        text = pd.Series(verse.text)
        text = verse.context + ' ' + ' '.join(list(text.drop(comb)))
        perplexities.append(perplexity(text,LLM_perplexity))                                       # mostly because sometimes spacy labels something wrong

    mean = np.mean(np.asarray(perplexities))

    combs = [item for i, item in enumerate(combs) if perplexities[i] <= mean]

    for comb in combs:

        rythm = pd.Series(verse.rythm_tokens)
        rythm = np.asarray(flatten(list(rythm.drop(comb))))

        _, error_split, splits, shift_0 = compare_rythm(rythm,target_rythm,verse)

        if error_split < n_splits: 
            n_splits = error_split
            best_splits = splits
            best_shift_0 = shift_0
            best_comb = comb
            
    
    verse = remove_verse_tokens(verse,best_comb)

    return verse 

def lst_insert(lst, ins, pos):

    return lst[:pos] + ins + lst[pos:]

def insert_syll(verse,pos,stress):
    verse.update_token_dict()

    if stress == 1:
        fill = random.choice(fill_stress)

    else: 
        fill = random.choice(fill_no_stress)
        stress = 0

    verse.tokens = lst_insert(verse.tokens,[fill],verse.token_dict[pos]) #verse.tokens[:pos] + [random.choice(fill_stress)] + verse.tokens
    verse.rythm_tokens = lst_insert(verse.rythm_tokens,[[stress]],verse.token_dict[pos]) #verse.rythm_tokens[:pos] [[1]] + verse.rythm_tokens
    verse.rythm = lst_insert(verse.rythm,[stress],pos)
    verse.update_token_dict()
    return verse


def get_pos_tok(pos,doc, similar_to = None):

    if type(pos) != list:
        pos = [pos]

    sel_token = []
    for token in doc:
        if token.pos_ in pos:
            sel_token.append(token)
        elif similar_to == 'conj_adv':
            
            if is_conj_adv(token.vector): 
                sel_token.append(token)

    return sel_token

def poss_removes(verse):
    #print('looking for poss removes')
    rm_ancestors = ['ADP']
    need_ancestors = ['SCONJ','DET']
    need_children = ['CONJ','CCONJ','ADP']
    rm_childs_of_ancestor = ['ADP','DET']

    only_del_by_ancestor = ['sb','ob']
    aux = ['AUX']
    doc = verse.doc
    rm_combinations_lst = []
    num_stress_lst = []
    num_syllabs_lst = []
    doc = nlp(str(doc))

    need_ancestors_tok = get_pos_tok(need_ancestors, doc,similar_to = 'conj_adv')
    need_children_tok = get_pos_tok(need_children, doc)
    blocked_idx = []
    not_alpha_lst = []
    #print(str(doc))
    for token in doc:
    
        if token.pos_ == 'AUX':

            if not doc[token.i - 1].is_alpha:
                
                for tok_tmp in doc:
                    if tok_tmp.pos_ == 'VERB':
                        if token.i in get_childs_idx(tok_tmp):
                            verb_idx = tok_tmp.i
                                
                        

                            for i in range(token.i,verb_idx):
                                if doc[i].pos_ not in ['AUX','ADV','CONJ','CCONJ']:
                                    blocked_idx.append(token.i)
                                  

        if not token.is_alpha:
            not_alpha_lst.append(token)

    #print('going through tokens')
    #print(str(doc))
    for token in doc:
        rm_lst = []
       
        
        if token.dep_ not in only_del_by_ancestor and token.dep_ != 'ROOT' and token.pos_ not in need_ancestors and token not in need_ancestors_tok and token.i not in blocked_idx:

            '''if token.pos_ in needed_by_ancestor:
                for ancestor in token.ancestors:
                        rm_lst.append(ancestor.i)
                        rm_lst += get_childs_idx(ancestor)
                  '''
        

            if token.pos_ == 'VERB':
                for ancestor in token.ancestors:
                    if ancestor.pos_ == 'AUX':
                        token = ancestor
                        break

            
            '''for ancestor in token.ancestors:
                if ancestor.pos_ in rm_ancestors:
                    rm_lst.append(ancestor.i)

                break'''

            '''if token.pos_ in rm_childs_of_ancestor:
                for ancestor in token.ancestors:
                    if ancestor.pos_ in rm_ancestors:
                        rm_lst.append(ancestor.i)
                        token = ancestor
                    break'''


            rm_lst += get_childs_idx(token)

            for tok in need_ancestors_tok:

                if tok.i not in rm_lst:
                    ancestor_lst = []
                    for ancestor in tok.ancestors:
                        ancestor_lst.append(ancestor.i)

                    intersection = list(np.intersect1d(np.asarray(ancestor_lst),np.asarray(rm_lst)))

                    if ancestor_lst and len(intersection) == len(ancestor_lst):
                        rm_lst.append(tok.i)
            

            for tok in need_children_tok:

                if tok.i not in rm_lst:
                    children_lst = []
              
                
                    for child in tok.children:
          
                        children_lst.append(child.i)

                    intersection = list(np.intersect1d(np.asarray(children_lst),np.asarray(rm_lst)))
            
                    if children_lst and len(intersection) == len(children_lst):
                        rm_lst.append(tok.i)
         
            for tok in doc:
                if tok.pos_ in ['CONJ','CCONJ']:
                    rm_lst.append(tok.i)
                    break
                if tok not in not_alpha_lst and tok.i not in rm_lst:
                    break
                
            '''num_stress = 0

            for idx in rm_lst:  
                num_stress += sum([int(stress) for stress in verse.rythm_tokens[idx]])             # if the stress is 0.5, it is not counted

            if num_stress != 0:
                num_stress_lst.append(int(num_stress))
                rm_combinations_lst.append(rm_lst)'''
            
            num_syllabs = 0
            rm_lst = set(rm_lst)
            for idx in rm_lst:  
                num_syllabs  += len(verse.rythm_tokens[idx])            

            

            if num_syllabs!= 0:
                num_syllabs_lst.append(int(num_syllabs))
                rm_combinations_lst.append(rm_lst)


    rm_combinations_sr = pd.Series(rm_combinations_lst)
   
    return num_syllabs_lst, rm_combinations_lst

def get_intersect_dict(combinations_lst):
    intersect_dict = {}
    for i, combination in enumerate(combinations_lst):
        intersect_lst = []
        for j in (list(range(i)) + list(range(i+1,len(combinations_lst)))):
            if (np.intersect1d(np.asarray(combination),np.asarray(combinations_lst[j]))):
                intersect_lst.append(j)
                
                intersect_dict[i] = intersect_lst

    return intersect_dict

def remove_token(verse,num_remove):
    num_syllabs_lst, rm_combinations_lst = poss_removes(verse)
    
    ''' intersect_dict = {}
    for i, combination in enumerate(rm_combinations_lst):
        intersect_lst = []
        for j in (list(range(i)) + list(range(i+1,len(rm_combinations_lst)))):
            if (np.intersect1d(np.asarray(combination),np.asarray(rm_combinations_lst[j]))):
                intersect_lst.append(j)
                
                intersect_dict[i] = intersect_lst
    '''

    intersect_dict = get_intersect_dict(rm_combinations_lst)
    print('rm_lst')
    print(rm_combinations_lst)
    print('intersect')
    print(intersect_dict)



    print(num_syllabs_lst)
    print(num_remove)
    all_comb = []
    while not all_comb:

        all_comb_idx = []
    
        all_comb_idx = get_all_comb(num_syllabs_lst,num_remove)
        num_remove -= 1
    
        all_comb = []
        intersect_arr = np.asarray(list(intersect_dict))
        for idx in all_comb_idx:
            intersect = np.intersect1d(np.asarray(idx),intersect_arr)
            take = True
            if len(intersect) > 0:
                for j in intersect:
                    if len(np.intersect1d(np.asarray(idx),intersect_dict[j])) > 0:
                        take = False

            if take:
                all_comb_temp = []
                for j in  idx:
                    all_comb_temp += rm_combinations_lst[j]

                all_comb.append(all_comb_temp)

    return all_comb
    
def shorten(verse,num_remove, min_syll = 3, toll = 8): # minimum syllables to remove; tollerance
    
    num_syllabs_lst, rm_combinations_lst = poss_removes(verse)
    intersection =  get_intersect_dict(rm_combinations_lst)

    rm_idx = get_single_comb(num_syllabs_lst, num_remove, intersection, min_syll,toll = toll)
    print(rm_idx)
    rm_all = []
    for idx in rm_idx:
        rm_all += rm_combinations_lst[idx]

    return rm_all



def fix_rythm(verse,target_rythm,num_syllabs,LLM_perplexity):
    '''
    fix the rythm of a verse
    takes:
    verse: verse to fix
    target_rythm: the metrum 
    num_syllabs: number of syllables the verse should have
    '''

    num_rm = len(verse.rythm) - num_syllabs

    if num_rm > 15:                                         # if the difference is too large, use a simpler method due to run time
        idx_shorten = shorten(verse,num_rm-15)
        print(idx_shorten)
        verse = remove_verse_tokens(verse,idx_shorten)
    
    difference = compare_verse_rythm(verse,target_rythm)
    num_rm = len(verse.rythm) - num_syllabs
    num_rm_0 = num_rm


   

    while (len(verse.rythm) > num_syllabs) or difference > 0: # while the metrum ist not correct
        print(target_rythm)
        print(num_syllabs)
        print(verse.text)
        print(verse.rythm)

        num_rm = len(verse.rythm) - num_syllabs+2               # number of syllables to remove
        if num_rm == num_rm_0:                                      #avoiding infinite loops of adding and removing
            num_rm += 1
        print(num_rm)
        if num_rm > 0:
            print('shortening')
            combs = remove_token(verse,num_rm)                    # get possible combinations of words that could be removed
            verse = find_opt_rythm(verse,combs,target_rythm,LLM_perplexity)      # find the metrically best option
        verse = fix_shifted_rythm(verse,target_rythm, LLM_perplexity, target_len = num_syllabs)    # correct metrically incorrect syllables
        print(verse.text)
        difference = compare_verse_rythm(verse,target_rythm)
        num_rm_0 = num_rm
        print(difference)

   

    if len(verse.rythm)  < num_syllabs:
        print('enter extension')
        verse = extend_verse(verse,target_rythm,num_syllabs,LLM_perplexity)  # exend the verse
    return verse




def check_rythm(vers, rythm_reff, last = None, num_stress = None):
    vers = get_rythm_sent(vers)
    rythm_raw = vers.rythm_raw
    rythm = list(itertools.chain.from_iterable(vers.rythm_raw))
    sentence = vers.text
    if num_stress:
        reff = rythm_reff*10
    #else:
        #reff = (rythm_reff*(math.ceil(len(rythm)/len(rythm_reff))))[:len(rythm)]

    if rythm_reff[0] == 0 and rythm[0] == 1:
        fill_word = random.choice(fill_no_stress)
        sentence = fill_word + ' ' + sentence
        rythm = [0] + rythm
        rythm_raw = [0].append(rythm_raw)

    if rythm_reff[0] == 1 and rythm[0] == 0:
        fill_word = random.choice(fill_stress)
        sentence = fill_word + ' ' + sentence
        rythm = [1] + rythm
        rythm_raw = [1].append(rythm_raw)

    rythm = np.asarray(rythm)
    reff = np.asarray(reff[:len(rythm)])

    diff = np.sum(np.abs(rythm - reff)*(rythm != 0.5) )                                       # if the wordstress is ambiguos we count it as correct

    if last:                                                                  # we check the refference since it is interpolated                 
        if reff[-1] != last:
            diff = 10
    
    if num_stress:
        if np.sum(reff) < num_stress:
            diff =  10 

    print('rythm:')
    print(rythm)

  
    # verse = verse_cl(sentence, rythm_reff, rythm_raw)

    return diff, vers
