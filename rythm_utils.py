import numpy as np
from itertools import combinations
from collections import Counter
import pandas as pd
import re
import pyphen
import os
from annotate_meter.ortho_to_ipa import ortho_to_ipa
import math

import spacy
from spacy.attrs import LOWER, POS, ENT_TYPE, IS_ALPHA, DEP, LEMMA, LOWER, IS_PUNCT, IS_DIGIT, IS_SPACE, IS_STOP
from spacy.tokens import Doc

from spacy_utils import remove_tokens_idx, get_childs_idx, is_conj_adv

from spacy.lang.de.examples import sentences 


from annotate_meter.ipa_hyphenate import hyphenate_ipa


regex_no_clean_ipa = r'[^aɐɑɒæɑʌbɓʙβcçɕɔɔdɗɖðdzdʒdʑɖʐeəɘɛɛɜfɸɡɠɢʛɣɤhħɦɧʜɥiĩɨɪɯɤijʝɟʄkkxlɫɬɭʟɮʎmɱɯɰnɲŋɳɴoõɵøɞœɶɔɔɤʊʘppfɸrɾɺɽɹɻʀʁrɐrɾɺɽɹɻʀʁrɐsʂʃsfɕtʈθtstʃtɕʈʂuũʉʊvvʋѵʌɣwwʍɰxχyʏɥʎɣɤzʑʐʒzvʡʔʢˈˌ,]'  # all but the accepted signs
vocal_sep = re.compile(r'(?<![ˈˌ^])[ɪiiʏyyʊuuieɛæɑɔouaəœ]')     #ɐ̯ is for reduced syllables; shwa seamingly counts
sec_stress_sep = re.compile(r'[ˌ]')


dirname = os.path.dirname(__file__)

otoi_path = os.path.join(dirname, 'annotate_meter/data/de_ortho_ipa.csv')
otoi_df = pd.read_csv(otoi_path)
otoi_df['word'] = otoi_df['word'].str.lower()


#m_path = os.path.join(dirname, 'ortho_to_ipa/model')

wiki_path = os.path.join(dirname, 'data_tools/wiktionary/wiktionary_data.csv')
wiktionary_df = pd.read_csv(wiki_path)

hyp_dic = pyphen.Pyphen(lang='de_DE')


otoi = ortho_to_ipa(load = True)
#####################################################################
stressed_list = ['NOUN','VERB','ADJ','ADV','PERSON']                   # removed PROPN since spacy declared unidentifiable tokens often as PROPN
unstressed_list = ['CCONJ','CONJ','DET','PART','CCONJ']
voca_list = ['ah','oh','a','o','u','uh','ach','nun']
#####################################################################

nlp = spacy.load("de_core_news_lg")

jambus = [0,1]


class verse_cl():

    '''
    stores a verse and it's metric and grammatical properties
    '''

    def __init__(self, text):
        if type(text) == list:
            self.text = text

        else:
            self.text = re.findall(r"[\w']+|[.,!?;:]", text)
        self.last_sign = ''

        self.token_pos = []
        self.update()
        #self.doc = nlp(self.text)
        #self.get_rythm_sent()       
        #self.update_token_dict()
        self.context = ''
        self.context_after = ''
        
        
    def shorten(self,idx):
        self.token_pos = self.token_pos[:idx]
        self.text = self.text[:idx]
        self.token_starts = self.token_starts[:idx]
        self.token_ends = self.token_ends[:idx]
        self.doc = self.doc[:idx]
        self.rythm_tokens = self.rythm_tokens[:idx]
        self.ipa = self.ipa[:idx]
        self.rythm = []
        dict_tmp = {}
        for i, key in enumerate(self.token_dict.keys()):
            dict_tmp[key] = self.token_dict[key]
            if i == idx - 1:
                break
        self.token_dict = dict_tmp
        for ryt in self.rythm_tokens:
            self.rythm += ryt


    def update_token_dict(self):
        offset = 0
        self.token_dict = {}
        self.token_starts = []
        self.token_ends = []
        for i, token in enumerate(self.rythm_tokens):
            self.token_starts.append(offset)
            for j,_ in enumerate(token):                                                       
                self.token_dict[j+ offset] = i 

            offset += len(token)
            self.token_ends.append(offset)


    def update_doc(self):
        self.doc = nlp(str(self.doc))

    def update(self, text = None):
        if text and type(text) == list:
            self.text = text

        if text and type(text) != list:
            self.text = re.findall(r"[\w']+|[.,!?;]", text)

        self.doc = nlp(' '.join(self.text))
        self.get_rythm_sent()
        self.update_token_dict()


    def get_rythm_sent(self): # get the rythm of a verse
        
        #stressed_list = ['NOUN','VERB','AUX','ADJ','PROPN','ADV','PERSON']
        #unstressed_list = ['CCONJ','CONJ','DET','PART','SCONJ','CCONJ']
        rythm = []
       
        doc = self.doc 
        rythm_tokens = []
        rythm = []
        ipa_lst = []
        for token in doc:
            self.token_pos.append(token.pos_)
            if token.text.isalpha():
                # ipa = ipa_from_ortho(token.text) 
                stress, ipa, _ = hyphenate_ipa(token.text)
                ipa_lst.append(ipa)
                #stress = list(get_rythm(token.text))
                #stress = list(get_rythm_ipa(token.text, ipa)) 
 
                
                try:
                    if stress == [0.5] and (token.pos_ in stressed_list):   # if the word contains meaning
                        stress = [1]
                except:
                    pass

                try:
                    if stress == [0.5] and (token.pos_ in unstressed_list):  # if the word contains no meaning
                        stress = [0]
                except:
                    pass

                '''try:
                    if token.text in voca_list:               # if the stress of the word is ambiguous
                        stress = [0.5]
                except:
                    pass'''
                rythm_tokens.append(stress)
                rythm += stress
            else:
                ipa_lst.append('')
                rythm_tokens.append([])

        self.rythm_tokens = rythm_tokens
        self.rythm = rythm
        self.ipa = ipa_lst


def clean_ipa(ipa_string):
    return re.sub(regex_no_clean_ipa,'',ipa_string)

def ipa_from_ortho(ortho):
    word_ortho = ortho.lower()
    word_ortho = re.sub(r'[^a-zäöüß]', '', word_ortho)
    try:
        ipa = (otoi_df.loc[otoi_df['word'] == ortho]['ipa']).values[0]
    except:
        ipa = otoi.translate(ortho)

    return ipa

def nearest_idx(arr, val):
    idx = (np.abs(arr-val)).argmin()
    return idx

def get_rythm(word_ortho):
    if type(word_ortho) == str:
        word_ortho = word_ortho.lower()
        word_ortho = re.sub(r'[^a-zäöüß]', '', word_ortho)
        word = ipa_from_ortho(word_ortho)     # convert the word into ipa symbols (if they are in the table, look them up, else do it with the neural net)
        rythm = get_rythm_ipa(word_ortho, word)
    else: 
        rythm = [2]
    return rythm

def get_start_idx (list_lists):
    starts = []
    start = 0
    for lst in list_lists: 
        starts.append(start)
        start += len(lst)

    return starts

def get_meter_difference(verse,target_rythm):
    rythm = verse.rythm

    rythm = np.asarray(rythm)
    target_rythm_ext = np.asarray(extend_target_rythm(rythm,target_rythm))

    comp = np.abs((target_rythm_ext-rythm) * (rythm != 0.5))

    return np.sum(comp)

def rate_candidate_meter(verse, target_rythm):
    rythm = verse.rythm
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

def hyphenate_word(word):

    syllabs_hyp = hyp_dic.inserted(word, ' ').split()        # hyphenate the word
    syllabs = []
    i = 0 
    while i < len(syllabs_hyp):

        if not (re.search("[aeiouäöüy]",syllabs_hyp[i])):
    
            if i < len(syllabs_hyp) - 1 and i > 0:
                if syllabs_hyp[i-1] > syllabs_hyp[i+1]:

                    syllabs.append(syllabs_hyp[i]+syllabs_hyp[i+1])
                    i += 2
                else: 
               
                    syllabs[i-1] += syllabs_hyp[i]
                    i += 1
            elif i == 0:
                syllabs.append(syllabs_hyp[i]+syllabs_hyp[i+1])
                i += 2
            else: 
                syllabs[i-1] += syllabs_hyp[i]
                i += 1
        else:
            syllabs.append(syllabs_hyp[i])
            i += 1
 
    return syllabs       

def get_rythm_ipa(word_ortho, word_ipa):        # get the rythm of a word
    #word_ortho = word_ortho.lower()
    #word_ortho = re.sub(r'[^a-zäöüß]', '', word_ortho)
    #word = ipa_from_ortho(word_ortho)     # convert the word into ipa symbols (if they are in the table, look them up, else do it with the neural net)
    word = clean_ipa(word_ipa)
    prim_stress = word.find("ˈ")
    if prim_stress == -1:
        return [0.5]
    
    else:
        sec_stress = [match.start(0) for match in re.finditer(sec_stress_sep, word)]

        syllabs = hyphenate_word(word_ortho)

        splits = [0]
        for syllab in syllabs[:-1]:
            splits.append(splits[-1] + len(syllab))

        splits = np.asarray(splits)
        rythm = np.zeros(len(splits))
        idx_prim_stress = nearest_idx(splits,prim_stress)    # approximate in which syllables the primary and secondary stress would be 
        rythm[idx_prim_stress] = 1
        for sec_idx in sec_stress:
            idx_sec_stress = nearest_idx(splits,sec_idx)
            rythm[idx_sec_stress] = 0.5

        return rythm

def rythm_comp_adaptive(rythm,target_rythms,adaptive = False): # flexible length
    match = False

    if type(target_rythms[0]) not in  [list, np.ndarray]:
        target_rythms = [target_rythms]

    if adaptive:
        factor = math.ceil((len(rythm)-len(target_rythms[0]))/2) + 1
        target_rythms = [(item*factor)[:len(rythm)] for item in target_rythms]
    
    for target_rythm in target_rythms:
        if len(rythm) == len(target_rythm):
     
            if np.sum(np.abs(np.asarray(rythm)-np.asarray(target_rythm))*(np.asarray(rythm) != 0.5)) == 0:
                match = True

    return match

def subset_sum(numbers, target, partial=[],partial_sum=0):
    # suggested by richard fern; stackoverflow question 4632322
    if partial_sum == target:
        yield partial
    if partial_sum >= target:
        return
    for i, n in enumerate(numbers):
        remaining = numbers[i+1:]
        yield from subset_sum(remaining, target, partial + [n],partial_sum + n)


def get_all_comb(value_lst,target):
    possible_combinations = []
    solutions = []

    all_comb = []
    value_arr = np.asarray(value_lst)
    solutions = []
    for solution in subset_sum(value_lst,target): 
       
        elem_cnt = dict(Counter(solution))

        if elem_cnt not in solutions:
            solutions.append(elem_cnt)
            arrays = []
            idx_lst = []
            offset = 0
            for number, count in elem_cnt.items():
            
                num_idx = np.where(value_arr == number)[0]
                
                all_num_comb = [list(item) for item in combinations(num_idx, count)]
                arrays += (all_num_comb)
                idx_lst.append(list(range(offset,offset+len(all_num_comb))))
                offset += len(all_num_comb)

            try:
                idx_comb = [list(item) for item in np.array(np.meshgrid(*idx_lst)).T.reshape(-1,len(idx_lst))]
                series = pd.Series(arrays)

                all_num_comb = [list(series[idx]) for idx in idx_comb]
                all_num_comb_tmp = []
                for comb in all_num_comb: 
                    all_num_comb_tmp.append([item for sublist in comb for item in sublist])
                    
                all_comb += all_num_comb_tmp
            except: 
                all_comb = []
    return all_comb


def get_single_comb(value_lst, amount, intersection, min_syll, toll = 5):        # this could be improved, the goal is an unperfect result with a minimum of computational time
    '''
    returns a single solution how a given sequence could be shortened to a given amount

    '''
    total_rm = 0
 
    rm_idx = []
    value_lst = np.asarray(value_lst)
    value_lst_new = value_lst
    removed = 0
    print(amount)
    for j in range(20):
        print(value_lst)
        best_idx = np.argmax(np.where(value_lst < amount, value_lst,0))
        print(best_idx)
        removed += value_lst[best_idx]
        value_lst[best_idx] = 1000

        rm_idx.append(best_idx)
        if best_idx in intersection.keys():
            for idx in intersection[best_idx]:
                value_lst[idx] = 1000
  
        if removed >= amount - toll:
            return rm_idx

    return rm_idx

def extend_target_rythm(rythm,target_rythm):
    factor = math.ceil((len(rythm)-len(target_rythm))/2) + 1
    return (target_rythm*factor)[:len(rythm)]

def get_best_shift(rythm,target_rythm, n=2,start = 0, end = None):
    target_rythm_ext = extend_target_rythm(rythm,target_rythm)
    rythm = np.asarray(rythm)
    diff_0 = 100
    best_shift = 0
    for i in range(n):
        target_rythm_tmp = target_rythm_ext[i:]+(target_rythm*math.ceil(i/len(target_rythm)))[:i]
        diff = np.sum(np.abs((np.asarray(target_rythm_tmp)-rythm) * (rythm != 0.5))[start:end])
        if diff < diff_0:
            best_shift = i
            diff_0 = diff

    return best_shift

def compare_verse_rythm(verse,target_rythm):
    target_rythm_ext = np.asarray(extend_target_rythm(verse.rythm,target_rythm))
    rythm = np.asarray(verse.rythm)
    return np.sum(np.abs((target_rythm_ext-rythm) * (rythm != 0.5)))



def get_best_comb(value_lst, target):
    #stackoverflow user trincot; question 44213144
    minCount = None
    result = None
    
    vals = [{'value':value,'count':value_lst.count(value)} for value in set(value_lst)] #modification

    def recurse(target, valIndex, valCount):
        nonlocal minCount
        if target == 0:
            
            if minCount == None or valCount < minCount:
                minCount = valCount
                return [] # success
            return None # not optimal
        if valIndex >= len(vals):
            return None # failure
        bestChange = None
        val = vals[valIndex]
        # Start by taking as many as possible from this val
        cantake = min(target // val["value"], val["count"])
        # Reduce the number taken from this val until 0
        for count in range(cantake, -1, -1):
            # Recurse, taking out this val as a possible choice
            change = recurse(target - val["value"] * count, valIndex + 1, 
                                                             valCount + count)

            # Do we have a solution that is better than the best so far?
            if change != None: 
                if count: # Does it involve this val?
                    change.append({ "value": val["value"], "count": count })
                bestChange = change # register this as the best so far
                
        return bestChange
 
    while not result:
        result = recurse(target,0,0)
        target -= 1
        
    idx_lst =  []
    for pair in result: 
        value = pair['value']
        count = pair['count']

        value_lst = np.asarray(value_lst)
        idx_lst += list(np.where(value_lst==value)[0][:count])
    return idx_lst




