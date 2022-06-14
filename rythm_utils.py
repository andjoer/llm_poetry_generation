import numpy as np
from itertools import combinations
from collections import Counter
import pandas as pd
import re
import pyphen
import os
from ortho_to_ipa.ortho_to_ipa import ortho_to_ipa
import math

import spacy
from spacy.attrs import LOWER, POS, ENT_TYPE, IS_ALPHA, DEP, LEMMA, LOWER, IS_PUNCT, IS_DIGIT, IS_SPACE, IS_STOP
from spacy.tokens import Doc

from spacy_utils import remove_tokens_idx, get_childs_idx, is_conj_adv

from spacy.lang.de.examples import sentences 

regex_no_clean_ipa = r'[^aɐɑɒæɑʌbɓʙβcçɕɔɔdɗɖðdzdʒdʑɖʐeəɘɛɛɜfɸɡɠɢʛɣɤhħɦɧʜɥiĩɨɪɯɤijʝɟʄkkxlɫɬɭʟɮʎmɱɯɰnɲŋɳɴoõɵøɞœɶɔɔɤʊʘppfɸrɾɺɽɹɻʀʁrɐrɾɺɽɹɻʀʁrɐsʂʃsfɕtʈθtstʃtɕʈʂuũʉʊvvʋѵʌɣwwʍɰxχyʏɥʎɣɤzʑʐʒzvʡʔʢˈˌ,]'  # all but the accepted signs
vocal_sep = re.compile(r'(?<![ˈˌ^])[ɪiiʏyyʊuuieɛæɑɔouaəœ]')     #ɐ̯ is for reduced syllables; shwa seamingly counts
sec_stress_sep = re.compile(r'[ˌ]')

otoi_df = pd.read_csv('ortho_to_ipa/data/de_ortho_ipa.csv')
otoi_df['word'] = otoi_df['word'].str.lower()

dirname = os.path.dirname(__file__)
m_path = os.path.join(dirname, 'ortho_to_ipa/model')

wiki_path = os.path.join(dirname, 'data_tools/wiktionary/wiktionary_data.csv')
wiktionary_df = pd.read_csv(wiki_path)

hyp_dic = pyphen.Pyphen(lang='de_DE')


otoi = ortho_to_ipa(load = True,fname_ortho= m_path +'/ortho.pt',fname_ipa=m_path +'/ipa.pt',fname_model=m_path +'/ortho_to_ipa.pt')
#####################################################################
stressed_list = ['NOUN','VERB','ADJ','PROPN','ADV','PERSON']
unstressed_list = ['CCONJ','CONJ','DET','PART','CCONJ']
voca_list = ['ah','oh','a','o','u','uh','ach','nun']
#####################################################################

nlp = spacy.load("de_core_news_lg")

jambus = [0,1]


class verse_cl():
    def __init__(self, text):
        if type(text) == list:
            self.text = text

        else:
            self.text = re.findall(r"[\w']+|[.,!?;]", text)
        self.last_sign = ''
        self.update()
        #self.doc = nlp(self.text)
        #self.get_rythm_sent()       
        #self.update_token_dict()
        self.context = ''
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


    def get_rythm_sent(self):
        
        #stressed_list = ['NOUN','VERB','AUX','ADJ','PROPN','ADV','PERSON']
        #unstressed_list = ['CCONJ','CONJ','DET','PART','SCONJ','CCONJ']
        rythm = []
       
        doc = self.doc 
        rythm_tokens = []
        rythm = []
        for token in doc:
            if token.text.isalpha():
                stress = list(get_rythm(token.text))
                try:
                    if stress == [0.5] and (token.pos_ in stressed_list):
                        stress = [1]
                except:
                    pass

                try:
                    if stress == [0.5] and (token.pos_ in unstressed_list):
                        stress = [0]
                except:
                    pass

                try:
                    if token.text in voca_list:
                        stress = [0.5]
                except:
                    pass
                rythm_tokens.append(stress)
                rythm += stress
            else:
                rythm_tokens.append([])

        self.rythm_tokens = rythm_tokens
        self.rythm = rythm


def clean_ipa(ipa_string):
    return re.sub(regex_no_clean_ipa,'',ipa_string)

def ipa_from_ortho(ortho):
    try:
        ipa = (otoi_df.loc[otoi_df['word'] == ortho]['ipa']).values[0]
    except:
        ipa = otoi.translate(ortho)

    return ipa

def nearest_idx(arr, val):
    idx = (np.abs(arr-val)).argmin()
    return idx


def get_rythm(word_ortho):
    word_ortho = word_ortho.lower()
    word_ortho = re.sub(r'[^a-zäöüß]', '', word_ortho)
    word = ipa_from_ortho(word_ortho)
    word = clean_ipa(word)
    prim_stress = word.find("ˈ")
    if prim_stress == -1:
        return [0.5]
    
    else:
        sec_stress = [match.start(0) for match in re.finditer(sec_stress_sep, word)]

        syllabs = hyp_dic.inserted(word_ortho, ' ').split()



        splits = [0]
        for syllab in syllabs[:-1]:
            splits.append(splits[-1] + len(syllab))

        splits = np.asarray(splits)
        rythm = np.zeros(len(splits))
        idx_prim_stress = nearest_idx(splits,prim_stress)    
        rythm[idx_prim_stress] = 1
        for sec_idx in sec_stress:
            idx_sec_stress = nearest_idx(splits,sec_idx)
            rythm[idx_sec_stress] = 0.5

        return rythm

def rythm_comp_adaptive(rythm,target_rythms,adaptive = False):
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
    # richard fern; stackoverflow question 4632322
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

'''def get_single_comb_rec(value_dict, amount, idx = 0,toll = 10):                        #
    if amount <= toll:
        return {} 
    if idx >= len(value_dict):
        return None 
   
    value = list(value_dict)[idx]
    count = list(value_dict.values())[idx]
    idx += 1
    
    canTake = min(amount // value, count)
  
    for count_ in range(canTake, -1, -1): 
        rest = get_single_comb_rec(value_dict, amount - value * count_, idx)
        if rest != None: 
            if count_: 
                rest[value] = count
                return rest
            return rest
'''

def get_single_comb(value_lst, amount, intersection, min_syll, toll = 5):        # this could be improved, the goal is an unperfect result with a minimum of computational time

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







ipa_groups = {                                                                                  # it are not the correct signs, but the cleaned version, correct would be:
                'plosives':'p,b,p,b,t,d,t,d,t,d,t,d,t,d,t,d,t,d,ʈ,ɖ,c,ɟ,k,ɢ,ʡ,ʔ',               # p,⁠b⁠,p̪⁠,b̪,⁠t̼⁠⁠,d̼,⁠t̟,d̟⁠⁠⁠,⁠t̺⁠,d̺,⁠t̪⁠,⁠d̪⁠,⁠t,d⁠,⁠t̻⁠,d̻⁠,t̠⁠,⁠d̠⁠,ṭ⁠,ḍ⁠,⁠ʈ⁠,ɖ⁠,⁠c,⁠ɟ⁠,⁠k,⁠g⁠,⁠q⁠,⁠ɢ⁠,⁠ʡ,ʔ⁠
                'fricatives': 'ɸ,β,f,v,θ,ð,s,z,ɬ,ɮ,ʃ,ʒ,ʂ,ʐ,ɕ,ʑ,ç,ʝ,x,ɣ,ʍ,ɧ,χ,ʁ,ħ,ʜ,ʢ,h,ɦ,s,f',  # ɸ⁠,β⁠,⁠⁠f⁠,⁠v⁠,θ⁠,⁠ð⁠,⁠s,z⁠,⁠ɬ⁠,⁠ɮ,ʃ⁠,ʒ⁠,⁠ʂ,⁠ʐ⁠,⁠ɕ,⁠ʑ⁠,⁠ç⁠,⁠ʝ⁠,⁠x,⁠ɣ,ʍ⁠,ɧ⁠,χ⁠,ʁ,ħ⁠⁠,⁠ʕ⁠,⁠ʜ⁠,⁠ʢ⁠,h⁠,ɦ,⁠s’,fʼ⁠
                'nasals':'m,m,ɱ,ɱ,n,n,ɳ,ɳ,ɲ,ɲ,ŋ,ŋ,ɴ,ɴ⁠',                                         # ⁠m,m̥⁠⁠,ɱ,ɱ̊⁠,⁠⁠n⁠,n̥⁠,⁠ɳ,⁠ɳ̊,ɲ⁠,ɲ̊⁠,ŋ⁠,ŋ̊⁠,ɴ⁠,⁠ɴ̥⁠
                'liquides':'ə,l,r', 
                'approximates':'j,w',
                'vowels_closed':'ɪ,i,i,ʏ,y,y,ʊ,u,u',                                            #ɪ,i,iː,ʏ,y,yː,ʊ,u,uː
                'vowels':'i,e,ɛ,æ,ɑ,ɔ,o,u,a,a,ɐ'                                                #i,e,ɛ,æ,ɑ,ɔ,o,u,a,aː,ɐ
            }