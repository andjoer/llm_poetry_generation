import re
import numpy as np 
from itertools import groupby
import pyphen

hyp_dic = pyphen.Pyphen(lang='de_DE')

def colone_phonetics(tokens):
    colone_phonetics = []
    tokens = tokens.lower()
    subst_list = []
    tokens = re.sub('[ä]', 'a', tokens) 
    tokens = re.sub('[ö]', 'o', tokens) 
    tokens = re.sub('[ü]', 'u', tokens) 
    tokens = re.sub('[ß]', 's', tokens) 
    tokens = re.sub('[^a-z]', '', tokens)
    subst_list.append(re.sub('[a,e,i,j,o,u,y]', '0.', tokens) )
    subst_list.append(re.sub('[b]', '1.', tokens) )
    subst_list.append(re.sub(r'[p](?![h])', '1.', tokens) )
    subst_list.append(re.sub(r'[p](?=[h])', '3.', tokens) )
    subst_list.append(re.sub(r'[d,t](?![c,s,z])', '2.', tokens) )
    subst_list.append(re.sub('[f,v,w]', '3.', tokens) )
    subst_list.append(re.sub(r'[p](=![h])', '3.', tokens) )
    subst_list.append(re.sub('[g,k,q]', '4.', tokens) )
    c_one = re.sub(r"(?<![s,z])[c](?=[a,h,k,o,q,u,x])", '4.', tokens) #no wordbeginning
    subst_list.append(re.sub(r'(?<![a-z])[c](?=[l])', '4.', c_one)  ) #beginning of word (some c are already replaced, but does not matter)
    subst_list.append(re.sub(r'(?<![c,k,q])[x]', ' 48. ', tokens))
    subst_list.append(re.sub(r'[l]', '5.', tokens))
    subst_list.append(re.sub(r'[m,n]', '6.', tokens))
    subst_list.append(re.sub(r'[r]', '7.', tokens))
    subst_list.append(re.sub(r'[h]', '64.', tokens))
    subst_list.append(re.sub(r'[s,z]', '8.', tokens))
    c_one = re.sub(r'(?<=[s,z])[c](?![a,h,k,o,q,u,x])', '8.', tokens) #no word beginning
    subst_list.append(re.sub(r'(?<![a-z])[c](?![a,h,k,l,o,q,r,u,x])', '8.', c_one))#word beginning
    subst_list.append(re.sub(r'[d,t](?=[c,s,z])', '8.', tokens) )
    subst_list.append(re.sub(r'(?<=[c,k,q])x', '8.', tokens) )

    for i in range(len(subst_list)):
        subst_list[i] = (re.sub(r'[a-z]','0.',subst_list[i])).split('.')[:-1]

    try:
        numerical_list = np.amax(np.asarray(subst_list).astype(int),axis=0).tolist()
    except:
        print(subst_list)
        print(tokens)
        raise(Exception)
    numerical_list = [x[0] for x in groupby(numerical_list) if len(x) > 0]
    try: 
        numerical_list.remove(64)
    except: 
        pass
        
    return numerical_list

def clean_word(word):
    return re.sub('[^a-zäöüß]', '', word.lower())
        
def get_last_two_vowels(word):
    
    #word = re.sub('[ä]', 'e', word) 
    #word = re.sub('[ö]', 'o', word) 
    #word = re.sub('[ü]', 'u', word) 
    word = re.sub('[y]', 'i', word) 
    word = re.sub('ie', 'i', word)
    word = re.sub('ae', 'a', word)
    word = re.sub('oe', 'o', word)
    word = re.sub('ue', 'u', word) 
    
    vowels = [x for x in re.split(r'[^aeiouyäöü]',word) if x]
    try:
        if vowels[-1] == word[-1]:
            last = True

        else: last = False
    except:
        last = False
        print('vowel error, word:')
        print(word)
    
    vowels = vowels[-2:]

    if len(vowels) < 2:
        vowels = None
    
    return vowels,last
    



def compare_words(word_1,word_2,last_stressed=-2):
    word_1 = clean_word(word_1)
    word_2 = clean_word(word_2)

    syllabs_1 = ''.join(hyp_dic.inserted(word_1, ' ').split()[last_stressed:])
    syllabs_2 = ''.join(hyp_dic.inserted(word_2, ' ').split()[last_stressed:])
    if len(syllabs_1) == len(syllabs_2):
        colone_1 = colone_phonetics(syllabs_1)
        colone_2 = colone_phonetics(syllabs_2)


        length = min(len(colone_1),len(colone_2))

        diff = np.sum(np.abs(np.asarray(colone_1[-length:])-np.asarray(colone_2[-length:])))

    else:
        diff = 50

    if word_1 == word_2:
        diff = 50

    
    return diff


def compare_last_vowels(word_1,word_2):
    word_1 = clean_word(word_1)
    word_2 = clean_word(word_2)

    vowels_1,last_1 = get_last_two_vowels(word_1)
    vowels_2,last_2 = get_last_two_vowels(word_2)
    
    if vowels_1 == vowels_2 and last_1 == last_2 and vowels_1 and word_1 != word_2:
        diff = 0
        print('matching vowels')

        return True

    else:
        return False
