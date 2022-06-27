from bert import bidirectional_synonyms

from gpt_poet import gpt_synonyms
from sia_rhyme.siamese_rhyme import siamese_rhyme
from rythm_utils import verse_cl
import numpy as np 
from rhyme_detection.word_spectral import wordspectrum
from rhyme_detection.utils import check_rhyme
from rhyme_detection.colone_phonetics import compare_words, compare_last_vowels, clean_word
import re
rhyme_model = siamese_rhyme()

def get_last(word_lst_inp):
    word_lst = word_lst_inp.copy()
    word_lst.reverse()
    for word in word_lst:
            if len(clean_word(word)) > 1:
                return word

    return None

def get_last_idx(word_lst_inp):
    word_lst = word_lst_inp.copy()
    word_lst.reverse()
    for i, word in enumerate(word_lst):
            if len(clean_word(word)) > 1:
                return len(word_lst) - i -1

    return None

def find_rhyme(verse_lst,idx1,idx2,target_rythm,last_stress = -2, detection_method ='neural',LLM='GPT2',use_tts = False,return_alternatives=False):

    print('--- looking for rhymes ---')
    print('using ' + str(LLM))
    print(' '.join(verse_lst[idx1].text))
    print(' '.join(verse_lst[idx2].text))
    print('--------------------------')

    context_aft = ''
    for i in range(idx1+1,idx2+1):
        context_aft += ' '.join(verse_lst[i].text) + '\n'

    bi_syns = bidirectional_synonyms(verse_lst[idx1],context_aft, target_rythm)

    if verse_lst[idx1].text[-1].isalpha():
        last = -1
    else:
        last = -2 

    bi_trunk = ' '.join(verse_lst[idx1].text[:last+1])

    if idx2 == len(verse_lst) - 1:
        causal = True
        causal_syns = gpt_synonyms(verse_lst[idx2],target_rythm,num_remove = 1,LLM=LLM)
        causal_syns += gpt_synonyms(verse_lst[idx2],target_rythm,LLM=LLM)[1:]        

        if len(causal_syns) < 20:
            causal_syns = gpt_synonyms(verse_lst[idx2],target_rythm,num_remove = 3,LLM=LLM)

    else:
        causal = False
        context_aft = ''
        for i in range(idx2,len(verse_lst)):
            context_aft += ' '.join(verse_lst[i].text) + '\n'

        syns_tmp = bidirectional_synonyms(verse_lst[idx1],context_aft, target_rythm)

        causal_syns = []

        last_idx = get_last_idx(verse_lst[idx2].text)

        verse_trunk = ' '.join(verse_lst[idx2].text[:last_idx])
        for syn in syns_tmp:
            causal_syns.append(verse_trunk + ' ' + ' '.join(syn))



    if causal: 
        print('number of found alternatives')
        print('causal:')
        print(len(causal_syns))
        print('bidirectional:')
        print(len(bi_syns))
        print('-----------------------------')

    else: 
        print('number of found alternatives')
        print('bidirectional first:')
        print(len(bi_syns))
        print('bidirectional second:')
        print(len(causal_syns))       
        print('-----------------------------')

    found = False

    differences = []
    sent_pairs = []
    word_pairs = []


    for sent_2 in causal_syns:
        for sent_1 in bi_syns:
            word_1 = get_last(sent_1)

            sent_2_split = sent_2.split()
            word_2 =  get_last(sent_2_split)
            if word_1 and word_2:
                word_pairs.append([word_1,word_2])              # [bidirectional, causal]
                sent_pairs.append([' '.join(sent_1), sent_2])

    if not word_pairs:      
        print('found no pair')
        print(word_1)
        print(word_2)
        print(causal_syns)
        print(bi_syns)


    for word_pair in word_pairs:
            word_1 = word_pair[0]
            word_2 = word_pair[1]

            if word_1 != word_2:
                difference =  compare_words(word_1,word_2, last_stressed = last_stress)
            else:
                difference = 50
            differences.append(difference)
            #idx.append([' '.join(sent_1),sent_2])

    if np.amin(np.asarray(differences)) < 1:
        best_idx = np.argmin(np.asarray(differences))
        bi_selection = sent_pairs[best_idx][0] #pairs[best_idx][0]
        causal_selection = sent_pairs[best_idx][1] #pairs[best_idx][1]
        found = True
        print('found via colone phonetics')

    '''if not found: 
        for idx, word_pair in enumerate(word_pairs):
            word_1 = word_pair[0]
            word_2 = word_pair[1]

            if word_1 != word_2:
   
                if compare_last_vowels(word_1,word_2):
                    bi_selection = sent_pairs[idx][0]
                    causal_selection = sent_pairs[idx][1]
                    found = True
            
                    break'''

    if found: 
        print('found via vowels or colone phonetics:')

        print(bi_selection)
        print(causal_selection)


    '''if not found:
        for word_pair in word_pairs:
            word_1 = word_pair[0]
            word_2 = word_pair[1]

            if word_1 != word_2:
                difference =  compare_words(word_1,word_2, last_stressed = last_stress)
            else:
                difference = 50
            differences.append(difference)

            #difference =  compare_words(word_1,word_2, last_stressed = -1)
            #differences.append(difference)
            #pairs.append([' '.join(sent_1),sent_2])


        if np.amin(np.asarray(differences)) < 1:
            best_idx = np.argmin(np.asarray(differences))
            bi_selection = sent_pairs[best_idx][0] #pairs[best_idx][0]
            causal_selection = sent_pairs[best_idx][1]
            found = True
            print('found via colone phonetics round 2')'''


    

    '''if not found:
        
        for idx, word_pair in enumerate(word_pairs):
            word_1 = word_pair[0]
            word_2 = word_pair[1]
            word_1 = re.sub('ch', '2', word_1.lower())
            word_2 = re.sub('ch', '2', word_2.lower())
            word_1 = re.sub('sch', '3', word_1.lower())
            word_2 = re.sub('sch', '3', word_2.lower())

            if word_1 != word_2 and word_1[-2:] == word_2[-2:]:
                bi_selection = sent_pairs[idx][0]
                causal_selection = sent_pairs[idx][1]
                print('matching last two letters')
                found = True
                break'''

        

    if not found and len(causal_syns)*len(bi_syns) < 10:            # leave it as it is; unprobable to find a rhyme
        print('rhyme not found')
        found = True
        bi_selection = ' '.join(bi_syns[-1])
        causal_selection = causal_syns[0]     

    if  True:#not found:
        vector_pairs = []
        for word_pair in word_pairs:

            vector_pairs.append([rhyme_model.get_word_vec(word_pair[0]),rhyme_model.get_word_vec(word_pair[1])])
            #causal_vecs.append(rhyme_model.get_word_vec(word_pair[0]))

        distances = []
        
        for vector_pair in vector_pairs:   
            distance = rhyme_model.vector_distance(vector_pair[0],vector_pair[1])
                                                                    
            distances.append(distance[0])

        distances = np.asarray(distances)

        candidate_idx = np.argsort(distances)[:200]

        if use_tts: 
            print(tts)
            spectral_diffs = []

            for idx in candidate_idx:
                idx = idx
                word_1 = word_pairs[idx][0]
                word_2 = word_pairs[idx][1]

                if word_1 != word_2:
                
                    try:
                        spec_1 = wordspectrum(word_1)
                        spec_2 = wordspectrum(word_2)
                        mean, _ = check_rhyme(spec_1,spec_2,
                                                            features = 'mfccs',
                                                            order=0,
                                                            length = 19, 
                                                            cut_off = 1,
                                                            min_matches=10,
                                                            pool=0)
                    except: 
                        mean = 100
                        print('failed to create spectrum')
                        print(word_1)
                        print(word_2)
                

                else: mean = 1000
            

                spectral_diffs.append(mean)

            best_idx = candidate_idx[np.argmin(np.asarray(spectral_diffs))]  

        else:
            candidates = np.argsort(distances)

            for candidate in candidates: 
                if distances[candidate] > 0:
                    best_idx = candidate
                    break
            
                
        print('found via sia rhyme')
        print(sent_pairs[best_idx][0])
        print(sent_pairs[best_idx][1])
        print('distance: ' + str(distances[best_idx]))
        
        if not found: 
            bi_selection = sent_pairs[best_idx][0]
            causal_selection = sent_pairs[best_idx][1]
        
              

    print('final choice:')
    print(' '.join(verse_lst[idx1].text[:last]) + ' ' + bi_selection)
    print(causal_selection)
    
    verse_lst[idx1] = verse_cl(' '.join(verse_lst[idx1].text[:last]) + ' ' + bi_selection)
    verse_lst[idx2] = verse_cl(causal_selection)
    
    if return_alternatives == False: 
        return verse_lst

    else:

        bi_syns = [' '.join(syn) for syn in bi_syns]
        
        bi_syns = [bi_trunk + ' ' + syn for syn in bi_syns]

        return verse_lst, bi_syns, causal_syns