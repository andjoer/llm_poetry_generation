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
            clean_word_out = clean_word(word)
            if len(clean_word_out) > 1:
                return clean_word_out

    return None

def get_last_idx(word_lst_inp):
    word_lst = word_lst_inp.copy()
    word_lst.reverse()
    for i, word in enumerate(word_lst):
            if len(clean_word(word)) > 1:
                return len(word_lst) - i -1

    return None

def get_spectral_diff(word_1, word_2):
    spec_1 = wordspectrum(word_1)                            # calculate the mfcc features for each word
    spec_2 = wordspectrum(word_2)
    mean, _ = check_rhyme(spec_1,spec_2,
                                        features = 'mfccs',
                                        order=0,
                                        length = 19, 
                                        cut_off = 1,
                                        min_matches=10,
                                        pool=0)
    return mean

def find_rhyme_last_verse(args,verse_lst,idx1,idx2,first_syn,LLM_perplexity,last_stress = -2, top_k = 25,LLM='', LLM2 = None, return_alternatives=False,force_rhyme=False):

    max_rhyme_dist = args.max_rhyme_dist
    use_colone = args.use_colone_phonetics
    use_tts = args.use_tts
    target_rythm = args.target_rythm
    top_p_dict_rhyme = args.top_p_dict_rhyme
    top_p_rhyme = args.top_p_rhyme
    stop_tokens = args.rhyme_stop_tokens
    rhyme_temperature = args.rhyme_temperature
    allow_pos_match = args.allow_pos_match
    invalid_verse_ends = args.invalid_verse_ends
    repetition_penalty = args.repetition_penalty

    min_dist = 0.01
    

    eol = True
    use_pos = False


    if type(LLM) != str:
        if LLM.sampling == 'systematic':
            sampling = 'systematic'
        else: 
            sampling = 'multinomial'
    else: 
        sampling = 'multinomial'

    if not verse_lst[idx2].text[-1].isalpha():
        sign_2 = verse_lst[idx2].text[-1]
    else: 
        sign_2 = ''

    context = ''
    for verse in verse_lst[:idx2]:
    
        new_text = re.sub(r'\s([,.!?;:](?:\s|$))', r'\1', ' '.join(verse.text))
        context += new_text + '\n'

    verse_lst[idx2].context = context

    print('context')
    print(context)

    if idx2 == len(verse_lst) - 1:
        causal = True
        causal_syns = gpt_synonyms(args,verse_lst[idx2],target_rythm,num_remove = 1,num_return_sequences = 200,LLM=LLM,eol=eol,use_pos = use_pos,top_k = top_k, top_p_dict =top_p_dict_rhyme ,temperature=rhyme_temperature,
                        stop_tokens=stop_tokens,allow_pos_match=allow_pos_match,invalid_verse_ends=invalid_verse_ends,repetition_penalty=repetition_penalty,replace_linebreaks=args.replace_linebreaks)
        
        causal_syns += gpt_synonyms(args,verse_lst[idx2],target_rythm,num_remove=2,num_return_sequences = 150,LLM=LLM,top_k = top_k,eol=eol,use_pos = use_pos,stop_tokens=stop_tokens,temperature=rhyme_temperature,top_p = top_p_rhyme,
                        allow_pos_match=allow_pos_match,invalid_verse_ends=invalid_verse_ends,repetition_penalty=repetition_penalty,replace_linebreaks=args.replace_linebreaks)[1:]        

        if len(causal_syns) < 10: 
            causal_syns = gpt_synonyms(args,verse_lst[idx2],target_rythm,num_remove = 3,num_return_sequences = 140,top_k = top_k,LLM=LLM,eol=eol,use_pos = use_pos,temperature=rhyme_temperature,stop_tokens=stop_tokens,top_p = top_p_rhyme,
            allow_pos_match=allow_pos_match,invalid_verse_ends=invalid_verse_ends,repetition_penalty=repetition_penalty,replace_linebreaks=args.replace_linebreaks)[1:]       # alternatives for the second verse

        if sampling == 'systematic':           # try as well with matching pos labels instead of correct prediction of new line
            eol = False
            use_pos = True
            causal_syns += gpt_synonyms(args,verse_lst[idx2],target_rythm,num_remove = 1,num_return_sequences = 200,LLM=LLM,top_k = top_k,eol=eol,use_pos = use_pos,temperature=rhyme_temperature,top_p_dict =top_p_dict_rhyme,
                        invalid_verse_ends=invalid_verse_ends,repetition_penalty=repetition_penalty,replace_linebreaks=args.replace_linebreaks)
        
            causal_syns += gpt_synonyms(args,verse_lst[idx2],target_rythm,num_remove=2,num_return_sequences = 150,LLM=LLM,top_k = top_k,eol=eol,use_pos = use_pos,temperature=rhyme_temperature,top_p = top_p_rhyme ,
                        invalid_verse_ends=invalid_verse_ends,repetition_penalty=repetition_penalty,replace_linebreaks=args.replace_linebreaks)[1:]        


    else:
        causal = False
        context_aft = ''
        for i in range(idx2+1,len(verse_lst)):
            context_aft += ' '.join(verse_lst[i].text) + '\n'

        syns_tmp = bidirectional_synonyms(args,verse_lst[idx2],context_aft, target_rythm,LLM_perplexity)

        causal_syns = []

        last_idx = get_last_idx(verse_lst[idx2].text)

        verse_trunk = ' '.join(verse_lst[idx2].text[:last_idx])
        for syn in syns_tmp:
            causal_syns.append(verse_trunk + ' ' + ' '.join(syn))


    print('number of found alternatives second verse')
    if causal: 
        
        print('causal:')
        print(len(causal_syns))
        print('-----------------------------')

    else: 
        print('bidirectional second:')
        print(len(causal_syns))       
        print('-----------------------------')

    found = False

    differences = []
    sent_pairs = []
    word_pairs = []


    for sent_2 in causal_syns:
        word_1 = first_syn

        sent_2_split = sent_2.split()
        word_2 =  get_last(sent_2_split)
        if word_1 and word_2:
            word_pairs.append([word_1,word_2])              # [bidirectional, causal]
            sent_pairs.append([' '.join(verse_lst[idx1].text), sent_2])

    if not word_pairs:      
        print('found no pair')
        if return_alternatives == False: 
            return verse_lst
        else:
            return verse_lst, [], []


    for word_pair in word_pairs:            # compare with colone phonetics
            word_1 = word_pair[0]
            word_2 = word_pair[1]

            if word_1 != word_2:
                difference =  compare_words(word_1,word_2, last_stressed = last_stress)
            else:
                difference = 50
            differences.append(difference)
            #idx.append([' '.join(sent_1),sent_2])

    if np.amin(np.asarray(differences)) < 1:           # a match in colone phonetics is only valid below a distance of 1
        best_idx = np.argmin(np.asarray(differences))                    # best match according to colone phonetics
        bi_selection = sent_pairs[best_idx][0] #pairs[best_idx][0]
        causal_selection = sent_pairs[best_idx][1] #pairs[best_idx][1]
        found = True
        print('found via colone phonetics')

    if not found and args.rhyme_last_two_vowels: 
        for idx, word_pair in enumerate(word_pairs):
            word_1 = word_pair[0]
            word_2 = word_pair[1]

            if word_1 != word_2:
   
                if compare_last_vowels(word_1,word_2):
                    bi_selection = sent_pairs[idx][0]
                    causal_selection = sent_pairs[idx][1]
                    found = True
            
                    break

    if found: 
        print('found via vowels or colone phonetics:')

        print(bi_selection)
        print(causal_selection)


    if  not found or not use_colone: # use the sia rhyme apporach
        spectral_diffs = None
        vector_pairs = []
        for word_pair in word_pairs:

            vector_pairs.append([rhyme_model.get_word_vec(word_pair[0]),rhyme_model.get_word_vec(word_pair[1])]) # vectorize the words
            #causal_vecs.append(rhyme_model.get_word_vec(word_pair[0]))

        distances = []
        
        for vector_pair in vector_pairs:   
            distance = rhyme_model.vector_distance(vector_pair[0],vector_pair[1])
                                                                    
            distances.append(distance[0])

        distances = np.asarray(distances) # distances between each possible combination

        candidate_idx = np.argsort(distances)[:args.size_tts_sample]
        if use_tts and np.amin(distances) <= max_rhyme_dist: 
            print('using tts')
            spectral_diffs = []

            for idx in candidate_idx:
                idx = idx
                word_1 = word_pairs[idx][0]
                word_2 = word_pairs[idx][1]

                if word_1.lower() != word_2.lower():
                
                    try:
                        mean = get_spectral_diff(word_1, word_2)
                    except: 
                        mean = 100
                        print('failed to create spectrum')
                        print(word_1)
                        print(word_2)
                

                else: mean = 1000
            

                spectral_diffs.append(mean)                 # calculate the distances between the mfcc vectors for each word

            print('spectral distance: ' + str(min(spectral_diffs)))
            best_idx = candidate_idx[np.argmin(np.asarray(spectral_diffs))]    # choose the pair with the lowest distance

        else:
            candidates = np.argsort(distances)
            best_idx = ''
            for candidate in candidates: 
                if distances[candidate] > min_dist:
                    best_idx = candidate                       # if no mffc features are used, use the minimum distance in the vectorspace of sia rhyme
                    break
            
        if best_idx and np.amin(distances) <= max_rhyme_dist:
                
            print('found via sia rhyme')
            print(sent_pairs[best_idx][0])
            print(sent_pairs[best_idx][1])
            print('distance: ' + str(distances[best_idx]))
        
            bi_selection = sent_pairs[best_idx][0]
            causal_selection = sent_pairs[best_idx][1]


            found = True

    if use_tts and found and spectral_diffs is not None:
        best_distance = np.amin(np.asarray(spectral_diffs))
    elif use_tts:
        best_distance = 100
    else: 
        best_distance = distances[best_idx]

    if found:   
        if causal_selection[-1].isalpha():
            causal_selection += sign_2

        print('final choice:')
        print(' '.join(verse_lst[idx1].text))
        print(causal_selection)                                                                    #otherwise don't change the verses 
        verse_lst[idx2] = verse_cl(causal_selection)


    if return_alternatives == False: 
        return verse_lst, best_distance, [], []

    else:

        causal_syns = []
        bi_syns = []
        idx = 0
        candidates = np.argsort(distances)
        items = 0
        while items < 50 and idx < len(candidates):
            candidate = candidates[idx]
            idx += 1
            if distances[candidate] > min_dist:
                items += 1
                bi_selection = sent_pairs[candidate][0]
                causal_selection = sent_pairs[candidate][1]
                causal_syns.append(causal_selection)


        return verse_lst, best_distance, bi_syns, causal_syns


def find_rhyme(args,verse_lst,idx1,idx2,LLM_perplexity,last_stress = -2, LLM='', LLM2 = None, return_alternatives=False,force_rhyme=False):
    
    '''
    finds rhyming endings for two verses 
    '''

    max_rhyme_dist = args.max_rhyme_dist
    use_colone = args.use_colone_phonetics
    use_tts = args.use_tts
    target_rythm = args.target_rythm
    top_p_dict_rhyme = args.top_p_dict_rhyme
    top_p_rhyme = args.top_p_rhyme
    stop_tokens = args.rhyme_stop_tokens
    rhyme_temperature = args.rhyme_temperature
    allow_pos_match = args.allow_pos_match
    invalid_verse_ends = args.invalid_verse_ends
    repetition_penalty = args.repetition_penalty
    top_k = args.top_k_rhyme

    min_dist = 0.01
    

    eol = True
    use_pos = False

    if type (LLM) != str:
            LLM_name = LLM.model_name
    else:
        LLM_name = LLM
    if type(LLM) != str:
        if LLM.sampling == 'systematic':
            sampling = 'systematic'
        else: 
            sampling = 'multinomial'
    else: 
        sampling = 'multinomial'


    print('--- looking for rhymes ---')
    print('using ' + str(LLM_name))
    print('sampling ' + str(sampling))
    print(' '.join(verse_lst[idx1].text))
    print(' '.join(verse_lst[idx2].text))
    print('--------------------------')

    if not verse_lst[idx1].text[-1].isalpha():
        sign_1 = verse_lst[idx1].text[-1]
    else:
        sign_1 = ''


    
    context_aft = ''
    for i in range(idx1+1,idx2+1):
        context_aft += ' '.join(verse_lst[i].text) + '\n'

    bi_syns = bidirectional_synonyms(args,verse_lst[idx1],context_aft, target_rythm,LLM_perplexity) # alternatives for the first verse

    print('number of found alternatives first verse')
    print('bidirectional:')
    print(len(bi_syns))
    print('-----------------------------')

    if verse_lst[idx1].text[-1].isalpha():
        last = -1
    else:
        last = -2 

    found = False
    
  
    min_idx = 0
    

    bi_syn_lst = []
    causal_syn_lst = []
    batch_size = 5

    while not found and min_idx < len(bi_syns):

        candidates = []
        distances = []
        

        for syn in bi_syns[min_idx:min_idx+batch_size]:
            verse_lst_tmp = verse_lst.copy()
            if type(syn) != list:
                syn = [syn]

            verse_lst_tmp[idx1] = verse_cl(' '.join(verse_lst[idx1].text[:last]) + ' ' + ' '.join(syn) + sign_1)
            verse_lst_tmp, distance, bi_syn, causal_syns = find_rhyme_last_verse(args,verse_lst_tmp,idx1,idx2, syn[-1],LLM_perplexity,last_stress = last_stress, LLM=LLM, LLM2 = LLM2, 
                                                                                top_k = top_k, return_alternatives=return_alternatives,force_rhyme=force_rhyme)
            candidates.append(verse_lst_tmp)

            distances.append(distance)
            bi_syn_lst += [' '.join(verse_lst[idx1].text[:last]) + ' ' + ' '.join(syn) + sign_1]*len(causal_syns)
            causal_syn_lst += causal_syns

        
        min_idx += batch_size
        distances = np.asarray(distances)
        best_idx = np.argmin(distances)

        verse_lst_chosen = candidates[best_idx]
        best_distance = distances[best_idx]

        if (best_distance < max_rhyme_dist) or (args.use_tts and best_distance < args.max_tts_dist):   
            print('final choice:')
            print(' '.join(verse_lst_chosen[idx1].text))
            print(' '.join(verse_lst_chosen[idx2].text))                                                                    #otherwise don't change the verses 
            print('distance: ' + str(best_distance))
            found = True


        

    if return_alternatives == False: 
        if found: 
            return verse_lst_chosen
        else: 
            print('no rhyme found')
            if force_rhyme:
                return verse_lst[:-1]
            else:
                return verse_lst
    else:
        return verse_lst_chosen, bi_syns, causal_syns


def find_rhyme_independent(args,verse_lst,idx1,idx2,LLM_perplexity,last_stress = -2, LLM='', LLM2 = None, return_alternatives=False,force_rhyme=False): # todo: integrate with new version above
    
    '''
    finds rhyming endings for two verses 
    '''

    max_rhyme_dist = args.max_rhyme_dist
    use_colone = args.use_colone_phonetics
    use_tts = args.use_tts
    target_rythm = args.target_rythm
    top_p_dict_rhyme = args.top_p_dict_rhyme
    top_p_rhyme = args.top_p_rhyme
    stop_tokens = args.rhyme_stop_tokens
    rhyme_temperature = args.rhyme_temperature
    allow_pos_match = args.allow_pos_match
    invalid_verse_ends = args.invalid_verse_ends
    repetition_penalty = args.repetition_penalty

    min_dist = 0.01
    

    eol = True
    use_pos = False

    if type (LLM) != str:
        LLM_name = LLM.model_name
    else:
        LLM_name = LLM

    if type(LLM) != str:
        if LLM.sampling == 'systematic':
            sampling = 'systematic'
        else: 
            sampling = 'multinomial'
    else: 
        sampling = 'multinomial'
    print('--- looking for rhymes ---')
    print('using ' + str(LLM_name))
    print('sampling ' + str(sampling))
    print(' '.join(verse_lst[idx1].text))
    print(' '.join(verse_lst[idx2].text))
    print('--------------------------')

    if not verse_lst[idx1].text[-1].isalpha():
        sign_1 = verse_lst[idx1].text[-1]
    else:
        sign_1 = ''

    if not verse_lst[idx2].text[-1].isalpha():
        sign_2 = verse_lst[idx2].text[-1]
    else: 
        sign_2 = ''
    
    context_aft = ''
    for i in range(idx1+1,idx2+1):
        context_aft += ' '.join(verse_lst[i].text) + '\n'

    bi_syns = bidirectional_synonyms(args,verse_lst[idx1],context_aft, target_rythm,LLM_perplexity) # alternatives for the first verse

    if verse_lst[idx1].text[-1].isalpha():
        last = -1
    else:
        last = -2 

    bi_trunk = ' '.join(verse_lst[idx1].text[:last+1])

    if idx2 == len(verse_lst) - 1:
        causal = True
        causal_syns = gpt_synonyms(args,verse_lst[idx2],target_rythm,num_remove = 1,num_return_sequences = 200,LLM=LLM,eol=eol,use_pos = use_pos,top_p_dict =top_p_dict_rhyme ,temperature=rhyme_temperature,top_k=50,
                        stop_tokens=stop_tokens,allow_pos_match=allow_pos_match,invalid_verse_ends=invalid_verse_ends,repetition_penalty=repetition_penalty,replace_linebreaks=args.replace_linebreaks)
        
        causal_syns += gpt_synonyms(args,verse_lst[idx2],target_rythm,num_remove=2,num_return_sequences = 150,LLM=LLM,eol=eol,use_pos = use_pos,stop_tokens=stop_tokens,temperature=rhyme_temperature,top_p = top_p_rhyme,
                        allow_pos_match=allow_pos_match,invalid_verse_ends=invalid_verse_ends,repetition_penalty=repetition_penalty,replace_linebreaks=args.replace_linebreaks)[1:]        

        if len(causal_syns) < 10: 
            causal_syns = gpt_synonyms(args,verse_lst[idx2],target_rythm,num_remove = 3,num_return_sequences = 140,LLM=LLM,eol=eol,use_pos = use_pos,temperature=rhyme_temperature,stop_tokens=stop_tokens,top_p = top_p_rhyme,
            allow_pos_match=allow_pos_match,invalid_verse_ends=invalid_verse_ends,repetition_penalty=repetition_penalty,replace_linebreaks=args.replace_linebreaks)[1:]       # alternatives for the second verse

        if sampling == 'systematic':           # try as well with matching pos labels instead of correct prediction of new line
            eol = False
            use_pos = True
            causal_syns += gpt_synonyms(args,verse_lst[idx2],target_rythm,num_remove = 1,num_return_sequences = 200,LLM=LLM,eol=eol,use_pos = use_pos,temperature=rhyme_temperature,top_p_dict =top_p_dict_rhyme,top_k=50,
                        invalid_verse_ends=invalid_verse_ends,repetition_penalty=repetition_penalty,replace_linebreaks=args.replace_linebreaks)
        
            causal_syns += gpt_synonyms(args,verse_lst[idx2],target_rythm,num_remove=2,num_return_sequences = 150,LLM=LLM,eol=eol,use_pos = use_pos,temperature=rhyme_temperature,top_p = top_p_rhyme ,
                        invalid_verse_ends=invalid_verse_ends,repetition_penalty=repetition_penalty,replace_linebreaks=args.replace_linebreaks)[1:]        


    else:
        causal = False
        context_aft = ''
        for i in range(idx2+1,len(verse_lst)):
            context_aft += ' '.join(verse_lst[i].text) + '\n'

        syns_tmp = bidirectional_synonyms(args,verse_lst[idx2],context_aft, target_rythm,LLM_perplexity)

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
            
            if type(sent_1) == str:
                sent_1 = [sent_1]
                
            word_1 = get_last(sent_1)

            sent_2_split = sent_2.split()
            word_2 =  get_last(sent_2_split)
            if word_1 and word_2:
                word_pairs.append([word_1,word_2])              # [bidirectional, causal]
                sent_pairs.append([' '.join(sent_1), sent_2])

    if not word_pairs:      
        print('found no pair')
        if return_alternatives == False: 
            return verse_lst
        else:
            return verse_lst, [], []


    for word_pair in word_pairs:            # compare with colone phonetics
            word_1 = word_pair[0]
            word_2 = word_pair[1]

            if word_1 != word_2:
                difference =  compare_words(word_1,word_2, last_stressed = last_stress)
            else:
                difference = 50
            differences.append(difference)
            #idx.append([' '.join(sent_1),sent_2])

    if np.amin(np.asarray(differences)) < 1:           # a match in colone phonetics is only valid below a distance of 1
        best_idx = np.argmin(np.asarray(differences))                    # best match according to colone phonetics
        bi_selection = sent_pairs[best_idx][0] #pairs[best_idx][0]
        causal_selection = sent_pairs[best_idx][1] #pairs[best_idx][1]
        found = True
        print('found via colone phonetics')

    if not found and args.rhyme_last_two_vowels: 
        for idx, word_pair in enumerate(word_pairs):
            word_1 = word_pair[0]
            word_2 = word_pair[1]

            if word_1 != word_2:
   
                if compare_last_vowels(word_1,word_2):
                    bi_selection = sent_pairs[idx][0]
                    causal_selection = sent_pairs[idx][1]
                    found = True
            
                    break

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

        

    ''' if not found and len(causal_syns)*len(bi_syns) < 10:            # leave it as it is; unprobable to find a rhyme
        print('rhyme not found')
        found = True
        bi_selection = ' '.join(bi_syns[-1])
        causal_selection = causal_syns[0]     '''

    if  not found or not use_colone: # use the sia rhyme apporach
        vector_pairs = []
        for word_pair in word_pairs:

            vector_pairs.append([rhyme_model.get_word_vec(word_pair[0]),rhyme_model.get_word_vec(word_pair[1])]) # vectorize the words
            #causal_vecs.append(rhyme_model.get_word_vec(word_pair[0]))

        distances = []
        
        for vector_pair in vector_pairs:   
            distance = rhyme_model.vector_distance(vector_pair[0],vector_pair[1])
                                                                    
            distances.append(distance[0])

        distances = np.asarray(distances) # distances between each possible combination

        candidate_idx = np.argsort(distances)[:args.size_tts_sample]
        if use_tts and np.amin(distances) <= max_rhyme_dist: 
            print('using tts')
            spectral_diffs = []

            for idx in candidate_idx:
                idx = idx
                word_1 = word_pairs[idx][0]
                word_2 = word_pairs[idx][1]

                if word_1.lower() != word_2.lower():
                
                    try:
                        spec_1 = wordspectrum(word_1)                            # calculate the mfcc features for each word
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
            

                spectral_diffs.append(mean)                 # calculate the distances between the mfcc vectors for each word

            print('spectral distance: ' + str(min(spectral_diffs)))
            best_idx = candidate_idx[np.argmin(np.asarray(spectral_diffs))]    # choose the pair with the lowest distance

        else:
            candidates = np.argsort(distances)
            best_idx = ''
            for candidate in candidates: 
                if distances[candidate] > min_dist:
                    best_idx = candidate                       # if no mffc features are used, use the minimum distance in the vectorspace of sia rhyme
                    break
            
        if best_idx and np.amin(distances) <= max_rhyme_dist:
                
            print('found via sia rhyme')
            print(sent_pairs[best_idx][0])
            print(sent_pairs[best_idx][1])
            print('distance: ' + str(distances[best_idx]))
        
            bi_selection = sent_pairs[best_idx][0]
            causal_selection = sent_pairs[best_idx][1]


            found = True

     
    if found:   
        if causal_selection[-1].isalpha():
            causal_selection += sign_2

        print('final choice:')
        print(' '.join(verse_lst[idx1].text[:last]) + ' ' + bi_selection)
        print(causal_selection)                                                                    #otherwise don't change the verses 
        verse_lst[idx1] = verse_cl(' '.join(verse_lst[idx1].text[:last]) + ' ' + bi_selection + sign_1)
        verse_lst[idx2] = verse_cl(causal_selection)
    else: 
        print('no rhyme found')
        if force_rhyme:
            return verse_lst[:-1]
    if return_alternatives == False: 
        return verse_lst

    else:

        causal_syns = []
        bi_syns = []
        idx = 0
        candidates = np.argsort(distances)
        items = 0
        while items < 50:
            candidate = candidates[idx]
            idx += 1
            if distances[candidate] > min_dist:
                items += 1
                bi_selection = sent_pairs[candidate][0]
                causal_selection = sent_pairs[candidate][1]
                causal_syns.append(causal_selection)
                bi_syns.append(' '.join(verse_lst[idx1].text[:last]) + ' ' + bi_selection)

        return verse_lst, bi_syns, causal_syns