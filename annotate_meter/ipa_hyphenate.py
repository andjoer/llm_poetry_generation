import re
import math 
import numpy as np

from convert_to_ipa import convert_to_ipa


ipa_groups = {                                                                                  # it are not the correct signs, but the cleaned version, correct would be:
                'plosives':'p,b,p,b,t,d,t,d,t,d,t,d,t,d,t,d,t,d,ʈ,ɖ,c,ɟ,k,ɢ,ʡ,ʔ,g,ɡ',           # p,⁠b⁠,p̪⁠,b̪,⁠t̼⁠⁠,d̼,⁠t̟,d̟⁠⁠⁠,⁠t̺⁠,d̺,⁠t̪⁠,⁠d̪⁠,⁠t,d⁠,⁠t̻⁠,d̻⁠,t̠⁠,⁠d̠⁠,ṭ⁠,ḍ⁠,⁠ʈ⁠,ɖ⁠,⁠c,⁠ɟ⁠,⁠k,⁠g⁠,⁠q⁠,⁠ɢ⁠,⁠ʡ,ʔ⁠
                'fricatives': 'ɸ,β,f,v,θ,ð,s,z,ɬ,ɮ,ʃ,ʒ,ʂ,ʐ,ɕ,ʑ,ç,ʝ,x,ɣ,ʍ,ɧ,χ,ʁ,ħ,ʜ,ʢ,h,ɦ,s,f,ɺ',  # ɸ⁠,β⁠,⁠⁠f⁠,⁠v⁠,θ⁠,⁠ð⁠,⁠s,z⁠,⁠ɬ⁠,⁠ɮ,ʃ⁠,ʒ⁠,⁠ʂ,⁠ʐ⁠,⁠ɕ,⁠ʑ⁠,⁠ç⁠,⁠ʝ⁠,⁠x,⁠ɣ,ʍ⁠,ɧ⁠,χ⁠,ʁ,ħ⁠⁠,⁠ʕ⁠,⁠ʜ⁠,⁠ʢ⁠,h⁠,ɦ,⁠s’,fʼ⁠
                'nasals':'m,m,ɱ,ɱ,n,n,ɳ,ɳ,ɲ,ɲ,ŋ,ŋ,ɴ,ɴ⁠',                                         # ⁠m,m̥⁠⁠,ɱ,ɱ̊⁠,⁠⁠n⁠,n̥⁠,⁠ɳ,⁠ɳ̊,ɲ⁠,ɲ̊⁠,ŋ⁠,ŋ̊⁠,ɴ⁠,⁠ɴ̥⁠
                'liquides':'l,r', 
                'approximates':'j,w',
                'vowels_closed':'ɪ,i,i,ʏ,y,y,ʊ,u,u',                                            #ɪ,i,iː,ʏ,y,yː,ʊ,u,uː
                'vowels':'ə,i,e,ɛ,æ,ɑ,ɔ,o,u,a,a,ɐ,ø,œ'                                          #i,e,ɛ,æ,ɑ,ɔ,o,u,a,aː,ɐ
            }

regex_ipa_letters = ''



for group in ipa_groups.keys():
    regex_ipa_letters += ipa_groups[group]


regex_no_clean_ipa = '[^' + re.sub(',','',regex_ipa_letters) + 'ˈˌ,]' #r'[^aɐɑɒæɑʌbɓʙβcçɕɔɔdɗɖðdzdʒdʑɖʐeəɘɛɛɜfɸɡɠɢʛɣɤhħɦɧʜɥiĩɨɪɯɤijʝɟʄkkxlɫɬɭʟɮʎmɱɯɰnɲŋɳɴoõɵøɞœɶɔɔɤʊʘppfɸrɾɺɽɹɻʀʁrɐrɾɺɽɹɻʀʁrɐsʂʃsfɕtʈθtstʃtɕʈʂuũʉʊvvʋѵʌɣwwʍɰxχyʏɥʎɣɤzʑʐʒzvʡʔʢˈˌ,]'  # all but the accepted signs

def clean_ipa(ipa_string):

    ipa_string = re.sub(regex_no_clean_ipa,'',ipa_string)
    ipa_string = re.sub(r"(.)\1", r'\1', ipa_string)       # fixes just a few remaining issues of the neural net that are mostly fixed now
    return ipa_string

def find_closest_idx(idx_1, idx_2):
    
    diff = np.abs(idx_1.shape[0]-idx_2.shape[0])
    keep = []
    for i in range(diff):
            
            dist_mat = np.abs(idx_1[:, None] - idx_2[None, :])

            arg_min = np.unravel_index(dist_mat.argmin(), dist_mat.shape)

            keep.append(arg_min[0])

            idx_2 = np.delete(idx_2,arg_min[1])

    return keep

def reverse_vocal_shortage(word,ipa,regex_vowel_letter,regex_letter,regex_ipa_letter, regex_ipa_vowel_letter, elastic = True ):

    
    ipa = clean_ipa(ipa)
    vowel_n_word = np.asarray([m.start(0) for m in re.finditer(regex_vowel_letter,word)])
    
    if vowel_n_word.shape[0] > 0:

        n_word = np.asarray([m.start(0) for m in re.finditer(regex_letter,word)])

        n_ipa = np.asarray([m.start(0) for m in re.finditer(regex_ipa_letter ,ipa)])

        if elastic:

            if len(n_word) > len(n_ipa) and len(n_ipa) > 0:
                try:
                    keep = find_closest_idx(n_word,n_ipa)
                    n_word = n_word[keep]
                except:
                    pass

        rel_idx_word_vowels = np.where(np.isin(n_word,vowel_n_word))[0]
 
        vowel_n_ipa = np.asarray([m.start(0) for m in re.finditer(regex_ipa_vowel_letter,ipa)])

        rel_idx_ipa_vowels = np.where(np.isin(n_ipa,vowel_n_ipa))[0]
    
        if len(vowel_n_ipa) != len(vowel_n_word):
            if len(n_word) == len(n_ipa):                                # otherwise it is not clear which n is meant: better do nothing
                replace_idx = []
                for idx in range(rel_idx_word_vowels.shape[0]):
                    
                    if not np.isin(idx,rel_idx_ipa_vowels):
                        replace_idx.append(n_ipa[idx])

                offset = 0
                for idx in replace_idx:
                    ipa = ipa[:idx+offset] + 'ɛ' + ipa[idx+offset:]
                    offset += 1

    return ipa

def reverse_vocal_shortage_n(word,ipa):
    regex_vowel_letter = '(?<=[aeiouäöü])n'
    regex_letter = 'n'
    regex_ipa_letter = '[nnɳɳɲɲŋŋɴɴ]'
    regex_ipa_vowel_letter = '(?:(?<=[əieɛæɑɔouaaɐøœɪiiʏyyʊuu][ˈˌ])|(?<=[əieɛæɑɔouaaɐøœɪiiʏyyʊuu]))[nnɳɳɲɲŋŋɴɴ]'
    return reverse_vocal_shortage(word,ipa,regex_vowel_letter,regex_letter,regex_ipa_letter, regex_ipa_vowel_letter )

def reverse_vocal_shortage_m(word,ipa):
    regex_vowel_letter = '(?<=[aeiouäöü])m'
    regex_letter = 'm'
    regex_ipa_letter = '[m,m,ɱ,ɱ]'
    regex_ipa_vowel_letter = '(?:(?<=[əieɛæɑɔouaaɐøœɪiiʏyyʊuu][ˈˌ])|(?<=[əieɛæɑɔouaaɐøœɪiiʏyyʊuu]))[m,m,ɱ,ɱ]'
    return reverse_vocal_shortage(word,ipa,regex_vowel_letter,regex_letter,regex_ipa_letter, regex_ipa_vowel_letter )

def reverse_vocal_shortage_l(word,ipa):
    regex_vowel_letter = '(?<=[aeiouäöü])l'
    regex_letter = 'l'
    regex_ipa_letter = 'l'
    regex_ipa_vowel_letter = '(?:(?<=[əieɛæɑɔouaaɐøœɪiiʏyyʊuu][ˈˌ])|(?<=[əieɛæɑɔouaaɐøœɪiiʏyyʊuu]))l'
    return reverse_vocal_shortage(word,ipa,regex_vowel_letter,regex_letter,regex_ipa_letter, regex_ipa_vowel_letter )

def reverse_vocal_shortage_shwa(word,ipa):
    regex_vowel_letter = '(?<=[aeiouäöü])r'
    regex_letter = 'r'
    regex_ipa_letter = 'ɐ'
    regex_ipa_vowel_letter = '(?:(?<=[əieɛæɑɔouaaɐøœɪiiʏyyʊuu][ˈˌ])|(?<=[əieɛæɑɔouaaɐøœɪiiʏyyʊuu]))ɐ'
    return reverse_vocal_shortage(word,ipa,regex_vowel_letter,regex_letter,regex_ipa_letter, regex_ipa_vowel_letter )

def hyphenate_ipa(word):
    word = re.sub('[ß]', 's', word)
    word = re.sub('[^A-Za-zäöüÄÖÜ]', '', word)
    ipa = clean_ipa(convert_to_ipa(word.lower()))
    
    if len(ipa) < 3:
        return [0.5],ipa,[ipa]


    ipa = reverse_vocal_shortage_n(word,ipa)
    ipa = reverse_vocal_shortage_l(word,ipa)
    ipa = reverse_vocal_shortage_m(word,ipa)
    #ipa = reverse_vocal_shortage_shwa(word,ipa)                          # it is now done with other workaround-rules

    word_ipa = ipa
    onsets = ipa
    for index, key in enumerate(ipa_groups):

        onsets = re.sub(str(list(ipa_groups[key])),str(index),onsets)

    vowel = False
    syllables = []
    last_onset = 0
    last_idx = 0
    stress_lst = []
    stress = 0

    onsets_np = np.asarray(list(re.sub('[ˈˌ]','0',onsets)),dtype = np.int8)

    vowels_np = np.where(onsets_np > 4)
    try:
        last_vowel = np.max(vowels_np)
    except:
        last_vowel = 0
    
    for idx, onset in enumerate(onsets):

        if onset == 'ˈ':                                                      # start of stressed syllable
            if vowel:                                                         # the last syllable is complete (contains vowel), so a new one is created
                syllables.append(word_ipa[last_idx:idx])
                stress_lst.append(stress)
                last_idx = idx
                vowel = False

            elif syllables and not vowel:                                    # the last syllable is not complete (contains not a vowel), so no new syllable is created
                syllables[-1] += word_ipa[last_idx:idx]
                last_idx = idx

            stress = 1

        elif onset == 'ˌ':                                                   # start of syllable with secondary stress
            if syllables and not vowel: 
                syllables[-1] += word_ipa[last_idx:idx]
                last_idx = idx
                
            elif vowel: 
                syllables.append(word_ipa[last_idx:idx])
                stress_lst.append(stress)
                last_idx = idx
                vowel = False


            stress = 0.5
            last_idx = idx
            
        else:
  
            dist = 1
            
            if onsets[min((idx+1),len(onsets)-1)] in ['ˌ','ˈ']:
                dist = 2


            if vowel and onset <= last_onset and onset < onsets[min((idx+dist),len(onsets)-1)] and idx < last_vowel and onset <= '4':      # local mainimum of onset; does not split at the end of a word_ipa
                                                                                                                      
              
                if syllables:
                    if (len(syllables[-1]) - math.ceil(stress_lst[-1]) > 1 ) or (len(word_ipa[last_idx:idx])- math.ceil(stress) > 1 ):      # cases like aeʁodyˈnaːmɪk; if the stress is larger then 0 the chunk of the word_ipas contains ˈ or ˌ
                                                                                                                                       # the markers for stress should not be a prob. but the algorithm is more resilient against mistakes then.
                        syllables.append(word_ipa[last_idx:idx])
                        vowel = False
                        last_idx = idx
                        stress_lst.append(stress)
                        stress = 0

                    else:
                        syllables[-1] += word_ipa[last_idx:idx]
                        vowel = False
                        last_idx = idx
                        stress_lst[-1] = max(stress,stress_lst[-1])
                        stress = 0
                else:
                        syllables.append(word_ipa[last_idx:idx])
                        vowel = False
                        last_idx = idx
                        stress_lst.append(stress)
                        stress = 0
             
            
            if int(onset) > 4 and vowel:                                    # two consecutive vowels
                
                if syllables:
                    condition = word_ipa[idx] not in ['ɪ','i','i','ʊ','u','ʏ','y','y','ɐ'] and idx - last_idx > 1

                else:
                    condition = word_ipa[idx] not in ['ɪ','i','i','ʊ','u','ʏ','y','y','ɐ']

                if word_ipa[idx] == 'ɐ' and re.match('[əieɛæɑɔouaaɐøœɪiiʏyyʊuu][əieɛæɑɔouaaɐøœɪiiʏyyʊuu]',word_ipa[max(0,idx -2):idx]):    #shwa could be separated after two other vowels
                    condition = True

                if word_ipa[idx] in ['ə'] and word_ipa[idx-1] in ['i'] and idx == len(word_ipa)-1:                         # not an ideal rule, but it fixed a few issues
                    condition = False

                if condition:
                    
                    syllables.append(word_ipa[last_idx:idx])
                    last_idx = idx
                    stress_lst.append(stress)
                    stress = 0
                    vowel = True
        
            elif int(onset) > 4 and not vowel:
               
                vowel = True

            last_onset = onset
 
    if len(''.join(syllables)) < len(word_ipa):
        stress_lst.append(stress)
        syllables.append(word_ipa[last_idx:])   


    if len(syllables) == 1 and stress_lst[-1] == 0:
        stress_lst = [0.5]

    return stress_lst, ipa, syllables



if __name__ == "__main__":  

    word = 'seinen'
    word = re.sub('[ß]', 's', word)
    word = re.sub('[^A-Za-zäöüÄÖÜ]', '', word)
    stress_lst, ipa, syllables = hyphenate_ipa(word)

    print(word)
    print(syllables)
    print(stress_lst)
    
  
