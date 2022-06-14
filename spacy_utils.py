from spacy.attrs import LOWER, POS, ENT_TYPE, IS_ALPHA, DEP, LEMMA, LOWER, IS_PUNCT, IS_DIGIT, IS_SPACE, IS_STOP
from spacy.tokens import Doc
import numpy as np
import pickle
from scipy import spatial
#import spacy
#nlp = spacy.load("de_core_news_lg")

with open('spacy_data/conj_adv_vec.lst', 'rb') as f:
   conj_adv_vecs = pickle.load(f)
 

def get_childs_idx(token):
    
    if token.n_lefts + token.n_rights > 0:
        lst = [token.i]
        for child in token.children:
                lst += (get_childs_idx(child))
        return lst
    else: 
        return [token.i]

def is_conj_adv(word_vector):
    distances = []
    for vector in conj_adv_vecs:
        distances.append(spatial.distance.cosine(word_vector,vector))

    if min(distances) < 0.35:
        return True
    else:
        return False



    

def remove_tokens_idx(doc, index_to_del, list_attr=[LOWER, POS, ENT_TYPE, IS_ALPHA, DEP, LEMMA, LOWER, IS_PUNCT, IS_DIGIT, IS_SPACE, IS_STOP]):
    """
    Copied from GithubUSer yrongon
    Remove tokens from a Spacy *Doc* object without losing 
    associated information (PartOfSpeech, Dependance, Lemma, extensions, ...)
    
    Parameters
    ----------
    doc : spacy.tokens.doc.Doc
        spacy representation of the text
    index_to_del : list of integer 
         positions of each token you want to delete from the document
    list_attr : list, optional
        Contains the Spacy attributes you want to keep (the default is 
        [LOWER, POS, ENT_TYPE, IS_ALPHA, DEP, LEMMA, LOWER, IS_PUNCT, IS_DIGIT, IS_SPACE, IS_STOP])
    Returns
    -------
    spacy.tokens.doc.Doc
        Filtered version of doc
    """
    
    np_array = doc.to_array(list_attr) # Array representation of Doc
    
    # Creating a mask: boolean array of the indexes to delete
    mask_to_del = np.ones(len(np_array), np.bool)
    mask_to_del[index_to_del] = 0
    
    np_array_2 = np_array[mask_to_del]
    doc2 = Doc(doc.vocab, words=[t.text for t in doc if t.i not in index_to_del])
    doc2.from_array(list_attr, np_array_2)
    
    # Handling user extensions
    #  The `doc.user_data` dictionary is holding the data backing user-defined attributes.
    #  The data is based on characters offset, so a conversion is needed from the
    #  old Doc to the new one.
    #  More info here: https://github.com/explosion/spaCy/issues/2532
    arr = np.arange(len(doc))
    new_index_to_old = arr[mask_to_del]
    doc_offset_2_token = {tok.idx : tok.i  for tok in doc}  # needed for the user data
    doc2_token_2_offset = {tok.i : tok.idx  for tok in doc2}  # needed for the user data
    new_user_data = {}
    for ((prefix, ext_name, offset, x), val) in doc.user_data.items():
        old_token_index = doc_offset_2_token[offset]
        new_token_index = np.where(new_index_to_old == old_token_index)[0]
        if new_token_index.size == 0:  # Case this index was deleted
            continue
        new_char_index = doc2_token_2_offset[new_token_index[0]]
        new_user_data[(prefix, ext_name, new_char_index, x)] = val
    doc2.user_data = new_user_data
    
    return doc2