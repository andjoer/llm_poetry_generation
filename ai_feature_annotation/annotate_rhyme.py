import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

from sia_rhyme import siamese_rhyme

import pandas as pd
import os
import numpy as np

import tqdm

from scipy.optimize import linear_sum_assignment
import networkx
from networkx.algorithms.components.connected import connected_components,number_connected_components

from preprocess_annotation import preprocess_ano
#from networkx.algorithms.components import number_strongly_connected_components
rhyme_model = siamese_rhyme.siamese_rhyme()

def to_graph(matches):
    G = networkx.Graph()
    for element in matches:
        G.add_nodes_from(element)
        G.add_edges_from(to_edges(element))
    return G

def to_edges(matches):
    it = iter(matches)
    last = next(it)

    for current in it: 
        yield last, current

def set_dist(lst_1,lst_2):
    return 1-(len(set(lst_1)&set(lst_2))/max(len(lst_1),len(lst_2)))


def distance_matrix(lst_1,lst_2,dist_func):
    return[[dist_func(elem_1,elem_2) for elem_1 in lst_1] for elem_2 in lst_2]

def lst_dist(lst_1,lst_2):
    cost = np.array(distance_matrix(lst_1,lst_2,set_dist))
    try:
        row_ind, col_ind = linear_sum_assignment(cost)
        total_cost = cost[row_ind,col_ind].sum()
    except:
        total_cost = 2

    return total_cost

def scheme_to_idx(rhyme_scheme):
    letters = (list(set(rhyme_scheme)))
    match_list = []
    for letter in letters:
        indices = [i for i, x in enumerate(rhyme_scheme) if x == letter]
        match_list.append(indices)

    return(match_list)

def idx_to_scheme(idx_lst):
    idx_lst = sorted(idx_lst,key=min)

    rhyme_scheme = pd.Series(['z']*sum([len(sublist) for sublist in idx_lst]))

    for count, sublist in enumerate(idx_lst):

        rhyme_scheme[sublist] = chr(97+count)

    return list(rhyme_scheme)

def get_best_match(dst_mat,thresh):

    dst_mat = np.maximum(np.rot90(np.fliplr(dst_mat)),dst_mat)
    np.fill_diagonal(dst_mat,100)

    minima = np.amin(dst_mat,axis=1)
    min_idx = np.argmin(dst_mat, axis=1)

    matches = []
    for i in range (len(minima)):
        if minima[i] < thresh:
            matches.append([i,min_idx[i]])

    return matches


def rhyme_pairs(verse_endings,thresh = 0.4, thresh_max = 0.7):
    

    vector_list = [rhyme_model.get_word_vec(word) for word in verse_endings]
    #print(verse_endings)
    dst_lst = []
    for cnt, vector_1 in enumerate(vector_list):
        dst_lst_int = [0]*(cnt+1)
        for vector_2 in vector_list[cnt+1:]:
                dst = rhyme_model.vector_distance(vector_1,vector_2)
                dst_lst_int.append(dst)
        dst_lst.append(dst_lst_int)
    dst_mat = np.asarray(dst_lst)  
    dst_mat = np.maximum(np.rot90(np.fliplr(dst_mat)),dst_mat)
    dst_str_lst = []
    for cnt, word_1 in enumerate(verse_endings):
        dst_lst_str_int = [0]*(cnt+1)
        for word_2 in verse_endings[cnt+1:]:
                if word_1[-2:] == word_2[-2:]:
                    dst = 7
                    #print(word_1 +' ' + word_2)
                else: 
                    dst = 100
                dst_lst_str_int.append(dst)
        dst_str_lst.append(dst_lst_str_int)
    dst_str_mat = np.asarray(dst_str_lst)  
    dst_str_mat = np.maximum(np.rot90(np.fliplr(dst_str_mat)),dst_str_mat)

    #print(dst_str_mat)
    dst_mat = np.minimum(dst_mat, dst_str_mat)    
    np.fill_diagonal(dst_mat,100)

    rhymes = np.where(dst_mat < thresh )
    
    minima = np.amin(dst_mat,axis=1)
    min_idx = np.argmin(dst_mat, axis=1)

    matches = []
    for i in range (len(minima)):
        if minima[i] < thresh:
            matches.append([i,min_idx[i]])
    

    #matches = [list(x) for x in list(rhymes)]
    
    #matches = get_best_match(dst_mat,thresh)        
    matched_elem = list(set([item for sublist in matches for item in sublist]))

    match_diff = len(vector_list) - len(matched_elem)

    if match_diff > 1 and len(matched_elem) > 0:
        dst_mat_2 = np.copy(dst_mat)
        dst_mat_2[matched_elem,:] = 100
        dst_mat_2[:,matched_elem] = 100
        
        min_idx_left = [x for x in range(len(vector_list)) if x not in matched_elem]

        minima = np.amin(dst_mat_2,axis=1)
        min_idx = np.argmin(dst_mat_2, axis=1)

        thresh_worst = thresh_max
        if match_diff % 2 > 0: 
            thresh_worst = np.amax(minima[np.where(minima<100)])

        for i in min_idx_left:

            if minima[i] < thresh_max and minima[i] < thresh_worst:
                matches.append([i,min_idx[i]])
            
    
    G = to_graph(matches)
    match_list = [list(x) for x in list(connected_components(G))]
    

    #if (match_diff % 2 > 0 and match_diff > 1) or match_diff >= 1:
    matched_elem = list(set([item for sublist in matches for item in sublist]))
    not_matched_elem = [[x] for x in range(len(vector_list)) if x not in matched_elem]
    if not_matched_elem:
        match_list += (not_matched_elem)

    match_list = sorted(match_list,key=min)
    
    return match_list

#print(number_strongly_connected_components(G))
#match_list = [list(x) for x in list(connected_components(G))]

def compare_rhyme_df(poem_df):
    differences = []
    rhyme_schemes = []
    for index, row in tqdm.tqdm(poem_df.iterrows(), total=poem_df.shape[0]):

            verse_endings = row['endings']
            rhyme_scheme = row['rhyme']

            if len(verse_endings) > 1:
                match_list = rhyme_pairs(verse_endings)

                gold_idx = scheme_to_idx(rhyme_scheme)
                diff = lst_dist(gold_idx,match_list)
                differences.append(diff)
                
                rhyme_scheme = idx_to_scheme(match_list)
                rhyme_schemes.append(rhyme_scheme)
            else: 
                differences.append(0)
                rhyme_schemes.append('a')
         
    
            
    return differences, rhyme_schemes

if __name__ == "__main__":  
    path = 'data'
    fname = 'rhyme_df.pkl'
    fname_save = 'rhyme_df_pred.csv'
    fname_save_pkl = 'rhyme_df_pred.pkl'

    rhyme_df_read = pd.read_pickle(os.path.join(path, fname))


    poem_df = preprocess_ano(rhyme_df_read)
    print(poem_df.head())
    diff_lst, rhyme_lst = compare_rhyme_df(poem_df)

    diff_lst_ext = diff_lst + [0]*(len(poem_df) - len(diff_lst))

    rhyme_lst_ext = rhyme_lst + ['z']*(len(poem_df) - len(diff_lst))

    poem_df['error'] = diff_lst_ext
    poem_df['prediction'] = rhyme_lst_ext

    poem_df.to_csv(os.path.join(path, fname_save))
    poem_df.to_pickle(os.path.join(path, fname_save_pkl))
    print(poem_df.head())

    print('sum of errors:')
    print(sum(diff_lst))