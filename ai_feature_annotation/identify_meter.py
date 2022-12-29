import numpy as np
import math
import tqdm
import pandas as pd
import os

meter_dict = {'jambus':[0,1],'trochee':[1,0],'daktylus':[1,0,0],'anapast':[0,0,1],'three_mid':[0,1,0]}



def compare(target_ext, rythms): 
    
    comp = (np.tile(target_ext, (rythms.shape[0], 1)))
  
    difference = np.abs(rythms - comp) * (rythms != 0.5)

    difference = np.sum(difference,axis = 1)
    return np.min(difference) , np.argmin(difference)

def compare_elastic(meter, rythm,rythm_tok):

    meter_ext = np.asarray(meter*math.ceil((len(rythm)+1)/len(meter)))[:(len(rythm)+1)]
    rythm_short = []

    rythm = list(rythm)
    
    poss_remove = []
    index = 0
    for syllable in rythm_tok:
    
        if len(syllable) > 1:
            poss_remove += list(range(index,index+len(syllable)))
        
        index += len(syllable)

    if poss_remove:
        for i in range(1, len(rythm)):

            if i in poss_remove:
                rythm_short.append(rythm[:i-1] + rythm[i:])
            
        rythm_short = np.asarray(rythm_short)

        diff_short, idx_short = compare(meter_ext[:-2],rythm_short)

        idx_short = - poss_remove[idx_short]    # the sign encodes that the rythm was shortened
    else:
        diff_short = 100
        idx_short = 0

    rythm_ext = []

    for i in range(1, len(rythm)):

        rythm_ext.append(rythm[:i] + [0.5] + rythm[i:])

    rythm_ext = np.asarray(rythm_ext)

    diff_ext, idx_ext = compare(meter_ext,rythm_ext)

    if diff_ext < diff_short:

        return diff_ext, idx_ext

    else: 

        return diff_short, idx_short


def printDistances(distances, token1Length, token2Length):
    for t1 in range(token1Length + 1):
        for t2 in range(token2Length + 1):
            print(int(distances[t1][t2]), end=" ")
        print()


def levenshteinDistanceDP(token1, token2):
    '''
    https://blog.paperspace.com/implementing-levenshtein-distance-word-autocomplete-autocorrect/
    copied and modified

    '''
    distances = np.zeros((len(token1) + 1, len(token2) + 1))

    for t1 in range(len(token1) + 1):
        distances[t1][0] = t1

    for t2 in range(len(token2) + 1):
        distances[0][t2] = t2
        
    a = 0
    b = 0
    c = 0
    
    for t1 in range(1, len(token1) + 1):
        for t2 in range(1, len(token2) + 1):
      
            if (token1[t1-1] == token2[t2-1] or token1[t1-1] == 0.5):
                distances[t1][t2] = distances[t1 - 1][t2 - 1]

            else:
                a = distances[t1][t2 - 1] 
                b = distances[t1 - 1][t2] 
                c = distances[t1 - 1][t2 - 1]
                
                if (a <= b and a <= c):
                    distances[t1][t2] = a + 1
                elif (b <= a and b <= c):
                    distances[t1][t2] = b + 1
                else:
                    distances[t1][t2] = c + 1

    #printDistances(distances, len(token1), len(token2))
    return distances[len(token1)][len(token2)]




def check_meter(meter, rythm,rythm_tok, penalty = 1):
    ext = meter*math.ceil((len(rythm))/len(meter))
    meter_ext = np.asarray(ext[:len(rythm)])
    diff= np.sum(np.absolute(meter_ext-rythm) * (rythm != 0.5))

    diff_el,idx_el = compare_elastic(meter, rythm,rythm_tok) 

    diff_el += penalty

    if diff < diff_el:
        return diff, None

    else: 
        return diff_el, idx_el

def identify_meter(rythm,rythm_tok):

    pred_meter = ''
    diff = 100
    pred_idx = 0
    for meter in meter_dict:
        diff_comp, idx = check_meter(meter_dict[meter],rythm,rythm_tok)
        if  diff_comp < diff:
            pred_meter = meter            
            diff = diff_comp
            pred_idx = idx

    return pred_meter, diff, pred_idx

if __name__ == "__main__":  
    path = 'data'
    fname = 'rythm_df_pred.pkl'
    fname_save = 'rythm_df_pred_2.csv'
    fname_save_pkl = 'rythm_df_pred_2.pkl'

    poem_df = pd.read_pickle(os.path.join(path, fname))

    predictions_lst = []
    diffs_lst = []
    idxs_lst = []
    for index, row in tqdm.tqdm(poem_df.iterrows(), total=poem_df.shape[0]):
        predictions = []
        diffs = []
        idxs = []
        for j, line in enumerate(row['rythm']):
            
            pred_meter, diff, idx = identify_meter(np.asarray(line),row['rythm_tok'][j])
            predictions.append(pred_meter)
            diffs.append(diff)
            idxs.append(idx)
        
        predictions_lst.append(predictions)
        diffs_lst.append(diffs)
        idxs_lst.append(idxs)

    poem_df['meter'] = predictions_lst
    poem_df['meter_difference'] = diffs_lst
    poem_df['insert/del idx'] = idxs_lst


    poem_df.to_csv(os.path.join(path, fname_save))
    poem_df.to_pickle(os.path.join(path, fname_save_pkl))



    #annotated_df = annotate_rythm(poem_df)
    #annotated_df.to_csv(os.path.join(path, fname_save))
    #annotated_df.to_pickle(os.path.join(path, fname_save_pkl))
