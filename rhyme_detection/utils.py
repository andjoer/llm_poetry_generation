
import numpy as np 
from scipy.ndimage.filters import maximum_filter, minimum_filter
from scipy.ndimage import binary_dilation, binary_erosion, generate_binary_structure
from skimage.feature import register_translation

def pool_array(array, size = 3):
    footprint = np.zeros((size,size))
    np.fill_diagonal(footprint,1)
    footprint = np.flip(footprint,0)
    return (minimum_filter(array,footprint = footprint))

def diff_phase_space(img_src, img_trg,tollerance):
    shift, error, diffphase = register_translation(img_src, img_trg)

    x_shift = int(shift[0])
    y_shift = int(shift[1])

    mat_1 = img_src
    mat_2 = img_trg
    if x_shift > 0: 
        
        mat_1 = img_src[:-x_shift,:]
        mat_2 = img_trg[x_shift:,:]

    elif x_shift < 0: 
        mat_1 = img_src[-x_shift:,:]
        mat_2 = img_trg[:x_shift,:]
        

    if y_shift > 0: 
        
        mat_1 = img_src[:,:-y_shift]
        mat_2 = img_trg[:,y_shift:]

    elif y_shift < 0: 
        mat_1 = img_src[:,-y_shift:]
        mat_2 = img_trg[:,:y_shift]

   
    mat_diff = np.absolute(mat_1-mat_2)
    

    column_mean = np.mean(mat_diff,axis = 0)
    

    min_value = np.amin(column_mean) 


    candidates = np.where(column_mean <= min_value*tollerance)

    total_diff = np.mean(column_mean[candidates[0]])

    best_matches_len = candidates[0].shape[0]
    return total_diff, best_matches_len

def diff_roll_mat(img_src, img_trg,pool,tollerance):
    len_time = img_src.shape[1]
    num_feat = img_src.shape[0]

    num_roll = len_time 

    roll_mat = np.zeros((num_roll, num_feat,len_time))    # we roll the smaller word over the larger word in order to make
                                                            # it translation invariant
    for i in range (num_roll):
        roll_mat[i,:,:] = np.roll(img_trg,i)
   
    img_src = np.reshape(img_src,(1,num_feat,len_time))
    img_src = np.repeat(img_src,num_roll,axis=0)
    
    mat_diff = np.absolute(roll_mat - img_src)

    mat_diff = np.linalg.norm(mat_diff, axis = 1)
    
    if max(pool) > 0: 
        mat_diff = pool_array(mat_diff,pool)     # we pool since the syllables are not spoken with same speed. This is a 
                                                 # suboptimal workaround. We should extract clusters and fit an optimal
                                                 # curve through the minima

    
    
   
    
    mat_diff = np.transpose(mat_diff)            # I think for most it is more intuitive to roll from left to right
    ####################################################################
    
    #sort = np.argsort(mat_diff, axis=0)
    min_rows = mat_diff #np.take_along_axis(mat_diff, sort, axis=0)[:10,:]
    
    min_rows_mean = np.mean(min_rows, axis=0)
    #indices = list(range(10))+list(range(-10,0))
    min_idx = np.argmin(min_rows_mean)
    
    min_mean_value = min_rows_mean[min_idx]

    

    min_column = mat_diff[:,min_idx]
    candidates = np.where(min_column < min_mean_value*tollerance)

    #splits = np.where(np.ediff1d(candidates) > max_dist)

    #idx_clusters = np.split(candidates[0], splits[0])
    
    #cluster_list_idx = np.argmax(np.asarray([x.shape[0] for x in idx_clusters]))# we are looking for the largest, not the best
                                                                                  # otherwise we would need to think about a metric
                                                                                  # to balance quality/size
   
    ##########################################################################
    
    
    #best_matches = idx_clusters[cluster_list_idx]
    best_values = min_column[candidates[0]] #min_column[best_matches]
    best_matches_mean = np.mean(best_values)
    best_matches_len = best_values.shape[0]
     #best_matches.shape[0]*unit_len
    
    

    return best_matches_mean,best_matches_len

def diff_roll_mat_2(img_src, img_trg,pool,min_matches):
    max_shift_perc = 0.15
    #max_shift = max(int(img_trg.shape[1]*max_shift_perc), min_matches)
    max_shift = int(img_trg.shape[1]*max_shift_perc)
    means = []
    
    iterations = range(max_shift,(img_src.shape[1]+img_trg.shape[1]-max_shift))
    num_iter = len(iterations)

    column_means = np.ones((img_trg.shape[1],num_iter))*np.Inf

    ctr = 0
    
    for i in iterations:

        
        img_roll = img_trg[:,np.clip(-i,-img_trg.shape[1],0):np.clip(img_src.shape[1]+img_trg.shape[1]-i,-img_trg.shape[1],img_trg.shape[1])]
       
        img_stat = img_src[:,np.clip(i-img_trg.shape[1],0,img_src.shape[1]):i]
       
        mat_diff = np.absolute(img_roll-img_stat)
    
        column_mean = np.mean(mat_diff,axis = 0)

        #min_value = np.amin(column_mean) 

        sort = np.argsort(column_mean, axis=0)
        min_rows = np.take_along_axis(column_mean, sort, axis=0)[:int(min_matches)]
        #candidates = np.where(column_mean <= min_value*tollerance)

        total_diff = np.mean(min_rows)
       
        #best_matches_len = 5 #candidates[0].shape[0]
        means.append(total_diff)
        
        if i <= img_trg.shape[1]:
            column_means[-column_mean.shape[0]:,ctr] = column_mean
        else: 
            column_means[:column_mean.shape[0],ctr] = column_mean
        ctr += 1

    if pool > 0:
        column_means = pool_array(column_means,pool)

    sort = np.argsort(column_means, axis=0)
    min_rows = np.take_along_axis(column_means, sort, axis=0)[:min_matches,:]

    min_rows_mean = np.mean(min_rows,axis=0)
   
    min_idx = np.argmin(min_rows_mean)

    total_min_mean = np.mean(min_rows[:,min_idx])

   # min_rows = np.take_along_axis(column_mean, sort, axis=0)[:int(min_matches)]

    return total_min_mean,min_rows

def diff_mat(img_src, img_trg,pool,min_matches):
    img_src = img_src[:,-img_trg.shape[1]:]

    mat_diff = np.absolute(img_trg-img_src)
    
    column_mean = np.mean(mat_diff,axis = 0)

        #min_value = np.amin(column_mean) 

    sort = np.argsort(column_mean, axis=0)
    min_rows = np.take_along_axis(column_mean, sort, axis=0)[:int(min_matches)]
        #candidates = np.where(column_mean <= min_value*tollerance)

    total_diff = np.mean(min_rows)
       
        #best_matches_len = 5 #candidates[0].shape[0]
    #means.append(total_diff)
    #column_means.append(list(column_mean))
   
    return total_diff,2

def check_rhyme(word_1,word_2,features = 'mel', length = 25, cut_off = 1, order=1,min_matches = 8,pool=(3,0),max_dist = 2):
    
    """ Compare the spectrum of two words and look for rhymes
    
        Keyword arguments: 
        word_1 -- first word of comparison
        word_2 -- second word of comparison
        features -- features to compare with, either 'mel' or 'mfccs'
        cut_off -- amount of entries the matrices get cut of on the right and left
        order -- order of derivative for the compared features
        pool -- amount of filtering in the comparison matrix
        tollerance -- factor which gets multiplied on top of the mean of minima in the comparison matrix as criterium for matches
        max_dist -- distance that could be between two found matching entries in a column to make it one single cluster
        
        returns: 
        mean of best matching range
        length of best matching range
    """
    word_list = [word_1,word_2]
    samples = word_1.samples
    
    if features == 'mel':
        spec_list =[item.mel[order] for item in word_list]
    elif features == 'mfccs':
        spec_list = [item.mfccs[order] for item in word_list]
    
    '''
    len_diff = spec_list[0].shape[1]-spec_list[1].shape[1]                             # difference of the spectrum-length

    concat_mat = np.ones((spec_list[1].shape[0],np.abs(len_diff)))*1

    idx = np.asarray(np.sign(len_diff)).clip(min=0)                                # idx of the shorter word/spectrum in list


    spec_roll = np.concatenate((spec_list[idx][:,cut_off:-cut_off],concat_mat),axis=1)   # first n and last n columns get cut
    spec_stat = spec_list[1-idx][:,cut_off:-cut_off]'''
    
    total_length = length + cut_off

    
    if spec_list[0].shape[1] < total_length and spec_list[1].shape[1] < total_length:
        length = max(spec_list[0].shape[1],spec_list[1].shape[1])

    idx = 0



    for i in range(2):
        item_shape = spec_list[i].shape[1]
        
        if item_shape < length:
            #concat_mat = np.ones((spec_list[i].shape[0],length - (item_shape)))
           # spec_list[i] = np.concatenate((spec_list[i][:,:item_shape-cut_off],concat_mat),axis=1)      # need to subtract cut_off from the shape in case cut_of = 0
            spec_list[i] = spec_list[i][:,:(spec_list[i].shape[1]-cut_off)]
            idx = i
               
        else: 
            
            spec_list[i] = spec_list[i][:,-length:(spec_list[i].shape[1]-cut_off)]
           

    '''shift, error, diffphase = register_translation(spec_list[0], spec_list[1])

    x_shift = int(shift[0])
    y_shift = int(shift[1])

    mat_1 = spec_list[0]
    mat_2 = spec_list[1]
    if x_shift > 0: 
        
        mat_1 = spec_list[0][:-x_shift,:]
        mat_2 = spec_list[0][x_shift:,:]

    elif x_shift < 0: 
        mat_1 = spec_list[0][-x_shift:,:]
        mat_2 = spec_list[0][:x_shift,:]
        

    if y_shift > 0: 
        
        mat_1 = spec_list[0][:,:-y_shift]
        mat_2 = spec_list[0][:,y_shift:]

    elif y_shift < 0: 
        mat_1 = spec_list[0][:,-y_shift:]
        mat_2 = spec_list[0][:,:y_shift]

    mat_diff = np.absolute(mat_1-mat_2)
    '''
    
    

    '''spec_roll = spec_list[idx]
    spec_stat = spec_list[1-idx]


    len_time = spec_stat.shape[1]
    num_feat = spec_stat.shape[0]

    num_roll = len_time 

    roll_mat = np.zeros((num_roll, num_feat,len_time))    # we roll the smaller word over the larger word in order to make
                                                            # it translation invariant
    for i in range (num_roll):
        roll_mat[i,:,:] = np.roll(spec_roll,i)
   
    spec_stat = np.reshape(spec_stat,(1,num_feat,len_time))
    spec_stat = np.repeat(spec_stat,num_roll,axis=0)
    
    mat_diff = np.absolute(roll_mat - spec_stat)

    mat_diff = np.linalg.norm(mat_diff, axis = 1)
    
    if max(pool) > 0: 
        mat_diff = pool_array(mat_diff,pool)     # we pool since the syllables are not spoken with same speed. This is a 
                                                 # suboptimal workaround. We should extract clusters and fit an optimal
                                                 # curve through the minima

    
    mat_diff = np.transpose(mat_diff)            # I think for most it is more intuitive to roll from left to right'''

   
                                                                                # it's not reasonable to assume that a word
                                                                                # rhymes beyound the borders of the other word
                                                                                # therefore: 
    #max_word_idx = spec_list[idx].shape[1]-2*cut_off

    #word_cut = mat_diff.shape[0]
    #if min_idx <= best_matches[0]:                                              # above or below main diagonal of comp. matrix
      #  word_roll_idx = [best_matches[0] - min_idx+cut_off, best_matches[-1]-min_idx+cut_off] 

   # else: 
        
       # word_roll_idx = [best_matches[0]  + mat_diff.shape[0] - min_idx+cut_off, best_matches[-1] + mat_diff.shape[0]- min_idx +cut_off] 
     
    #word_stat_idx = [best_matches[0]+cut_off,best_matches[-1]+cut_off]''''''
    
    #return_idx = [word_roll_idx, word_stat_idx]'''

    best_matches_mean, img = diff_roll_mat_2(spec_list[1-idx],spec_list[idx],pool,min_matches)
    #best_matches_mean, img = diff_mat(spec_list[1-idx],spec_list[idx],pool,tollerance)
    #best_matches_mean, best_matches_len = diff_phase_space(spec_list[0],spec_list[1],tollerance)

    #unit_len = word_list[1-idx].duration/spec_list[1-idx].shape[0]
    
    #rhyme_len = best_matches_len * unit_len


    return best_matches_mean, img #, shift
    #return  error,shift, diffphase