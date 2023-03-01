import os
import sys
import inspect
import argparse
import glob
import re
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

from sia_rhyme import siamese_rhyme
from rhyme_detection.word_spectral import wordspectrum
from rhyme_detection.utils import check_rhyme

import pandas as pd
import os
import numpy as np

import tqdm

from scipy.optimize import linear_sum_assignment
import networkx
from networkx.algorithms.components.connected import connected_components,number_connected_components

from preprocess_annotation import preprocess_ano

#from networkx.algorithms.components import number_strongly_connected_components
def str_eval(string):
    return ast.literal_eval(str(string))

class rhyme_vectors():
    
    def __init__(self, method):
        self.method = method

        if method == 'siamese':
            self.rhyme_model = siamese_rhyme.siamese_rhyme()
            self.get_vector = self.get_vector_sia
            self.get_distance = self.get_distance_sia
            self.thresh = 0.55
            self.thresh_max = 0.55

        else:
            self.get_vector = self.get_vector_tts
            self.get_distance = self.get_distance_tts
            self.thresh = 8
            self.thresh_max = 10
    def get_vector_sia(self,word):

        return self.rhyme_model.get_word_vec(word)

    def get_vector_tts(self,word):

        return wordspectrum(word)


    def get_distance_sia(self,vec_1,vec_2):

        return self.rhyme_model.vector_distance(vec_1,vec_2)

    def get_distance_tts(self,vec_1,vec_2):
        dist, _ = check_rhyme(vec_1,vec_2,
                            features = 'mfccs',
                            order=0,
                            length = 19, 
                            cut_off = 1,
                            min_matches=10,
                            pool=0)
        return dist



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


def rhyme_pairs(verse_endings,rhyme_detection):
    

    #vector_list = [rhyme_model.get_word_vec(word) for word in verse_endings]
    vector_list = [rhyme_detection.get_vector(word) for word in verse_endings]
    #print(verse_endings)
    dst_lst = []
    for cnt, vector_1 in enumerate(vector_list):
        dst_lst_int = [0]*(cnt+1)
        for vector_2 in vector_list[cnt+1:]:
                dst = rhyme_detection.get_distance(vector_1,vector_2)
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

    rhymes = np.where(dst_mat < rhyme_detection.thresh )
    
    minima = np.amin(dst_mat,axis=1)
    min_idx = np.argmin(dst_mat, axis=1)

    matches = []
    for i in range (len(minima)):
        if minima[i] < rhyme_detection.thresh:
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

        thresh_worst = rhyme_detection.thresh_max
        if match_diff % 2 > 0: 
            thresh_worst = np.amax(minima[np.where(minima<100)])

        for i in min_idx_left:

            if minima[i] < rhyme_detection.thresh_max and minima[i] < thresh_worst:
                matches.append([i,min_idx[i]])
            
    
    G = to_graph(matches)
    match_list = [list(x) for x in list(connected_components(G))]
    

    #if (match_diff % 2 > 0 and match_diff > 1) or match_diff >= 1:
    matched_elem = list(set([item for sublist in matches for item in sublist]))
    not_matched_elem = [[x] for x in range(len(vector_list)) if x not in matched_elem]
    if not_matched_elem:
        match_list += (not_matched_elem)

    match_list = sorted(match_list,key=min)


    diff = []
    
    for match in match_list:
        if len(match) > 1:
            coordinates = [(a, b) for idx, a in enumerate(match) for b in match[idx + 1:]]
            coordinates_array = np.moveaxis(np.array(coordinates), -1, 0)

            diff.append(dst_mat[tuple(coordinates_array)].tolist())

    return match_list,diff

#print(number_strongly_connected_components(G))
#match_list = [list(x) for x in list(connected_components(G))]

def compare_rhyme_df(poem_df,gold,rhyme):
    differences = []
    for index, row in tqdm.tqdm(poem_df.iterrows(), total=poem_df.shape[0]):

            rhyme_scheme_gold = row[gold]
            rhyme_scheme = row[rhyme]

            if len(rhyme_scheme) > 1:
                rhyme_idx = scheme_to_idx(rhyme_scheme)
                gold_idx = scheme_to_idx(rhyme_scheme_gold)
                diff = lst_dist(gold_idx,rhyme_idx)
                differences.append(diff)

            else: 
                differences.append(0)
               
    return differences

def load_txt_files(args):

    if not args.fname_input == 'all':
        fnames_input = [args.fname_input] 
    else:
        files = glob.glob(args.data_dir+"/*.txt")
        fnames_input = [file.split('/')[-1] for file in files]

    poem_lst = []
    author_lst = []
    for fname in fnames_input:
        input_file = os.path.join(args.data_dir, fname)
        with open(input_file) as f:
            text = f.read()
        
        text_lst = text.split(args.sep)
        poem_lst += text_lst
        author_lst += [fname]*len(text_lst)

    return pd.DataFrame(zip(author_lst,poem_lst),columns=[args.author_column,args.text_column])
            


class Annotate_rhyme():
    def __init__(self,args):
        self.args = args
        self.rhyme_schemes = []
        self.differences = []
        self.probability = []
        

        self.rhyme_detection = rhyme_vectors(args.method)

        if args.fname_input[-3:] == 'txt' or args.fname_input == 'all':
            self.poem_df = load_txt_files(args)
            

        else:
            
            input_file = os.path.join(self.args.data_dir, self.args.fname_input)

            try: 
                self.poem_df = pd.read_csv(input_file)
            except: 
                try: 
                    self.poem_df = pd.read_pickle(input_file)
                except: 
                    print('file not found or not a csv or pickle file')
                    raise Exception


        if args.concat_strophes:
            self.poem_df['text'] = self.poem_df['text'].apply(lambda x: str(x) + '\n')
            text_df_strophes = self.poem_df[['ID','text']].groupby(['ID']).sum()[args.text_column]

            text_df_authors = self.poem_df.groupby(['ID']).first()[args.author_column]

            self.poem_df = pd.DataFrame(list(zip(list(text_df_authors),list(text_df_strophes))),columns=[args.author_column,args.text_column])

        print(self.poem_df.head())
        self.poem_df = preprocess_ano(self.poem_df,col_text = args.text_column)
        
        if args.load_checkpoint:
            
            fname_ckp = os.path.join(args.data_dir+'/checkpoints', args.load_checkpoint)
            if fname_ckp[-3:] != 'pkl':
                fname_ckp += '.pkl'
            try: 
                ckp_df = pd.read_pickle(fname_ckp)
            except: 
                print('unable to load checkpoint')
                raise Exception

            self.probability = ckp_df['prob_rhyme'].to_list()
            self.rhyme_schemes = ckp_df[self.args.rhyme_column].to_list()
            self.differences = ckp_df[self.args.difference_column].to_list()

        self.offset = len(self.rhyme_schemes)

        files = glob.glob(args.data_dir+'/'+args.fname_output+'*')
        max_idx = 1
        if files: 
            for file in files: 
                try: 
                    max_idx = max(int(re.findall(r'\d+', file)[0]),max_idx)    # find the number of the last file
                except: 
                    pass
        
            self.args.fname_output += '_' + str(max_idx +1)

    def annotate_rhyme(self):
        
        for index, row in tqdm.tqdm(self.poem_df[self.offset:].iterrows(), total=self.poem_df[self.offset:].shape[0]):
            verse_endings = row['endings']

            if len(verse_endings) > 1:
                match_list, diff = rhyme_pairs(verse_endings,self.rhyme_detection)
                rhyme_scheme = idx_to_scheme(match_list)
                self.probability.append(len(set(rhyme_scheme))/len(verse_endings))
                self.rhyme_schemes.append(rhyme_scheme)
                self.differences.append(diff)
            else: 
                self.rhyme_schemes.append('a')
                self.probability.append(1)
            if (index+1)%self.args.save_every == 0: 
                self.save_ckp()

        self.poem_df[self.args.rhyme_column] = self.rhyme_schemes
        self.poem_df['rhyme_prob'] = self.probability
        #self.save_df_at(self.args.data_dir,self.args.fname_output, annotated_df)

    
    def compare_to_gold(self):
        print(self.poem_df.head())
        self.poem_df['difference'] = compare_rhyme_df(self.poem_df,self.args.rhyme_column,self.args.gold_column)

    def get_dataframe(self):
        text = self.poem_df['text'].tolist()[:len(self.rhyme_schemes)]
        annotated_df = pd.DataFrame(list(zip(text, self.rhyme_schemes, self.differences,self.probability)),
                columns =['text', self.args.rhyme_column,self.args.difference_column,'prob_rhyme'])
        return annotated_df

    def output_statistics(self):

        if self.args.flatten_for_stat:
            self.statistic_df = pd.DataFrame([self.poem_df['rhyme_prob'].mean()],columns=['rhyme_prob'])
        else:
            self.statistic_df = self.poem_df[['author','rhyme_prob']]
            self.statistic_df = self.statistic_df.groupby('author').mean()


        lower_05 = []
        lower_06 = []
        lower_07= []
        lower_08= []
        for author in list(self.statistic_df.index.values):
            print('author')
            if self.args.flatten_for_stat:
                author_df = self.poem_df
            else:
                author_df = self.poem_df[self.poem_df['author'] == author]
            lower_05.append(len(author_df[author_df['rhyme_prob']<= 0.5])/len(author_df))
            lower_06.append(len(author_df[author_df['rhyme_prob']<= 0.6])/len(author_df))
            lower_07.append(len(author_df[author_df['rhyme_prob']<= 0.7])/len(author_df))
            lower_08.append(len(author_df[author_df['rhyme_prob']<= 0.8])/len(author_df))

        self.statistic_df['lower_05'] = lower_05
        self.statistic_df['lower_06'] = lower_06
        self.statistic_df['lower_07'] = lower_07
        self.statistic_df['lower_08'] = lower_08
        print(self.statistic_df.head()) 
        self.save_df_at(self.args.data_dir,self.args.fname_output+'_statistic',self.statistic_df)


    def save_ckp(self):
        annotated_dataframe = self.get_dataframe()
        fname = self.args.fname_output +'_ckp_' + str(len(self.rhyme_schemes))
        self.save_df_at(args.data_dir+'/checkpoints',fname,annotated_dataframe)

    def save_df(self):
        self.save_df_at(self.args.data_dir,self.args.fname_output,self.poem_df)

    def save_df_at(self,path,fname,dataframe):
        dataframe.to_csv(os.path.join(path, fname +'.csv'))
        dataframe.to_pickle(os.path.join(path, fname +'.pkl'))
        print('saved file: '+ os.path.join(path, fname))

if __name__ == "__main__":  

    parser = argparse.ArgumentParser()
    

    parser.add_argument("--data_dir", type=str,default='data/rhyme_detection/Gutenberg',help="subdirectory for the input and output data")
    parser.add_argument("--fname_output", type=str,default='gutenberg_rhyme_df',help="filename of the output file")
    parser.add_argument("--fname_input", type=str,default='gutenberg.csv',help="filename of the created file")

    parser.add_argument("--text_column", type=str,default='text',help="name of the column that contains the texts")
    parser.add_argument("--gold_column", type=str,default=None,help="name of the column with gold annotation")
    parser.add_argument("--rhyme_column", type=str,default='rhyme_scheme',help="column of the ai annotation")
    parser.add_argument("--author_column", type=str,default='author',help="column of the author name")
    parser.add_argument("--difference_column", type=str,default='difference',help="column for the vector differences")
    parser.add_argument("--save_every", type=int,default=10000,help="number of verses after which a checkpoint will be saved")
    parser.add_argument("--load_checkpoint", type=str,default=None,help="filename of the checkpoint to load")
    parser.add_argument("--method", type=str,default='siamese',help="method used to detect the rhymes (tts or siamese)")
    parser.add_argument("--flatten_for_stat", type=str_eval,default=True,help="statistics for all authors not separate")

    parser.add_argument("--concat_strophes", type=str_eval,default=False,help="concatenate the strophes of a poem")

    parser.add_argument("--sep", type=str,default='\n\n',help="separator of poems in text file")

    args = parser.parse_args()

    args.rhyme_column += '_'+args.method
    args.fname_output += '_'+args.method
    args.difference_column += '_'+args.method


    annotation = Annotate_rhyme(args)
    annotation.annotate_rhyme()
    annotation.save_df()
    annotation.output_statistics()


    if args.gold_column:
        annotation.compare_to_gold()
        annotation.save_df()

