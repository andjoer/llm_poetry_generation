import pandas as pd
import itertools
import os


def rhyming_words(rhyme_df,file,path='data'):

    ''' Takes a dataframe that contains annotated strophes of poems and returns shuffled annotaded pairs 
        of rhyming and non rhyming words. 

        Args: 

        rhyme_df (df): Dataframe of annotated strophes. The name of the column with the strophes is 'text' 
                       the column with the annotation is "rhyme"
        file (str): name of the filename for the output
        path (str): filepath to the output file

    '''
    file_path = os.path.join(path,file) 
    rhyme_pairs = []
    no_rhyme_pairs = []


    for index, row in rhyme_df.iterrows():
        
        scheme = row['rhyme']
        sent = row['text']

        new_pairs = []
        letters = set(scheme)
        for letter in letters:
            indices = [idx for idx, char in enumerate(scheme) if char == letter]
            
            if len(indices) >= 2: 
                try:
                    new_pair = [sent[i].split()[-1] for i in indices]
                    new_pairs.append(new_pair)
                        
                    if len(indices) > 2:
                        rhyme_pairs += [list(item) for item in list(itertools.combinations(new_pair,2))]
                    elif len(indices) == 2:
                        rhyme_pairs.append(new_pair)
                except: 
                    print(indices)
                    print(sent)
                    print(scheme)

            else:
                pass
            
        for idx in range(len(new_pairs)-1):                                          # ugly, but does the job fast
            try:
                no_rhyme_pairs.append([new_pairs[idx][0],new_pairs[idx+1][0]])
                no_rhyme_pairs.append([new_pairs[idx][1],new_pairs[idx+1][1]])
            except:
                pass

    pairs = rhyme_pairs + no_rhyme_pairs
    rhyme_stat = [1]*len(rhyme_pairs) + [0]*len(no_rhyme_pairs)
    for pair in pairs:
        
        if len(pair) < 1:
            print(pair)

    word_1 = [pair[0] for pair in pairs]
    word_2 = [pair[1] for pair in pairs]

    rhyme_word_df = pd.DataFrame(list(zip(word_1, word_2,rhyme_stat)),
                columns =['word1', 'word2','rhyme'])

    shuffled_df = rhyme_word_df.sample(frac=1).reset_index(drop=True)
    shuffled_df.to_csv(file_path, index=False)
    
    return rhyme_word_df

    def shuffle_df(df,frac = 0.75):
        shuffled_df = rhyme_word_df.sample(frac=1).reset_index(drop=True)

        