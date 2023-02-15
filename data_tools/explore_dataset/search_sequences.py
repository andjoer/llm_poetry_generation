from trie import Trie
import time
import glob
from multiprocessing import Pool, cpu_count, Process
import re
import itertools
import pickle


text_path = 'cc_100_vocab'

files = glob.glob(text_path+"/*.txt")

text = ''
for file in files[:2]:
    with open(file) as f:
        text += ' ' + f.read()

def preprocess_text(text):
    if type(text) == list:
        text = ' '.join(text)
    return re.sub('[^a-zäöüß ]', ' ', text.lower())

print('loaded text')

def regex_trie(keys):

    start = time.time()
    trie = Trie()
    for key in keys:
        trie.add(key)
    regex = re.compile(trie.pattern())
    found = regex.findall(text)

    current = Process()
    file_ext = str(current._identity[0])+'_'+str(current._identity[1])
    with open('sequence_search/found_keys_'+file_ext+'.pkl', 'wb') as f:
            pickle.dump(found, f)
    print(time.time() - start)
    return(found)

with open('gutenberg/gutenberg_corpus_complete.txt') as f:
    gutenberg_lines = f.readlines()

keys = []
for line in gutenberg_lines:
    if len(line.strip().split()) > 2:
        keys.append(preprocess_text(line).strip())


print(len(keys))
print(len(text))

step = 1500
workers = min(30,int(cpu_count()-6))

with Pool(workers) as p:
    start = time.time()
    results = p.map(regex_trie,[keys[start:start+step] for start in range(0,len(keys),step)])  # due to high ram usage usually the program will not manage to combine all the results at the end
                                                                                                         # but the files written by the workers itself are saved and need to be concatenated
results = list(itertools.chain.from_iterable(results))

with open('sequence_search/found_keys_all.pkl', 'wb') as f:
    pickle.dump(results, f)
    
    
print(time.time() - start)


#benchmark_regex_trie(text,keys[-1000:])
#benchmark_pyahocorasick(text,keys[-5000:])