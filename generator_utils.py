import re
import numpy as np
import sys

def conv_rhyme_scheme(rhyme_scheme):
    scheme = -np.ones(len(rhyme_scheme))
    for char in ''.join(set(rhyme_scheme)):
        found = [char_.start(0) for char_ in re.finditer(char, rhyme_scheme)]
        for idx,_ in enumerate(found[:-1]): 
           
            scheme[found[idx+1]] = found[idx] 

    return scheme.tolist()

class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        self.log = ''
   
    def write(self, message):
        self.terminal.write(message)
        self.log += message   

    def save(self,i):
        with open('logs/poem_' + str(i)+'.log', 'w') as f:
             f.write(self.log)
             self.log = ''

    def flush(self):

        pass  

