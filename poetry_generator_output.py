import glob
import re
from time import sleep
import os
import argparse


fast_display = ['result','rhyme','rating','using','---']
def output_file(file,start_gen = 4,factor = 1):
    with open(file) as f:
        lines = f.readlines()

    start_print = False
    for idx, line in enumerate(lines):
        
        if '*** final output ***' in line:
            break

        print(line[:-1])
        
        if idx > start_gen:
            if line.split():                                        # no letter
                if line.split()[0] not in fast_display:
                    sleep(2/factor)
                else:
                    sleep(0.3/factor)
       
        
    
    print_text = ''
    for line in lines[idx:]:
        print_text += line 

    print('only for test, this needs to be send to the printer:')
    print(print_text)


    for line in lines[idx:]:
        print(line[:-1])
        sleep(0.5/factor)



    

if __name__ == "__main__":  
    parser = argparse.ArgumentParser()
    

    parser.add_argument("--start", type=int,default=0,help="input the file index to start with")
    parser.add_argument("--timelapse", type=int,default=1,help="a factor how much faster the program should run")

    args = parser.parse_args()
    
    start_index = args.start
    factor = args.timelapse
    files = glob.glob("logs/*.txt")

    indexed_files = []
    for file in files: 
        indexed_files.append([file,int(re.findall(r'\d+', file)[0])])

    for file, index in indexed_files:
        if index > start_index:
            print(index)
            output_file(file,factor=factor)

