
import os
import glob
import re
import numpy as np 
from rhyme import find_rhyme
from rythm import fix_rythm, verse_cl
import copy 
import argparse

from gpt2 import LLM_class
from gpt_poet import gpt_poet
from generator_utils import conv_rhyme_scheme, Logger
from perplexity import perplexity
import random
import torch

import sys

from transformers import pipeline
from transformers import GPT2Tokenizer, GPT2LMHeadModel


######################################
jambus = [0,1]                        # defining the metric patterns
trochee = [1,0]
######################################


#Joseph Karl Benedikt, Freiherr von Eichendorff  Laue Luft kommt blau geflossen 
prompt_1 = ['''Laue Luft kommt blau geflossen,
Frühling, Frühling soll es sein!
Waldwärts Hörnerklang geschossen,
Mutger Augen lichter Schein;
Und das Wirren bunt und bunter
''',[10],trochee]                                     # a prompt is a list of a text followed by the syllable list and the meter

# Friedrich Schiller Die Künstler
prompt_2 = ['''Nur durch das Morgentor des Schönen
Drangst du in der Erkenntnis Land.
An höhern Glanz sich zu gewöhnen,
Übt sich am Reize der Verstand.
''',[9,8],jambus]

# Friedrich Schiller Die Künstler
prompt_3 = ['''Als der Erschaffende von seinem Angesichte
Den Menschen in die Sterblichkeit verwies
Und eine späte Wiederkehr zum Lichte
Auf schwerem Sinnenpfad ihn finden hieß,
''',[11],jambus]

# Friedrich Schiller Genialität
prompt_4 = ['''Wodurch gibt sich der Genius kund? Wodurch sich der Schöpfer
Kund gibt in der Natur, in dem unendlichen All:
Klar ist der Äther und doch von unermeßlicher Tiefe;
Offen dem Aug, dem Verstand bleibt er doch ewig geheim.
''',[12,13],trochee]

# Johannes Daniel Falk An das Nichts
prompt_5 = ['''Selbst philosophische Systeme –
Kants Lieblingsjünger, Reinhold, spricht’s –
Von Plato bis auf Jakob Böhme,
Sie waren samt und sonders – Nichts.

Was bin ich selbst? – Ein Kind der Erde,
Der Schatten eines Traumgesichts,
Der halbe Weg von Gott zum Werde,
Ein Engel heut, und morgen – Nichts.
''',[9],jambus]

# Johann Wolfgang von Goethe Vermächtnis
prompt_6 = ['''Kein Wesen kann zu nichts zerfallen!
Das Ewge regt sich fort in allen,
Am Sein erhalte dich beglückt!
Das Sein ist ewig: denn Gesetze
Bewahren die lebendgen Schätze,
Aus welchen sich das All geschmückt.
''',[9,10],jambus]

# Johann Wolfgang von Goethe Parabase
prompt_7 = ['''Freudig war, vor vielen Jahren,
Eifrig so der Geist bestrebt,
Zu erforschen, zu erfahren,
Wie Natur im Schaffen lebt.
Und es ist das ewig Eine,
Das sich vielfach offenbart
''',[8],trochee]

# Giacomo Graf Leopardi Palinodie an den Marchese Gino Capponi
prompt_8 = ['''O Geist, o Einsicht, Scharfsinn, übermenschlich,
Der Zeit, in der wir leben! Welches sichre
Philosophiren, welche Weisheit lehrt
In den geheimsten, höchsten, feinsten Dingen
Den kommenden Jahrhunderten das unsre!
Mit welcher ärmlichen Beständigkeit
Wirft heut der Mensch vor das, was gestern er
Verspottet, sich auf's Knie, um morgen wieder
Es zu zertrümmern, dann aufs neu die Trümmer
Zu sammeln, es auf den Altar zurück
Zu setzen, es mit Weihrauch zu bequalmen!
''',[10,11],jambus]

# Friedrich Hebbel Philosophenschicksal
prompt_9 = ['''Salomons Schlüssel glaubst du zu fassen und Himmel und Erde
Aufzuschließen, da löst er in Figuren sich auf,
Und du siehst mit Entsetzen das Alphabet sich erneuern,
Tröste dich aber, es hat währende der Zeit sich erhöht.
''',[12],trochee]

# Heinrich Heine Himmelfahrt
prompt_10 = ['''Die Philosophie ist ein schlechtes Metier.
 Wahrhaftig, ich begreife nie,
Warum man treibt Philosophie.
Sie ist langweilig und bringt nichts ein,
Und gottlos ist sie obendrein;
''',[11],jambus]

# Friedrich Schiller Jeremiade
prompt_11 = ['''Alles in Deutschland hat sich in Prosa und Versen verschlimmert,
Ach, und hinter uns liegt weit schon die goldene Zeit!
Philosophen verderben die Sprache, Poeten die Logik.
''',[12,13],trochee]

# Robert Gernhardt, Trost und Rat
prompt_12 = ['''Ja wer wird denn gleich verzweifeln,
weil er klein und laut und dumm ist?
Jedes Leben endet. Leb so,
daß du, wenn dein Leben um ist

von dir sagen kannst: Na wenn schon!
Ist mein Leben jetzt auch um,
habe ich doch was geleistet:
ich war klein und laut und dumm.
''',[8,9],trochee]

# Robert Gernhardt, Ach!
prompt_13 =['''Woran soll es gehn? Ans Sterben?
Hab ich zwar noch nie gemacht,
doch wir werd’n das Kind schon schaukeln —
na, das wäre ja gelacht!

Interessant so eine Sanduhr!
Ja, die halt ich gern mal fest.
Ach – und das ist Ihre Sense?
Und die gibt mir dann den Rest?
''',[9,8],trochee]

# Antonio Cho
prompt_14 = ['''Gibt es einen Dadageist?
Ich behaupte, er sei
das große Gelächter
das einzig umfassende metaphysische Gelächter
das große Gelächter
über den Witz der Schöpfung
das große Lachen
über den Witz der eigenen Existenz.
''',[10,9],jambus]

prompt_15 = ['''über den Feldhamster Karl und den Philosophen Kant:
''',[9,8],trochee]


def get_LLM_name(LLM):
    if type (LLM) == str:
        LLM_name = LLM
    else:
        LLM_name = LLM.model_name

    return LLM_name


def generate_poetry(args, 
                    shots = 1, 
                    LLM=None, 
                    LLM_rhyme=None, 
                    LLM_2  = None,
                    LLM_perplexity = None):
    '''
    contains the main loop that generates the peoms
    takes: 
    prompt: the prompt that will be forwarded to the llm
    target_rythm: the target rythm (jambus or trochee)
    num_syll_lst: a list that contains the number of syllables in each verse. Once the list ends it will be read 
                 from the beginning again
    rhyme_scheme: the rhyme scheme
    shots: number of candidate verses to generate for each resulting verse
    LLM: the llm that should be used 
    LLM_rhyme: the llm that should be used to generate alternative verse endings
    use_tts: Use the mffc-feature method in order to check the results from the sia-rhyme method
    num_lines: number of verses to generate
    '''


    prompt = args.prompt
    target_rythm = args.target_rythm
    num_syll_lst = args.num_syll_list
    rhyme_scheme = args.rhyme_scheme
    num_lines = args.generated_lines

    if rhyme_scheme:
        rhyme_scheme_print = rhyme_scheme
        
        rhyme_scheme = conv_rhyme_scheme(rhyme_scheme)
        
        
    else:
        rhyme_scheme_print = 'No Rhyme'

    freq = len(rhyme_scheme)  # number of verses per strophe

    input_text = prompt

    lines = []

    verse_lst = []
    input_text_0 = input_text
    print_text = ''
    cnt = 0
    title = ''
    title_accepted = True
    offset = 0
    rating = 'pending'
    for i in range(num_lines):

        shots = args.verse_versions

        tmp_lst = []
        perp_lst = []
        is_title = False

        num_syll = num_syll_lst[cnt%len(num_syll_lst)]

        for j in range(shots):
            line = gpt_poet(args,input_text,num_syll,title_accepted,LLM=LLM,LLM_2=LLM_2)  # generate a new verse

            if line[:2] == '##':                                          # no text has been created, but maybe a new title
                offset += cnt
                cnt = 0
                input_text += line +'\n'
                title = line[2:]
                title_accepted = False
                is_title = True
                break

            elif line =='**':                                   
                if (j > 1 and not tmp_lst) or shots == 1:                         # we could try again but  we basically tried enough  -> stop           
                    return print_text, rating

            else:        
                verse_tmp = verse_cl(line)
                verse_tmp.context = input_text[-400:]
                verse_tmp = fix_rythm(verse_tmp,target_rythm,num_syll,LLM_perplexity)          # fix the rythm of the generated verse

                perplexity_check_text = '\n'.join(input_text.split('\n')[-2:]) + ' '.join((re.sub('[^a-zA-ZäöüÄÖÜß ]','',' '.join(verse_tmp.text)).split()))
                
                if args.check_end == True: 
                    perplexity_check_text += '.'

                perp = perplexity(perplexity_check_text,LLM_perplexity)
                perp_lst.append(perp) # measure the perplexity of the verse

                print('generated verse: ')
                print(' '.join(verse_tmp.text))
                print('perplexity: ' + str(perp))
                tmp_lst.append(verse_tmp)

        if not is_title:
            title_accepted = False
            if perp_lst:                                            
                best_idx = np.argmin(np.asarray(perp_lst))          # choose the candidate verse with the least perplexity
                verse = tmp_lst[best_idx]

            else: 
                return print_text, rating 

            verse_lst.append(verse)
            
            if rhyme_scheme and rhyme_scheme[cnt%freq] != -1:  # if rhyme partner already exists
     
          
                verse_lst = find_rhyme(args,verse_lst,offset + int(int(cnt/freq)*freq+rhyme_scheme[cnt%freq]),cnt,LLM_perplexity,LLM=LLM_rhyme,LLM2=LLM_2)
                input_text = input_text_0
                print_text = ''
                for verse in verse_lst:
                    input_text += ' '.join(verse.text) + '\n'
                    print_text += ' '.join(verse.text) + '\n'

            else:
                input_text = input_text + str(verse.doc).strip() + ' \n'
                print_text += str(verse.doc).strip() + ' \n'

            cnt += 1

        if print_text:
            rating = 1000/perplexity(print_text,LLM_perplexity)
        else:
            rating = 'pending'

    
        print('result by: ' + str(get_LLM_name(LLM)))
        print('rhyme scheme: ' + str(rhyme_scheme_print))
        print('rating: ' + str(rating))
        print(title)
        print(print_text)

    return print_text, rating


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str,default=None,help="initial input prompt")
    parser.add_argument("--title", type=str,default='Die Regierung',help="title of the poem") 
    parser.add_argument("--generated_lines", type=int,default=8,help="number of lines that will be generated")
    parser.add_argument("--verse_versions", type=int,default=1,help="number of versions for one verse will be generated; the one with lowest perplexity will be chosen")
    parser.add_argument("--check_end", type=bool,default=True,help="append '.' after verse to check perplexity") # experimental, might help to avoid invalid end of verses

    parser.add_argument("--LLM", type=str,default='Anjoe/german-poetry-gpt2-large',help="generative language model to use from the huggingface library or GPT3")
    parser.add_argument("--LLM_sampling", type=str,default='systematic',help="sampling method for gpt2 models - systematic or multinomial")
    parser.add_argument("--LLM_random_first", type=bool,default=True,help="mix the top p filtered logits at the first position by multinomial sampling")
    parser.add_argument("--LLM_random_all", type=str,default=True,help="mix the top p filtered logits at every position by multinomial sampling")
    parser.add_argument("--LLM_temperature", type=int,default=0.9,help="sampling temperature for systematic verse sampling")
    parser.add_argument("--trunkate_after", type=int,default=50,help="number of tries after which the search beam will be trunkated when sampling = systematic")
    parser.add_argument("--LLM_top_p", type=int,default=0.6,help="top p filter value used if sampling = systematic for the initial verse")
    parser.add_argument("--syllable_count_toll", type=int,default=0.65,help="precentage of the allowed difference between target syllables and delivered syllables by gpt_poet")
    parser.add_argument("--dividable_rest", type=bool,default=True,help="if the number of pending syllables left by gpt_poet need to be devidable by the length of the target rythm")   # then bert would not need to find a different ending
    parser.add_argument("--verse_stop_tokens", type=list,default=['\n','.'],help="list of tokens after which a verse could end (only applies when sampling = systematic)")
    parser.add_argument("--verse_alpha_only_after", type=int,default=4,help="blocks non alphabetic tokens close to the verse ending which is calculated by num_syll*num_syll_toll-verse_alpha_only_after")


    parser.add_argument("--LLM_2", type=str,default=None,help="model to replace words with certain pos tags")
    parser.add_argument("--LLM_2_pos", type=list,default=['NOUN','PROPN'],help="pos token of the words that should be replaced by the second language model")
    parser.add_argument("--LLM_2_sampling", type=str,default='systematic',help="sampling method for the second language model - systematic or multinomial")
    parser.add_argument("--top_p_dict_replace", type=dict,default={0:0.8,1:0.4},help="top p dictionary used for the words replaced by the second model")

    parser.add_argument("--LLM_rhyme", type=str,default=None,help="generative language model to use from the huggingface library or gpt3")
    parser.add_argument("--LLM_rhyme_sampling", type=str,default='systematic',help="sampling method for the rhyme model - systematic or multinomial")
    parser.add_argument("--rhyme_temperature", type=int,default=1,help="sampling temperature for rhyming words sampling")
    
    parser.add_argument("--use_pos_rhyme_syns", type=bool,default=True,help="synonyms with the same pos tokens are allowed when looking for rhymes (only if sampling = systematic)")
    parser.add_argument("--top_p_dict_rhyme", type=dict,default={0:0.65,2:0.5},help="top p dictionary used to find rhyming alternatives for a single word")
    parser.add_argument("--top_p_rhyme", type=int,default=0.6,help="top p value used to find rhyming alternatives for longer sequences")
    parser.add_argument("--max_rhyme_dist", type=int,default=0.5,help="maximum siamese vector distances of two words in order to be considered rhyming")
    parser.add_argument("--rhyme_stop_tokens", type=list,default=['\n','.'],help="list of tokens after which a verse could end (only applies when sampling = systematic)")  

    parser.add_argument("--rhyme_scheme", type=str,default=None,help="rhyme scheme for the created poem")
    parser.add_argument("--num_syll_list", type=list,default=None,help="list of the syllable count of each line; when more lines than items in the list are generated it iterates")
    parser.add_argument("--target_rythm", type=str,default=None,help="rythm of the poem: jambus or trochee")
    parser.add_argument("--use_tts", type=bool,default=False,help="use also text to speech to fine-select the best rhyming pair")
    parser.add_argument("--use_colone_phonetics", type=bool,default=False,help="if a rhyme is detected by using colone phonetics, prefer this one over sia rhyme/tts")
    parser.add_argument("--allow_pos_match", type=bool,default=True,help="Ignores the end of an alternative verse ending if the pos tags match the original verse")

    parser.add_argument("--log_stdout", type=bool,default=True,help="if a rhyme is detected by using colone phonetics, prefer this one over sia rhyme/tts")
    
    args = parser.parse_args()

    return args

def initialize_llms(args):

    if torch.cuda.device_count() >= 1:

        LLM_device = 'cuda:0'
    else: 
        LLM_device = 'cpu'

    default_llm = 'Anjoe/german-poetry-gpt2-large'
    if args.LLM == 'GPT2-large':                                                        # backwards compatibility
        LLM = LLM_class(default_llm,sampling='multinomial')


    if len(args.LLM) > 5:          # ohterwise it is string for an api, not a huggingface link
        LLM = LLM_class(args.LLM,device=LLM_device,sampling=args.LLM_sampling)
        
  
    if args.LLM_2:
        if torch.cuda.device_count() > 1 and type(LLM) != str:
            LLM_2_device =  'cuda:1'
        elif type(LLM) == str and torch.cuda.device_count() == 1:
            LLM_2_device = 'cuda:0'
        else: 
            LLM_2_device = 'cpu'
        LLM_2 = LLM_class(args.LLM_2,device=LLM_2_device,sampling=args.LLM_2_sampling)

    else:
        LLM_2 = None

    if LLM == 'GPT3':
        args.prompt = 'schreibe ein Gedicht auf Deutsch \n' + args.prompt

    if LLM_2 and not args.LLM_rhyme:
        if args.LLM_rhyme_sampling == 'multinomial' or args.LLM_2_sampling == 'multinomial':
            LLM_rhyme = LLM_class(LLM_2.model_name,sampling=args.LLM_rhyme_sampling, device='cpu')
        else:
            LLM_rhyme = LLM

    elif not args.LLM_rhyme and args.LLM_rhyme_sampling != 'multinomial':
        if (LLM.sampling != args.LLM_rhyme_sampling and args.LLM_rhyme_sampling) and torch.cuda.device_count() > 1:
            LLM_rhyme = LLM_class(LLM.model_name,sampling=args.LLM_rhyme_sampling, device='cuda:1')

        elif (LLM.sampling == args.LLM_rhyme_sampling and args.LLM_rhyme_sampling) and args.LLM_rhyme_sampling == 'systematic':
            LLM_rhyme = LLM

        elif torch.cuda.device_count() > 1:             
            LLM_rhyme = LLM_class(LLM.model_name,sampling=args.LLM_rhyme_sampling,device='cuda:1')
        else: 
            LLM_rhyme = LLM_class(LLM.model_name,sampling=args.LLM_rhyme_sampling,device='cpu')

    
    elif not args.LLM_rhyme and args.LLM_rhyme_sampling == 'multinomial':
        LLM_rhyme = LLM_class(LLM.model_name,sampling=args.LLM_rhyme_sampling,device='cpu')

    else: 
        if args.LLM_rhyme_sampling == 'multinomial':
            LLM_rhyme = LLM_class(args.LLM_rhyme,sampling=args.LLM_rhyme_sampling,device='cpu')

        elif not LLM_2 and type(LLM) == str and torch.cuda.device_count() > 0:                        # LLM via API
            LLM_rhyme = LLM_class(args.LLM_rhyme,sampling=args.LLM_rhyme_sampling,device = 'cuda:0')

        elif not LLM_2 and torch.cuda.device_count() > 1:
            LLM_rhyme = LLM_class(args.LLM_rhyme,sampling=args.LLM_rhyme_sampling,device = 'cuda:1') 

        elif LLM_2 and torch.cuda.device_count() > 1 and type(LLM) == str:
            LLM_rhyme = LLM_class(args.LLM_rhyme,sampling=args.LLM_rhyme_sampling,device = 'cuda:1') 
        else:
            LLM_rhyme = LLM_class(args.LLM_rhyme,sampling=args.LLM_rhyme_sampling,device = 'cpu')


    LLM_perplexity = None

    LLMs = [LLM,LLM_2,LLM_rhyme]

    gpu_lst = []
    free_gpu = ''

    for item in LLMs:
        if type(item) != str and item: 
            gpu_lst.append(item.device)

    for i in range(torch.cuda.device_count()):

        if 'cuda:'+str(i) not in gpu_lst:
            free_gpu = 'cuda:'+str(i)
            break 
    if not free_gpu:
        perplexity_device = 'cpu'
    else: 
        perplexity_device = free_gpu

    if LLM_2:
        LLM_perplexity = LLM_2
    else:
        if args.LLM_rhyme_sampling == 'systematic' and args.LLM_sampling != 'systematic':
            LLM_perplexity = LLM_rhyme
        elif type(LLM) != str: 
            LLM_perplexity = LLM

    if not LLM_perplexity:
        LLM_perplexity = LLM_class(default_llm,device=perplexity_device)

    elif LLM_perplexity.sampling != 'systematic':
        LLM_perplexity = LLM_class(LLM_perplexity.model_name,device=perplexity_device)

    print('perp device')
    print(LLM_perplexity.device)

    return LLM, LLM_perplexity, LLM_rhyme, LLM_2


if __name__ == "__main__":  
    '''
    start the program
    
    '''
    args = parse_arguments()


    sys.stdout = Logger()
    files = glob.glob("logs/*.log")
    max_idx = 0
    for file in files: 
        max_idx = max(int(re.findall(r'\d+', file)[0]),max_idx)    # find the number of the last log file

    
    start_idx = max_idx + 1
    print(start_idx)

    

    if args.target_rythm == 'jambus':
        args.target_rythm = jambus
    elif args.target_rythm == 'trochee':
        args.target_rythm = trochee
    elif args.target_rythm: 
        print('rythm not supported, using jambus')
        args.target_rythm = jambus

    prompt_0 = args.prompt
    title_0 = args.title
    rhyme_scheme_0 = args.rhyme_scheme
    num_syll_lst_0 = args.num_syll_list
    target_rythm_0 = args.target_rythm

    LLM, LLM_perplexity, LLM_rhyme, LLM_2 = initialize_llms(args)

    for i in range(1000):  
        rhyme_schemes = ['aabb','abba','abab','']              # rhyme schemes to sample from

        args.prompt = prompt_0
        prompts = [prompt_2,prompt_3,prompt_5,prompt_7,prompt_8,prompt_10,prompt_11,prompt_12,prompt_13] # prompt to sample from
  
        prompt = random.choice(prompts)
        num_syll = prompt[1]                       # the metric properties are defined in the prompt
        rythm = prompt[2] 
        prompt_text = prompt[0] # + ' \n Titel: Warum ist etwas und nicht nur nichts \n' # title of the created poem

        if not rhyme_scheme_0: 
            args.rhyme_scheme = random.choice(rhyme_schemes)
        
        if (not prompt_0 and not title_0) or prompt_0 == 'ran': 
            prompt = random.choice(prompts)
            
            if title_0:
                title = 'titel: ' + title_0 + '\n'
            else: 
                title = ''
            args.prompt = prompt[0] + '\n' +title

            if not num_syll_lst_0:
                args.num_syll_list = prompt[1]
            if not target_rythm_0:
                args.target_rythm = prompt[2]
        else: 
            if not prompt_0: 
                args.prompt = 'titel: ' + args.title + '\n'
            
            if not num_syll_lst_0:
                args.num_syll_list = [10,11]
            if not target_rythm_0:
                args.target_rythm = jambus
            

        print('LLM: ' + str(get_LLM_name(LLM)))
        print('LLM_rhyme: ' + str(get_LLM_name(LLM_rhyme)))

        if LLM_2:
            print('LLM_2: ' + LLM_2.model_name)
        print(args.prompt)
        
        text, rating = generate_poetry(args,LLM=LLM,LLM_rhyme=LLM_rhyme,LLM_2=LLM_2,LLM_perplexity=LLM_perplexity)

        print('*** final output ***')
        print('\n')
        print('rating: ' + str(rating))
        print('\n')
        print(text)
        print('\n')
        print('***')

       
        sys.stdout.save(start_idx+i)
        