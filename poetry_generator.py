
import os
import glob
import re
import numpy as np 
from rhyme import find_rhyme
from rythm import fix_rythm, verse_cl

from gpt_poet import gpt_poet
from generator_utils import conv_rhyme_scheme, Logger
from perplexity import perplexity
import random

import sys




######################################
jambus = [0,1]
trochee = [1,0]
######################################


#Joseph Karl Benedikt, Freiherr von Eichendorff  Laue Luft kommt blau geflossen 
prompt_1 = ['''Laue Luft kommt blau geflossen,
Frühling, Frühling soll es sein!
Waldwärts Hörnerklang geschossen,
Mutger Augen lichter Schein;
Und das Wirren bunt und bunter
''',[10],trochee]

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
Ein Engel heut, und morgen – Nichts.''',[9],jambus]

# Johann Wolfgang von Goethe Vermächtnis
prompt_6 = ['''Kein Wesen kann zu nichts zerfallen!
Das Ewge regt sich fort in allen,
Am Sein erhalte dich beglückt!
Das Sein ist ewig: denn Gesetze
Bewahren die lebendgen Schätze,
Aus welchen sich das All geschmückt. ''',[9,10],jambus]

# Johann Wolfgang von Goethe Parabase
prompt_7 = ['''Freudig war, vor vielen Jahren,
Eifrig so der Geist bestrebt,
Zu erforschen, zu erfahren,
Wie Natur im Schaffen lebt.
Und es ist das ewig Eine,
Das sich vielfach offenbart''',[8],trochee]

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
Zu setzen, es mit Weihrauch zu bequalmen!''',[10,11],jambus]

# Friedrich Hebbel Philosophenschicksal
prompt_9 = ['''Salomons Schlüssel glaubst du zu fassen und Himmel und Erde
Aufzuschließen, da löst er in Figuren sich auf,
Und du siehst mit Entsetzen das Alphabet sich erneuern,
Tröste dich aber, es hat währende der Zeit sich erhöht.''',[12],trochee]

# Heinrich Heine Himmelfahrt
prompt_10 = ['''Die Philosophie ist ein schlechtes Metier.
 Wahrhaftig, ich begreife nie,
Warum man treibt Philosophie.
Sie ist langweilig und bringt nichts ein,
Und gottlos ist sie obendrein;''',[11],jambus]

# Friedrich Schiller Jeremiade
prompt_11 = ['''Alles in Deutschland hat sich in Prosa und Versen verschlimmert,
Ach, und hinter uns liegt weit schon die goldene Zeit!
Philosophen verderben die Sprache, Poeten die Logik.''',[12,13],trochee]

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
Und die gibt mir dann den Rest?''',[9,8],trochee]

# Antonio Cho
prompt_14 = ['''Gibt es einen Dadageist?
Ich behaupte, er sei
das große Gelächter
das einzig umfassende metaphysische Gelächter
das große Gelächter
über den Witz der Schöpfung
das große Lachen
über den Witz der eigenen Existenz.''',[10,9],jambus]

prompt_15 = ['''über den Feldhamster Karl und den Philosophen Kant:
''',[9,8],trochee]

def generate_poetry(prompt,target_rythm, num_syll_lst, rhyme_scheme, shots = 1, LLM='GPT2-large', LLM_rhyme='GPT2-large', use_tts = True,num_lines = 15):
    
    if rhyme_scheme:
        rhyme_scheme_print = rhyme_scheme
        
        rhyme_scheme = conv_rhyme_scheme(rhyme_scheme)
        
        
    else:
        rhyme_scheme_print = 'No Rhyme'
    freq = len(rhyme_scheme)  

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

        if LLM == 'GPT2-large_top_p':
            shots = 1
        if LLM == 'GPT3':
            shots = 1

        tmp_lst = []
        perp_lst = []
        is_title = False

        num_syll = num_syll_lst[cnt%len(num_syll_lst)]

        for j in range(shots):
            line = gpt_poet(input_text,target_rythm,num_syll,title_accepted,LLM=LLM)

            if line[:2] == '##':
                offset += cnt
                cnt = 0
                input_text += line +'\n'
                title = line[2:]
                title_accepted = False
                is_title = True
                break

            elif line =='**':                                   
                if j > 1 and not tmp_lst or shots == 1:                         # we could try again but  we basically tried enough             
                    return print_text, rating

            else:        
                verse_tmp = verse_cl(line)
                verse_tmp.context = input_text[-100:]
                verse_tmp = fix_rythm(verse_tmp,target_rythm,num_syll)
                perp_lst.append(perplexity(input_text[-200:] + ' '.join(verse_tmp.text)))

                tmp_lst.append(verse_tmp)

        if not is_title:
            title_accepted = False
            if perp_lst:
                best_idx = np.argmin(np.asarray(perp_lst))
                verse = tmp_lst[best_idx]

            else: 
                return print_text, rating 

            #verse = verse_cl(line)
            #verse.context = input_text[-100:]

            #verse = fix_rythm(verse,target_rythm,num_syll)
            
            verse_lst.append(verse)
            
            if rhyme_scheme and rhyme_scheme[cnt%freq] != -1:
     
                verse_lst = find_rhyme(verse_lst,offset + int(int(cnt/freq)*freq+rhyme_scheme[cnt%freq]),cnt,target_rythm,LLM=LLM_rhyme,use_tts=use_tts)
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
            rating = 1000/perplexity(print_text)
        else:
            rating = 'pending'
        print('result by: ' + str(LLM))
        print('rhyme scheme: ' + str(rhyme_scheme_print))
        print('rating: ' + str(rating))
        print(title)
        print(print_text)

    return print_text, rating


def start_poetry_generation(prompt,target_rythm, num_syll_lst, rhyme_scheme, shots = 1, loops = 1, LLM='GPT2-large', LLM_rhyme='GPT2-large', use_tts = False):

    if LLM == 'GPT3':
        prompt = 'schreibe ein Gedicht auf Deutsch ' + prompt

    if type(num_syll_lst) != list:
        num_syll_lst = [num_syll_lst]
   
    files = glob.glob("output/*.txt")
    max_idx = 0
    for file in files: 
        max_idx = max(int(re.findall(r'\d+', file)[0]),max_idx)

    start_idx = max_idx + 1

    for i in range(loops):
        text, rating = generate_poetry(prompt,target_rythm, num_syll_lst, rhyme_scheme,LLM=LLM)
    
        with open('output/poem_' + str(start_idx + i)+'.txt', 'w') as f:
            f.write(text)


if __name__ == "__main__":  
    sys.stdout = Logger()
    files = glob.glob("logs/*.log")
    max_idx = 0
    for file in files: 
        max_idx = max(int(re.findall(r'\d+', file)[0]),max_idx)

    
    start_idx = max_idx + 1
    print(start_idx)
    for i in range(1000):  
        rhyme_schemes = ['aabb','abba','abab']
        LLMS = ['GPT2-large']
        prompts = [prompt_2,prompt_3,prompt_5,prompt_7,prompt_8,prompt_10,prompt_11,prompt_12,prompt_13]
  
        prompt = random.choice(prompts)
        num_syll = prompt[1]
        rythm = prompt[2] 
        prompt_text = prompt[0] + ' \n Titel: Warum ist etwas und nicht nur nichts \n'
        
 

        #LLM = random.choice(LLMS)
        LLM = 'GPT2-large'            # debug

        if LLM == 'GPT3':
            prompt_text = 'schreibe ein Gedicht auf Deutsch \n' + prompt_text + ' \n Titel: Warum ist etwas und nicht nur nichts \n'
        rhyme_scheme = random.choice(rhyme_schemes)

        print(prompt_text)
        text, rating = generate_poetry(prompt_text,rythm, num_syll, rhyme_scheme,LLM=LLM,use_tts = False)

        print('*** final output ***')
        print('\n')
        print('rating: ' + str(rating))
        print('\n')
        print(text)
        print('\n')
        print('***')

        sys.stdout.save(start_idx+i)
       