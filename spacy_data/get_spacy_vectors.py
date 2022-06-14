import re
import spacy
import pickle

conjunctional_adverbs = '''außerdem, zudem, dazu, daneben, darüber hinaus, desgleichen, ebenso, ferner, weiter, zusätzlich,davor, währenddessen, indessen, danach, anschließend,
folglich, demzufolge, demnach, damit, somit, mithin, also, deswegen, deshalb, daher, nämlich,notfalls, sonst,ansonsten, andernfalls, gegebenenfalls, so, dann,obwohl,trotzdem, 
dennoch, dessen ungeachtet, gleichwohl, immerhin, allerdings, sowieso, nichtsdestoweniger,insofern, so weit, freilich,hingegen, dafür, dagegen, jedoch, 
doch, dennoch, indes, indessen, allerdings, nur, vielmehr, demgegenüber, stattdessen, wie'''

word_list = re.findall(r"[\w']+", conjunctional_adverbs)

nlp = spacy.load("de_core_news_lg")

vec_lst = []
for word in word_list:
    doc = nlp(word)
    vec_lst.append(doc[0].vector)


with open('conj_adv_vec.lst', 'wb') as f:
    pickle.dump(vec_lst, f)
