from transformers import AutoTokenizer
import argparse, ast

from gpt2 import LLM_class

import torch
import pandas as pd

def str_eval(string):
    return ast.literal_eval(str(string))

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str,default=None,help="initial input prompt")
    parser.add_argument("--title", type=str,default='Die Regierung',help="title of the poem") 
    parser.add_argument("--add_title_indicator", type=str_eval,default=True,help="add the word titel in front of the title")
    parser.add_argument("--generated_lines", type=int,default=12,help="number of lines that will be generated")
    parser.add_argument("--generated_poems", type=int,default=1000,help="number of poems that will be generated")
    parser.add_argument("--replace_linebreaks", type=str_eval,default=False,help="replaces linebreaks with spaces in the prompts")

    parser.add_argument("--vocab", type=str,default=None,help="restrict words to a vocabulary when sampling is systematic; relative path to the vocabulary in csv format")
    parser.add_argument("--check_vocab_all", type=str_eval,default=False,help="if false only the first word of each line is restricted to the vocabulary")

    parser.add_argument("--verse_versions", type=int,default=1,help="number of versions for one verse will be generated; the one with lowest perplexity will be chosen")
    parser.add_argument("--check_end", type=str_eval,default=False,help="append '.' after verse to check perplexity") # experimental, might help to avoid invalid end of verses
    parser.add_argument("--invalid_verse_ends", type=str_eval,default=['CONJ','CCONJ'],help="pos tokens that should not appear at the end of a verse")
    parser.add_argument("--repetition_penalty", type=float,default=1.2,help="repetition penalty according to CTRL paper")

    parser.add_argument("--LLM", type=str,default='Anjoe/german-poetry-gpt2-large',help="generative language model to use from the huggingface library or GPT3")
    parser.add_argument("--LLM_sampling", type=str,default='systematic',help="sampling method for gpt2 models - systematic or multinomial")
    parser.add_argument("--LLM_random_first", type=str_eval,default=True,help="mix the top p filtered logits at the first position by multinomial sampling")
    parser.add_argument("--LLM_random_all", type=str,default=True,help="mix the top p filtered logits at every position by multinomial sampling")
    parser.add_argument("--LLM_temperature", type=float,default=0.9,help="sampling temperature for systematic verse sampling")
    parser.add_argument("--trunkate_after", type=int,default=150,help="number of tries after which the search beam will be trunkated when sampling = systematic")
    parser.add_argument("--LLM_top_p", type=float,default=None,help="top p filter value used if sampling = systematic for the initial verse")
    parser.add_argument("--syllable_count_toll", type=float,default=0.65,help="precentage of the allowed difference between target syllables and delivered syllables by gpt_poet")
    parser.add_argument("--dividable_rest", type=str_eval,default=True,help="if the number of pending syllables left by gpt_poet need to be devidable by the length of the target rythm")   # then bert would not need to find a different ending
    parser.add_argument("--verse_stop_tokens", type=str_eval,default=[',','.','!','?',';',':'],help="list of tokens after which a verse could end (only applies when sampling = systematic)")
    parser.add_argument("--verse_alpha_only_after", type=int,default=4,help="blocks non alphabetic tokens close to the verse ending which is calculated by num_syll*num_syll_toll-verse_alpha_only_after")

    parser.add_argument("--bidirectional_model", type=str_eval,default=['Anjoe/gbert-large','Anjoe/german-poetry-xlm-roberta'],help="Bidirectional model for bidirectional synonyms/inserted words")

    parser.add_argument("--LLM_2", type=str,default=None,help="model to replace words with certain pos tags")
    parser.add_argument("--LLM_2_pos", type=str_eval,default=['NOUN','PROPN','VERB'],help="pos token of the words that should be replaced by the second language model")
    parser.add_argument("--LLM_2_sampling", type=str,default='systematic',help="sampling method for the second language model - systematic or multinomial")
    parser.add_argument("--LLM_2_temperature", type=float,default=0.9,help="temperature for the second LLM")
    parser.add_argument("--LLM_2_top_p", type=float,default=0.7,help="top p for the second LLM when the sampling is multinomial")
    parser.add_argument("--top_p_dict_replace", type=str_eval,default={0:0.8,1:0.4},help="top p dictionary used for the words replaced by the second model")

    parser.add_argument("--LLM_rhyme", type=str,default=None,help="ge--force_rhymenerative language model to use from the huggingface library or gpt3")
    parser.add_argument("--LLM_rhyme_sampling", type=str,default='systematic',help="sampling method for the rhyme model - systematic or multinomial")
    parser.add_argument("--rhyme_temperature", type=float,default=0.9,help="sampling temperature for rhyming words sampling")
    
    parser.add_argument("--use_pos_rhyme_syns", type=str_eval,default=True,help="synonyms with the same pos tokens are allowed when looking for rhymes (only if sampling = systematic)")
    parser.add_argument("--top_p_dict_rhyme", type=str_eval,default={0:0.75,2:0.5},help="top p dictionary used to find rhyming alternatives for a single word")
    parser.add_argument("--top_k_rhyme", type=int,default=40,help="top p dictionary used to find rhyming alternatives for a single word")

    parser.add_argument("--top_p_rhyme", type=float,default=None,help="top p value used to find rhyming alternatives for longer sequences")
    parser.add_argument("--max_rhyme_dist", type=float,default=None,help="maximum siamese vector distances of two words in order to be considered rhyming")
    parser.add_argument("--rhyme_stop_tokens", type=str_eval,default=[',','.','!','?',';',':'],help="list of tokens after which a verse could end (only applies when sampling = systematic)")
    parser.add_argument("--force_rhyme", type=str_eval,default=False,help="list of tokens after which a verse could end (only applies when sampling = systematic)")  
    parser.add_argument("--failed_rhyme_count_limit", type=int,default=3,help="number of alternative verses is created if no rhyme is found when force_rhyme is enabled")
    parser.add_argument("--sample_rhymes_independent", type=str_eval,default=False,help="sample the first and the last verse end of a rhyme-pair independently from each other")  


    parser.add_argument("--rhyme_scheme", type=str,default=None,help="rhyme scheme for the created poem")
    parser.add_argument("--num_syll_list", type=str_eval,default=None,help="list of the syllable count of each line; when more lines than items in the list are generated it iterates")
    parser.add_argument("--target_rythm", type=str,default=None,help="rythm of the poem: jambus or trochee")
    parser.add_argument("--use_tts", type=str_eval,default=False,help="use also text to speech to fine-select the best rhyming pair")
    parser.add_argument("--max_tts_dist", type=float,default=9.5,help="maximum spectral vector distances of two words in order to be considered rhyming")
    parser.add_argument("--size_tts_sample", type=int,default=10,help="number of best candidats forwarded by sia rhyme to the tts algorithm")
    parser.add_argument("--use_colone_phonetics", type=str_eval,default=False,help="if a rhyme is detected by using colone phonetics, prefer this one over sia rhyme/tts")
    parser.add_argument("--rhyme_last_two_vowels", type=str_eval,default=False,help="if a rhyme is not detected by colone phonetics but has the last two vowels in common it is considered a rhyme and prefered over sia rhyme/tts")
    parser.add_argument("--allow_pos_match", type=str_eval,default=True,help="Ignores the end of an alternative verse ending if the pos tags match the original verse")

    parser.add_argument("--log_stdout", type=str_eval,default=True,help="if a rhyme is detected by using colone phonetics, prefer this one over sia rhyme/tts")
    
    args = parser.parse_args()

    if type(args.bidirectional_model) == str:
        args.bidirectional_model = [args.bidirectional_model]
    
    args.mask_tok = []
    for model_id, model in enumerate(args.bidirectional_model): 
        bi_tokenizer = AutoTokenizer.from_pretrained(model)
        args.mask_tok.append(str(bi_tokenizer.mask_token))

    if args.vocab: 
        args.vocab = pd.read_csv(args.vocab).iloc[:, 0].tolist()

    if not args.max_rhyme_dist:
        if not args.use_tts:
            args.max_rhyme_dist = 0.35
        elif args.force_rhyme:
            args.max_rhyme_dist = 0.35
        else: 
            args.max_rhyme_dist = 0.6

    #LLM dependent defaults

    if args.LLM_2 and not args.LLM_rhyme: 
        args.LLM_rhyme = args.LLM_2

    if not (args.LLM_rhyme or args.LLM_2): 
        if len(args.LLM) <= 5:                                        # API
            args.LLM_rhyme = args.LLM
            args.LLM_rhyme_sampling = 'multinomial'

    if args.LLM_rhyme_sampling != 'systematic':
        args.top_p_rhyme = 1 
    else: 
        args.top_p_rhyme = 0.5

    if len(args.LLM) < 5:
        args.LLM_sampling = 'multinomial'

    if args.LLM_sampling == 'multinomial' and not args.LLM_top_p:
        args.LLM_top_p = 1
    elif args.LLM_sampling != 'multinomial' and not args.LLM_top_p:
        args.LLM_top_p = 0.4


    return args

def get_LLM_name(LLM):
    if type (LLM) == str:
        LLM_name = LLM
    else:
        LLM_name = LLM.model_name

    return LLM_name

    
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
    else: LLM = args.LLM
  
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
        if len(args.LLM_rhyme) <= 5:
            LLM_rhyme = args.LLM_rhyme
        elif args.LLM_rhyme_sampling == 'multinomial':
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
    elif type(LLM) != str: 
        LLM_perplexity = LLM

    if not LLM_perplexity:
        LLM_perplexity = LLM_class(default_llm,device=perplexity_device)


    return LLM, LLM_perplexity, LLM_rhyme, LLM_2

