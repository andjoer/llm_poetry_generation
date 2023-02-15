from transformers import AutoTokenizer
import argparse, ast, re

from gpt2 import LLM_class

import torch
import pandas as pd

def str_eval(string):
    return ast.literal_eval(str(string))

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--poem", type=str,default=None,help="the poem that should be modified")
    parser.add_argument("--fix_meter", type=str_eval,default=False,help="fixes the meter of the input poem")
    parser.add_argument("--replace_linebreaks", type=str_eval,default=True,help="replaces linebreaks with spaces in the prompts")

    parser.add_argument("--vocab", type=str,default=None,help="restrict words to a vocabulary when sampling is systematic; relative path to the vocabulary in csv format")
    parser.add_argument("--check_vocab_all", type=str_eval,default=False,help="if false only the first word of each line is restricted to the vocabularyidimode")

    parser.add_argument("--repetition_penalty", type=float,default=1.2,help="repetition penalty according to CTRL paper")

    parser.add_argument("--LLM_rhyme", type=str,default=None,help="ge--force_rhymenerative language model to use from the huggingface library or gpt3")
    parser.add_argument("--bidirectional_model", type=str,default='Anjoe/gbert-large',help="Bidirectional model for bidirectional synonyms/inserted words")

    parser.add_argument("--LLM_rhyme_sampling", type=str,default='systematic',help="sampling method for the rhyme model - systematic or multinomial")
    parser.add_argument("--rhyme_temperature", type=float,default=0.9,help="sampling temperature for rhyming words sampling")
    
    parser.add_argument("--use_pos_rhyme_syns", type=str_eval,default=True,help="synonyms with the same pos tokens are allowed when looking for rhymes (only if sampling = systematic)")
    parser.add_argument("--top_p_dict_rhyme", type=str_eval,default={0:0.65,2:0.5},help="top p dictionary used to find rhyming alternatives for a single word")
    parser.add_argument("--top_p_rhyme", type=float,default=None,help="top p value used to find rhyming alternatives for longer sequences")
    parser.add_argument("--max_rhyme_dist", type=float,default=None,help="maximum siamese vector distances of two words in order to be considered rhyming")
    parser.add_argument("--rhyme_stop_tokens", type=str_eval,default=[',','.','!','?',';',':'],help="list of tokens after which a verse could end (only applies when sampling = systematic)")
    parser.add_argument("--invalid_verse_ends", type=str_eval,default=['CONJ','CCONJ'],help="pos tokens that should not appear at the end of a verse")

    parser.add_argument("--rhyme_scheme", type=str,default=None,help="rhyme scheme for the created poem")
    parser.add_argument("--target_rythm", type=str,default=None,help="rythm of the poem: jambus or trochee")

    parser.add_argument("--use_tts", type=str_eval,default=False,help="use also text to speech to fine-select the best rhyming pair")
    parser.add_argument("--size_tts_sample", type=int,default=10,help="number of best candidats forwarded by sia rhyme to the tts algorithm")
    parser.add_argument("--use_colone_phonetics", type=str_eval,default=False,help="if a rhyme is detected by using colone phonetics, prefer this one over sia rhyme/tts")
    parser.add_argument("--rhyme_last_two_vowels", type=str_eval,default=False,help="if a rhyme is not detected by colone phonetics but has the last two vowels in common it is considered a rhyme and prefered over sia rhyme/tts")
    parser.add_argument("--allow_pos_match", type=str_eval,default=True,help="Ignores the end of an alternative verse ending if the pos tags match the original verse")

    parser.add_argument("--log_stdout", type=str_eval,default=True,help="if a rhyme is detected by using colone phonetics, prefer this one over sia rhyme/tts")
    
    args = parser.parse_args()


    bi_tokenizer = AutoTokenizer.from_pretrained(args.bidirectional_model)
    args.mask_tok = str(bi_tokenizer.mask_token)

    if args.vocab: 
        args.vocab = pd.read_csv(args.vocab).iloc[:, 0].tolist()

    if not args.max_rhyme_dist:
        if not args.use_tts:
            args.max_rhyme_dist = 0.35

        else: 
            args.max_rhyme_dist = 0.5

    #LLM dependent defaults

    if args.LLM_rhyme_sampling != 'systematic':
        args.top_p_rhyme = 1 
    else: 
        args.top_p_rhyme = 0.65

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

    
    if args.LLM_rhyme == 'GPT2-large':                                                        # backwards compatibility
        LLM_rhyme = LLM_class(default_llm,sampling=args.LLM_rhyme_sampling,device=LLM_device)

    else:
        LLM_rhyme = LLM_class(default_llm,sampling=args.LLM_rhyme_sampling,device=LLM_device)


    LLM_perplexity = LLM_rhyme

    return LLM_rhyme, LLM_perplexity

