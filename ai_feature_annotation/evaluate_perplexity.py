import torch
from datasets import load_dataset
from tqdm.auto import tqdm
import pandas as pd
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import numpy as np
from collections import Counter
import argparse
from utils import texts
import re

class perplexity():


    def __init__(self,model,text1=None, text2=None,path='data/perplexity',num_token=None,prep_text=False):

        self.device = torch.device(1) if torch.cuda.is_available() else torch.device("cpu")
        self.model = GPT2LMHeadModel.from_pretrained(model).to(self.device)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model)
        
        self.model_name = model
        self.text_name = text1

        with open(path+'/'+text1) as f:
            self.text = f.read()
            if prep_text:
                self.text = preprocess_text(self.text)

        if text2 is not None:
            with open(path+'/'+text2) as f:
                self.text2 = f.read()

            if prep_text:
                self.text2 = preprocess_text(self.text2)


    def compare_perp_cnt(self):
        self.perplexity_ptk()
        self.word_cnt(self.text2)
        self.cnt_lst = []
        self.nlls_lst = []
        print_lst = []
        for idx, word in enumerate(self.sorted_words):
            if word.strip().isalpha():
                cnt = self.word_reff_cnt.get(word.strip().lower())
                if cnt is None:
                    cnt = 0
                self.cnt_lst.append(cnt)
                self.nlls_lst.append(self.sorted_word_nllns[idx])
                print_lst.append((word,cnt))


        print(print_lst[:50])

    def word_cnt(self, text):
        text = re.sub('[^A-Za-zäöüÄÖÜß]', ' ',text)
        text_lst = [word.lower() for word in text.split() if word]
        self.word_reff_cnt = Counter(text_lst)

    def perplexity_corpus(self):
        

        encodings = self.tokenizer(self.text, return_tensors="pt")

        # taken from https://huggingface.co/transformers/perplexity.html
        max_length = self.model.config.n_positions
        stride = 512
        seq_len = encodings.input_ids.size(1)
        nlls = []
        prev_end_loc = 0
        processed_token = 512
        for begin_loc in tqdm(range(0, seq_len, stride)):
            end_loc = min(begin_loc + max_length, seq_len)
            trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
            input_ids = encodings.input_ids[:, begin_loc:end_loc].to(self.device)
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100
            with torch.no_grad():
                outputs = self.model(input_ids, labels=target_ids)

                # loss is calculated using CrossEntropyLoss which averages over input tokens.
                # Multiply it with trg_len to get the summation instead of average.
                # We will take average over all the tokens to get the true average
                # in the last step of this example.
                neg_log_likelihood = outputs.loss * trg_len
  
            nlls.append(neg_log_likelihood)
            
            prev_end_loc = end_loc
            processed_token += trg_len
            if end_loc == seq_len:
                break

        self.ppl = torch.exp(torch.stack(nlls).sum() / processed_token).item()
        print('model: ', self.model_name)
        print('text file: ', self.text_name)
        print("Perplexity:", self.ppl)

    def get_word_beginnings(self, tokens):
        beginnings = [0]
        for idx, token in enumerate(tokens[1:],start=1):
            chunk = self.tokenizer.decode(token)

            if not chunk.isalpha():
                beginnings.append(idx)

            else: 
                if self.tokenizer.decode(tokens[idx-1]).strip().isalpha():
                    if chunk[0] == ' ':
                        beginnings.append(idx)

                else: 
                    beginnings.append(idx)


        return beginnings

    def perplexity_ptk(self):
        

        encodings = self.tokenizer(self.text, return_tensors="pt")

        # taken from https://huggingface.co/transformers/perplexity.html
        max_length = self.model.config.n_positions
        stride = 512
        seq_len = encodings.input_ids.size(1)
        nlls = []
        prev_end_loc = 0
        for begin_loc in tqdm(range(0, seq_len)):
            end_loc = min(begin_loc + stride, seq_len)
            trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
            input_ids = encodings.input_ids[:, begin_loc:end_loc].to(self.device)
            target_ids = input_ids.clone()
            target_ids[:, :-1] = -100
            with torch.no_grad():
                outputs = self.model(input_ids, labels=target_ids)

                # loss is calculated using CrossEntropyLoss which averages over input tokens.
                # Multiply it with trg_len to get the summation instead of average.
                # We will take average over all the tokens to get the true average
                # in the last step of this example.
                neg_log_likelihood = outputs.loss 


            nlls.append(neg_log_likelihood)

            prev_end_loc = end_loc
            if end_loc == seq_len:
                break

        test_set = encodings.input_ids[0][stride:]
        word_beginnings = self.get_word_beginnings(test_set)
        words = []
        token_nllns = []

        for idx, beginning in enumerate(word_beginnings[:-1]):
            words.append(self.tokenizer.decode(test_set[beginning:word_beginnings[idx+1]]))
            token_nllns.append([val for val in nlls[beginning:word_beginnings[idx+1]]])

        word_nllns = np.asarray([torch.max(torch.vstack(val)).item() for val in token_nllns])
        sorted_idx = np.argsort(-word_nllns)

        self.sorted_words = [words[i] for i in sorted_idx ]
        self.sorted_word_nllns = [word_nllns[i] for i in sorted_idx ]
        
        ppl = torch.exp(torch.stack(nlls).sum() / torch.stack(nlls).size(0)).item()
        print("Perplexity:", ppl)

    def perplexity_sequence(self, max_len_in = 512,max_len_out = 512):
        encodings1 = self.tokenizer(self.text1 , return_tensors="pt").input_ids[:,-max_len_in:]
        encodings2 = self.tokenizer(' '+self.text2, return_tensors="pt").input_ids[:,:max_len_out]
        
        encodings = torch.cat((encodings1,encodings2),dim=1)
        print(len(encodings[0]))
        trg_len = len(encodings2)
        input_ids = encodings.to(self.device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100
        with torch.no_grad():
            outputs = self.model(input_ids, labels=target_ids)
            log_likelihood = outputs[0] * trg_len

        ppl = torch.exp(log_likelihood.sum() / trg_len).item()
        print("Perplexity:", ppl)        

def preprocess_text_lb(text):
    if type(text) == list:
        text = ' '.join(text)
    
    return ' '.join((re.sub('[^A-ZÄÖÜa-zäöüß,.!? ]', ' ', text)).split())

def preprocess_text(text):
    if type(text) == list:
        text = ' '.join(text)

    lines = []
    for line in text.split('\n'):
        lines.append(' '.join((re.sub('[^A-ZÄÖÜa-zäöüß,.!? ]', ' ', line)).split()))

    return '\n'.join(lines)


def remove_titel(text):
    text_lst = text.split('\n')

    print(len(text_lst))
    text_lst = [text for text in text_lst if 'Titel' not in text]
    print(len(text_lst))
    text = '\n'.join(text_lst)
    print(text)
    return text


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str,default='data/perplexity/evaluation_3',help="relative path to the text file to evaluate")
    parser.add_argument("--fname_input", type=str,default='gpt2_training_gutenberg_only_hoelderlin_no_train.txt',help="filename of the input file")
    parser.add_argument("--fname_input_2", type=str,default='Anjoe/poetry-gpt2-large-no_schiller',help="filename of the input file")
    parser.add_argument("--model", type=str,default='Anjoe/poetry-gpt2-large-no_schiller',help="name of the model to evaluate")
    args = parser.parse_args()

    models = ['benjamin/gerpt2-large','Anjoe/poetry-gpt2-large-complete_3']
    text_lst = ['bajohr_halbzeug_perplexity.txt','john_bock.txt','gpt2_training_gutenberg_only_goethe_no_train.txt']

    results = []
    for idx,model in enumerate(models):
        results.append([])
        for text in text_lst:

            perp = perplexity(model,text1=text, path=args.path,prep_text=True)
            perp.perplexity_corpus()
            results[idx].append(perp.ppl)

    result_df = pd.DataFrame(results,index=models,columns=text_lst)

    print(result_df.head())
    result_df.to_csv(args.path+'/evaluation_prep.csv')


    #perp = perplexity(args.model,text1=args.fname_input, path=args.path)