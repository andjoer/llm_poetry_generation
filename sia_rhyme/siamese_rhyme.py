from torch.optim import Adam
import torch

from . utils import train, sequence_to_text, text_to_sequence, predict, word_to_vec, get_distance, get_distance_vec
from . import hyperparameters as hp

from . models import SiameseRNN, vec_distance

import os
#from sgdr import SGDRScheduler
#from utils import train, translate, return_vectors

def tokenize(text):
            return [char for char in text]

from torchtext.legacy.data import LabelField, Field, BucketIterator, TabularDataset

class siamese_rhyme:
    def __init__(self,load = True, fname_data='data', fname_vocab='model/vocab.pt',fname_model='model/rhyme_model.pt'):
        
        self.fpath = os.path.dirname(__file__)
        self.fpath_vocab = os.path.join(self.fpath, fname_vocab)
        self.fpath_data = os.path.join(self.fpath, fname_data)
        self.fpath_model = os.path.join(self.fpath, fname_model)

        if not load: 
            self.get_training_data(self.fpath_data)
            self.init_model()
            

        else:
            self.load_vocabs()
            self.load_model()
            

            pass
    

    def init_model(self):
        self.Siamese = SiameseRNN(source_vocab_size=len(self.words.vocab),
                  embed_dim=hp.embed_dim, hidden_dim=hp.hidden_dim,
                  n_layers=hp.n_layers, dropout=hp.dropout)

        self.Siamese.to(hp.device)
        
        #self.Siamese.init_weights()


    def model_test(self):
        output = 1

    def get_training_data(self,fpath):
       
        batch_size = hp.batch_size
        device = hp.device

        self.words = Field(tokenize=tokenize, init_token='<sos>', eos_token='<eos>',fix_length=20)
        self.labels = LabelField(dtype=torch.long, batch_first=True, sequential=False)


        fields = {'word1':('word1',self.words),'word2':('word2',self.words),'rhyme':('trg',self.labels)}

        train_data, val_data = TabularDataset.splits(path=fpath,
                                                    train='train.csv',
                                                    #test='val3.csv',
                                                    test='val.csv',
                                                    format='csv',
                                                    fields=fields)

        self.words.build_vocab(train_data.word1)
        self.labels.build_vocab(train_data.trg)

        self.train_iter, self.val_iter = BucketIterator.splits(
                (train_data, val_data), batch_size=batch_size, device=device, 
            repeat=False,
            sort_key = lambda x: len(x.word1),
            sort_within_batch=True)
       
        print(text_to_sequence('0',self.labels))


    def train(self):
    
        #optimizer = Adam(self.Siamese.parameters(), lr=0.1)
        optimizer = Adam(self.Siamese.parameters())
       

        self.Siamese = train(self.Siamese, optimizer, self.train_iter, self.val_iter, hp.num_epochs,self.labels,self.words,self.fpath)

    def predict(self, word1,word2):
        return predict(self.Siamese,self.words, word1,word2)

    def get_word_vec(self, word,numpy=False):
        return word_to_vec(self.Siamese,self.words, word,numpy)

    def get_distance(self, word1,word2):
        return get_distance(self.Siamese, self.words, word1,word2)

    def get_distance_vec(self, word1,vector):
        return get_distance_vec(self.Siamese, self.words, word1,vector)

    def load_model(self):
        self.init_model()
        self.Siamese.load_state_dict(torch.load(self.fpath_model))

    def save_vocabs(self):
        torch.save(self.words,self.fpath_vocab)

    def load_vocabs(self):

        self.words = torch.load(self.fpath_vocab)
    
    def vector_distance(self,vector_1,vector_2):
        return vec_distance(vector_1,vector_2).cpu().detach().numpy()

    def get_vocab(self, vocabulary = 'vocab'):
        
        if vocabulary == 'vocab':
            vocab = self.vocab.vocab

        return vocab.itos

        
