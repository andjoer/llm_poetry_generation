from torch.optim import Adam
import torch

#import hyperparams as hp
import hyperparams as hp
import os
from  models import Encoder, Decoder, Seq2Seq
from sgdr import SGDRScheduler
from utils import train, translate, return_vectors


from torchtext.legacy.data import Field, BucketIterator, TabularDataset

def tokenize(text):
        return [char for char in text]

class ortho_to_ipa:
    def __init__(self,load = True, path = '', fpath = 'data', fname_ortho='model/ortho.pt',fname_ipa='model/ipa.pt',fname_model='model/ortho_to_ipa.pt'):

        self.path = path
        if load: 
            self.load_vocabs(os.path.join(path, fname_ortho),os.path.join(path, fname_ipa))
            self.load_model(fname_model)
        else:
            self.fpath = os.path.join(path, fpath)
            self.get_training_data()

     

    def init_model(self):
        self.encoder = Encoder(source_vocab_size=len(self.ortho.vocab),
                  embed_dim=hp.embed_dim, hidden_dim=hp.hidden_dim,
                  n_layers=hp.n_layers, dropout=hp.dropout)
        self.decoder = Decoder(target_vocab_size=len(self.ipa.vocab),
                        embed_dim=hp.embed_dim, hidden_dim=hp.hidden_dim,
                        n_layers=hp.n_layers, dropout=hp.dropout)
        self.seq2seq = Seq2Seq(self.encoder, self.decoder)

        self.seq2seq.to(hp.device)


    def get_training_data(self):
       
        batch_size = hp.batch_size
        device = hp.device

        self.ortho = Field(tokenize=tokenize, init_token='<sos>', eos_token='<eos>')
        self.ipa = Field(tokenize=tokenize, init_token='<sos>', eos_token='<eos>')

        fields = {'words':('src',self.ortho),'ipa':('trg',self.ipa)}

        train_data, val_data = TabularDataset.splits(path=self.fpath,
                                                    train='train.csv',
                                                    test='val.csv',
                                                    format='csv',
                                                    fields=fields)

        self.ortho.build_vocab(train_data.src)
        self.ipa.build_vocab(train_data.trg)

        self.train_iter, self.val_iter = BucketIterator.splits(
                (train_data, val_data), batch_size=batch_size, device=device, 
            repeat=False,
            sort_key = lambda x: len(x.src),
            sort_within_batch=True)
       

    def load_vocabs(self,fname_ortho='vocab/ortho.pt',fname_ipa='vocab/ipa.pt'):
        self.ortho = torch.load(fname_ortho)
        self.ipa = torch.load(fname_ipa)
    

    def save_vocabs(self,fname_ortho='vocab/ortho.pt',fname_ipa='vocab/ipa.pt'):
        torch.save(self.ortho,fname_ortho)
        torch.save(self.ipa,fname_ipa)

    def train(self):
        self.init_model()
        optimizer = Adam(self.seq2seq.parameters(), lr=hp.max_lr)
        scheduler = SGDRScheduler(optimizer, max_lr=hp.max_lr, cycle_length=hp.cycle_length)

        self.seq2seq = train(self.seq2seq, optimizer, scheduler, self.train_iter, self.val_iter, num_epochs=hp.num_epochs)

    def load_model(self,fname):
        self.init_model()
        self.seq2seq.load_state_dict(torch.load(os.path.join(self.path, fname)))

    def translate(self,text):
        prediction = translate(self.seq2seq,self.ortho,self.ipa,text)
        return prediction

    def get_vectors(self,text):
        prediction, pred_idx, output,encoder_hidden, encoder_out = return_vectors(self.seq2seq,self.ortho,self.ipa,text)

        return prediction, pred_idx.cpu().detach().numpy()[:,0], output.cpu().detach().numpy(),encoder_hidden.cpu().detach().numpy(),encoder_out.cpu().detach().numpy()[:,0,:]
    def get_vocab(self, vocabulary = 'ipa'):
        
        if vocabulary == 'ipa':
            vocab = self.ipa.vocab
        else: vocab = self.ortho.vocab
        
        return vocab.itos

        
