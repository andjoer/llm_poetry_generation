import torch
from torch import nn
from torch import Tensor
import sia_hyperparameters as hp

import torch.nn.functional as F
  
def exponent_neg_manhattan_distance(vec_1, vec_2):
        dist = torch.exp(-torch.sum(torch.abs(vec_1-vec_2),dim=1))

        return dist

def vec_distance(output1, output2):


    distance = 1- F.cosine_similarity(output1, output2)  

    #distance = 1 - exponent_neg_manhattan_distance(output1,output2)  

    return distance

class ContrastiveLoss(torch.nn.Module):
    

      def __init__(self, margin=1.05):    
            super(ContrastiveLoss, self).__init__()
            self.margin = margin

      def forward(self, output1_, output2_, label):
            # Find the pairwise distance or eucledian distance of two output feature vectors
            

            output1 = output1_
            output2 = output2_


            euclidean_distance = vec_distance(output1,output2)
       
            loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
            (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

            return loss_contrastive

class SiameseRNN(nn.Module):
    def __init__(self, source_vocab_size, embed_dim, hidden_dim,
                 n_layers, dropout, state_dict = None):
        super(SiameseRNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.embed = nn.Embedding(source_vocab_size, embed_dim, padding_idx=hp.pad_idx)
        self.rnn = nn.LSTM(embed_dim, hidden_dim, n_layers,
                          dropout=hp.dropout, bidirectional=hp.bidirectional)
       
       
        if hp.bidirectional: 
           bidim = 2
        else: 
            bidim = 1
        
        
        
        # Setting up the Fully Connected Layers
        self.fc1 = nn.Sequential(

            nn.Linear(bidim*hidden_dim*hp.max_len, 128),
            #nn.Linear(256, 128),

        )

    ##//def forward(self, word_1,word_2, hidden):
    def forward(self, word_1 ,*args):                   # args is word_2; it is not needed for embedding    
                                                        # we don't use the DRY approach for readability
        if args:
            word_2 = args[0]
  
            emb_1 = self.embed(word_1)  
            emb_2 = self.embed(word_2)
        
            encoder_1_out , encoder_1_hidden = self.rnn(
                emb_1) 
            encoder_2_out , encoder_2_hidden = self.rnn(
                emb_2) 
    
            
    
            #encoder_1_out = torch.mean(encoder_1_out,1)   #temporal average method 1
            #encoder_2_out = torch.mean(encoder_2_out,1)   #temporal average
            
            encoder_1_out = torch.reshape(encoder_1_out[:,-hp.max_len:,:],(encoder_1_out.size()[0],-1))
            encoder_2_out = torch.reshape(encoder_2_out[:,-hp.max_len:,:],(encoder_1_out.size()[0],-1))
            
            output1 = self.fc1(encoder_1_out)
            output2 = self.fc1(encoder_2_out)
           
    
            return output1, output2 

        else:                                                   
            emb = self.embed(word_1)      
            encoder_out , encoder_hidden = self.rnn(
                emb) 
          
            #encoder_out = torch.mean(encoder_out,1)   #temporal average
            encoder_out = torch.reshape(encoder_out[:,-20:,:],(encoder_out.size()[0],-1))

            output = self.fc1(encoder_out)
    
            return output
    
    

    
