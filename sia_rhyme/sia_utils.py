import io

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
from PIL import Image
from tensorboardX import SummaryWriter
from torch import nn
from tqdm import tqdm
from sia_models import ContrastiveLoss, vec_distance
import sia_hyperparameters as hp
import os

def tokenize(text):
            return [char for char in text]

def sequence_to_text(sequence, field):
    return " ".join([field.vocab.itos[int(i)] for i in sequence])

def sequence_to_final(sequence, field):
    output = []
    for i in sequence: 
        
        token = field.vocab.itos[int(i)]
        if token == '<eos>':
            break
        else: output.append(token)
    return "".join(output)

def word_to_tensor(vocab, word):
    word = tokenize(word[::-1])

    sequence = [2]+[vocab.vocab.stoi[char] for char in word]+[3]
    pad = hp.max_len - len(sequence)
    padder = torch.nn.ConstantPad1d((0,pad),1)
    

    #torch.nn.functional.pad(sequence, pad, mode='constant', value=1)
    inp_tensor = torch.LongTensor(sequence).to(hp.device)
    inp_tensor = padder(inp_tensor)
    inp_tensor = torch.reshape(inp_tensor,(len(inp_tensor),1))
    
    return inp_tensor

def text_to_sequence(text, field):
    return [field.vocab.stoi[word] for word in text]

def predict(model, vocab, word1, word2):
    word1_inp = torch.transpose(word_to_tensor(vocab, word1),0,1)
    word2_inp = torch.transpose(word_to_tensor(vocab, word2),0,1)
    model.eval()
    output_1, output_2 = model(word1_inp,word2_inp)
    distance = vec_distance(output_1,output_2)
    pred = (distance<0.7).float()

    return pred

def get_distance(model, vocab, word1, word2):
    word1_inp = torch.transpose(word_to_tensor(vocab, word1),0,1)
    word2_inp = torch.transpose(word_to_tensor(vocab, word2),0,1)
    model.eval()

    output_1, output_2 = model(word1_inp,word2_inp)

    distance = vec_distance(output_1,output_2).cpu().detach().numpy()

    return distance

def get_distance_vec(model, vocab, word1, vector):
    word1_inp = torch.transpose(word_to_tensor(vocab, word1),0,1)
    model.eval()
    output_1 = model(word1_inp)
    distance = vec_distance(output_1,vector).cpu().detach().numpy()

    return distance

def word_to_vec(model, vocab, word, numpy = False):
    word_inp = torch.transpose(word_to_tensor(vocab, word),0,1)
    model.eval()
    vec = model(word_inp)
    if numpy: 
        return vec.cpu().detach().numpy()
    else:
        return vec


def evaluate(model, val_iter, writer, step):
    model.eval()
    criterion = ContrastiveLoss()

    step = 0
    eval_loss_total = 0
    acc_total = 0
    
    batch_sum = 0
    for i, batch in enumerate(val_iter):
        batch_sum += len(batch)
        
        batch.word1 = torch.transpose(batch.word1,0,1) 
        batch.word2 = torch.transpose(batch.word2,0,1) 
        output_1, output_2 = model(batch.word1, batch.word2)

        
        distance = vec_distance(output_1,output_2)
  
        pred = (distance<0.5).float()

        acc = torch.sum((pred==(batch.trg)).float())
        acc_total = acc_total +acc
        
        step +=1

        
    return (acc_total/batch_sum*100).item()





def train(model, optimizer, train_iter, val_iter, num_epochs,labels, words, fpath, step=0):

    eval_accs = [0]
    writer = SummaryWriter()
    criterion = ContrastiveLoss()

    step = 0
    acc = 0
    for i in range(num_epochs):
        print('start')
        model.train()
        model.zero_grad()
        total_loss = 0
        step2 = 0
        #pbar = tqdm(train_iter, total=len(train_iter), unit=' batches')
        pbar = tqdm(train_iter, total=len(train_iter), unit=' batches')
        for b, batch in enumerate(pbar):
                optimizer.zero_grad()

                batch.word1 = torch.transpose(batch.word1,0,1)    ##//
                batch.word2 = torch.transpose(batch.word2,0,1)    ##//    

                output_1, output_2 = model(batch.word1, batch.word2)
               
                loss = criterion(output_1,output_2,1-batch.trg)
                loss.backward()

                nn.utils.clip_grad_norm_(model.parameters(), 10.0, norm_type=2)
               
                optimizer.step()
                pbar.set_description(f'loss: {loss.item():.4f}')
                writer.add_scalar('loss', loss.item(), step)
                writer.add_scalar('accuracy', acc, step)
                #writer.add_scalar('lr', scheduler.lr, step)
                step += 1
                step2 += 1
                total_loss += loss.item()
         
        
        print('epoch:', i)
        print(total_loss/len(pbar))
        
        print('validation')
        acc = evaluate(model, val_iter, writer, step)
        
        print("accuracy : ", acc, '%')
        if acc > max(eval_accs):
            print('validation accuracy improved, model saved')
            torch.save(model.state_dict(), os.path.join(fpath, f'checkpoints/rhyme_model_{step}.pt'))
        else:
            print('validation accuracy did not improve')

        eval_accs.append(acc)

    return(model)