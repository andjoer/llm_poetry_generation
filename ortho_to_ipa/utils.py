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

import hyperparams as hp
from decoding_helpers import Greedy, Teacher, Greedy_vectors

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

def text_to_tensor(ortho,ipa, text):
    text = tokenize(text)
    text = ['<sos>'] + text + ['<eos>']

    sequence = text_to_sequence(text,ortho)
    inp_tensor = torch.LongTensor(sequence).to(hp.device)
    inp_tensor = torch.reshape(inp_tensor,(len(inp_tensor),1))
    return inp_tensor

def text_to_sequence(text, field):
    return [field.vocab.stoi[word] for word in text]

def translate(model,ortho,ipa, text):

    inp_tensor = text_to_tensor(ortho,ipa, text)

    model.eval()
    greedy = Greedy()
    greedy.set_maxlen(hp.max_len_translate)
    outputs, attention,_,_ = model(inp_tensor, greedy)
    preds = outputs.topk(1)[1]
    prediction = sequence_to_final(preds[:, 0].data, ipa)

    return prediction

def return_vectors(model, ortho,ipa,text):

    inp_tensor = text_to_tensor(ortho,ipa, text)

    model.eval()
    greedy = Greedy_vectors()
    greedy.set_maxlen(hp.max_len_translate)
    outputs, attention, hidden, encoder_out = model(inp_tensor, greedy)
    preds = outputs.topk(1)[1]
    prediction = sequence_to_final(preds[:, 0].data, ipa)
    length = len(prediction)

    return prediction, preds[:, 0].data, outputs[:length,:,:],hidden, encoder_out

def evaluate(model, val_iter, writer, step):
    model.eval()
    total_loss = 0
    fields = val_iter.dataset.fields
    greedy = Greedy()
    random_batches = []
    for i in range (3):
        random_batch = np.random.randint(0, len(val_iter) - 1)
        random_batches.append(random_batch)


    for i, batch in enumerate(val_iter):
        greedy.set_maxlen(len(batch.trg[1:]))
        outputs, attention,_,_ = model(batch.src, greedy)
        seq_len, batch_size, vocab_size = outputs.size()
        loss = F.cross_entropy(outputs.view(seq_len * batch_size, vocab_size),
                               batch.trg[1:].view(-1),
                               ignore_index=hp.pad_idx)
        total_loss += loss.item()

        # tensorboard logging
        if i in random_batches:
            preds = outputs.topk(1)[1]
            source = sequence_to_text(batch.src[:, 0].data, fields['src'])
            prediction = sequence_to_text(preds[:, 0].data, fields['trg'])
            target = sequence_to_text(batch.trg[1:, 0].data, fields['trg'])
            attention_plot = show_attention(attention[0],
                                            prediction, source, return_array=True)

            #writer.add_image('Attention', attention_plot, step)
            writer.add_text('Source: ', source, step)
            writer.add_text('Prediction: ', prediction, step)
            writer.add_text('Target: ', target, step)
    eval_loss = total_loss / len(val_iter)
    writer.add_scalar('val_loss', eval_loss, step)
    return eval_loss


def train(model, optimizer, scheduler, train_iter, val_iter,
          num_epochs, teacher_forcing_ratio=0.5, step=0):
    
    eval_losses = []
    writer = SummaryWriter()
    teacher = Teacher(teacher_forcing_ratio)
    for _ in tqdm(range(num_epochs), total=num_epochs, unit=' epochs'):
        model.train()
        pbar = tqdm(train_iter, total=len(train_iter), unit=' batches')
        for b, batch in enumerate(pbar):
            optimizer.zero_grad()

            teacher.set_targets(batch.trg)

            outputs, masks,_,_ = model(batch.src, teacher)
            seq_len, batch_size, vocab_size = outputs.size()
            loss = F.cross_entropy(outputs.view(seq_len * batch_size, vocab_size),
                                   batch.trg[1:].view(-1),
                                   ignore_index=hp.pad_idx)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 10.0, norm_type=2)  # prevent exploding grads
            scheduler.step()
            optimizer.step()
            # tensorboard logging
            pbar.set_description(f'loss: {loss.item():.4f}')
            writer.add_scalar('loss', loss.item(), step)
            writer.add_scalar('lr', scheduler.lr, step)
            step += 1
      
        eval_loss = evaluate(model, val_iter, writer, step)
        eval_losses.append(eval_loss)

        if eval_loss <= min(eval_losses):
            print('val loss improved, model saved')
            torch.save(model.state_dict(), f'checkpoints/seq2seq_{step}.pt')
        else:
            print('val loss not improved')

    return(model)



def show_attention(attention, prediction=None, source=None, return_array=False):
    plt.figure(figsize=(14, 6))
    sns.heatmap(attention,
                xticklabels=prediction.split(),
                yticklabels=source.split(),
                linewidths=.05,
                cmap="Blues")
    plt.ylabel('Source (Orthographic)')
    plt.xlabel('Prediction (IPA)')
    plt.xticks(rotation=60)
    plt.yticks(rotation=0)
    if return_array:
        plt.tight_layout()
        buff = io.BytesIO()
        plt.savefig(buff, format='png')
        buff.seek(0)
        return np.array(Image.open(buff))
